import torch
import torch.nn as nn

from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

from hyper_lrrnn.rnn import LowRankRNN, LowRankRNNWithReadout


class Regularizer(nn.Module):
    def __init__(self, pr_weight=0.0, orth_weight=0.0, nll_weight=0.0, num_mixtures=1, ema_gamma=0.9):
        super().__init__()
        self.pr_weight = pr_weight
        self.orth_weight = orth_weight
        self.nll_weight = nll_weight
        self.num_mixtures = num_mixtures
        self.ema_gamma = ema_gamma

        if self.nll_weight > 0:
            self.register_buffer('mu', None)
            self.register_buffer('cov', None)
            self.register_buffer('mix_weights', None)
    
    def participation_ratio(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        norms = model.m.norm(dim=0) * model.n.norm(dim=0)
        pr = torch.square(torch.sum(norms)) / torch.sum(torch.square(norms)).clamp(min=1e-8)
        return pr

    def input_latent_orthogonality(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        dots = model.m.T @ model.I  # shape (rank, input_size)
        norms = model.m.norm(dim=0)[:, None] * model.I.norm(dim=0)[None, :]  # shape (rank, input_size)
        sim = dots / norms.clamp(min=1e-8)  # shape (rank, input_size)
        orth = torch.square(sim).mean()
        return orth
    
    def ema_gaussian_nll(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        params = torch.cat([model.m, model.n, model.I], dim=-1)
        new_mean = params.mean(dim=0).detach()
        new_cov = ((params - new_mean).T @ (params - new_mean) / params.shape[0]).detach()
        if self.mu is None:
            self.mu = new_mean
            self.cov = new_cov
        else:
            self.mu = self.ema_gamma * self.mu + (1 - self.ema_gamma) * new_mean
            self.cov = self.ema_gamma * self.cov + (1 - self.ema_gamma) * new_cov
        nll = torch.distributions.MultivariateNormal(self.mu, self.cov).log_prob(params).mean()
        return nll
    
    def ema_mog_nll(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        params = torch.cat([model.m, model.n, model.I], dim=-1)
        gm = GaussianMixture(n_components=self.num_mixtures, covariance_type='full')
        gm.fit(params.detach())
        device = model.m.device
        new_mix_weights = torch.tensor(gm.weights_, device=device)
        new_means = torch.tensor(gm.means_, device=device)
        new_covs = torch.tensor(gm.covariances_, device=device)
        if self.mix_weights is None:
            self.mix_weights = new_mix_weights
            self.mix_weights = self.mix_weights / self.mix_weights.sum()
            self.means = new_means
            self.cov = new_covs
        else:
            dists = torch.cdist(self.means, new_means)
            assignment = linear_sum_assignment(dists.detach().cpu())[1]
            self.mix_weights = self.mix_weights * (1 - self.ema_gamma) + new_mix_weights[assignment] * self.ema_gamma
            self.mix_weights = self.mix_weights / self.mix_weights.sum()
            self.means = self.means * (1 - self.ema_gamma) + new_means[assignment] * self.ema_gamma
            self.cov = self.cov * (1 - self.ema_gamma) + new_covs[assignment] * self.ema_gamma
        nll = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(self.mix_weights),
            torch.distributions.MultivariateNormal(self.means, self.cov)
        ).log_prob(params).mean()
        return nll
      
    def forward(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        loss = 0.0
        if self.pr_weight > 0.0:
            pr = self.participation_ratio(model)
            loss += self.pr_weight * pr
        if self.orth_weight > 0.0:
            orth = self.input_latent_orthogonality(model)
            loss += self.orth_weight * orth
        if self.nll_weight > 0.0:
            if self.num_mixtures == 1:
                nll = self.ema_gaussian_nll(model)
            else:
                nll = self.ema_mog_nll(model)
            loss += self.nll_weight * nll
        return loss