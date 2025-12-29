import torch
import torch.nn as nn

from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

from hyper_lrrnn.rnn import LowRankRNN, LowRankRNNWithReadout
from hyper_lrrnn.gen.mmd import compute_rbf_mmd_median_heuristic
from hyper_lrrnn.gen.sliced_wasserstein import sliced_wasserstein_distance


class Regularizer(nn.Module):
    def __init__(self, pr_weight=0.0, orth_weight=0.0, nll_weight=0.0, dist_weight=0.0, kurt_weight=0.0, num_mixtures=1, ema_gamma=0.9, dist_type="mmd"):
        super().__init__()
        self.pr_weight = pr_weight
        self.orth_weight = orth_weight
        self.nll_weight = nll_weight
        self.kurt_weight = kurt_weight
        self.num_mixtures = num_mixtures
        self.ema_gamma = ema_gamma
        self.dist_weight = dist_weight
        assert dist_type in ["mmd", "swd"], "dist_type must be either 'mmd' or 'swd'"
        self.dist_type = dist_type
        assert num_mixtures >= 1, "num_mixtures must be at least 1"
        if kurt_weight > 0:
            assert num_mixtures == 1, "kurtosis regularization only supported for single Gaussian EMA"

        if self.nll_weight > 0 or self.dist_weight > 0 or self.kurt_weight > 0:
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
    
    def update_ema(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return
        if self.num_mixtures == 1:
            self._update_ema_gaussian(model)
        else:
            self._update_ema_mog(model)
    
    def _update_ema_gaussian(self, model):
        params = torch.cat([model.m, model.n, model.I], dim=-1)
        new_mean = params.mean(dim=0).detach()
        new_cov = ((params - new_mean).T @ (params - new_mean) / params.shape[0]).detach()
        if self.mu is None:
            self.mu = new_mean
            self.cov = new_cov
        else:
            self.mu = self.ema_gamma * self.mu + (1 - self.ema_gamma) * new_mean
            self.cov = self.ema_gamma * self.cov + (1 - self.ema_gamma) * new_cov
    
    def _update_ema_mog(self, model):
        params = torch.cat([model.m, model.n, model.I], dim=-1)
        gm = GaussianMixture(n_components=self.num_mixtures, covariance_type='full')
        gm.fit(params.detach())
        device = model.m.device
        new_mix_weights = torch.tensor(gm.weights_, device=device, dtype=torch.float32)
        new_means = torch.tensor(gm.means_, device=device, dtype=torch.float32)
        new_covs = torch.tensor(gm.covariances_, device=device, dtype=torch.float32)
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
    
    def ema_gaussian_nll(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        params = torch.cat([model.m, model.n, model.I], dim=-1)
        nll = -torch.distributions.MultivariateNormal(self.mu, self.cov).log_prob(params).mean()
        return nll
    
    def ema_mog_nll(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        params = torch.cat([model.m, model.n, model.I], dim=-1)
        nll = -torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(self.mix_weights),
            torch.distributions.MultivariateNormal(self.means, self.cov)
        ).log_prob(params).mean()
        return nll

    def ema_distribution_distance(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        params = torch.cat([model.m, model.n, model.I], dim=-1)
        if self.num_mixtures == 1:
            with torch.no_grad():
                gm_samples = torch.distributions.MultivariateNormal(self.mu, self.cov).sample((params.shape[0],))
        else:
            with torch.no_grad():
                gm_samples = torch.distributions.MixtureSameFamily(
                    torch.distributions.Categorical(self.mix_weights),
                    torch.distributions.MultivariateNormal(self.means, self.cov)
                ).sample((params.shape[0],))
        if self.dist_type == "mmd":
            dist = compute_rbf_mmd_median_heuristic(params, gm_samples)
        else:  # swd
            dist = sliced_wasserstein_distance(params, gm_samples, device=params.device)
        return dist
    
    def ema_kurtosis(self, model):
        if not isinstance(model, (LowRankRNN, LowRankRNNWithReadout)):
            return 0.0
        params = torch.cat([model.m, model.n, model.I], dim=-1)
        diffs = params - self.mu
        inv_cov = torch.linalg.pinv(self.cov)
        mahalanobis_sq = torch.einsum('bi,ij,bj->b', diffs, inv_cov, diffs)
        kurtosis = mahalanobis_sq ** 2
        kurtosis = kurtosis.mean()
        return torch.square(kurtosis - (params.shape[1] * (params.shape[1] + 2)))
      
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
        if self.nll_weight > 0.0 or self.dist_weight > 0.0 or self.kurt_weight > 0.0:
            self.update_ema(model)
        if self.nll_weight > 0.0:
            if self.num_mixtures == 1:
                nll = self.ema_gaussian_nll(model)
            else:
                nll = self.ema_mog_nll(model)
            loss += self.nll_weight * nll
        if self.dist_weight > 0.0:
            dist = self.ema_distribution_distance(model)
            loss += self.dist_weight * dist
        if self.kurt_weight > 0.0:
            kurt = self.ema_kurtosis(model)
            loss += self.kurt_weight * kurt
        return loss
