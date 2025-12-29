import math
import numpy as np
import torch
import torch.nn as nn


def is_psd(mat):
    return bool((mat == mat.transpose(-2, -1)).all() and (torch.linalg.eigvals(mat).real>=0).all())


class TrainableMixtureLowRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rank=2, num_mixtures=1, alpha=0.1, activation="tanh", output_size=None, base_scale=500):
        super(TrainableMixtureLowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_mixtures = num_mixtures
        self.alpha = alpha
        self.base_scale = base_scale

        self.mixture_weights = nn.Parameter(torch.ones(num_mixtures) / num_mixtures, requires_grad=True)
        # self._mixture_sizes = np.diff(np.round(np.linspace(0, hidden_size, num_mixtures + 1)[1:]), prepend=0).astype(int)

        self.means = nn.Parameter(torch.zeros(num_mixtures, 2 * rank + input_size))

        init_cov = None
        retry_count = 0
        while (init_cov is None or not is_psd(init_cov)) and retry_count < 5:
            # sample random symmetric matrix
            init_cov = torch.rand(num_mixtures, 2 * rank + input_size, 2 * rank + input_size)
            init_cov = init_cov + init_cov.transpose(-2, -1)  # uniform[0, 2]
            init_cov = init_cov / 2 - 0.5  # uniform[-0.5, 0.5]
            # ensure positive diagonal entries
            init_cov += torch.eye(2 * rank + input_size, 2 * rank + input_size)[None, ...]
            init_cov /= math.sqrt(hidden_size)
            # enforce positive real eigenvalues
            out = torch.linalg.eigh(init_cov)
            init_cov = (out.eigenvectors @ torch.diag_embed(torch.abs(out.eigenvalues - 1e-12) + 1e-12) @ out.eigenvectors.transpose(-2, -1))
            # symmetrize again
            init_cov = 0.5 * (init_cov + init_cov.transpose(-2, -1) )
            retry_count += 1
        if retry_count >= 5:
            raise ValueError("Failed to initialize positive definite covariance matrices.")
        init_tril = torch.linalg.cholesky(init_cov)
        self.scale_tril = nn.Parameter(init_tril)
        self._clamp_tril()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.last_m = None
        self.last_n = None
        self.last_I = None

    def _clamp_tril(self):
        """Clamp matrix to be lower triangular and have positive diagonal elements."""
        with torch.no_grad():
            clamped = torch.zeros_like(self.scale_tril)
            for i in range(self.num_mixtures):
                matrix = self.scale_tril.data[i]
                tril_matrix = torch.tril(matrix, diagonal=-1)
                tril_matrix += torch.diagflat(torch.abs(torch.diagonal(matrix) - 1e-12) + 1e-12)
                clamped[i] = tril_matrix
            self.scale_tril.data.copy_(clamped)
    
    def _normalize_weights(self):
        """Normalize mixture weights to sum to 1."""
        with torch.no_grad():
            weights = self.mixture_weights.data
            weights = torch.clamp(weights, min=1e-6)  # prevent negative or zero weights
            weights = weights / torch.sum(weights)
            self.mixture_weights.data.copy_(weights)

    def forward(self, x):
        """Forward pass

        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
        """
        self._clamp_tril()
        self._normalize_weights()
        # sample weight matrices
        m = []
        n = []
        I = []
        for i in range(self.num_mixtures):
            mean = self.means[i]
            scale_tril = self.scale_tril[i]
            mixture_size = self.hidden_size
            samp = torch.distributions.MultivariateNormal(
                loc=mean, scale_tril=scale_tril).rsample((mixture_size,))
            m_mixture = samp[:, :self.rank]
            n_mixture = samp[:, self.rank:2*self.rank]
            i_mixture = samp[:, 2*self.rank:]
            m.append(m_mixture)
            n.append(n_mixture)
            I.append(i_mixture)
        m = torch.sum(torch.stack(m, dim=0) * self.mixture_weights[:, None, None], dim=0)
        n = torch.sum(torch.stack(n, dim=0) * self.mixture_weights[:, None, None], dim=0)
        I = torch.sum(torch.stack(I, dim=0) * self.mixture_weights[:, None, None], dim=0)

        J = m @ n.T  # (N, r) x (r, N) = (N, N)
        J = J * self.base_scale / self.hidden_size
        span = torch.cat([m, I], axis=1)  # (N, r + I)

        # run RNN dynamics over input sequence
        batch_size, seq_len, input_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        hiddens = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            dh = -h + (J @ self.activation(h).T).T + (I @ x_t.T).T
            h = h + self.alpha * dh
            hiddens.append(h)
        hiddens = torch.stack(hiddens, dim=1)  # (B, T, N)
        hiddens = (torch.linalg.pinv(span) @ hiddens.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, r + I)
        # self.last_m = m
        # self.last_n = n
        # self.last_I = I
        return hiddens


class TrainableMixtureLowRankRNNWithReadout(TrainableMixtureLowRankRNN):
    def __init__(self, input_size, hidden_size, output_size, rank=2, num_mixtures=1, alpha=0.1, activation="tanh"):
        super(TrainableMixtureLowRankRNNWithReadout, self).__init__(
            input_size, hidden_size, rank, num_mixtures, alpha, activation
        )
        self.readout = nn.Linear(rank, output_size)  # project to latent space before readout for stability

    def forward(self, x):
        hiddens = super().forward(x)
        readout = self.readout(hiddens[..., :self.rank])
        return readout
    

class FixedSeedMixtureLowRankRNN(TrainableMixtureLowRankRNN):
    def __init__(self, input_size, hidden_size, rank=2, num_mixtures=1, alpha=0.1, activation="tanh", output_size=None, base_scale=500, num_seeds=1, down_project=False):
        super(FixedSeedMixtureLowRankRNN, self).__init__(input_size, hidden_size, rank, num_mixtures, alpha, activation, output_size, base_scale)
        self.num_seeds = num_seeds
        self.register_buffer(
            "seeds",
            torch.randn(num_seeds, self.hidden_size, 2 * rank + input_size)
        )
        self.cur_seeds = None
        self.down_project = down_project

    def forward(self, x):
        self._clamp_tril()
        self._normalize_weights()
        # sample weight matrices
        m = []
        n = []
        I = []
        for i in range(self.num_mixtures):
            mean = self.means[i]  # (d,)
            scale_tril = self.scale_tril[i] # (d,d)
            samp = mean + (scale_tril @ self.seeds[:, :, :].unsqueeze(-1)).squeeze(-1)
            m_mixture = samp[:, :, :self.rank]
            n_mixture = samp[:, :, self.rank:2*self.rank]
            i_mixture = samp[:, :, 2*self.rank:]
            m.append(m_mixture)
            n.append(n_mixture)
            I.append(i_mixture)
        m = torch.sum(torch.stack(m, dim=0) * self.mixture_weights[:, None, None, None], dim=0)
        n = torch.sum(torch.stack(n, dim=0) * self.mixture_weights[:, None, None, None], dim=0)
        I = torch.sum(torch.stack(I, dim=0) * self.mixture_weights[:, None, None, None], dim=0)

        J = m @ n.transpose(-2, -1) * self.base_scale / self.hidden_size  # (S, N, r) x (S, r, N) = (S, N, N)
        self.cur_seeds = torch.randint(low=0, high=self.num_seeds, size=(x.shape[0],)).to(x.device)
        span = torch.cat([m, I], axis=2)  # (S, N, r + I)

        # run RNN dynamics over input sequence
        batch_size, seq_len, input_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        hiddens = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            dh = torch.zeros_like(h)
            for s in range(self.num_seeds):
                mask = (self.cur_seeds == s)
                if mask.sum() == 0:
                    continue
                dh[mask] = -h[mask] + (J[s] @ self.activation(h[mask]).T).T + (I[s] @ x_t[mask].T).T
            h = h + self.alpha * dh
            hiddens.append(h)
        hiddens = torch.stack(hiddens, dim=1)  # (B, T, N)
        if self.down_project:
            down_hiddens = torch.zeros(batch_size, seq_len, self.rank + self.input_size).to(x.device)
            for s in range(self.num_seeds):
                mask = (self.cur_seeds == s)
                if mask.sum() == 0:
                    continue
                down_hiddens[mask] = (torch.linalg.pinv(span[s]) @ hiddens[mask].permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, r + I)
            hiddens = down_hiddens
        # self.last_m = m
        # self.last_n = n
        # self.last_I = I
        return hiddens


class FixedSeedMixtureLowRankRNNWithReadout(FixedSeedMixtureLowRankRNN):
    def __init__(self, input_size, hidden_size, output_size, rank=2, num_mixtures=1, alpha=0.1, activation="tanh", num_seeds=1, down_project=False):
        super(FixedSeedMixtureLowRankRNNWithReadout, self).__init__(
            input_size, hidden_size, rank, num_mixtures, alpha, activation, num_seeds, down_project
        )
        self.readout = nn.ModuleList([
            nn.Linear(rank if down_project else hidden_size, output_size)
            for _ in range(self.num_seeds)
        ])

    def forward(self, x):
        hiddens = super().forward(x)
        readout = torch.zeros(hiddens.shape[0], hiddens.shape[1], self.readout[0].out_features).to(x.device)
        for s in range(self.num_seeds):
            mask = (self.cur_seeds == s)
            if mask.sum() == 0:
                continue
            readout[mask] = self.readout[s](hiddens[mask])
        return readout


class FixedSeedResampler:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.old_seed = self.model.seeds.data
        with torch.no_grad():
            self.model.seeds.data = torch.randn(
                self.model.num_seeds,
                self.model.hidden_size,
                2 * self.model.rank + self.model.input_size
            )
        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.seeds.data = self.old_seed
        return True