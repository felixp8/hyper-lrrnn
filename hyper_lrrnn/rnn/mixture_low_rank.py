import math
import numpy as np
import torch
import torch.nn as nn


def is_psd(mat):
    return bool((mat == mat.transpose(-2, -1)).all() and (torch.linalg.eigvals(mat).real>=0).all())


class MixtureLowRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rank=2, num_mixtures=1, alpha=0.1, activation="tanh", output_size=None, base_scale=500):
        super(MixtureLowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_mixtures = num_mixtures
        self.alpha = alpha
        self.base_scale = base_scale

        self.mixture_weights = nn.Parameter(torch.ones(num_mixtures) / num_mixtures, requires_grad=False)
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

    def forward(self, x):
        """Forward pass

        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
        """
        self._clamp_tril()
        # sample weight matrices
        m = []
        n = []
        I = []
        mixture_sizes = torch.diff(torch.round(
            torch.cumsum((self.mixture_weights / torch.sum(self.mixture_weights)) * self.hidden_size, dim=0)
        ), prepend=torch.tensor([0.0], device=self.mixture_weights.device)).long().tolist()
        for i in range(self.num_mixtures):
            mean = self.means[i]
            scale_tril = self.scale_tril[i]
            mixture_size = mixture_sizes[i]
            samp = torch.distributions.MultivariateNormal(
                loc=mean, scale_tril=scale_tril).rsample((mixture_size,))
            m_mixture = samp[:, :self.rank]
            n_mixture = samp[:, self.rank:2*self.rank]
            i_mixture = samp[:, 2*self.rank:]
            m.append(m_mixture)
            n.append(n_mixture)
            I.append(i_mixture)
        m = torch.cat(m, dim=0)
        n = torch.cat(n, dim=0)
        I = torch.cat(I, dim=0)

        # NOTE: should divide by hidden_size here but skipping for consistency with LowRankRNN
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


class MixtureLowRankRNNWithReadout(MixtureLowRankRNN):
    def __init__(self, input_size, hidden_size, output_size, rank=2, num_mixtures=1, alpha=0.1, activation="tanh"):
        super(MixtureLowRankRNNWithReadout, self).__init__(
            input_size, hidden_size, rank, num_mixtures, alpha, activation
        )
        self.readout = nn.Linear(rank, output_size)  # project to latent space before readout for stability

    def forward(self, x):
        hiddens = super().forward(x)
        readout = self.readout(hiddens[..., :self.rank])
        return readout