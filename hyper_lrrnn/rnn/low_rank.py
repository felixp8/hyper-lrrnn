import math
import torch
import torch.nn as nn


class LowRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rank=2, alpha=0.1, activation="tanh", output_size=None, orth_gain=1.0, pr_gain=1.0):
        super(LowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.orth_gain = orth_gain
        self.pr_gain = pr_gain

        self.m = nn.Parameter(torch.randn(hidden_size, rank) / math.sqrt(hidden_size))
        self.n = nn.Parameter(torch.randn(hidden_size, rank) / math.sqrt(hidden_size))
        self.I = nn.Parameter(torch.randn(hidden_size, input_size) / math.sqrt(hidden_size))

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        hiddens = []
        J = self.m @ self.n.T
        for t in range(seq_len):
            x_t = x[:, t, :]
            dh = -h + (J @ self.activation(h).T).T + (self.I @ x_t.T).T
            h = h + self.alpha * dh
            hiddens.append(h)
        hiddens = torch.stack(hiddens, dim=1)
        return hiddens

    def _participation_ratio(self):
        norms = self.m.norm(dim=0) * self.n.norm(dim=0)
        pr = torch.square(torch.sum(norms)) / torch.sum(torch.square(norms)).clamp(min=1e-8)
        return pr

    def _reg_loss(self):
        dots = self.m.T @ self.I  # shape (rank, input_size)
        norms = self.m.norm(dim=0)[:, None] * self.I.norm(dim=0)[None, :]  # shape (rank, input_size)
        sim = dots / norms.clamp(min=1e-8)  # shape (rank, input_size)

        pr = self._participation_ratio()
        return (sim ** 2).mean() * self.orth_gain + pr * self.pr_gain

class LowRankRNNWithReadout(LowRankRNN):
    def __init__(self, input_size, hidden_size, output_size, rank=2, alpha=0.1, activation="tanh", orth_gain=1.0, pr_gain=1.0):
        super(LowRankRNNWithReadout, self).__init__(
            input_size, hidden_size, rank, alpha, activation, output_size, orth_gain, pr_gain
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hiddens = super().forward(x)
        readout = self.readout(hiddens)
        return readout