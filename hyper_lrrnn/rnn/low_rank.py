import math
import torch
import torch.nn as nn


class LowRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rank=2, alpha=0.1, activation="tanh", output_size=None, orth_gain=1.0, pr_gain=1.0, dropout_rate=0.0):
        super(LowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.orth_gain = orth_gain
        self.pr_gain = pr_gain
        self.dropout_rate = dropout_rate

        # self.m = nn.Parameter(torch.randn(hidden_size, rank))
        # self.n = nn.Parameter(torch.randn(hidden_size, rank))
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
        if self.dropout_rate > 0 and self.training:
            mask = torch.bernoulli(1 - self.dropout_rate * torch.ones(self.hidden_size)).to(x.device)
            m = self.m * mask[:, None]
            n = self.n * mask[:, None]
            I = self.I * mask[:, None]
            J = m @ n.T * self.hidden_size / (self.hidden_size * (1 - self.dropout_rate))
        else:
            J = self.m @ self.n.T # / self.hidden_size  # (N, r) x (r, N) = (N, N)
            I = self.I
        for t in range(seq_len):
            x_t = x[:, t, :]
            dh = -h + (J @ self.activation(h).T).T + (I @ x_t.T).T
            h = h + self.alpha * dh
            hiddens.append(h)
        hiddens = torch.stack(hiddens, dim=1)
        return hiddens

    # def forward(self, x):
    #     batch_size, seq_len, input_size = x.size()
    #     h = torch.zeros(batch_size, self.hidden_size).to(x.device)
    #     hiddens = []
    #     if self.dropout_rate > 0 and self.training:
    #         mask = torch.bernoulli(1 - self.dropout_rate * torch.ones(self.hidden_size)).to(x.device)
    #         J = self.m @ self.n.T / (self.hidden_size * (1 - self.dropout_rate))
    #     else:
    #         J = self.m @ self.n.T / self.hidden_size
    #     for t in range(seq_len):
    #         x_t = x[:, t, :]
    #         dh = -h + (J @ self.activation(h).T).T + (self.I @ x_t.T).T
    #         h = h + self.alpha * dh
    #         if self.dropout_rate > 0 and self.training:
    #             h = h * mask
    #         hiddens.append(h)
    #     hiddens = torch.stack(hiddens, dim=1)
    #     return hiddens

class LowRankRNNWithReadout(LowRankRNN):
    def __init__(self, input_size, hidden_size, output_size, rank=2, alpha=0.1, activation="tanh", orth_gain=1.0, pr_gain=1.0, dropout_rate=0.0):
        super(LowRankRNNWithReadout, self).__init__(
            input_size, hidden_size, rank, alpha, activation, output_size, orth_gain, pr_gain, dropout_rate
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hiddens = super().forward(x)
        readout = self.readout(hiddens)
        return readout