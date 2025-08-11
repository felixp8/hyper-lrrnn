import math
import torch
import torch.nn as nn


class LowRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rank=2, alpha=0.1, activation="tanh"):
        super(LowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha

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


class LowRankRNNWithReadout(LowRankRNN):
    def __init__(self, input_size, hidden_size, output_size, rank=2, alpha=0.1, activation="tanh"):
        super(LowRankRNNWithReadout, self).__init__(
            input_size, hidden_size, rank, alpha, activation
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hiddens = super().forward(x)
        readout = self.readout(hiddens)
        return readout