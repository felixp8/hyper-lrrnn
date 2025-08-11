import math
import torch
import torch.nn as nn


class FullRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, alpha=0.1, activation="tanh"):
        super(FullRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) / math.sqrt(hidden_size))
        self.U = nn.Parameter(torch.randn(hidden_size, input_size) / math.sqrt(hidden_size))
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        hiddens = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            dh = -h + (self.W @ self.activation(h).T).T + (self.U @ x_t.T).T
            h = h + self.alpha * dh
            hiddens.append(h)
        hiddens = torch.stack(hiddens, dim=1)
        return hiddens


class FullRankRNNWithReadout(FullRankRNN):
    def __init__(self, input_size, hidden_size, output_size, alpha=0.1, activation="tanh"):
        super(FullRankRNNWithReadout, self).__init__(
            input_size, hidden_size, alpha, activation
            )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hiddens = super().forward(x)
        readout = self.readout(hiddens)
        return readout