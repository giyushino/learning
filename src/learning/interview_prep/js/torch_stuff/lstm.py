import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class LSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, hidden_size * 4)
        self.h2h = nn.Linear(input_size, hidden_size * 4)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.x2h(x) + self.h2h(h)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() + g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, c 

class LSTM(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.cell = LSTMCell(hidden_size, input_size)
        self.hidden_size = hidden_size

    def forward(self, x, state=None):
        N, L, _ = x.shape
        if state is None:
            h = c = x.new_zeros(N, self.hidden_size)
        else:
            h, c = state
        outs = []
        for t in range(L):
            h, c = self.cell(x[:, t], (h, c))
            outs.append(h)

        return torch.stack(outs, dim=1), (h, c)



if __name__ == "__main__":
    lstm = LSTM()
