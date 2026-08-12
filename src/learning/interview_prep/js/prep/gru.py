import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.x2h = nn.Linear(input_dim, hidden_dim*3)
        self.h2h = nn.Linear(input_dim, hidden_dim*3)

    def forward(self, x, hidden):
        x_reset, x_update, x_new = self.x2h(x).chunk(3, dim=-1)
        h_reset, h_update, h_new = self.h2h(hidden).chunk(3, dim=-1)
        reset = torch.sigmoid(x_reset + h_reset)
        update = torch.sigmoid(x_update + h_update)
        new = torch.sigmoid_(x_new + reset * h_new)
        return (1 - reset) * new + update * hidden


