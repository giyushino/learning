import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size
        #  4 channels, input gate, forget gate, output gate, input node
        self.x2h = nn.Linear(input_size, hidden_size * 4)
        self.h2h = nn.Linear(hidden_size, hidden_size * 4)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)

    def forward(self, x, state):
        hidden, cell = state
        input_gate, forget_gate, input_node, output_gate = (self.x2h(x) + self.h2h(hidden)).chunk(4, dim=1)
        cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * input_node.tanh()
        hidden = output_gate.sigmoid() * cell.tanh()
        return hidden, cell


class LSTM(nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(hidden_size, input_size)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, state):
        hidden, cell = self.lstm_cell(x, state)
        return self.output_proj(hidden), (hidden, cell)

       
def train():
    hidden_size = 128
    output_dim = 10
    model = LSTM(hidden_size, 32, output_dim)
    optimizer = optim.AdamW(model.parameters(), lr = 3e-4, betas=(0.9, 0.98), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    dataloader = [] # we don't have real data to work with right now
    
    for epoch in range(10):
        for batch in dataloader:
            x, y = batch["input"], batch["ground_truth"]
            batch_size, seq_length, _ = x.shape
            hidden, cell = torch.zeros(batch_size, hidden_size), torch.zeros(batch_size, hidden_size)
            state = (hidden, cell)

            logits = []
            for t in range(seq_length):
                out, state = model(x[:, t], state)
                logits.append(out)

            logits = torch.stack(logits, dim=1)
            # y is reshaped to (batch_size * seq_length)
            # we need to reshape loss,
            loss = criterion(logits.reshape(-1, output_dim), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()



