import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.recurrence = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, x, prev_hidden):
        x = self.layer1(x)
        h = F.relu(self.recurrence(prev_hidden) + x)
        return self.layer2(x), h


def train():
    hidden_state_dim = 128
    model = RNNCell(12, hidden_state_dim, 10)
    optimizer = optim.AdamW(model.parameters(), 1e-3, (0.9, 0.98), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    hidden_state = torch.zeros(hidden_state_dim)
    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            x = batch["input"]
            y = batch["ground_truth"]
            B, T, _ = x.shape

            h = torch.zeroes(B, hidden_state_dim)
            logits = []
            for t in range(T):
                out, h = model(x[:, t], h)
                logits.append(out)
            
            logits = torch.stack(logits, dim=1)
            loss = criterion(logits.rehape(-1, 10), y.reshape(-1))
