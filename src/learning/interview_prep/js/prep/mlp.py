import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def train_reg():
    model = MLP(10, 128, 1, 0.2)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=3e-4, weight_decay=0.1)
    criterion = nn.MSELoss()
    dataloader = []

    for epoch in range(10):
        for batch in dataloader:
            x, y = batch["input"], batch["ground_truth"]
            out = model(x)
            loss = criterion(out, y.unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_clf():
    model = MLP(10, 128, 5, 0.2)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=3e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    dataloader = []

    for epoch in range(10):
        for batch in dataloader:
            x, y = batch["input"], batch["ground_truth"]
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
