import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


def train():
    input_dim, output_dim = 10, 10
    model = LinearRegression(input_dim, output_dim)
    optimizer = optim.SGD(model.paramters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()
    dataloader = []

    for epoch in range(10):
        for batch in dataloader:
            x, y = batch["input"], batch["ground_truth"]
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

