import torch
import torch.nn as nn
import torch.optim as optim

# linear regession works well when we assume noise is well behaved and follows a guassian
# generally if we have weak signal, regression is fine since it won't overfit to noise?
# we can differentiate between types of regession through the loss function
# Lasso (l1) and Ridge (l2). Ridge has an easy api exposed, where we can set the
# weight decay in the optimizer

class LinearReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x):
        return self.linear(x) 


if __name__ == "__main__":
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

    model = LinearReg()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    epochs = 10
    for epoch in range(epochs):
        pred_y = model(X)
        loss = criterion(pred_y, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"{loss=}")




