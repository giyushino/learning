import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__() 
        self.l1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.l2(x)


def train():
    model = MLP(10, 128, 4)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    for epoch in range(epochs):
        batch = {"inputs": None, "ground_truth": None} # dataloader not implemented
        preds = model(batch["inputs"])
        loss = criterion(preds, batch["ground_truth"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)



if __name__ == "__main__":
    train()

    


