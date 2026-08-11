import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.recurrence = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hidden):
        x = self.fc1(x)
        hidden = torch.tanh(x + self.recurrence(hidden))
        return self.fc2(hidden), hidden


def train():
    input_dim, hidden_dim, output_dim = 10, 128, 10
    model = RNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)  
    dataloader = []

    for epoch in range(10):
        for batch in dataloader:
            # we can assume the samples have time dependence
            inputs, ground_truth = batch["input"], batch["ground_truth"]
            batch_size, seq_length, _ = inputs.shape
            # assume padded?
            logits = []
            hidden = inputs.new_zoers(batch_size, hidden_dim)
            for t in range(seq_length):
                out, hidden = model(inputs[:, t], hidden)
                logits.append(out)

            logits = torch.stack(logits, dim=1)
            loss = criterion(logits.reshape(-1, output_dim), ground_truth.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()








