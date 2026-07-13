"""Debugging challenge: tiny transformer that should learn to copy sequences.

Task
----
Each training example is a sequence of the form:

    [BOS] t1 t2 ... t8 [SEP] t1 t2 ... t8

i.e. the model sees 8 random tokens, then a separator, and must reproduce
the 8 tokens autoregressively. A correct implementation of this script
reaches >99% eval accuracy within 500 steps (well under a minute on CPU).

This implementation does not. The architecture, task, and hyperparameters
are all sensible -- every problem is an implementation bug, not a tuning
issue. There are multiple independent bugs (fewer than 10). Find and fix
all of them. Fixing one may change the symptoms of the others, so re-run
as you go and be suspicious of anything that looks "almost right".

Watch out for the classic trap: a training loss near zero does NOT mean
the model works.

Run with:  python debug_pytorch.py
Solutions: debug_pytorch_SOLUTIONS.md (don't peek until you're done)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB_SIZE = 16  # 0 = BOS, 1 = SEP, 2..15 = data tokens
SEQ_LEN = 8
MAX_LEN = 2 * SEQ_LEN + 2

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

BATCH_SIZE = 64
STEPS = 500
LR = 1e-3


def make_batch(batch_size: int) -> torch.Tensor:
    """Returns (batch_size, MAX_LEN) sequences: [BOS] data [SEP] data."""
    data = torch.randint(2, VOCAB_SIZE, (batch_size, SEQ_LEN))
    bos = torch.zeros(batch_size, 1, dtype=torch.long)
    sep = torch.ones(batch_size, 1, dtype=torch.long)
    batch = torch.cat([bos, data, sep, data], dim=1)
    return batch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, S, _ = x.shape
        return x.transpose(1, 2).reshape(B, S, self.n_heads * self.d_head) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        # print(causal_mask)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, n_heads, T, d_head)
        out = self.combine_heads(out)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        x = F.dropout(x, DROPOUT, training=self.training)
        return x


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(MAX_LEN, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList(Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:T]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


@torch.no_grad()
def evaluate(model: nn.Module, n: int = 256) -> float:
    """Greedy-decodes the copy and returns token-level accuracy."""
    model.eval()
    batch = make_batch(n)
    prompt = batch[:, : SEQ_LEN + 2]  # [BOS] data [SEP]
    answer = batch[:, SEQ_LEN + 2 :]

    seq = prompt
    for _ in range(SEQ_LEN):
        out = model(seq)
        next_tok = out[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)

    pred = seq[:, SEQ_LEN + 2 :]
    acc = (pred == answer).float().mean().item()
    model.train()
    return acc


def train():
    model = TinyLM()
    print(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    running_loss = 0.0
    for step in range(1, STEPS + 1):
        batch = make_batch(BATCH_SIZE)
        inputs, targets = batch[:, :-1], batch[:, 1:]

        out = model(inputs)
        loss = F.cross_entropy(out.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        if step % 100 == 0:
            acc = evaluate(model)
            print(
                f"step {step:4d} | train loss {running_loss / 100:.4f} "
                f"| eval copy accuracy {acc:.2%}"
            )
            running_loss = 0.0

    final_acc = evaluate(model)
    print(f"\nfinal eval copy accuracy: {final_acc:.2%}")
    if final_acc > 0.99:
        print("PASSED -- the model copies correctly.")
    else:
        print("FAILED -- keep debugging.")


if __name__ == "__main__":
    train()
