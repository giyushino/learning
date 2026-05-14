"""
self contained qwen3 arch llm
"""

import torch
import torch.nn as nn


class Qwen3RotaryEmbedding(nn.Module):
    """
    The goal of this is to encode position by
    rotating the Q and K vectors inside attention
    """
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"

        # inv_freq gives us theta, which determines how much
        # we should rotate the Q K vectors in attention
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, False)


    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        tensor = [a, b, c, d, e, f, g, h]
        rotate_half(tensor) = [-e, -f, -g, -h, a, b, c, d]
        """
        x1 = x[...:x.shape[-1] // 2]
        x2 = x[...,x.shape[-1] // 2:]

        return torch.cat((-x2, x1), dim=-1)


    def foward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # add new dimension to end of position ids, get (B, S, 1)
        # add 2 new dimentions to inv_freq, get (1, 1, D/2)
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]

        # freqs shape: (B, S, D/2)
        emb = torch.cat(freqs, freqs ,dim=-1)

        # cos/sin shape: (B, 1, S, D), broadcast over heads
        cos = emb.cos()[:, None, :, :].to(dtype=x.dtype)
        sin = emb.sin()[:, None, :, :].to(dtype=x.dtype)

        return (x * cos) + (self.rotate_half(x) * sin)
        

class Qwen3TransformerBlock(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        

class Qwen3(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        


if __name__ == "__main__":
    pass

