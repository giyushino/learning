"""
implementation of transformers with
multihead attention in pytorch
"""

import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"

        inv_freq = 1.0 / (
          base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # x shape: (B, H, S, D)
        # position_ids shape: (B, S)

        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]

        # freqs shape: (B, S, D / 2)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        # cos/sin shape: (B, 1, S, D), broadcast over heads
        cos = emb.cos()[:, None, :, :].to(dtype=x.dtype)
        sin = emb.sin()[:, None, :, :].to(dtype=x.dtype)

        return (x * cos) + (self.rotate_half(x) * sin)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int):
        super().__init__()
        # for multi head attention, we split
        # the embeddings across heads
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = self.emb_dim // self.num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.o_proj = nn.Linear(emb_dim, emb_dim)

        self.rope = RotaryEmbedding(self.head_dim)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # reshape input tensor for mha attention
        # x is (batch_size, seq_length, emb_dim)
        B, S, _ = x.shape
    
        # this is so that each head sees a slice of the embeddings
        # new shape is (batch_size, num_heads, seq_length, head_dim)
        return x.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # combine outputs from each head back together
        B, _, S, _ = x.shape
        # simply reverse the transformation we
        # applied when splitting the input tensor
        return x.transpose(1, 2).reshape(B, S, self.emb_dim)
        
    def scaled_self_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None
    ):
        # Q matrix contains what each token is looking for
        # K matrix contains how each token should be matched
        # V matrix contains the information each token can contribute

        # Q and K are (batch_size, num_heads, seq_length, head_dim)
        # for matrix mult, we'll take the transpose of K, where we
        # can simply swap head_dim and seq_length
        # divide by sqrt of head dim to reduce variance  
        # matmul automatically broadcasts over the first two dims
        # shape is now (batch_size, num_heads, seq_length, seq_length)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_value = torch.finfo(attn_scores.dtype).min

        S = Q.size(-2)
        causal_mask = torch.triu(
            torch.ones(S, S, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, mask_value)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            key_padding_mask = ~attention_mask[:, None, None, :]
            query_padding_mask = ~attention_mask[:, None, :, None]
            attn_scores = attn_scores.masked_fill(key_padding_mask, mask_value)
            attn_scores = attn_scores.masked_fill(query_padding_mask, 0.0)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        if attention_mask is not None:
            attn_probs = attn_probs.masked_fill(query_padding_mask, 0.0)
       
        # this final matrix mult mixes information between the embeddings
        # of a token with the tokens it attended to
        return torch.matmul(attn_probs, V) 

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self.split_heads(self.q_proj(x))
        K = self.split_heads(self.k_proj(x))
        V = self.split_heads(self.v_proj(x))

        if position_ids is not None:
            Q = self.rope(Q, position_ids)
            K = self.rope(K, position_ids)

        attn_output = self.scaled_self_attention(Q, K, V, attention_mask)

        return self.o_proj(self.combine_heads(attn_output))



class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.silu(gate)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, ffn_mult: int):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, emb_dim)
        self.norm1 = nn.RMSNorm(emb_dim)
        self.norm2 = nn.RMSNorm(emb_dim)

        # maybe we make param to change the
        # activation function?
        ffn_dim = emb_dim * ffn_mult

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim * 2),
            SwiGLU(),
            nn.Linear(ffn_dim, emb_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # can't do += in pytorch since this
        # updates values in place, preventing
        # gradients from flowing properly
        x = x + self.mha(self.norm1(x), attention_mask, position_ids)
        x = x + self.ffn(self.norm2(x))

        return x


if __name__ == "__main__":
    config = {
        "num_heads": 4,
        "emb_dim": 728,
        "ffn_mult": 4
    }
    
    mha = TransformerBlock(**config)
    rand_Tensor = torch.rand((2, 10, 728))
    output = mha(rand_Tensor)
    print(output)
