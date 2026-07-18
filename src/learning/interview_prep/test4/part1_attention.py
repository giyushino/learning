"""
PART 1 — Attention from scratch (~25 min)

This is the single most common ML coding screen at OpenAI, Anthropic, and the
smaller labs (Liquid AI, Zyphra, Arcee): implement attention from memory, in
PyTorch, with only basic tensor ops. No F.scaled_dot_product_attention, no
nn.MultiheadAttention, no einops. torch.einsum is allowed.

Task A: `scaled_dot_product_attention` — the functional core.
Task B: `MultiHeadAttention.forward` — full MHA with grouped-query attention
        (GQA). The __init__ (projection layout) is given; you write forward.

Grade yourself:  uv run python tests/grade_part1.py   (from the test4/ dir)
Do NOT open the grader — it contains a reference implementation.
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False):
    """Compute softmax(q @ k^T / sqrt(d)) @ v.

    Shapes:
      q: (..., T_q, d)     k: (..., T_k, d)     v: (..., T_k, d_v)
      returns (..., T_q, d_v)

    attn_mask: optional *boolean* tensor broadcastable to (..., T_q, T_k).
      True  = position may be attended to
      False = position is masked out (gets zero attention weight)

    is_causal: if True, apply a causal mask (query i attends to keys 0..i).
      You may assume attn_mask is None when is_causal=True, and T_q == T_k.

    Match torch.nn.functional.scaled_dot_product_attention semantics.
    """
    attn_scores = torch.matmul(q, k.tranpose(-2, -1)) / math.sqrt(q.size(-1))
    mask_value = torch.finfo(attn_scores.dtype).min

    if is_causal:
        S = q.size(-2)
        causal_mask = torch.triu(
            torch.ones((S, S), device=q.device, dtype=torch.bool),
            diagonal = 1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, mask_value)

    if attn_mask is not None:
        key_padding_mask = ~attn_mask[:, None, None, :].bool()
        attn_scores.masked_fill(key_padding_mask, mask_value)

    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, v)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional grouped-query attention (GQA).

    GQA: there are n_heads query heads but only n_kv_heads key/value heads
    (n_kv_heads divides n_heads). Each group of consecutive query heads shares
    one kv head: query head i uses kv head  i // (n_heads // n_kv_heads).
    n_kv_heads == n_heads is ordinary MHA; n_kv_heads == 1 is MQA.
    """

    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def split_heads(self, x: torch.Tensor):
        B, S, V = x.shape
        return x.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def split_kv_heads(self, x: torch.Tensor):
        B, S, V = x.shape
        return x.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor):
        B, S, _ = x.shape
        return x.tranpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        
    def forward(self, x, is_causal=True):
        """x: (B, T, d_model) -> (B, T, d_model).

        Use your scaled_dot_product_attention above for the core computation.
        """
        Q = self.split_heads(self.q_proj(x))
        K = self.split_kv_heads(self.k_proj(x))
        V = self.split_kv_heads(self.v_proj(x))
        
        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=1)
        V = V.repeat_interleave(repeat_factor, dim=1)

        attn_output = scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
        return self.o_proj(self.combine_heads(attn_output))
