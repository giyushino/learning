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
    raise NotImplementedError


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

    def forward(self, x, is_causal=True):
        """x: (B, T, d_model) -> (B, T, d_model).

        Use your scaled_dot_product_attention above for the core computation.
        """
        raise NotImplementedError
