"""Grader for Part 1. DO NOT OPEN until you're done — contains the reference."""

import torch
import torch.nn.functional as F

from _harness import run
from part1_attention import MultiHeadAttention, scaled_dot_product_attention

torch.manual_seed(0)


def _assert_close(a, b, msg=""):
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=msg or None)


def test_sdpa_basic():
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 10, 8)
    v = torch.randn(2, 4, 10, 8)
    _assert_close(
        scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
    )


def test_sdpa_causal():
    q = torch.randn(2, 4, 7, 8)
    k = torch.randn(2, 4, 7, 8)
    v = torch.randn(2, 4, 7, 8)
    _assert_close(
        scaled_dot_product_attention(q, k, v, is_causal=True),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
    )


def test_sdpa_bool_mask():
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 9, 8)
    v = torch.randn(2, 4, 9, 8)
    mask = torch.rand(2, 1, 6, 9) > 0.4
    mask[..., 0] = True  # every query row keeps at least one key
    _assert_close(
        scaled_dot_product_attention(q, k, v, attn_mask=mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
    )


def test_sdpa_no_softmax_over_masked():
    # masked positions must get exactly zero weight, not just tiny weight
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 3, 4) * 50  # huge logits amplify any leakage
    v = torch.randn(1, 1, 3, 4)
    mask = torch.tensor([[[[True, False, False], [True, True, False]]]])
    out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    _assert_close(out[0, 0, 0], v[0, 0, 0], msg="row 0 should equal v[0] exactly")


def _mha_reference(mha, x, is_causal):
    B, T, _ = x.shape
    hd, nh, nkv = mha.head_dim, mha.n_heads, mha.n_kv_heads
    q = mha.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
    k = mha.k_proj(x).view(B, T, nkv, hd).transpose(1, 2)
    v = mha.v_proj(x).view(B, T, nkv, hd).transpose(1, 2)
    k = k.repeat_interleave(nh // nkv, dim=1)
    v = v.repeat_interleave(nh // nkv, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    return mha.o_proj(out.transpose(1, 2).reshape(B, T, nh * hd))


def test_mha_causal():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 9, 32)
    _assert_close(mha(x, is_causal=True), _mha_reference(mha, x, True))


def test_mha_bidirectional():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 9, 32)
    _assert_close(mha(x, is_causal=False), _mha_reference(mha, x, False))


def test_gqa():
    mha = MultiHeadAttention(d_model=64, n_heads=8, n_kv_heads=2)
    x = torch.randn(2, 11, 64)
    _assert_close(mha(x, is_causal=True), _mha_reference(mha, x, True))


def test_mqa():
    mha = MultiHeadAttention(d_model=48, n_heads=6, n_kv_heads=1)
    x = torch.randn(3, 5, 48)
    _assert_close(mha(x, is_causal=True), _mha_reference(mha, x, True))


if __name__ == "__main__":
    print("Part 1 — attention from scratch")
    ok = run([
        ("sdpa_basic", test_sdpa_basic),
        ("sdpa_causal", test_sdpa_causal),
        ("sdpa_bool_mask", test_sdpa_bool_mask),
        ("sdpa_masked_rows_exact", test_sdpa_no_softmax_over_masked),
        ("mha_causal", test_mha_causal),
        ("mha_bidirectional", test_mha_bidirectional),
        ("gqa_8q_2kv", test_gqa),
        ("mqa_6q_1kv", test_mqa),
    ])
    raise SystemExit(0 if ok else 1)
