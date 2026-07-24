"""Grader for Part 2. DO NOT OPEN until you're done — contains the reference."""

import math

import torch
import torch.nn.functional as F

from _harness import run
from part2_sampling import (
    CausalSelfAttention,
    MiniGPT,
    generate,
    sample,
    top_k_filter,
    top_p_filter,
)

torch.manual_seed(0)


def _assert_close(a, b, msg=""):
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=msg or None)


def _model(vocab=50, d=32, heads=4, layers=2, seq=32):
    torch.manual_seed(123)
    m = MiniGPT(vocab, d, heads, layers, seq)
    m.eval()
    return m


def test_attn_no_cache():
    torch.manual_seed(1)
    attn = CausalSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 7, 32)
    out, (k, v) = attn(x)
    B, T, _ = x.shape
    hd, nh = attn.head_dim, attn.n_heads
    q_r = attn.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
    k_r = attn.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
    v_r = attn.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q_r, k_r, v_r, is_causal=True)
    ref = attn.o_proj(ref.transpose(1, 2).reshape(B, T, nh * hd))
    _assert_close(out, ref)
    assert k.shape == (2, 4, 7, 8), f"bad cache k shape {tuple(k.shape)}"
    _assert_close(k, k_r, msg="returned k cache is wrong")
    _assert_close(v, v_r, msg="returned v cache is wrong")


def test_cache_matches_full_forward():
    m = _model()
    idx = torch.randint(0, 50, (2, 12))
    with torch.no_grad():
        full_logits, _ = m(idx)
        # prefill on first 8 tokens, then 4 single-token steps
        logits, kvs = m(idx[:, :8])
        _assert_close(logits, full_logits[:, :8], msg="prefill logits differ")
        for t in range(8, 12):
            logits, kvs = m(idx[:, t : t + 1], kvs)
            _assert_close(
                logits[:, 0],
                full_logits[:, t],
                msg=f"cached logits differ at position {t}",
            )


def test_cache_multi_token_chunk():
    # feeding several new tokens at once against an existing cache
    m = _model()
    idx = torch.randint(0, 50, (2, 10))
    with torch.no_grad():
        full_logits, _ = m(idx)
        logits1, kvs = m(idx[:, :4])
        logits2, kvs = m(idx[:, 4:10], kvs)  # 6 new tokens at once
        _assert_close(logits2, full_logits[:, 4:10],
                      msg="chunked decode with cache differs (mask offset bug?)")


def test_top_k_filter():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0],
                           [-1.0, -5.0, -3.0, -2.0, -4.0]])
    out = top_k_filter(logits, 2)
    inf = float("-inf")
    expected = torch.tensor([[inf, 5.0, inf, inf, 4.0],
                             [-1.0, inf, inf, -2.0, inf]])
    assert torch.equal(out, expected), f"got {out}"
    assert logits[0, 0] == 1.0, "input was modified in place"


def test_top_p_filter():
    # probs: [0.5, 0.3, 0.15, 0.05] (log gives exact softmax recovery)
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    inf = float("-inf")

    out = top_p_filter(logits, 0.79)  # 0.5+0.3 = 0.8 >= 0.79 -> keep 2
    assert out[0, 2] == inf and out[0, 3] == inf, f"p=0.79 should keep 2: {out}"
    assert out[0, 0] == logits[0, 0] and out[0, 1] == logits[0, 1]

    out = top_p_filter(logits, 0.81)  # need third token
    assert out[0, 2] == logits[0, 2] and out[0, 3] == inf, \
        f"p=0.81 should keep 3: {out}"

    out = top_p_filter(logits, 0.1)  # always keep at least top-1
    assert out[0, 0] == logits[0, 0], "must always keep top-1"
    assert all(out[0, i] == inf for i in (1, 2, 3)), f"p=0.1 keeps only top-1: {out}"

    # order independence: shuffled columns, same decision
    perm = torch.tensor([2, 0, 3, 1])
    out = top_p_filter(logits[:, perm], 0.79)
    kept = (out != inf).nonzero()[:, 1].tolist()
    assert sorted(kept) == [1, 3], f"shuffled p=0.79 should keep cols 1,3: {out}"


def test_sample_greedy():
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, -1.0, 0.5]])
    out = sample(logits, temperature=0.0)
    assert out.tolist() == [1, 0], f"greedy should be argmax, got {out.tolist()}"
    assert out.dtype == torch.long


def test_sample_respects_top_k():
    torch.manual_seed(7)
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(4, 20)
    allowed = set()
    for row in range(4):
        allowed |= {(row, i) for i in torch.topk(logits[row], 3).indices.tolist()}
    for _ in range(200):
        out = sample(logits, temperature=1.5, top_k=3, generator=g)
        for row, tok in enumerate(out.tolist()):
            assert (row, tok) in allowed, f"sampled token {tok} outside top-3"


def test_sample_temperature_direction():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]]).repeat(2000, 1)
    hot = (sample(logits, temperature=5.0, generator=g) == 0).float().mean()
    cold = (sample(logits, temperature=0.2, generator=g) == 0).float().mean()
    assert cold > hot, (
        f"low temperature must concentrate on argmax (T=0.2: {cold:.2f}, "
        f"T=5: {hot:.2f}) — are you multiplying instead of dividing?"
    )


def test_generate_greedy_matches_naive():
    m = _model()
    prompt = torch.randint(0, 50, (2, 5))
    out = generate(m, prompt, max_new_tokens=10, temperature=0.0)
    assert out.shape == (2, 15), f"bad output shape {tuple(out.shape)}"
    assert torch.equal(out[:, :5], prompt), "prompt must be preserved"
    # naive reference: full recompute each step
    seq = prompt.clone()
    with torch.no_grad():
        for _ in range(10):
            logits, _ = m(seq)
            seq = torch.cat([seq, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, seq), (
        f"cached greedy decode diverges from full recompute\n"
        f"got      {out.tolist()}\nexpected {seq.tolist()}"
    )


def test_generate_actually_uses_cache():
    m = _model()
    seen = []
    for block in m.blocks:
        orig = block.attn.forward
        def wrapped(x, past_kv=None, _orig=orig):
            seen.append(x.shape[1])
            return _orig(x, past_kv)
        block.attn.forward = wrapped
    prompt = torch.randint(0, 50, (1, 6))
    generate(m, prompt, max_new_tokens=8, temperature=0.0)
    per_layer = seen[:: len(m.blocks)]
    assert per_layer[0] == 6, f"first call should prefill 6 tokens, saw {per_layer}"
    # 7 forwards if you skip the useless one after the last sampled token, 8 if not
    assert all(t == 1 for t in per_layer[1:]) and len(per_layer) in (8, 9), (
        f"each decode step must feed exactly 1 new token; attention saw "
        f"{per_layer} — you're recomputing the prefix"
    )


def test_generate_sampled_reproducible():
    m = _model()
    prompt = torch.randint(0, 50, (2, 4))
    a = generate(m, prompt, 6, temperature=0.9, top_k=10,
                 generator=torch.Generator().manual_seed(42))
    b = generate(m, prompt, 6, temperature=0.9, top_k=10,
                 generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b), "same generator seed must give same tokens"


if __name__ == "__main__":
    print("Part 2 — KV-cache inference & sampling")
    ok = run([
        # ("attn_no_cache", test_attn_no_cache),
        # ("cache_matches_full_forward", test_cache_matches_full_forward),
        # ("cache_multi_token_chunk", test_cache_multi_token_chunk),
        # ("top_k_filter", test_top_k_filter),
        # ("top_p_filter", test_top_p_filter),
        ("sample_greedy", test_sample_greedy),
        ("sample_respects_top_k", test_sample_respects_top_k),
        ("sample_temperature_direction", test_sample_temperature_direction),
        ("generate_greedy_matches_naive", test_generate_greedy_matches_naive),
        ("generate_actually_uses_cache", test_generate_actually_uses_cache),
        ("generate_reproducible", test_generate_sampled_reproducible),
    ])
    raise SystemExit(0 if ok else 1)
