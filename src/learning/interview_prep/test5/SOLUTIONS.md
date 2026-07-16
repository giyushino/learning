# SOLUTIONS — do not open until you're done (or truly stuck)

Graduated hints first; full code at the bottom. Read one level at a time.

---

## Hints, level 1 (nudges)

- **Part 1:** the top-p mask is cleanest computed in *sorted* order, then
  scattered back to the original token order. Watch the boundary: the token
  that pushes the cumulative sum across `top_p` is **kept**.
- **Part 2:** two things must change outside `KVCache` itself. What does
  `torch.arange(T)` in `TinyLM.forward` assume? What does the square
  `triu` mask in `Attention.forward` assume?
- **Part 3:** don't special-case `temperature=0`. Represent the greedy
  "distribution" as a one-hot vector and the general algorithm handles it —
  acceptance ratio becomes 0 or 1, and the residual becomes one-hot at the
  target argmax.

## Hints, level 2 (the actual gotchas)

- **Part 2, RoPE:** a cached decode step feeds 1 token, but that token is
  *not* at position 0 — rotate it with `positions = arange(start_pos,
  start_pos + T)`. Keys already in the cache were rotated when they were
  inserted, so cache *post*-RoPE k. If greedy matches for prompt length 1
  but diverges for longer prompts, this is your bug.
- **Part 2, mask:** with `Tq` queries and `Tk` cached+new keys, query `i`
  sits at absolute position `Tk - Tq + i` and may attend to keys `j <=
  Tk - Tq + i`. The square `triu(T, T)` mask is the special case `Tq == Tk`.
  Single-token decode (`Tq=1`) needs *no* mask, which is why bugs here only
  show up in the chunked-prefill test.
- **Part 3, residual:** on rejection you must sample from
  `norm(max(p - q, 0))` — not from `p`. Resampling from `p` double-counts
  mass where `p < q` and the 1500-run distribution test will catch it
  (that's what it's for).
- **Part 3, bookkeeping:** each round appends between 1 and `n_draft + 1`
  tokens (accepted prefix + replacement, or all accepted + bonus). Loop
  until you have enough, then truncate to exactly `n_new_tokens`.

---

## Full solution

### Part 1

```python
def sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        # drop a token iff the cumulative mass BEFORE it already reaches top_p
        remove = (cum - probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
```

The `(cum - probs) >= top_p` trick implements "keep the smallest prefix
whose cumulative sum reaches top_p" without an off-by-one: a token is
dropped only if the mass strictly before it already suffices.

### Part 2

`KVCache` — append along the sequence dim, return the full k/v:

```python
class KVCache:
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    @property
    def seq_len(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer_idx, k, v):
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k, v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=2)
        return self.k[layer_idx], self.v[layer_idx]
```

(Production engines preallocate `(B, Hkv, max_seq, D)` and write in place —
`torch.cat` reallocates every step. Fine here, worth *saying* in an
interview.)

`Attention.forward` — replace the cache branch and the mask with a general
rectangular version (it degenerates to the square causal mask when there is
no cache):

```python
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)   # post-RoPE!

        Tk = k.shape[2]
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)  # (B, H, T, Tk)
        # query i sits at absolute position Tk - T + i
        q_pos = torch.arange(Tk - T, Tk, device=x.device)
        k_pos = torch.arange(Tk, device=x.device)
        scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))

        out = scores.softmax(-1) @ v
        return self.wo(out.transpose(1, 2).reshape(B, T, H * D))
```

`TinyLM.forward` — one line: `positions = torch.arange(start_pos,
start_pos + T, device=tokens.device)`.

`generate_fast`:

```python
@torch.no_grad()
def generate_fast(model, prompt, n_new_tokens, temperature=0.0, top_k=None,
                  top_p=None, generator=None):
    cache = KVCache(model.cfg.n_layers)
    logits = model(prompt, cache=cache, start_pos=0)[:, -1, :]   # prefill
    pieces = [prompt]
    cur = sample_token(logits, temperature, top_k, top_p, generator)
    pieces.append(cur[:, None])
    for _ in range(n_new_tokens - 1):
        logits = model(cur[:, None], cache=cache, start_pos=cache.seq_len)[:, -1, :]
        cur = sample_token(logits, temperature, top_k, top_p, generator)
        pieces.append(cur[:, None])
    return torch.cat(pieces, dim=1)
```

Measured on the grading benchmark: ~5–6x over `generate_slow`.

### Part 3

```python
def _dist(logits, temperature):
    """Token distribution; one-hot argmax for T=0 so greedy needs no special case."""
    if temperature == 0.0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return (logits / temperature).softmax(-1)


@torch.no_grad()
def generate_speculative(target, draft, prompt, n_new_tokens, n_draft=4,
                         temperature=1.0, generator=None):
    assert prompt.shape[0] == 1
    tokens = prompt.clone()
    target_len = prompt.shape[1] + n_new_tokens
    n_accepted, n_proposed = 0, 0

    while tokens.shape[1] < target_len:
        # 1) draft proposes n_draft tokens autoregressively
        ctx = tokens
        draft_toks, draft_dists = [], []
        for _ in range(n_draft):
            q = _dist(draft(ctx)[:, -1, :], temperature)          # (1, V)
            tok = torch.multinomial(q, 1, generator=generator)    # (1, 1)
            draft_toks.append(tok)
            draft_dists.append(q)
            ctx = torch.cat([ctx, tok], dim=1)

        # 2) target scores every proposal in ONE forward pass
        tgt_logits = target(ctx)[:, -(n_draft + 1):, :]           # (1, k+1, V)

        # 3) accept/reject left to right
        new_toks, rejected = [], False
        for i in range(n_draft):
            x = draft_toks[i]
            p = _dist(tgt_logits[:, i, :], temperature)
            q = draft_dists[i]
            n_proposed += 1
            ratio = p[0, x[0, 0]] / q[0, x[0, 0]]   # q(x) > 0: x was drawn from q
            if torch.rand((), generator=generator) < ratio:
                new_toks.append(x)
                n_accepted += 1
            else:
                residual = (p - q).clamp(min=0.0)
                residual = residual / residual.sum(-1, keepdim=True)
                new_toks.append(torch.multinomial(residual, 1, generator=generator))
                rejected = True
                break
        if not rejected:  # all accepted: bonus token from the target's last dist
            p_last = _dist(tgt_logits[:, n_draft, :], temperature)
            new_toks.append(torch.multinomial(p_last, 1, generator=generator))
        tokens = torch.cat([tokens] + new_toks, dim=1)

    return tokens[:, :target_len], n_accepted / max(n_proposed, 1)
```

With the provided model pair, acceptance is ~0.4 greedy / ~0.9 at
temperature 1.

Why greedy falls out for free: with one-hot p and q, the ratio is 1 when
the argmaxes agree (accept) and 0 when they don't (reject); the residual
`max(p - q, 0)` is then one-hot at the *target's* argmax. So every emitted
token is the target argmax given its prefix — exactly greedy decoding.

### Part 4 — reference answers

**1. Unbiasedness.** For any token x, P(emit x) = P(draft proposes x and
it's accepted) + P(rejection and x drawn from residual)
= q(x)·min(1, p(x)/q(x)) + P(reject)·(p(x) − q(x))⁺ / Σ_y (p(y) − q(y))⁺.
The first term is min(q(x), p(x)). Total rejection probability is
Σ_y q(y)(1 − min(1, p(y)/q(y))) = Σ_y (q(y) − min(p(y), q(y)))
= Σ_y (q(y) − p(y))⁺ = Σ_y (p(y) − q(y))⁺ (both equal 1 − Σ min(p,q)).
So the second term is exactly (p(x) − q(x))⁺, and
min(p(x), q(x)) + (p(x) − q(x))⁺ = p(x). ∎

**2. KV memory.** Per token: 32 layers × 8 KV heads × 128 dims × 2 (K and
V) × 2 bytes = **131,072 B = 128 KiB/token**. A 32k-context request needs
32,768 × 128 KiB = **4 GiB**. With 24 − 16 = 8 GB free: **2 concurrent
requests** (ignoring activations — mention that caveat). This is why KV
cache, not weights, bounds batch size at long context, and why GQA/MQA and
cache quantization exist.

**3. GQA.** KV cache size and KV memory bandwidth scale with the number of
KV heads, not query heads. MHA (KV heads = query heads) maximizes quality
but the cache is huge; MQA (1 KV head) shrinks it 64x but measurably hurts
quality and hurts tensor-parallel sharding (the single head must be
replicated). GQA is the middle point: e.g. 8 KV heads for 64 query heads
gives an 8x smaller cache at near-MHA quality, and shards cleanly across 8
TP ranks.

**4. When speculation hurts.** (a) Low acceptance rate — target and draft
disagree (out-of-domain prompts, high temperature): you pay k draft
forwards + 1 target forward and mostly keep 1 token. Break-even needs
roughly acceptance × k × c_target > k × c_draft + verification overhead.
(b) Compute-bound serving — at large batch sizes the GPU is already
saturated with useful work; speculation's extra FLOPs (draft passes +
target verification of tokens that get thrown away) displace real
throughput. Speculation wins in the memory-bandwidth-bound, small-batch,
latency-sensitive regime. (c) Also: draft too expensive relative to target,
or draft KV/weights evicting cache the target needs.
