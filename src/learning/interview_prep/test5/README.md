# Test 5 — Inference Engineering: Sampling, KV Cache, Speculative Decoding

**Format:** frontier-lab research-engineering onsite ("ML coding from scratch"
round, in the style of Anthropic / OpenAI / DeepMind inference-team loops).
**Difficulty:** moderate–hard. **Time budget:** ~2.5–3.5 hours total.

## Scenario

You've just joined the inference team. `challenge.py` contains a small but
architecturally realistic Llama-style LM (RMSNorm, RoPE, grouped-query
attention, SwiGLU) and `generate_slow`, a generation loop that is *correct*
but recomputes the entire prefix on every step. Serving is on fire. Ship a
fast decoder **without changing what the model outputs** — the graders
(`test_challenge.py`) compare your fast paths against the slow reference at
the token level, exactly.

## Rules

- No internet, no LLMs. PyTorch API docs (`help(...)`) are fine.
- **Do not modify `test_challenge.py`.** Everything else in `challenge.py`
  is yours to restructure — you are *expected* to modify `Attention.forward`,
  `TinyLM.forward`, and `KVCache`.
- CPU only, and everything is deterministic — if a test flakes, it's a bug.

```sh
cd test5
uv run --project ~/nvim/learning --with pytest pytest test_challenge.py -v
uv run --project ~/nvim/learning --with pytest pytest test_challenge.py -k Part1   # one part
```

## Part 1 — `sample_token` (~20–30 min)

Temperature / top-k / top-p (nucleus) sampling, vectorized over the batch.
The exact semantics are in the docstring. The tests check the *distribution*
of your samples statistically, including the exact nucleus boundary and
top-k + top-p composed together. `generate_slow` depends on this, so do it
first.

## Part 2 — KV-cache incremental decoding (~45–75 min)

Implement `KVCache` and `generate_fast`, and extend the model's forward pass
to accept `(cache, start_pos)`. The graders check:

- greedy `generate_fast` output `==` `generate_slow` output, token-exact,
  for prompt lengths 1, 5, 33, 64;
- *sampled* generation matches too, given identically-seeded generators;
- **chunked prefill**: pushing a 50-token prompt through the cache in chunks
  of 17 / 1 / 24 / 8 must reproduce the full uncached forward's logits to
  1e-4 — so your cached path must handle multi-token chunks, not just
  single-token decode;
- ≥ 2x wall-clock speedup on prompt=384, 128 generated tokens (the
  reference solution gets ~5x).

Think carefully about (a) which *absolute* positions get which RoPE
rotation, and (b) what the causal mask looks like when there are more keys
than queries. These two details are where almost everyone loses time.

## Part 3 — speculative decoding (~60–90 min)

Implement `generate_speculative` per Leviathan et al. 2023 (the algorithm is
specified in the docstring — implementing it *correctly* is the hard part,
not remembering it). `build_models()` gives you a target and a cheap
"early-exit" draft that agrees with it often but not always. The graders
check:

- `temperature=0` reproduces the target's greedy decode **exactly**, no
  matter what the draft proposes (this must fall out of the algorithm — no
  special-casing the output);
- output length is exactly `n_new_tokens` even though rounds emit a
  variable number of tokens;
- **unbiasedness**: over 1500 runs at temperature 0.25 (where target and
  draft sharply disagree), the empirical distribution of the first
  generated token must match the *target's* distribution. Shortcuts like
  "keep the draft's token on rejection" or "resample from p" fail this.

You may implement it with full (uncached) forward passes; correctness is
graded, not speed. Stretch goal: run both models through KV caches — note
you'll need cache *rollback* on rejection.

## Part 4 — written questions (~20 min, no code)

Answer in a few sentences each (solutions file has reference answers):

1. Prove that speculative decoding's accept/reject scheme yields samples
   distributed exactly as the target distribution p, for one step.
2. An 8B model has 32 layers, 8 KV heads, head dim 128, fp16 cache. How many
   bytes of KV cache per token? After 16 GB of weights on a 24 GB GPU, how
   many concurrent 32k-context requests fit?
3. Why do modern models use GQA instead of MHA or MQA? What does the KV-head
   count trade off?
4. When does speculative decoding *hurt* throughput? Give at least two
   distinct regimes.

## Grading yourself

- **Strong hire:** Parts 1–3 all green in ~3h, clean code, solid Part 4.
- **Hire:** Parts 1–2 green, Part 3 greedy test green, distribution test
  needed hints; Part 4 mostly right.
- **No hire:** Part 2 equivalence not achieved, or resorted to peeking.

`SOLUTIONS.md` has graduated hints at the top — read one level at a time.
