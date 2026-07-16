# SOLUTIONS — do not open until you've graded yourself

## Part 1 — Attention

```python
import math
import torch

def scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if is_causal:
        T_q, T_k = q.shape[-2], k.shape[-2]
        causal = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    return scores.softmax(dim=-1) @ v

class MultiHeadAttention(nn.Module):
    # __init__ as given
    def forward(self, x, is_causal=True):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        rep = self.n_heads // self.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)
```

**Gotchas the graders probe:**
- Mask with `-inf` **before** softmax, not by zeroing probabilities after
  (`sdpa_masked_rows_exact` uses huge logits to amplify any leakage).
- `view(B, T, H, hd).transpose(1, 2)` — the classic wrong move is
  `view(B, H, T, hd)` directly, which silently shuffles features. (You debugged
  this in test3; here you have to *produce* it correctly.)
- GQA sharing must be `repeat_interleave` (query head `i` → kv head
  `i // (n_heads//n_kv_heads)`), not `repeat`, which interleaves the groups
  the wrong way.
- Scale by `sqrt(head_dim)`, not `sqrt(d_model)`.
- Talking points if asked "why GQA?": KV cache size scales with n_kv_heads;
  GQA/MQA cut cache memory (and memory bandwidth at decode) by n_heads/n_kv_heads
  with minor quality loss — this is *the* small-lab efficiency question.

## Part 2 — KV cache & sampling

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, past_kv=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        T_total = k.shape[2]
        t_past = T_total - T
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        q_pos = torch.arange(T, device=x.device).unsqueeze(1) + t_past
        k_pos = torch.arange(T_total, device=x.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        out = scores.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), (k, v)

def top_k_filter(logits, k):
    vals, idx = torch.topk(logits, k, dim=-1)
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, idx, vals)
    return out

def top_p_filter(logits, p):
    probs = logits.softmax(dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    # drop a token iff cumulative prob BEFORE it already reached p
    drop_sorted = (cum - sorted_probs) >= p
    drop = drop_sorted.gather(-1, sorted_idx.argsort(dim=-1))
    return logits.masked_fill(drop, float("-inf"))

def sample(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=None,
             top_p=None, generator=None):
    logits, kvs = model(idx)              # one prefill pass
    for i in range(max_new_tokens):
        next_tok = sample(logits[:, -1], temperature=temperature,
                          top_k=top_k, top_p=top_p, generator=generator)
        idx = torch.cat([idx, next_tok.unsqueeze(1)], dim=1)
        if i < max_new_tokens - 1:        # last token needs no forward
            logits, kvs = model(next_tok.unsqueeze(1), kvs)
    return idx
```

**Gotchas the graders probe:**
- **The mask offset** (`cache_multi_token_chunk`): with a cache, query at
  local index `i` sits at *global* position `t_past + i` and may see keys
  `0..t_past+i`. Using a plain `tril(T, T_total)` without the offset blocks
  new tokens from seeing the cached prefix (or lets them see the future).
  For the common T=1 decode step any mask bug is invisible — that's why the
  grader feeds a 6-token chunk against a 4-token cache.
- Return the **pre-softmax-filtered original logits** in the filters (kept
  values unchanged), and don't mutate the input in place.
- top-p is defined on **sorted probabilities**: keep the smallest prefix with
  cumulative prob ≥ p, always ≥ 1 token. The `(cum - probs) >= p` trick avoids
  the off-by-one where the crossing token gets dropped.
- Temperature **divides** the logits. `sample_temperature_direction` catches
  multiplying.
- `generate` must feed exactly one token per decode step — the grader wraps
  the attention modules and counts what they see. Recomputing the full prefix
  each step passes every *correctness* test but fails this one; that's the
  difference between knowing what a KV cache is and having built one.
- Talking point: cache memory = 2 · n_layers · n_heads · head_dim · T · bytes
  per token per sequence — why long-context serving is memory-bound and why
  GQA/MQA/sliding-window exist.

## Part 3 — Training mechanics

```python
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params = [p for p in params]
        self.lr, self.betas, self.eps = lr, betas, eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.weight_decay)      # decoupled decay
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            p.sub_(self.lr * (m / bc1) / ((v / bc2).sqrt() + self.eps))

def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def accumulated_grads(model, x, y, loss_fn, micro_batch_size):
    for p in model.parameters():
        p.grad = None
    N = x.shape[0]
    total_loss = 0.0
    for i in range(0, N, micro_batch_size):
        xb, yb = x[i:i + micro_batch_size], y[i:i + micro_batch_size]
        micro_loss = loss_fn(model(xb), yb)
        (micro_loss * (xb.shape[0] / N)).backward()   # weight by chunk size
        total_loss += micro_loss.item() * xb.shape[0] / N
    return total_loss
```

**Gotchas the graders probe:**
- **Decoupled** weight decay: shrink `p` directly by `lr·wd·p`, *before* the
  Adam update, and never fold `wd·p` into the gradient (that's plain Adam+L2,
  and it fails both `adamw_matches_torch` and `adamw_decay_is_decoupled` —
  the zero-grad test isolates it: one step must be exactly `p *= 1 - lr·wd`).
- Bias correction with a step counter that starts at 1. Forgetting it drifts
  from torch within a couple of steps.
- The update must run under `torch.no_grad()` (or on `.data`) or autograd
  tracks it / errors on leaf in-place ops.
- Skip params with `p.grad is None` — frozen params are everywhere in
  finetuning (LoRA!).
- Grad accumulation: with mean-reduction loss, scale each micro-loss by
  `n_micro / N`, **not** `1 / n_chunks`. Mean-of-means only equals the mean
  when chunks are equal-sized — the N=10, micro=4 test (chunks 4/4/2) is the
  trap. Also clear stale grads first (`accum_clears_stale_grads`).
- Talking point: why warmup at all — Adam's second-moment estimate is garbage
  for the first ~1/(1-β₂) steps, so early LR must be small; and why min_lr
  floor instead of decaying to 0.

## Rubric recap

| Signal | Where it shows |
|---|---|
| Head plumbing fluency | part 1 view/transpose, GQA repeat_interleave |
| Masking correctness under pressure | sdpa bool mask, cache offset |
| Knows inference beyond `model.generate()` | KV cache, one-token decode steps |
| Optimizer literacy | decoupled decay, bias correction |
| "Same math, less memory" engineering | ragged gradient accumulation |
