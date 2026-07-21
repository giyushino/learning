"""
PART 2 — KV-cache inference & sampling (~35 min)

The inference round. Labs shipping models (and especially the small-model labs
— Liquid AI, Zyphra, Arcee — whose whole pitch is cheap inference) ask exactly
this: make autoregressive decoding *not* recompute the whole prefix each step,
and implement the standard sampling strategies.

A tiny GPT is provided below and is CORRECT — do not modify anything except
the four TODO functions:

  1. CausalSelfAttention.forward  — attention that reads/extends a KV cache
  2. top_k_filter / top_p_filter  — logit filtering
  3. sample                       — temperature / top-k / top-p sampling
  4. generate                     — incremental decoding using the cache

Rules: no F.scaled_dot_product_attention, no HF. torch.multinomial is allowed.

Grade yourself:  uv run python tests/grade_part2.py   (from the test4/ dir)
Do NOT open the grader — it contains a reference implementation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. Cached causal self-attention
# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        B, S, _ = x.shape
        return x.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, _, S, _ = x.shape
        return x.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        

    def forward(self, x, past_kv=None):
        """Causal self-attention over (cached prefix + x).

        x:       (B, T_new, d_model) — embeddings for the NEW tokens only.
        past_kv: None, or a tuple (k, v) each of shape
                 (B, n_heads, T_past, head_dim) — the cache for tokens already
                 processed. The new tokens' global positions are
                 T_past .. T_past + T_new - 1.

        Returns: (out, (k_full, v_full))
          out:    (B, T_new, d_model)
          k_full: (B, n_heads, T_past + T_new, head_dim)  — cache to reuse
          v_full: same shape as k_full

        Masking: new query at global position t may attend to every key at
        global position <= t (all cached keys, plus causal within the new
        block). Get this right — the off-by-one here is the whole question.
        """
        Q = self.split_heads(self.q_proj(x))
        # this is the number of new tokens
        S = Q.size(-2)

        K = self.split_heads(self.k_proj(x))
        V = self.split_heads(self.v_proj(x))
        if past_kv is None:
            past_offset = 0
            curr_kv = (K, V)
        else:
            past_offset = past_kv[0].size(2)
            cached_k = torch.cat([past_kv[0], K], dim=2)
            cached_v = torch.cat([past_kv[1], V], dim=2)
            curr_kv = (cached_k, cached_v)

        attn_scores = torch.matmul(Q, curr_kv[0].transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_value = torch.finfo(attn_scores.dtype).min

        causal_mask = torch.triu(
            torch.ones(S, past_offset + S, device=Q.device, dtype=torch.bool),
            diagonal=past_offset + 1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, mask_value)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = self.combine_heads(torch.matmul(attn_probs, curr_kv[1]))
        return self.o_proj(attn_out), curr_kv
       


# ----------------------------------------------------------------------------
# Given, correct — do not modify
# ----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, past_kvs=None):
        """idx: (B, T_new) token ids for the new tokens only.

        past_kvs: None, or list (one per block) of (k, v) cache tuples.
        Returns (logits, new_kvs): logits (B, T_new, vocab), updated caches.
        """
        B, T = idx.shape
        t_past = 0 if past_kvs is None else past_kvs[0][0].shape[2]
        assert t_past + T <= self.max_seq_len
        pos = torch.arange(t_past, t_past + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
        new_kvs = []
        for block, past_kv in zip(self.blocks, past_kvs):
            x, kv = block(x, past_kv)
            new_kvs.append(kv)
        return self.lm_head(self.ln_f(x)), new_kvs


# ----------------------------------------------------------------------------
# 2. Logit filtering
# ----------------------------------------------------------------------------

def top_k_filter(logits, k):
    """logits: (B, V). Return a copy where everything outside the top-k
    logits per row is set to -inf. Kept logits are unchanged.
    Assume 1 <= k <= V. Do not modify the input in place.
    """
    kth = torch.topk(logits, k, dim=1, largest=True, sorted=True).values[..., -1, None]
    top_k = logits.masked_fill(logits < kth, float('-inf'))
    return top_k


def top_p_filter(logits, p):
    """Nucleus filtering. logits: (B, V), 0 < p <= 1.

    Per row: sort tokens by probability (softmax of logits) descending, keep
    the smallest prefix whose cumulative probability is >= p (i.e. include the
    token that crosses the threshold), set every other logit to -inf. Always
    keep at least the top-1 token. Kept logits are unchanged.
    Do not modify the input in place.
    """
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = probs.cumsum(dim=-1)
    remove = (cum_probs - probs) >= p
    sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))

    return torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)



# ----------------------------------------------------------------------------
# 3. Sampling
# ----------------------------------------------------------------------------

def sample(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """Sample one token id per row. logits: (B, V) -> LongTensor (B,).

    - temperature == 0.0 means greedy (argmax); ignore top_k/top_p.
    - Otherwise: scale logits by 1/temperature, then apply top_k_filter (if
      top_k is not None), then top_p_filter (if top_p is not None), then
      sample from the resulting distribution with torch.multinomial, passing
      `generator` through for reproducibility.
    """
    if temperature == 0.0:
        greedy_token_pos = torch.argmax(logits, -1)
        return greedy_token_pos
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    prob_dist = torch.softmax(logits, dim=-1)
    return torch.multinomial(input=prob_dist, num_samples=1, replacement=False, generator=generator).squeeze(-1)


# ----------------------------------------------------------------------------
# 4. Incremental decoding
# ----------------------------------------------------------------------------

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=None,
             top_p=None, generator=None):
    """Autoregressive decoding WITH the KV cache.

    idx: (B, T_prompt) prompt token ids. Returns (B, T_prompt + max_new_tokens).

    Requirements:
      - One prefill forward over the prompt, then exactly one forward per new
        token, each seeing only ONE new token (the grader checks this).
      - Use `sample` above to pick each next token.
    """
    raise NotImplementedError
