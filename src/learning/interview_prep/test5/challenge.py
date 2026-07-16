"""
Inference Engineering Challenge — see README.md for the full brief.

You are given a small but realistic Llama-style decoder-only LM (RMSNorm,
RoPE, grouped-query attention, SwiGLU) and a correct-but-quadratic
generation loop. Your job: sampling, KV-cache decoding, and speculative
decoding — WITHOUT changing what the model outputs.

Implement everything marked  >>> YOUR CODE <<< . You may (and will need to)
modify `Attention.forward`, `TinyLM.forward`, and `KVCache`. Do not change
the provided math of the no-cache forward pass — the tests compare your
fast paths against it bit-for-bit at the token level.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Part 0 — PROVIDED MODEL. Read it carefully; you will need to understand it.
# ============================================================================


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 1024
    rope_base: float = 10000.0

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def rope_cos_sin(positions: torch.Tensor, d_head: int, base: float):
    """cos/sin tables for a batch of ABSOLUTE positions. positions: (T,) int."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, d_head // 2, dtype=torch.float32) * 2.0 / d_head)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]  # (T, d_head//2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (T, D//2). Half-split rotation (Llama style)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Required public contract (the tests use exactly this much):
      - constructed as `KVCache(n_layers)`
      - passed to `model(tokens, cache=cache, start_pos=...)`
    Storage layout and update API are entirely up to you.
    """

    def __init__(self, n_layers: int):
        # ======================= >>> YOUR CODE (Part 2) <<< ==================
        raise NotImplementedError("Part 2")


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        d, dh = cfg.d_model, cfg.d_head
        self.wq = nn.Linear(d, cfg.n_heads * dh, bias=False)
        self.wk = nn.Linear(d, cfg.n_kv_heads * dh, bias=False)
        self.wv = nn.Linear(d, cfg.n_kv_heads * dh, bias=False)
        self.wo = nn.Linear(cfg.n_heads * dh, d, bias=False)

    def forward(self, x, cos, sin, cache: "KVCache | None" = None):
        B, T, _ = x.shape
        H, Hkv, D = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.d_head

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)      # (B, H, T, D)
        k = self.wk(x).view(B, T, Hkv, D).transpose(1, 2)    # (B, Hkv, T, D)
        v = self.wv(x).view(B, T, Hkv, D).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            # =================== >>> YOUR CODE (Part 2) <<< ==================
            # Attend over the cached keys/values as well as the new ones.
            # Think hard about what the causal mask looks like when the
            # number of queries and the number of keys differ.
            raise NotImplementedError("Part 2")

        # grouped-query attention: each kv head serves H // Hkv query heads
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)      # (B, H, T, T)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

        out = scores.softmax(-1) @ v                         # (B, H, T, D)
        return self.wo(out.transpose(1, 2).reshape(B, T, H * D))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = 2 * cfg.d_model
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, cache: KVCache | None = None,
                start_pos: int = 0) -> torch.Tensor:
        """tokens: (B, T) -> logits (B, T, vocab).

        With a cache, `tokens` are the NEW tokens only, and `start_pos` is
        the absolute position of the first of them (i.e. the number of
        tokens already in the cache).
        """
        B, T = tokens.shape
        # NOTE (Part 2): as written, this is only correct for the no-cache
        # case (cache=None, start_pos=0). Generalize it.
        positions = torch.arange(T, device=tokens.device)
        cos, sin = rope_cos_sin(positions, self.cfg.d_head, self.cfg.rope_base)
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        return self.lm_head(self.norm(x))


def build_models(seed: int = 0):
    """Deterministically build the (target, draft) model pair.

    The draft is an "early-exit" version of the target: same embedding and
    LM head, but only the first 2 of 4 blocks, plus a little weight noise.
    It agrees with the target often enough for speculation to be useful,
    but disagrees often enough that correctness bugs are visible.
    """
    torch.manual_seed(seed)
    target = TinyLM(Config()).eval()
    draft = TinyLM(Config(n_layers=2)).eval()
    draft.embed.load_state_dict(target.embed.state_dict())
    draft.norm.load_state_dict(target.norm.state_dict())
    draft.lm_head.load_state_dict(target.lm_head.state_dict())
    for i in range(2):
        draft.blocks[i].load_state_dict(target.blocks[i].state_dict())
    g = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for p in draft.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)
    return target, draft


@torch.no_grad()
def generate_slow(model, prompt, n_new_tokens, temperature=0.0, top_k=None,
                  top_p=None, generator=None):
    """Reference generation: recomputes the FULL prefix every step. Correct
    but O(T^2) forward work. prompt: (B, T) -> (B, T + n_new_tokens).

    (Uses your Part 1 `sample_token`, so finish that first.)
    """
    tokens = prompt.clone()
    for _ in range(n_new_tokens):
        logits = model(tokens)[:, -1, :]
        next_tok = sample_token(logits, temperature, top_k, top_p, generator)
        tokens = torch.cat([tokens, next_tok[:, None]], dim=1)
    return tokens


# ============================================================================
# Part 1 — sampling
# ============================================================================


def sample_token(logits: torch.Tensor, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one token id per row. logits: (B, V) -> (B,), dtype long.

    Semantics (in this order):
      1. temperature == 0.0 means greedy: return the argmax.
      2. Otherwise scale logits by 1/temperature.
      3. If top_k is set: keep only the k highest-logit tokens.
      4. If top_p is set: nucleus sampling — keep the SMALLEST prefix of
         tokens, in descending probability order, whose cumulative
         probability is >= top_p. (Applied after top-k, on the
         renormalized distribution.)
      5. Sample from the renormalized remaining distribution, drawing
         randomness from `generator` (pass it to torch.multinomial).

    Must be vectorized over the batch — the tests call this with B = 20000.
    """
    # ========================= >>> YOUR CODE (Part 1) <<< ====================
    raise NotImplementedError("Part 1")


# ============================================================================
# Part 2 — KV-cache decoding
# ============================================================================


@torch.no_grad()
def generate_fast(model, prompt, n_new_tokens, temperature=0.0, top_k=None,
                  top_p=None, generator=None):
    """Exactly the same contract and OUTPUT as generate_slow — same tokens
    for the same arguments and generator state — but O(T) total forward
    work, using a KVCache: prefill the prompt once, then feed one token at
    a time. Must beat generate_slow by >= 2x on the benchmark in the tests.
    """
    # ========================= >>> YOUR CODE (Part 2) <<< ====================
    raise NotImplementedError("Part 2")


# ============================================================================
# Part 3 — speculative decoding
# ============================================================================


@torch.no_grad()
def generate_speculative(target, draft, prompt, n_new_tokens, n_draft=4,
                         temperature=1.0, generator=None):
    """Speculative decoding (Leviathan et al. 2023, arXiv:2211.17192).
    Batch size is always 1.

    Each round:
      1. The draft model proposes `n_draft` tokens autoregressively, sampling
         each from its own distribution q at `temperature`.
      2. The target model scores the context + all proposals in ONE forward
         pass, giving its distribution p at each of the n_draft + 1 positions.
      3. Proposals are accepted left to right: proposal x is accepted with
         probability min(1, p(x) / q(x)). On the first rejection, sample a
         replacement token from the residual distribution
         norm(max(p - q, 0)) and end the round. If all n_draft proposals are
         accepted, sample one bonus token from the target's distribution at
         the final position.

    Requirements:
      - Output must have EXACTLY n_new_tokens new tokens (rounds emit a
        variable number — truncate the overshoot).
      - The output distribution must be identical to sampling from the
        target model alone. In particular temperature=0 must reproduce the
        target's greedy decode exactly, whatever the draft does.
      - Draw all randomness from `generator`.

    Returns (tokens, acceptance_rate):
      tokens: (1, prompt_len + n_new_tokens)
      acceptance_rate: fraction of draft proposals accepted, in [0, 1].
    """
    # ========================= >>> YOUR CODE (Part 3) <<< ====================
    raise NotImplementedError("Part 3")
