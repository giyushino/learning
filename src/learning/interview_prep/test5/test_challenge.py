"""Grading tests for the inference engineering challenge.

Run:  uv run --with pytest pytest test_challenge.py -v
Run one part:  ... -k part1

Do not modify this file. All tests are CPU-only and deterministic.
"""

import time

import pytest
import torch

from challenge import (
    build_models,
    generate_fast,
    generate_slow,
    generate_speculative,
    sample_token,
    KVCache,
)

torch.set_num_threads(max(1, torch.get_num_threads()))

V = 8
BASE_LOGITS = torch.log(
    torch.tensor([0.50, 0.30, 0.15, 0.02, 0.01, 0.01, 0.005, 0.005])
)


def _freqs(samples: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.bincount(samples, minlength=vocab).float() / samples.numel()


def _tv(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * (p - q).abs().sum().item()


# ---------------------------------------------------------------- Part 1


class TestPart1Sampling:
    def test_greedy(self):
        logits = torch.tensor([[0.1, 3.0, -1.0], [2.0, 0.0, 1.9]])
        out = sample_token(logits, temperature=0.0)
        assert out.dtype == torch.long
        assert out.shape == (2,)
        assert out.tolist() == [1, 0]

    def test_temperature_distribution(self):
        n = 20000
        g = torch.Generator().manual_seed(1)
        logits = BASE_LOGITS.expand(n, V)
        samples = sample_token(logits, temperature=0.5, generator=g)
        expected = (BASE_LOGITS / 0.5).softmax(-1)
        assert _tv(_freqs(samples, V), expected) < 0.03

    def test_top_k(self):
        n = 20000
        g = torch.Generator().manual_seed(2)
        logits = BASE_LOGITS.expand(n, V)
        samples = sample_token(logits, temperature=1.0, top_k=3, generator=g)
        assert set(samples.tolist()) <= {0, 1, 2}, "sampled outside top-k"
        expected = torch.zeros(V)
        expected[:3] = BASE_LOGITS[:3].softmax(-1)
        assert _tv(_freqs(samples, V), expected) < 0.03

    def test_top_p(self):
        n = 20000
        g = torch.Generator().manual_seed(3)
        logits = BASE_LOGITS.expand(n, V)
        # probs are [.5, .3, .15, ...] so the p=0.75 nucleus is exactly {0, 1}
        samples = sample_token(logits, temperature=1.0, top_p=0.75, generator=g)
        assert set(samples.tolist()) <= {0, 1}, "sampled outside the nucleus"
        freqs = _freqs(samples, V)
        assert abs(freqs[0].item() - 0.625) < 0.02  # 0.5 / 0.8

    def test_top_k_and_top_p_compose(self):
        n = 20000
        g = torch.Generator().manual_seed(4)
        logits = BASE_LOGITS.expand(n, V)
        # top_k=4 then top_p=0.9 over the renormalized top-4 -> {0, 1, 2}
        samples = sample_token(logits, temperature=1.0, top_k=4, top_p=0.9,
                               generator=g)
        assert set(samples.tolist()) <= {0, 1, 2}


# ---------------------------------------------------------------- Part 2


@pytest.fixture(scope="module")
def models():
    return build_models(seed=0)


def _prompt(seed: int, length: int, vocab: int = 256) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, length), generator=g)


class TestPart2KVCache:
    @pytest.mark.parametrize("prompt_len", [1, 5, 33, 64])
    def test_greedy_matches_slow(self, models, prompt_len):
        target, _ = models
        prompt = _prompt(10 + prompt_len, prompt_len)
        slow = generate_slow(target, prompt, 64, temperature=0.0)
        fast = generate_fast(target, prompt, 64, temperature=0.0)
        assert torch.equal(slow, fast), (
            f"divergence at index "
            f"{(slow[0] != fast[0]).nonzero()[0].item() if not torch.equal(slow, fast) else -1}"
        )

    def test_sampled_matches_slow_with_same_seed(self, models):
        target, _ = models
        prompt = _prompt(7, 20)
        outs = []
        for fn in (generate_slow, generate_fast):
            g = torch.Generator().manual_seed(1234)
            outs.append(fn(target, prompt, 48, temperature=0.8, top_p=0.95,
                           generator=g))
        assert torch.equal(outs[0], outs[1])

    def test_chunked_prefill_matches_full_forward(self, models):
        """Feeding the prompt through the cache in uneven chunks must give
        the same final logits as one full uncached forward pass."""
        target, _ = models
        prompt = _prompt(99, 50)
        full = target(prompt)  # (1, 50, V)

        cache = KVCache(target.cfg.n_layers)
        chunks, start = [17, 1, 24, 8], 0
        outs = []
        for c in chunks:
            outs.append(target(prompt[:, start:start + c], cache=cache,
                               start_pos=start))
            start += c
        chunked = torch.cat(outs, dim=1)
        torch.testing.assert_close(chunked, full, atol=1e-4, rtol=1e-4)

    def test_speedup(self, models):
        target, _ = models
        prompt = _prompt(42, 384)
        # warmup
        generate_fast(target, prompt, 8)
        generate_slow(target, prompt, 8)

        t0 = time.perf_counter()
        generate_slow(target, prompt, 128)
        slow_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        generate_fast(target, prompt, 128)
        fast_t = time.perf_counter() - t0

        assert fast_t * 2 < slow_t, (
            f"expected >=2x speedup, got {slow_t / fast_t:.2f}x "
            f"(slow={slow_t:.3f}s fast={fast_t:.3f}s)"
        )


# ---------------------------------------------------------------- Part 3


class TestPart3Speculative:
    @pytest.mark.parametrize("prompt_len,n_new", [(8, 40), (30, 63)])
    def test_greedy_matches_target_exactly(self, models, prompt_len, n_new):
        """With temperature=0, speculative decoding must be EXACTLY the
        target model's greedy output — regardless of the draft model."""
        target, draft = models
        prompt = _prompt(prompt_len, prompt_len)
        g = torch.Generator().manual_seed(0)
        spec, acceptance = generate_speculative(
            target, draft, prompt, n_new, n_draft=4, temperature=0.0,
            generator=g,
        )
        ref = generate_slow(target, prompt, n_new, temperature=0.0)
        assert spec.shape == ref.shape
        assert torch.equal(spec, ref)
        assert 0.0 <= acceptance <= 1.0

    def test_output_length_exact(self, models):
        """Rounds emit 1..k+1 tokens; output must still be exactly n_new."""
        target, draft = models
        prompt = _prompt(3, 12)
        for n_new in (1, 2, 5, 17):
            g = torch.Generator().manual_seed(5)
            out, _ = generate_speculative(target, draft, prompt, n_new,
                                          n_draft=4, temperature=1.0,
                                          generator=g)
            assert out.shape == (1, 12 + n_new)
            assert torch.equal(out[:, :12], prompt)

    def test_unbiased_at_low_temperature(self, models):
        """The distribution of the FIRST generated token must match the
        target model's distribution, not the draft's. At low temperature
        the two models disagree sharply, so shortcuts (e.g. keeping draft
        samples on rejection) fail this test."""
        target, draft = models
        temp = 0.25
        prompt = _prompt(77, 16)
        with torch.no_grad():
            p_target = (target(prompt)[0, -1, :] / temp).softmax(-1)
            p_draft = (draft(prompt)[0, -1, :] / temp).softmax(-1)
        # sanity: the test only has power if the models actually disagree
        assert _tv(p_target, p_draft) > 0.3

        n_trials = 1500
        g = torch.Generator().manual_seed(11)
        firsts = torch.empty(n_trials, dtype=torch.long)
        for i in range(n_trials):
            out, _ = generate_speculative(target, draft, prompt, 1,
                                          n_draft=3, temperature=temp,
                                          generator=g)
            firsts[i] = out[0, -1]
        tv = _tv(_freqs(firsts, target.cfg.vocab_size), p_target)
        assert tv < 0.12, f"first-token TV distance to target dist: {tv:.3f}"
