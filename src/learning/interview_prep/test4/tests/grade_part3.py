"""Grader for Part 3. DO NOT OPEN until you're done — contains the reference."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from _harness import run
from part3_optim import AdamW, accumulated_grads, get_lr


def _mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 3))


def test_adamw_matches_torch():
    m1, m2 = _mlp(), _mlp()
    m2.load_state_dict(m1.state_dict())
    opt1 = AdamW(m1.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=3e-3, betas=(0.9, 0.95),
                             eps=1e-8, weight_decay=0.1)
    torch.manual_seed(42)
    for _ in range(25):
        x = torch.randn(16, 10)
        y = torch.randint(0, 3, (16,))
        for m, opt in ((m1, opt1), (m2, opt2)):
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(
            p1, p2, rtol=1e-4, atol=1e-6,
            msg="drifted from torch.optim.AdamW after 25 steps "
                "(decay coupled into the gradient? bias correction off?)",
        )


def test_adamw_decay_is_decoupled():
    # single param, zero gradient: only decay should act -> geometric shrink
    p = nn.Parameter(torch.tensor([10.0]))
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    # decoupled: p = 10 * (1 - 0.1*0.5) = 9.5; Adam term is 0/(0+eps) = 0
    torch.testing.assert_close(
        p.detach(), torch.tensor([9.5]), rtol=0, atol=1e-5,
        msg="with zero grad, one step should be p *= (1 - lr*wd) exactly",
    )


def test_adamw_skips_none_grads():
    p1 = nn.Parameter(torch.ones(3))
    p2 = nn.Parameter(torch.ones(3))
    opt = AdamW([p1, p2], lr=0.1, weight_decay=0.0)
    p1.grad = torch.ones(3)
    opt.step()  # p2.grad is None — must not crash, must not move p2
    assert torch.equal(p2.detach(), torch.ones(3)), "p2 had no grad but moved"
    assert not torch.equal(p1.detach(), torch.ones(3)), "p1 should have moved"


def test_adamw_zero_grad():
    p = nn.Parameter(torch.ones(3))
    opt = AdamW([p], lr=0.1)
    p.grad = torch.ones(3)
    opt.zero_grad()
    assert p.grad is None or torch.equal(p.grad, torch.zeros(3)), \
        "zero_grad must clear gradients"


def test_lr_schedule():
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup_steps=10, total_steps=100)
    cases = [
        (0, 1e-3 * 1 / 10),
        (4, 1e-3 * 5 / 10),
        (9, 1e-3),
        (10, 1e-3),                                   # cosine progress = 0
        (55, 1e-4 + 0.5 * 9e-4),                      # progress = 0.5
        (100, 1e-4),                                  # progress = 1
        (250, 1e-4),                                  # past the end
    ]
    for step, expected in cases:
        got = get_lr(step, **kw)
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12), \
            f"step {step}: expected {expected:.6e}, got {got:.6e}"
    # quarter point sanity: cos curve, not linear
    got = get_lr(32, **kw)  # progress ~0.2444
    expected = 1e-4 + 0.5 * 9e-4 * (1 + math.cos(math.pi * 22 / 90))
    assert math.isclose(got, expected, rel_tol=1e-9), \
        f"step 32: expected {expected:.6e}, got {got:.6e}"


def _grad_reference(model, x, y, loss_fn):
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    return loss.item(), grads


def _check_accum(N, micro):
    model = _mlp(seed=3)
    torch.manual_seed(N)
    x = torch.randn(N, 10)
    y = torch.randint(0, 3, (N,))
    ref_loss, ref_grads = _grad_reference(model, x, y, F.cross_entropy)
    loss = accumulated_grads(model, x, y, F.cross_entropy, micro)
    assert isinstance(loss, float), f"must return a float, got {type(loss)}"
    assert math.isclose(loss, ref_loss, rel_tol=1e-5), \
        f"returned loss {loss:.6f} != full-batch loss {ref_loss:.6f}"
    for p, g_ref in zip(model.parameters(), ref_grads):
        assert p.grad is not None, "a parameter ended up with no grad"
        torch.testing.assert_close(
            p.grad, g_ref, rtol=1e-4, atol=1e-6,
            msg=f"accumulated grads != full-batch grads (N={N}, micro={micro})",
        )


def test_accum_divisible():
    _check_accum(N=12, micro=4)


def test_accum_single_chunk():
    _check_accum(N=8, micro=16)  # micro bigger than batch


def test_accum_ragged_tail():
    _check_accum(N=10, micro=4)  # chunks of 4, 4, 2 — the classic trap


def test_accum_clears_stale_grads():
    model = _mlp(seed=3)
    for p in model.parameters():
        p.grad = torch.full_like(p, 100.0)  # poison
    torch.manual_seed(5)
    x = torch.randn(8, 10)
    y = torch.randint(0, 3, (8,))
    _, ref_grads = _grad_reference(model, x, y, F.cross_entropy)
    # poison again after reference pass
    for p in model.parameters():
        p.grad = torch.full_like(p, 100.0)
    accumulated_grads(model, x, y, F.cross_entropy, 4)
    for p, g_ref in zip(model.parameters(), ref_grads):
        torch.testing.assert_close(
            p.grad, g_ref, rtol=1e-4, atol=1e-6,
            msg="stale .grad from a previous step leaked into the result",
        )


if __name__ == "__main__":
    print("Part 3 — training mechanics from scratch")
    ok = run([
        ("adamw_matches_torch_25_steps", test_adamw_matches_torch),
        ("adamw_decay_is_decoupled", test_adamw_decay_is_decoupled),
        ("adamw_skips_none_grads", test_adamw_skips_none_grads),
        ("adamw_zero_grad", test_adamw_zero_grad),
        ("lr_schedule", test_lr_schedule),
        ("accum_divisible", test_accum_divisible),
        ("accum_single_chunk", test_accum_single_chunk),
        ("accum_ragged_tail", test_accum_ragged_tail),
        ("accum_clears_stale_grads", test_accum_clears_stale_grads),
    ])
    raise SystemExit(0 if ok else 1)
