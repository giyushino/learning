"""
PART 3 — Training mechanics from scratch (~30 min)

The training-loop round. Every lab that pretrains or finetunes (all of them)
cares that you know what the optimizer actually does to a parameter, how the
LR schedule is shaped, and how to fit a big batch on a small GPU without
changing the math.

  1. AdamW      — from scratch, no torch.optim. Must match torch exactly.
  2. get_lr     — linear warmup + cosine decay schedule.
  3. accumulated_grads — gradient accumulation that is EXACTLY equivalent to
                  a full-batch backward, including ragged final micro-batch.

Grade yourself:  uv run python tests/grade_part3.py   (from the test4/ dir)
Do NOT open the grader — it contains a reference implementation.
"""

import math

import torch


class AdamW:
    """Decoupled-weight-decay Adam (Loshchilov & Hutter), matching
    torch.optim.AdamW semantics bit-for-bit (up to float associativity).

    Per parameter p with gradient g at step t (1-indexed):
        p     <- p - lr * weight_decay * p          # decoupled decay FIRST
        m     <- beta1 * m + (1 - beta1) * g
        v     <- beta2 * v + (1 - beta2) * g^2
        m_hat <- m / (1 - beta1^t)
        v_hat <- v / (1 - beta2^t)
        p     <- p - lr * m_hat / (sqrt(v_hat) + eps)

    Parameters whose .grad is None are skipped. Use torch.no_grad() (or .data)
    so the update itself isn't tracked by autograd.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params = [p for p in params]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # TODO: any state you need (m, v, step count)

    def zero_grad(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    """LR schedule used by essentially every modern pretraining run.

    - Warmup:  for step < warmup_steps:
                 lr = max_lr * (step + 1) / warmup_steps
    - Cosine:  for warmup_steps <= step <= total_steps:
                 progress = (step - warmup_steps) / (total_steps - warmup_steps)
                 lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    - After:   for step > total_steps:  lr = min_lr
    """
    raise NotImplementedError


def accumulated_grads(model, x, y, loss_fn, micro_batch_size):
    """Populate model .grads via gradient accumulation.

    x: (N, ...) inputs, y: (N, ...) targets. loss_fn(pred, target) uses MEAN
    reduction. Process the batch in chunks of micro_batch_size (the last chunk
    may be smaller!) so that afterwards every parameter's .grad EXACTLY equals
    (up to float noise) what

        model.zero_grad(); loss_fn(model(x), y).backward()

    would have produced. Zero/clear existing grads first. Do not call any
    optimizer. Return the full-batch loss value as a float (for logging).

    Hint: mean-of-means is not the mean when the chunks aren't equal sized.
    """
    raise NotImplementedError
