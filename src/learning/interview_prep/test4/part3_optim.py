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
        self.step_count = 0
        self.states = {}

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        beta1, beta2 = self.betas
        for param in self.params:
            if param.grad is None:
                continue
            # weight decay
            param.mul_(1 - self.lr * self.weight_decay)
            if param not in self.states:
                self.states[param] = {
                    "exp_avg": torch.zeros_like(param),
                    "exp_avg_sq": torch.zeros_like(param),
                }
            states = self.states[param]
            grad = param.grad

            m, v = states["exp_avg"], states["exp_avg_sq"]
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            m_corr = 1 - beta1 ** self.step_count
            v_corr = 1 - beta2 ** self.step_count
            step_size = self.lr / m_corr

            denominator = (v.sqrt() / math.sqrt(v_corr)).add_(self.eps)
            param.addcdiv_(m, denominator, value=-step_size)

        return
    

def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    """LR schedule used by essentially every modern pretraining run.

    - Warmup:  for step < warmup_steps:
                 lr = max_lr * (step + 1) / warmup_steps
    - Cosine:  for warmup_steps <= step <= total_steps:
                 progress = (step - warmup_steps) / (total_steps - warmup_steps)
                 lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    - After:   for step > total_steps:  lr = min_lr
    """
    if step < warmup_steps:
        lr = max_lr * (step + 1) / warmup_steps
    elif warmup_steps <= step and step <= total_steps:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    else:
        lr = min_lr

    return lr


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
    batch_size = x.size(0)
    model.zero_grad(set_to_none=True)
    loss = 0.0
    for start in range(0, batch_size, micro_batch_size):
        end = start + micro_batch_size
        x_chunk = x[start:end]
        y_chunk = y[start:end]
        mini_batch_size = x_chunk.size(0)
        chunked_loss = loss_fn(model(x_chunk), y_chunk)
        weighted = (chunked_loss * mini_batch_size) / batch_size
        weighted.backward()
        loss += weighted.item()

    return loss

        
