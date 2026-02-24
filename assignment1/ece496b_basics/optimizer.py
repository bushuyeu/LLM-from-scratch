from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Key difference from vanilla Adam: weight decay is applied directly to
    the parameters (p -= lr * wd * p) rather than being folded into the
    gradient. This decoupling makes weight decay behave consistently
    regardless of the adaptive learning rate.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize state on first step
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)       # first moment (mean)
                    state["v"] = torch.zeros_like(p)       # second moment (variance)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # Update biased moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)  # m = β1·m + (1-β1)·g
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v = β2·v + (1-β2)·g²

                # Bias correction — compensates for zero-initialization
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Adam update: p -= lr · m̂ / (√v̂ + ε)
                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

                # Decoupled weight decay: p -= lr · λ · p
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    Three phases:
      1. Linear warmup:  0 → max_lr over [0, warmup_iters)
      2. Cosine decay:   max_lr → min_lr over [warmup_iters, cosine_cycle_iters)
      3. Constant:        min_lr for [cosine_cycle_iters, ∞)
    """
    if it < warmup_iters:                                  # Phase 1: linear warmup
        return max_learning_rate * it / warmup_iters
    elif it >= cosine_cycle_iters:                         # Phase 3: constant min
        return min_learning_rate
    else:                                                  # Phase 2: cosine decay
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
