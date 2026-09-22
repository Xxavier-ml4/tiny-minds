"""RMSNorm: ``y = x / sqrt(mean(x^2, axis=-1) + eps) * weight``.

No mean-subtraction (that's LayerNorm) — RMSNorm normalizes only by scale,
which is why it needs no bias term and one fewer reduction than LayerNorm.
Implemented from scratch on ``tinymind.model.tensor.Tensor`` (see that
module's docstring for why); correctness is verified against a numerical
gradient in ``tests/model/test_norm.py``, not assumed from the formula
looking right.
"""
from __future__ import annotations

import numpy as np

from tinymind.model import fused
from tinymind.model.module import Module
from tinymind.model.tensor import Tensor


class RMSNorm(Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Tensor(np.ones(hidden_size, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if fused.enabled():
            return fused.rmsnorm(x, self.weight, self.eps)
        # Reference path. mean(x^2, axis=-1, keepdims=True) + eps, then rsqrt — every step
        # here is a Tensor op, so this whole computation is differentiable
        # through the autograd engine with no special-cased backward of its
        # own; the engine's own primitives (pow/mean/add/rsqrt/mul) already
        # each have a checked gradient (tests/model/test_tensor.py).
        variance = (x ** 2).mean(axis=-1, keepdims=True)
        normalized = x * (variance + self.eps).rsqrt()
        return normalized * self.weight
