"""SwiGLU MLP: ``down(SiLU(gate(x)) * up(x))`` — a gated variant of the
standard transformer feedforward block. Three projections instead of two
(gate/up/down), each without bias (matching ``tinymind.model.layers.Linear``'s
default and current Llama-family convention).
"""
from __future__ import annotations

import numpy as np

from tinymind.model import fused
from tinymind.model.config import ModelConfig
from tinymind.model.linear import Linear
from tinymind.model.module import Module
from tinymind.model.tensor import Tensor


class SwiGLUMLP(Module):
    def __init__(self, config: ModelConfig, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, rng=rng)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, rng=rng)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        if fused.enabled():
            return self.down_proj(fused.swiglu(self.gate_proj(x), self.up_proj(x)))
        gate = self.gate_proj(x).silu()  # reference path
        up = self.up_proj(x)
        return self.down_proj(gate * up)
