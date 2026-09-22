"""``TransformerBlock``: the pre-norm residual block that stacks attention
and MLP. ``Linear`` used to live in this file too; see
``tinymind/model/linear.py`` for why it moved (circular import).

Per the engineering brief section 8: ``x = x + attention(norm(x)); x = x +
mlp(norm(x))`` — pre-norm, residual, nothing else. No tool/runtime logic
belongs here (brief section 8's own instruction, repeated in section 18);
this file only ever produces hidden states, never interprets them.
"""
from __future__ import annotations

import numpy as np

from tinymind.model.attention import CausalSelfAttention
from tinymind.model.config import ModelConfig
from tinymind.model.linear import Linear  # re-exported for convenience/backward compatibility
from tinymind.model.mlp import SwiGLUMLP
from tinymind.model.module import Module
from tinymind.model.norm import RMSNorm
from tinymind.model.tensor import Tensor

__all__ = ["Linear", "TransformerBlock"]


class TransformerBlock(Module):
    def __init__(self, config: ModelConfig, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        self.attention = CausalSelfAttention(config, rng=rng)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        self.mlp = SwiGLUMLP(config, rng=rng)

    def forward(self, x: Tensor, cos: np.ndarray = None, sin: np.ndarray = None,
               kv_cache=None, layer_idx: int = 0, position_ids: np.ndarray = None,
               attention_bias: np.ndarray = None) -> Tensor:
        x = x + self.attention(self.attn_norm(x), cos, sin, kv_cache=kv_cache,
                               layer_idx=layer_idx, position_ids=position_ids,
                               attention_bias=attention_bias)
        x = x + self.mlp(self.mlp_norm(x))
        return x
