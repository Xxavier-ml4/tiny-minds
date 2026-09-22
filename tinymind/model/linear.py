"""``Linear``: ``y = x @ W^T + b``.

Kept in its own module (rather than ``layers.py``, where the brief's file
list originally suggested it) because ``attention.py``, ``mlp.py``, and
``layers.py`` (``TransformerBlock``) all need it, and ``layers.py`` also
needs ``attention.py``/``mlp.py`` — putting ``Linear`` there created a
circular import. This file has no TinyMind-internal dependencies beyond
``tensor.py``/``module.py``, so it can sit underneath everything else.
"""
from __future__ import annotations

import numpy as np

from tinymind.model import fused
from tinymind.model.module import Module
from tinymind.model.tensor import Tensor


class Linear(Module):
    """Weight orientation ``[out_features, in_features]`` — see
    ``docs/architecture/native-model-contract.md`` for why (a native GEMM
    kernel wants each output row contiguous along the reduction axis, the
    same reasoning the Needle analysis noted for the `.cact` format's
    tensor layout, needle-analysis.md section 14).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        # Standard "scaled" initialization (a la GPT-2/nanoGPT): normal with
        # std = 1/sqrt(in_features), which keeps activation variance roughly
        # constant across a matmul at init regardless of layer width — see
        # docs/architecture/model-implementation.md's initialization section.
        std = 1.0 / np.sqrt(in_features)
        weight_data = rng.normal(0.0, std, size=(out_features, in_features)).astype(np.float32)
        self.weight = Tensor(weight_data, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if fused.enabled():
            out = fused.linear(x, self.weight)
        else:  # reference path: a transpose node plus a batched matmul
            out = x @ self.weight.transpose(1, 0)
        if self.bias is not None:
            out = out + self.bias
        return out
