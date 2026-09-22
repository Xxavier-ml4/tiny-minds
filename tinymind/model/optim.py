"""AdamW — the brief's requested optimizer (section 19). Implemented
directly against ``tinymind.model.tensor.Tensor``'s ``.data``/``.grad``
NumPy arrays; there is no framework optimizer to wrap (see
``tinymind/model/tensor.py``'s module docstring).

Standard decoupled weight decay (Loshchilov & Hutter): weight decay is
applied directly to the parameter, not folded into the gradient the way
plain L2 regularization would be — the actual difference AdamW makes over
plain Adam+L2.

Phase 3B additions: moments are addressable **by parameter name**
(``state_dict``/``load_state_dict`` refuse any mismatch of names, shapes or
hyper-parameters and never load partially), weight decay can skip 1-D tensors
(norm gains), and a non-finite gradient raises *before* any state or weight is
touched — Phase 3A would have trained on a NaN.
"""
from __future__ import annotations

import numpy as np

from tinymind.model.tensor import Tensor


class NonFiniteGradientError(FloatingPointError):
    """The global gradient norm was NaN/Inf; the step was NOT applied."""


class OptimizerStateError(ValueError):
    """A stored optimizer state does not match this optimizer."""


class AdamW:
    def __init__(self, parameters: list[Tensor], learning_rate: float = 3e-4,
                betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                weight_decay: float = 0.01, names: list[str] | None = None,
                decay_min_ndim: int = 0) -> None:
        self.parameters = list(parameters)
        self.names = list(names) if names is not None else [f"param_{i}" for i in range(len(self.parameters))]
        if len(self.names) != len(self.parameters) or len(set(self.names)) != len(self.names):
            raise ValueError("names must be unique and one per parameter")
        self.lr = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # tensors with fewer dimensions than this get no decay (2 => RMSNorm gains are exempt)
        self.decay_min_ndim = int(decay_min_ndim)
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]
        self.step_count = 0

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()

    def step(self, grad_clip_norm: float | None = None) -> float:
        """Applies one optimizer step. Returns the (pre-clip) global
        gradient norm, useful for logging (brief section 19: "Log: ...").
        Raises ``NonFiniteGradientError`` — with weights and moments untouched —
        if that norm is NaN/Inf."""
        grads = [p.grad if p.grad is not None else np.zeros_like(p.data) for p in self.parameters]

        global_norm = float(np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads)))
        if not np.isfinite(global_norm):
            raise NonFiniteGradientError(f"global gradient norm is {global_norm}; step {self.step_count + 1} not applied")
        self.step_count += 1
        if grad_clip_norm is not None and global_norm > grad_clip_norm:
            scale = grad_clip_norm / (global_norm + 1e-6)
            grads = [g * scale for g in grads]

        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count

        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[i] / bias_correction1
            v_hat = self._v[i] / bias_correction2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0 and param.data.ndim >= self.decay_min_ndim:
                param.data -= self.lr * self.weight_decay * param.data  # decoupled weight decay
            param.data -= self.lr * update

        return global_norm

    # ---- checkpointing --------------------------------------------------
    def hyperparameters(self) -> dict[str, float | int]:
        return {"beta1": self.beta1, "beta2": self.beta2, "eps": self.eps, "weight_decay": self.weight_decay,
                "decay_min_ndim": self.decay_min_ndim}

    def state_dict(self) -> dict:
        """``{"hyper": {...}, "lr": float, "step_count": int, "names": [...],
        "arrays": {"m.<name>": ndarray, "v.<name>": ndarray}}`` — arrays are
        copies, so later steps cannot alter a saved state."""
        arrays = {}
        for name, m, v in zip(self.names, self._m, self._v):
            arrays[f"m.{name}"] = m.copy()
            arrays[f"v.{name}"] = v.copy()
        return {"hyper": self.hyperparameters(), "lr": float(self.lr), "step_count": int(self.step_count),
                "names": list(self.names), "arrays": arrays}

    def load_state_dict(self, state: dict) -> None:
        """All-or-nothing: every check runs before any field is assigned."""
        if list(state.get("names", [])) != self.names:
            missing = sorted(set(self.names) - set(state.get("names", [])))
            extra = sorted(set(state.get("names", [])) - set(self.names))
            raise OptimizerStateError(f"optimizer parameter names differ (missing {missing[:3]}, unexpected {extra[:3]})")
        if state["hyper"] != self.hyperparameters():
            raise OptimizerStateError(f"optimizer hyper-parameters differ: stored {state['hyper']} vs {self.hyperparameters()}")
        arrays = state["arrays"]
        for name, p in zip(self.names, self.parameters):
            for prefix in ("m", "v"):
                arr = arrays.get(f"{prefix}.{name}")
                if arr is None or tuple(arr.shape) != tuple(p.data.shape) or arr.dtype != p.data.dtype:
                    raise OptimizerStateError(f"optimizer array {prefix}.{name} missing or has the wrong shape/dtype")
                if not np.isfinite(arr).all():
                    raise OptimizerStateError(f"optimizer array {prefix}.{name} contains NaN/Inf")
        step_count = int(state["step_count"])
        if step_count < 0:
            raise OptimizerStateError("negative step_count")
        self._m = [arrays[f"m.{n}"].copy() for n in self.names]
        self._v = [arrays[f"v.{n}"].copy() for n in self.names]
        self.step_count = step_count
        self.lr = float(state["lr"])
