"""A minimal, from-scratch reverse-mode autodiff engine, in plain NumPy.

**Why this file exists at all** (read this before anything else in
``tinymind/model/``): the brief this phase was built from assumes an
existing tensor/autodiff framework (PyTorch, JAX) to build a transformer
on top of. This sandbox has neither installed, and has no network access
to install one — verified directly (``pip install torch`` reports "No
matching distribution found"; no CUDA/GPU is present either). Rather than
silently downgrading the whole phase to a non-differentiable forward-pass-
only "model" (which would fail the brief's own acceptance criteria —
section 31, items 5/6/20 all require real gradients and a real decreasing
loss), this module implements exactly the reverse-mode autodiff a small
transformer needs, in the same spirit as micrograd/tinygrad's early
versions: every op wraps a NumPy array, records how it was produced, and
knows its own local gradient rule. ``docs/architecture/
model-implementation.md`` section 0 explains this decision and its
consequences (chiefly: no GPU/vectorized-kernel performance, so real
training only happens at toy scale in this delivery — see ``STATUS.md``).

Every operation below is checked against numerical (finite-difference)
gradients in ``tests/model/test_tensor.py`` — that test file is the actual
correctness spec for this engine, not just a formality.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

_DTYPE = np.float32

# Global switch consulted by ``Tensor.__init__``. Inside ``no_grad()`` no op
# records its inputs or keeps a backward closure, so a forward-only pass
# (evaluation, generation) retains no autodiff graph: without this, every
# intermediate activation stays alive through the closures for as long as
# the output tensor does. See docs/architecture/training-system.md, section 3.
_GRAD_ENABLED = True


class no_grad:
    """Context manager: ops executed inside record no graph. Re-entrant and
    exception-safe; the previous state is always restored."""

    def __enter__(self) -> "no_grad":
        global _GRAD_ENABLED
        self._previous = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc_info) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._previous


def grad_enabled() -> bool:
    return _GRAD_ENABLED


def _noop() -> None:
    return None


def _is_basic_index(index) -> bool:
    """True when ``index`` selects with slices/ints/Ellipsis/None only, i.e.
    it can never address the same element twice (so ``grad[index] += g`` is
    exact and much faster than the unbuffered ``np.add.at``)."""
    parts = index if isinstance(index, tuple) else (index,)
    return all(part is Ellipsis or part is None or isinstance(part, (slice, int, np.integer))
               for part in parts)


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` down to ``shape`` by reducing over any axis that numpy
    broadcast up from ``shape`` — the standard companion operation to
    NumPy broadcasting in a forward pass: whatever dimension broadcasting
    duplicated a value across, backward must sum the incoming gradient
    back across that same dimension.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "requires_grad", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad: bool = False, _children: Iterable["Tensor"] = (),
                _op: str = ""):
        self.data = np.asarray(data, dtype=_DTYPE)
        if not _GRAD_ENABLED:
            requires_grad = False
            _children = ()
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward = lambda: None
        # A tuple, not a set: Tensor has no custom __hash__, so a set of
        # Tensors orders by the default identity hash (memory-address-
        # based), which varies between process runs even for identical
        # code and inputs. That let the reverse-mode topological sort in
        # backward() visit multi-parent nodes in a different order run to
        # run, which — because floating-point addition isn't exactly
        # associative — showed up as tiny (~1e-7) run-to-run differences in
        # accumulated gradients and, after enough training steps, in the
        # loss itself (caught by
        # tests/training/test_training.py::test_reproducible_given_seed).
        # A tuple's iteration order is construction order, which is fully
        # deterministic; see tests/model/test_tensor.py::
        # TestDeterminism for this checked directly.
        self._prev: tuple["Tensor", ...] = tuple(_children)
        self._op = _op

    # -- bookkeeping -------------------------------------------------
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate(self, grad: np.ndarray) -> None:
        grad = _unbroadcast(np.asarray(grad, dtype=_DTYPE), self.data.shape)
        self.grad = grad if self.grad is None else self.grad + grad

    @staticmethod
    def _as_tensor(value) -> "Tensor":
        return value if isinstance(value, Tensor) else Tensor(value)

    def item(self):
        return float(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    # -- graph traversal / backward -----------------------------------
    def _toposort(self) -> list["Tensor"]:
        """Children-before-parents order via an explicit stack. (The Phase 3A
        version recursed once per graph level and would hit Python's
        recursion limit on a deep enough model.) The visiting order is the
        same as the recursive version's — depth-first over ``_prev`` in
        construction order — so gradient accumulation order, and therefore
        float rounding, is unchanged."""
        order: list[Tensor] = []
        visited = {id(self)}
        stack = [(self, iter(self._prev))]
        while stack:
            node, children = stack[-1]
            for child in children:
                if id(child) not in visited:
                    visited.add(id(child))
                    stack.append((child, iter(child._prev)))
                    break
            else:
                order.append(node)
                stack.pop()
        return order

    def backward(self, grad: np.ndarray | None = None, retain_graph: bool = True) -> None:
        """Reverse-mode pass from this tensor.

        ``retain_graph=False`` releases every interior node as soon as its
        gradient has been pushed to its inputs (its ``grad``, closure and
        ``_prev`` are dropped, and the list holding the order is consumed),
        so activations are freed progressively instead of all living until
        the whole graph is garbage collected. Leaf tensors (parameters)
        keep their ``grad``. The default stays ``True`` because Phase 3A
        callers/tests read intermediate gradients and call ``backward``
        repeatedly.
        """
        order = self._toposort()

        if grad is None:
            if self.data.size != 1:
                raise ValueError(
                    f"backward() with no explicit grad requires a scalar tensor, got shape {self.shape}")
            grad = np.ones_like(self.data)
        self._accumulate(grad)

        if retain_graph:
            for t in reversed(order):
                if t.requires_grad:
                    t._backward()
            return

        while order:
            t = order.pop()
            if t.requires_grad:
                t._backward()
            if t._prev:  # interior node: everything it held is now dead weight
                t._backward = _noop
                t._prev = ()
                t.grad = None

    # -- elementwise arithmetic ---------------------------------------
    def __add__(self, other) -> "Tensor":
        other = self._as_tensor(other)
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad),
                    _children=(self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad)
            if other.requires_grad:
                other._accumulate(out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self._accumulate(-out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    def __sub__(self, other) -> "Tensor":
        return self + (-self._as_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        return self._as_tensor(other) + (-self)

    def __mul__(self, other) -> "Tensor":
        other = self._as_tensor(other)
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad),
                    _children=(self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * other.data)
            if other.requires_grad:
                other._accumulate(out.grad * self.data)
        if out.requires_grad:
            out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Tensor":
        other = self._as_tensor(other)
        return self * other.reciprocal()

    def reciprocal(self) -> "Tensor":
        out_data = 1.0 / self.data
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="reciprocal")

        def _backward():
            if self.requires_grad:
                self._accumulate(-out.grad * out_data * out_data)
        if out.requires_grad:
            out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Tensor":
        out_data = self.data ** power
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op=f"pow{power}")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * power * (self.data ** (power - 1)))
        if out.requires_grad:
            out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        out_data = np.sqrt(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sqrt")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * 0.5 / out_data)
        if out.requires_grad:
            out._backward = _backward
        return out

    def rsqrt(self) -> "Tensor":
        out_data = 1.0 / np.sqrt(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="rsqrt")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (-0.5) * out_data ** 3)
        if out.requires_grad:
            out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * out_data)
        if out.requires_grad:
            out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / self.data)
        if out.requires_grad:
            out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        out_data = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * out_data * (1.0 - out_data))
        if out.requires_grad:
            out._backward = _backward
        return out

    def silu(self) -> "Tensor":
        """x * sigmoid(x) — SiLU/Swish, used by the SwiGLU MLP."""
        sig = self.sigmoid()
        return self * sig

    # -- shape ops ------------------------------------------------------
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original_shape = self.data.shape
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad,
                    _children=(self,), _op="reshape")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.reshape(original_shape))
        if out.requires_grad:
            out._backward = _backward
        return out

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad,
                    _children=(self,), _op="transpose")
        inverse_axes = np.argsort(axes)

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.transpose(*inverse_axes))
        if out.requires_grad:
            out._backward = _backward
        return out

    def swapaxes(self, a: int, b: int) -> "Tensor":
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    # -- reductions -------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sum")
        input_shape = self.data.shape

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis if isinstance(axis, int) else tuple(axis))
                self._accumulate(np.broadcast_to(grad, input_shape))
        if out.requires_grad:
            out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            count = self.data.size
        elif isinstance(axis, int):
            count = self.data.shape[axis]
        else:
            count = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # -- matmul -----------------------------------------------------------
    def __matmul__(self, other) -> "Tensor":
        other = self._as_tensor(other)
        out_data = self.data @ other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                    _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
                self._accumulate(_unbroadcast(grad_self, self.data.shape))
            if other.requires_grad:
                grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
                other._accumulate(_unbroadcast(grad_other, other.data.shape))
        if out.requires_grad:
            out._backward = _backward
        return out

    # -- indexing / embedding ----------------------------------------------
    def __getitem__(self, index) -> "Tensor":
        out_data = self.data[index]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="getitem")
        input_shape = self.data.shape

        basic = _is_basic_index(index)

        def _backward():
            if self.requires_grad:
                grad = np.zeros(input_shape, dtype=_DTYPE)
                if basic:
                    grad[index] += out.grad  # no repeated targets: exact, and far faster than add.at
                else:
                    np.add.at(grad, index, out.grad)
                self._accumulate(grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    def embedding_lookup(self, indices: np.ndarray) -> "Tensor":
        """``self`` is a ``[vocab_size, hidden_size]`` embedding table;
        ``indices`` is an integer array of any shape. Returns
        ``[*indices.shape, hidden_size]``. Repeated indices correctly
        accumulate (scatter-add) their gradient, via ``np.add.at``."""
        return self[indices]

    # -- softmax / cross entropy (fused for numerical stability) -----------
    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=axis, keepdims=True)
        out = Tensor(probs, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _backward():
            if self.requires_grad:
                # d(softmax)/dx standard identity: grad_x = p * (grad_y - sum(grad_y * p, axis))
                dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
                self._accumulate(probs * (out.grad - dot))
        if out.requires_grad:
            out._backward = _backward
        return out


def cross_entropy(logits: Tensor, targets: np.ndarray, ignore_index: int = -100,
                  normalizer: float | None = None) -> Tensor:
    """Average next-token cross-entropy loss.

    ``logits``: ``Tensor`` of shape ``[..., vocab_size]``.
    ``targets``: integer array of shape ``[...]`` (one fewer dim than
    ``logits``), with ``ignore_index`` marking positions to exclude (e.g.
    padding) from the average — brief section 10's "ignore_index" and
    "padding masks" requirement.

    Implemented as one fused op (not composed from ``.softmax()``/``.log()``)
    for both numerical stability (the standard log-sum-exp trick) and a
    simple, directly-verifiable backward: ``d(loss)/d(logits) = (softmax(logits)
    - one_hot(targets)) / N`` at valid positions, exactly zero at ignored
    ones — the standard, well-known cross-entropy-softmax gradient identity.
    """
    targets = np.asarray(targets)
    valid_mask = (targets != ignore_index)
    n_valid = max(int(valid_mask.sum()), 1)
    # ``normalizer`` replaces the per-call valid-token count as the divisor.
    # Gradient accumulation needs this: dividing every micro-batch by the
    # token count of the WHOLE effective batch makes the summed gradient
    # independent of how that batch was split (see training/engine.py).
    divisor = float(normalizer) if normalizer is not None else float(n_valid)
    if divisor <= 0:
        raise ValueError(f"cross_entropy normalizer must be positive, got {divisor}")

    flat_logits = logits.data.reshape(-1, logits.data.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_valid = valid_mask.reshape(-1)

    shifted = flat_logits - np.max(flat_logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
    safe_targets = np.where(flat_valid, flat_targets, 0)
    picked = shifted[np.arange(flat_logits.shape[0]), safe_targets]
    per_token_loss = (log_sum_exp - picked) * flat_valid
    loss_value = per_token_loss.sum() / divisor

    out = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), _op="cross_entropy")

    def _backward():
        if logits.requires_grad:
            probs = np.exp(shifted - log_sum_exp[:, None])  # softmax, reusing the shift already computed
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(flat_logits.shape[0]), safe_targets] = 1.0
            grad_flat = (probs - one_hot) * flat_valid[:, None] / divisor
            logits._accumulate((out.grad * grad_flat).reshape(logits.data.shape))
    if out.requires_grad:
        out._backward = _backward
    return out


def concat(tensors: list[Tensor], axis: int = -1) -> Tensor:
    datas = [t.data for t in tensors]
    out_data = np.concatenate(datas, axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=requires_grad, _children=tuple(tensors), _op="concat")
    sizes = [d.shape[axis] for d in datas]

    def _backward():
        offset = 0
        slices = [slice(None)] * out_data.ndim
        for t, size in zip(tensors, sizes):
            if t.requires_grad:
                slices[axis] = slice(offset, offset + size)
                t._accumulate(out.grad[tuple(slices)])
            offset += size
    if out.requires_grad:
        out._backward = _backward
    return out
