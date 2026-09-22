"""Fused kernels for the transformer's hot operations.

**Why these exist** (evidence: ``docs/architecture/training-system.md``
section 3): profiling the Phase 3A step showed graph-construction overhead is
0.3 % of a step, but the attention core is ~55 % of the time for ~25 % of the
FLOPs, because the composed ops materialise about eight ``[B, H, T, T]``
arrays per layer and copy K/V for every query-head group. These kernels do the
same mathematics with one node per operation, in-place softmax, no K/V repeat
and 2-D GEMMs for weight gradients.

**Contract with the reference implementation.** The composed ops
(``Tensor.__matmul__``, ``RMSNorm``'s pow/mean/rsqrt chain, ``silu``,
``apply_rotary_pos_emb``, ``_repeat_kv`` + softmax) remain in the code base and
are the reference: this module changes *how* the numbers are computed, never
*which* numbers. ``tests/model/test_fused.py`` checks forward outputs and
every gradient of every fused op against the composed path, plus finite
differences. There is one architecture, one parameter layout and one ``.tm``
export contract; ``use_fused(False)`` (or ``TINYMIND_FUSED=0``) switches the
whole model back to the reference ops.

All functions take/return :class:`tinymind.model.tensor.Tensor` and honour
``no_grad()`` (no closure is kept when the output cannot receive gradient).
"""
from __future__ import annotations

import os

import numpy as np

from tinymind.model.tensor import Tensor

_ENABLED = os.environ.get("TINYMIND_FUSED", "1") not in ("0", "false", "False", "")
_NEG_INF = np.float32(-1e9)  # same finite mask value as the reference attention (see native-model-contract.md)


def enabled() -> bool:
    return _ENABLED


class use_fused:
    """Context manager: ``with use_fused(False): ...`` runs the reference ops."""

    def __init__(self, flag: bool) -> None:
        self._flag = bool(flag)

    def __enter__(self) -> "use_fused":
        global _ENABLED
        self._previous = _ENABLED
        _ENABLED = self._flag
        return self

    def __exit__(self, *exc_info) -> None:
        global _ENABLED
        _ENABLED = self._previous


# --------------------------------------------------------------------------
# linear:  y = x @ W^T
# --------------------------------------------------------------------------
def linear(x: Tensor, weight: Tensor) -> Tensor:
    """``x``: ``[..., in]``; ``weight``: ``[out, in]`` (the stored, native
    orientation — no transpose node is created). Both directions are single
    2-D GEMMs over the flattened leading axes: the composed path multiplies a
    batched ``[B, T, in]`` and then un-broadcasts the weight gradient by
    summing a ``[B, in, out]`` intermediate."""
    out_f, in_f = weight.data.shape
    lead = x.data.shape[:-1]
    x2 = x.data.reshape(-1, in_f)
    y = (x2 @ weight.data.T).reshape(lead + (out_f,))
    out = Tensor(y, requires_grad=x.requires_grad or weight.requires_grad,
                 _children=(x, weight), _op="linear")
    if out.requires_grad:
        def _backward():
            g2 = out.grad.reshape(-1, out_f)
            if x.requires_grad:
                x._accumulate((g2 @ weight.data).reshape(x.data.shape))
            if weight.requires_grad:
                weight._accumulate(g2.T @ x2)
        out._backward = _backward
    return out


# --------------------------------------------------------------------------
# RMSNorm:  y = x * rsqrt(mean(x^2) + eps) * w
# --------------------------------------------------------------------------
def rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    xd = x.data
    n = xd.shape[-1]
    mean_sq = np.einsum("...i,...i->...", xd, xd)[..., None] * np.float32(1.0 / n)
    r = (np.float32(1.0) / np.sqrt(mean_sq + np.float32(eps))).astype(np.float32, copy=False)
    normed = xd * r
    out = Tensor(normed * weight.data, requires_grad=x.requires_grad or weight.requires_grad,
                 _children=(x, weight), _op="rmsnorm")
    if out.requires_grad:
        def _backward():
            g = out.grad
            if weight.requires_grad:
                weight._accumulate((g * normed).reshape(-1, n).sum(axis=0))
            if x.requires_grad:
                gw = g * weight.data
                dot = np.einsum("...i,...i->...", gw, normed)[..., None] * np.float32(1.0 / n)
                x._accumulate(r * (gw - normed * dot))
        out._backward = _backward
    return out


# --------------------------------------------------------------------------
# SwiGLU gate:  h = silu(gate) * up
# --------------------------------------------------------------------------
def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    gd, ud = gate.data, up.data
    with np.errstate(over="ignore"):
        sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-gd))
    silu = gd * sig
    out = Tensor(silu * ud, requires_grad=gate.requires_grad or up.requires_grad,
                 _children=(gate, up), _op="swiglu")
    if out.requires_grad:
        def _backward():
            g = out.grad
            if up.requires_grad:
                up._accumulate(g * silu)
            if gate.requires_grad:
                gate._accumulate(g * ud * (sig * (np.float32(1.0) + gd * (np.float32(1.0) - sig))))
        out._backward = _backward
    return out


# --------------------------------------------------------------------------
# RoPE (rotate-half):  out = x * cos + rotate_half(x) * sin
# --------------------------------------------------------------------------
def rope(x: Tensor, cos_b: np.ndarray, sin_b: np.ndarray) -> Tensor:
    """``x``: ``[B, heads, T, d]``. ``cos_b``/``sin_b`` are already gathered and
    broadcastable to ``x`` (``[1, 1, T, d]`` or ``[B, 1, T, d]``). With
    ``x = (x1 | x2)`` halves, ``out = (x1 c1 - x2 s1 | x2 c2 + x1 s2)`` and the
    gradient is the transposed rotation ``(g1 c1 + g2 s2 | g2 c2 - g1 s1)``.
    One node instead of the composed path's slice/neg/concat/mul/add chain."""
    xd = x.data
    half = xd.shape[-1] // 2
    c1, c2 = cos_b[..., :half], cos_b[..., half:]
    s1, s2 = sin_b[..., :half], sin_b[..., half:]
    x1, x2 = xd[..., :half], xd[..., half:]
    y = np.empty(xd.shape, dtype=np.float32)
    y[..., :half] = x1 * c1 - x2 * s1
    y[..., half:] = x2 * c2 + x1 * s2
    out = Tensor(y, requires_grad=x.requires_grad, _children=(x,), _op="rope")
    if out.requires_grad:
        def _backward():
            g = out.grad
            g1, g2 = g[..., :half], g[..., half:]
            dx = np.empty(g.shape, dtype=np.float32)
            dx[..., :half] = g1 * c1 + g2 * s2
            dx[..., half:] = g2 * c2 - g1 * s1
            x._accumulate(dx)
        out._backward = _backward
    return out


# --------------------------------------------------------------------------
# Grouped-query attention core: softmax(scale * Q K^T + mask) V
# --------------------------------------------------------------------------
def causal_bias(t: int, t_kv: int) -> np.ndarray:
    """Additive causal mask ``[t, t_kv]`` for queries that sit after
    ``t_kv - t`` already-cached keys (0 where allowed, -1e9 above the
    diagonal). Identical to the reference attention's mask."""
    return np.triu(np.full((t, t_kv), _NEG_INF, dtype=np.float32), k=(t_kv - t) + 1)


def attention(q: Tensor, k: Tensor, v: Tensor, *, scale: float, causal: bool,
              bias: np.ndarray | None = None) -> Tensor:
    """``q``: ``[B, H, T, d]``; ``k``/``v``: ``[B, Hkv, Tk, d]`` with
    ``H = Hkv * n_rep``. Returns ``[B, H, T, d]``.

    GQA without repeating K/V: the ``n_rep`` query heads that share a KV head
    are stacked along the query axis, ``[B, Hkv, n_rep*T, d]``, so one batched
    GEMM per KV head computes all their scores against the un-repeated keys
    (and the K/V gradients, being a contraction over that stacked axis, sum
    over the group for free). Head ``h`` reads KV head ``h // n_rep`` —
    exactly the reference convention (native-model-contract.md section 3).

    ``causal=True`` applies the triangular mask (offset for cached keys);
    ``bias`` is an extra additive mask ``[B, 1, T, Tk]`` (block-diagonal
    causal masks for packed sequences are passed this way, with
    ``causal=False``). The scale is folded into Q so the ``[.., T, Tk]`` score
    array is written once, softmaxed in place, and is the only large tensor
    kept for backward.
    """
    b, h, t, d = q.data.shape
    h_kv, t_kv = k.data.shape[1], k.data.shape[2]
    n_rep = h // h_kv
    scale32 = np.float32(scale)

    qg = np.ascontiguousarray(q.data.reshape(b, h_kv, n_rep * t, d)) * scale32  # scaled, contiguous
    kd, vd = k.data, v.data
    s = np.matmul(qg, kd.transpose(0, 1, 3, 2))                                  # [B, Hkv, n_rep*T, Tk]
    s5 = s.reshape(b, h_kv, n_rep, t, t_kv)
    if causal and not (t == 1):
        s5 += causal_bias(t, t_kv)
    if bias is not None:
        s5 += bias[:, :, None, :, :]
    s -= s.max(axis=-1, keepdims=True)
    np.exp(s, out=s)
    s /= s.sum(axis=-1, keepdims=True)
    p = s
    ctx = np.matmul(p, vd)                                                       # [B, Hkv, n_rep*T, d]
    out = Tensor(ctx.reshape(b, h, t, d), requires_grad=q.requires_grad or k.requires_grad or v.requires_grad,
                 _children=(q, k, v), _op="attention")

    if out.requires_grad:
        def _backward():
            g = out.grad.reshape(b, h_kv, n_rep * t, d)
            dp = np.matmul(g, vd.transpose(0, 1, 3, 2))                          # [B, Hkv, n_rep*T, Tk]
            if v.requires_grad:
                v._accumulate(np.matmul(p.transpose(0, 1, 3, 2), g))
            dot = np.einsum("...j,...j->...", dp, p)[..., None]
            dp -= dot
            dp *= p                                                              # gradient w.r.t. pre-softmax scores
            if q.requires_grad:
                dq = np.matmul(dp, kd) * scale32                                 # scale folded into Q above
                q._accumulate(dq.reshape(b, h, t, d))
            if k.requires_grad:
                k._accumulate(np.matmul(dp.transpose(0, 1, 3, 2), qg))
        out._backward = _backward
    return out
