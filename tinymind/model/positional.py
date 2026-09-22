"""Rotary positional embeddings (RoPE), GPT-NeoX/Llama-style "rotate-half"
convention.

Frequencies are precomputed once per ``(head_dim, max_seq_len, theta)`` and
cached (``precompute_rope_cache``) rather than recomputed per forward call;
applying them (``apply_rotary_pos_emb``) is pure NumPy broadcasting — no
Python loop over individual token positions, per the brief's explicit
requirement in section 5.

Formula, applied identically to Q and K:

    rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])
    x_rope = x * cos + rotate_half(x) * sin

with ``cos``/``sin`` built by duplicating a ``[T, d/2]`` frequency table
into ``[T, d]`` (``concat([freqs, freqs], axis=-1)``) so this single
elementwise formula implements the standard pairwise 2D rotation on
``(x_i, x_{i+d/2})`` for every ``i`` — see
``docs/architecture/model-implementation.md``'s positional-encoding section
for the derivation, and ``tests/model/test_rope.py`` for the shape/
determinism/position-sensitivity checks this is verified against.
"""
from __future__ import annotations

import numpy as np

from tinymind.model import fused
from tinymind.model.tensor import Tensor, concat

_DTYPE = np.float32


def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0
                          ) -> tuple[np.ndarray, np.ndarray]:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(max_seq_len, dtype=np.float64)
    freqs = np.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]
    freqs_full = np.concatenate([freqs, freqs], axis=-1)  # [max_seq_len, head_dim]
    return np.cos(freqs_full).astype(_DTYPE), np.sin(freqs_full).astype(_DTYPE)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x: Tensor, cos: np.ndarray, sin: np.ndarray,
                         position_ids: np.ndarray | None = None) -> Tensor:
    """``x``: ``[B, num_heads, T, head_dim]``. ``cos``/``sin``: the full
    ``[max_seq_len, head_dim]`` cache from ``precompute_rope_cache``.
    ``position_ids``, if given, indexes into the cache per-token (needed for
    KV-cache decode, where the single new token's absolute position isn't
    simply "0" — see ``tinymind/model/generation.py``); otherwise positions
    ``0..T-1`` are used.
    """
    seq_len = x.shape[-2]
    if position_ids is None:
        cos_slice = cos[:seq_len]
        sin_slice = sin[:seq_len]
    else:
        cos_slice = cos[position_ids]
        sin_slice = sin[position_ids]
    # Broadcast [T, head_dim] (2D: no position_ids, or a single unbatched
    # position_ids vector) or [B, T, head_dim] (2D position_ids batched per
    # sequence) up to x's [B, num_heads, T, head_dim] by inserting the
    # heads axis; Tensor.__mul__'s broadcasting (numpy semantics) does the
    # rest.
    cos_b = cos_slice[None, None, :, :] if cos_slice.ndim == 2 else cos_slice[:, None, :, :]
    sin_b = sin_slice[None, None, :, :] if sin_slice.ndim == 2 else sin_slice[:, None, :, :]
    if fused.enabled():
        return fused.rope(x, cos_b, sin_b)
    return x * cos_b + _rotate_half(x) * sin_b  # reference path
