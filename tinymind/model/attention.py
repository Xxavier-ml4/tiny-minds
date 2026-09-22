"""Causal self-attention: Q/K/V projections, RoPE on Q/K, grouped-query
attention (GQA) support, causal masking, softmax, output projection.

Shapes follow ``docs/architecture/model-implementation.md``'s attention
section exactly:

    input:            [B, T, hidden_size]
    Q:                [B, num_heads,    T, head_dim]
    K, V:             [B, num_kv_heads, T, head_dim]  (repeated to num_heads for GQA)
    attention scores: [B, num_heads, T, T]
    output:           [B, T, hidden_size]

GQA (``num_kv_heads < num_heads``) is supported via ``_repeat_kv``, which
degenerates to a no-op when ``num_kv_heads == num_heads`` (plain MHA) — one
code path for both, per ``tinymind.model.config.ModelConfig``'s own
docstring on that field. MQA (``num_kv_heads == 1``) falls out of the same
mechanism for free.
"""
from __future__ import annotations

import numpy as np

from tinymind.model import fused
from tinymind.model.config import ModelConfig
from tinymind.model.linear import Linear
from tinymind.model.module import Module
from tinymind.model.positional import apply_rotary_pos_emb, precompute_rope_cache
from tinymind.model.tensor import Tensor

_NEG_INF = np.float32(-1e9)  # finite, not -inf: avoids 0*inf==nan if a fully-masked row ever occurs


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """``[B, num_kv_heads, T, head_dim] -> [B, num_kv_heads * n_rep, T,
    head_dim]``, each kv head repeated ``n_rep`` times contiguously (so head
    ``i`` of the expanded tensor reads from kv head ``i // n_rep``, matching
    how ``num_heads`` query heads are grouped in blocks of ``n_rep`` per kv
    head — the standard GQA head-grouping convention)."""
    if n_rep == 1:
        return x
    b, num_kv_heads, t, head_dim = x.shape
    tiled_data = np.broadcast_to(
        x.data.reshape(b, num_kv_heads, 1, t, head_dim),
        (b, num_kv_heads, n_rep, t, head_dim)).copy()
    out = Tensor(tiled_data.reshape(b, num_kv_heads * n_rep, t, head_dim),
                requires_grad=x.requires_grad, _children=(x,), _op="repeat_kv")

    def _backward():
        if x.requires_grad:
            grad = out.grad.reshape(b, num_kv_heads, n_rep, t, head_dim).sum(axis=2)
            x._accumulate(grad)
    out._backward = _backward
    return out


class CausalSelfAttention(Module):
    def __init__(self, config: ModelConfig, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim, rng=rng)
        self.k_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim, rng=rng)
        self.v_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim, rng=rng)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size, rng=rng)

        self._cos, self._sin = precompute_rope_cache(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: Tensor, cos: np.ndarray | None = None, sin: np.ndarray | None = None,
               kv_cache=None, layer_idx: int = 0, position_ids: np.ndarray | None = None,
               attention_bias: np.ndarray | None = None) -> Tensor:
        """``attention_bias`` (optional, ``[B, 1, T, T_kv]`` additive) REPLACES
        the plain causal mask: the caller builds it with causality already in
        it (see ``block_causal_bias``), which is how packed sequences are kept
        from attending to each other."""
        cos = self._cos if cos is None else cos
        sin = self._sin if sin is None else sin
        b, t, _hidden = x.shape

        q = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = apply_rotary_pos_emb(q, cos, sin, position_ids=position_ids)
        k = apply_rotary_pos_emb(k, cos, sin, position_ids=position_ids)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        scale = 1.0 / np.sqrt(self.head_dim)

        if fused.enabled():
            use_causal = attention_bias is None and (kv_cache is None or t > 1)
            out = fused.attention(q, k, v, scale=scale, causal=use_causal, bias=attention_bias)
            out = out.transpose(0, 2, 1, 3).reshape(b, t, self.num_heads * self.head_dim)
            return self.o_proj(out)

        # ---- reference path (kept as the numerical ground truth) ----
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B, num_heads, T, T_kv]

        t_kv = k.shape[2]
        if attention_bias is not None:
            scores = scores + attention_bias
        elif kv_cache is None or t > 1:
            # Full/prefill causal mask: query position i may attend to key
            # position j iff j <= i (offset by however much cache already
            # holds, so a prefill after a nonempty cache still masks
            # correctly — see tests/model/test_attention.py).
            already_cached = t_kv - t
            causal = np.triu(np.full((t, t_kv), _NEG_INF, dtype=np.float32), k=already_cached + 1)
            scores = scores + causal[None, None, :, :]
        # else: single-token decode step (t == 1) attending to the whole
        # cache needs no mask — every cached position is, by construction,
        # already at or before the current position.

        weights = scores.softmax(axis=-1)
        out = weights @ v  # [B, num_heads, T, head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out)


def block_causal_bias(segment_ids: np.ndarray) -> np.ndarray:
    """Additive attention mask ``[B, 1, T, T]`` for rows that hold several
    independent sequences back to back: position ``i`` may attend to ``j``
    iff ``j <= i`` **and** both belong to the same segment. ``segment_ids`` is
    ``[B, T]`` of ints (padding gets its own id, so it never mixes with real
    tokens). Uses the same finite ``-1e9`` as the plain causal mask."""
    seg = np.asarray(segment_ids)
    t = seg.shape[1]
    same = seg[:, :, None] == seg[:, None, :]
    causal = np.tril(np.ones((t, t), dtype=bool))
    return np.where(same & causal[None], np.float32(0.0), _NEG_INF).astype(np.float32)[:, None, :, :]


def segment_positions(segment_ids: np.ndarray) -> np.ndarray:
    """Position of every token *within its own segment* (0 at each segment's
    first token), ``[B, T]`` int64. Feeds RoPE so a packed sequence sees the
    same positions it would see alone."""
    seg = np.asarray(segment_ids)
    b, t = seg.shape
    idx = np.broadcast_to(np.arange(t), (b, t))
    is_start = np.ones((b, t), dtype=bool)
    is_start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    starts = np.maximum.accumulate(np.where(is_start, idx, 0), axis=1)
    return (idx - starts).astype(np.int64)
