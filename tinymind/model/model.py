"""The full model: token embedding -> N transformer blocks -> final RMSNorm
-> LM head -> logits, per the brief section 9. Softmax is never applied
inside ``forward()`` — callers get raw logits, exactly as specified.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from tinymind.model import fused
from tinymind.model.attention import block_causal_bias, segment_positions
from tinymind.model.config import ModelConfig
from tinymind.model.layers import TransformerBlock
from tinymind.model.linear import Linear
from tinymind.model.loss import causal_lm_loss
from tinymind.model.module import Module
from tinymind.model.norm import RMSNorm
from tinymind.model.tensor import Tensor


class KVCache:
    """Per-layer key/value cache for autoregressive decoding — the Python-
    level counterpart to ``native/src/kv_cache.h``'s design (see that
    file's docstring for the native-side version of the same contract:
    fixed-capacity, indexed by layer and position). ``update()`` writes the
    newest keys/values for one layer at the cache's current length and
    returns the full keys/values up to (and including) them; ``advance()``
    moves the shared length forward once per full model forward pass (not
    per layer — every layer sees the same ``length`` throughout one
    ``model()`` call).
    """

    def __init__(self, config: ModelConfig, batch_size: int = 1) -> None:
        self.max_seq_len = config.max_seq_len
        self.length = 0
        shape = (config.num_layers, batch_size, config.num_kv_heads, config.max_seq_len, config.head_dim)
        self._keys = np.zeros(shape, dtype=np.float32)
        self._values = np.zeros(shape, dtype=np.float32)

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        t_new = k.shape[2]
        if self.length + t_new > self.max_seq_len:
            raise ValueError(
                f"KVCache overflow: {self.length} + {t_new} exceeds max_seq_len={self.max_seq_len}")
        self._keys[layer_idx, :, :, self.length:self.length + t_new, :] = k.data
        self._values[layer_idx, :, :, self.length:self.length + t_new, :] = v.data
        full_len = self.length + t_new
        return (Tensor(self._keys[layer_idx, :, :, :full_len, :]),
                Tensor(self._values[layer_idx, :, :, :full_len, :]))

    def advance(self, t_new: int) -> None:
        self.length += t_new

    def reset(self) -> None:
        self.length = 0


@dataclasses.dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: KVCache | None = None
    hidden_states: Tensor | None = None


class TinyMindTransformer(Module):
    def __init__(self, config: ModelConfig, seed: int | None = None) -> None:
        super().__init__()
        config.require_supported()  # refuse configs that would be silently ignored
        self.config = config
        rng = np.random.default_rng(seed)

        embed_std = 1.0 / np.sqrt(config.hidden_size)
        self.embed_tokens = Tensor(
            rng.normal(0.0, embed_std, size=(config.vocab_size, config.hidden_size)).astype(np.float32),
            requires_grad=True)

        self.blocks = [TransformerBlock(config, rng=rng) for _ in range(config.num_layers)]
        for i, block in enumerate(self.blocks):
            setattr(self, f"block_{i}", block)  # registers each block's params via Module.__setattr__

        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_epsilon)

        self.lm_head: Linear | None
        if config.tie_embeddings:
            self.lm_head = None  # forward() reuses embed_tokens directly — see below
        else:
            self.lm_head = Linear(config.hidden_size, config.vocab_size, rng=rng)

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray | None = None,
               position_ids: np.ndarray | None = None, labels: np.ndarray | None = None,
               use_cache: bool = False, past_key_values: KVCache | None = None,
               output_hidden_states: bool = False, segment_ids: np.ndarray | None = None,
               loss_normalizer: float | None = None) -> ModelOutput:
        """``labels`` uses the Hugging Face convention: same shape as
        ``input_ids``, shifted inside the loss, ``-100`` = no loss at that
        token. ``segment_ids`` (``[B, T]``) marks independent sequences packed
        into one row: attention becomes block-diagonal causal and RoPE
        positions restart in every segment. ``loss_normalizer`` overrides
        the loss divisor (gradient accumulation)."""
        input_ids = np.asarray(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        if int(input_ids.max(initial=0)) >= self.config.vocab_size or int(input_ids.min(initial=0)) < 0:
            raise ValueError(
                f"input_ids contains a token id outside [0, {self.config.vocab_size}) — "
                "tokenizer/model vocabulary mismatch (see TinyMindTransformer.check_tokenizer_compatibility)")

        attention_bias = None
        if segment_ids is not None:
            segment_ids = np.asarray(segment_ids)
            if segment_ids.shape != input_ids.shape:
                raise ValueError(f"segment_ids shape {segment_ids.shape} != input_ids shape {input_ids.shape}")
            if use_cache or past_key_values is not None:
                raise ValueError("segment_ids (packed training rows) cannot be combined with a KV cache")
            attention_bias = block_causal_bias(segment_ids)
            if position_ids is None:
                position_ids = segment_positions(segment_ids)

        if position_ids is not None:
            position_ids = np.asarray(position_ids)
            if int(position_ids.max(initial=0)) >= self.config.max_seq_len:
                raise ValueError(
                    f"position_ids contains a position >= max_seq_len ({self.config.max_seq_len}) — "
                    "the sequence (including any KV cache already filled) has exceeded the "
                    "positions this model's RoPE cache and KVCache were sized for")

        hidden = self.embed_tokens.embedding_lookup(input_ids)  # [B, T, hidden_size]

        cache = past_key_values if use_cache else None
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(hidden, cos=None, sin=None, kv_cache=cache, layer_idx=layer_idx,
                          position_ids=position_ids, attention_bias=attention_bias)
        if cache is not None:
            cache.advance(input_ids.shape[1])

        hidden = self.final_norm(hidden)

        if self.config.tie_embeddings:
            logits = (fused.linear(hidden, self.embed_tokens) if fused.enabled()
                      else hidden @ self.embed_tokens.transpose(1, 0))
        else:
            logits = self.lm_head(hidden)

        loss = (causal_lm_loss(logits, labels, attention_mask, normalizer=loss_normalizer)
                if labels is not None else None)

        return ModelOutput(logits=logits, loss=loss, past_key_values=cache if use_cache else None,
                          hidden_states=hidden if output_hidden_states else None)

    def check_tokenizer_compatibility(self, tokenizer) -> None:
        if tokenizer.vocab_size > self.config.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size ({tokenizer.vocab_size}) exceeds the model's "
                f"vocab_size ({self.config.vocab_size}) — this tokenizer cannot be used with "
                "this model; every token id it can produce must be a valid embedding row")

    def num_parameters(self, trainable_only: bool = False) -> int:
        return self.count_parameters()
