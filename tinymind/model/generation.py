"""Autoregressive generation for ``TinyMindTransformer``: greedy decoding
and sampling (temperature/top-k/top-p/repetition penalty), with KV caching.

This extends the existing ``tinymind/runtime/generation.py``
(``GenerationConfig``, which is backend-agnostic) rather than duplicating
it — see that module. This file is specifically the real-model generation
loop, which needs an actual model and tokenizer to drive, unlike the
backend-agnostic config dataclass.

Default generation is deterministic (greedy, ``do_sample=False``) — per
the brief section 14: "Default generation should be deterministic. Do not
make sampling the default."
"""
from __future__ import annotations

import dataclasses

import numpy as np

from tinymind.model.model import KVCache, TinyMindTransformer
from tinymind.model.tensor import no_grad
from tinymind.model.tokenizer import Tokenizer
from tinymind.runtime.sampling import softmax, top_k_filter, top_p_filter


@dataclasses.dataclass
class ModelGenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = False
    repetition_penalty: float = 1.0
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    seed: int | None = None


def _apply_repetition_penalty(logits: np.ndarray, generated_ids: list[int], penalty: float) -> np.ndarray:
    if penalty == 1.0 or not generated_ids:
        return logits
    logits = logits.copy()
    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def _select_next_token(logits: np.ndarray, generated_ids: list[int], config: ModelGenerationConfig,
                       rng: np.random.Generator) -> int:
    logits = _apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)
    if not config.do_sample:
        return int(np.argmax(logits))
    probs = softmax((logits / max(config.temperature, 1e-6)).tolist())
    if config.top_k is not None:
        probs = top_k_filter(probs, config.top_k)
    if config.top_p is not None:
        probs = top_p_filter(probs, config.top_p)
    probs_arr = np.array(probs)
    probs_arr = probs_arr / probs_arr.sum()
    return int(rng.choice(len(probs_arr), p=probs_arr))


def generate(model: TinyMindTransformer, tokenizer: Tokenizer, prompt: str,
            config: ModelGenerationConfig | None = None) -> str:
    """Full text-in, text-out generation: encode, run prefill + incremental
    decode with a ``KVCache``, decode back to text. See
    ``generate_with_cache_ids`` for the token-id-level version used by
    ``tests/model/test_cache.py`` to check cache/non-cache equivalence
    directly, without a tokenizer round-trip in the way.
    """
    config = config or ModelGenerationConfig()
    input_ids = np.array([tokenizer.encode(prompt, add_bos=True)])
    output_ids = generate_with_cache_ids(model, input_ids, config)
    return tokenizer.decode(output_ids[0].tolist())


def generate_with_cache_ids(model: TinyMindTransformer, input_ids: np.ndarray,
                            config: ModelGenerationConfig | None = None) -> np.ndarray:
    with no_grad():  # inference records no autodiff graph
        return _generate_with_cache_ids(model, input_ids, config)


def _generate_with_cache_ids(model: TinyMindTransformer, input_ids: np.ndarray,
                             config: ModelGenerationConfig | None = None) -> np.ndarray:
    config = config or ModelGenerationConfig()
    rng = np.random.default_rng(config.seed)
    batch_size = input_ids.shape[0]
    if batch_size != 1:
        raise NotImplementedError(
            "generate_with_cache_ids currently supports batch_size=1 only — see STATUS.md")

    cache = KVCache(model.config, batch_size=batch_size)
    generated = list(input_ids[0])

    if len(generated) > model.config.max_seq_len:
        raise ValueError(
            f"prompt is {len(generated)} tokens, longer than this model's max_seq_len "
            f"({model.config.max_seq_len}) — it cannot be prefilled at all, regardless of "
            "max_new_tokens")

    # A small model's max_seq_len is a real, physical limit (it bounds both
    # the RoPE cache and the KVCache — see KVCache/apply_rotary_pos_emb),
    # not a tunable default like max_new_tokens. Running out of room to
    # continue in is an ordinary generation-stopping condition (the same
    # role EOS or max_new_tokens itself plays), not an error — so this
    # clamps rather than raising, and does it once, upfront, rather than
    # letting the loop below fail confusingly on whichever step first
    # crosses the boundary.
    max_new_tokens = min(config.max_new_tokens, max(model.config.max_seq_len - len(generated), 0))

    # Prefill: process the whole prompt at once.
    position_ids = np.arange(len(generated))[None, :]
    out = model(np.array([generated]), use_cache=True, past_key_values=cache, position_ids=position_ids)
    next_logits = out.logits.data[0, -1, :]

    for _ in range(max_new_tokens):
        next_token = _select_next_token(next_logits, generated, config, rng)
        generated.append(next_token)
        if config.eos_token_id is not None and next_token == config.eos_token_id:
            break
        # Decode: process only the new token, at its true absolute position.
        position_ids = np.array([[cache.length]])
        out = model(np.array([[next_token]]), use_cache=True, past_key_values=cache,
                   position_ids=position_ids)
        next_logits = out.logits.data[0, -1, :]

    return np.array([generated])


def generate_full_sequence_no_cache(model: TinyMindTransformer, input_ids: np.ndarray,
                                    config: ModelGenerationConfig | None = None) -> np.ndarray:
    """Reference implementation with NO cache: re-runs the full sequence
    from scratch at every step. Deliberately inefficient — it exists only
    so ``tests/model/test_cache.py`` has an independent ground truth to
    check the cached path against (brief section 13: "prefill + incremental
    decode logits" must match "full-sequence logits" — this function
    computes the latter at every generation step).
    """
    config = config or ModelGenerationConfig()
    rng = np.random.default_rng(config.seed)
    generated = list(input_ids[0])

    for _ in range(config.max_new_tokens):
        out = model(np.array([generated]), use_cache=False)
        next_logits = out.logits.data[0, -1, :]
        next_token = _select_next_token(next_logits, generated, config, rng)
        generated.append(next_token)
        if config.eos_token_id is not None and next_token == config.eos_token_id:
            break

    return np.array([generated])
