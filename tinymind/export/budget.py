"""Mobile size budget: what a model costs in bytes, computed from its config.

Everything here is arithmetic on the config (no model is instantiated), and
``tests/export/test_budget.py`` pins each formula to the real object it
describes: parameter bytes to ``np.float32`` weights, KV bytes to the arrays
``KVCache`` allocates, INT8 bytes to what ``quantize_model`` produces.

*Scratch* (activation) figures are **estimates** of the fp32 working buffers a
straightforward runtime needs; they are labelled as such. The measured numbers
(peak RSS of the Python runtime, latency) come from ``benchmarks/`` and the
native figures must be measured on the target device.
"""
from __future__ import annotations

from typing import Any

from tinymind.model.config import ModelConfig, count_parameters, parameter_shapes

MiB = 1024 * 1024


def weight_bytes_fp32(cfg: ModelConfig) -> int:
    return 4 * count_parameters(cfg)


def weight_bytes_int8(cfg: ModelConfig, quantize_embeddings: bool = False) -> int:
    """Per-row symmetric INT8 (``Int8Scheme``): one int8 per element of every quantized 2-D tensor plus one fp32
    scale per row; 1-D tensors (norm gains) stay fp32; the embedding table stays fp32 unless
    ``quantize_embeddings`` (it is also the tied output head, so its error lands directly on the logits)."""
    total = 0
    for name, shape in parameter_shapes(cfg).items():
        n = 1
        for d in shape:
            n *= d
        is_embedding = name == "embed_tokens"
        if len(shape) == 2 and (quantize_embeddings or not is_embedding):
            total += n + 4 * shape[0]
        else:
            total += 4 * n
    return total


def kv_cache_bytes(cfg: ModelConfig, seq_len: int, dtype_bytes: int = 4) -> int:
    """Keys and values for every layer, KV head and position up to ``seq_len``."""
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * seq_len * dtype_bytes


def scratch_decode_bytes(cfg: ModelConfig, seq_len: int) -> int:
    """Estimate: fp32 buffers for one decode step (residual, normed, q, k, v, scores over ``seq_len`` positions,
    attention context, MLP gate/up/act, logits)."""
    h, d = cfg.hidden_size, cfg.head_dim
    floats = 4 * h + 2 * cfg.num_heads * d + 2 * cfg.num_kv_heads * d + cfg.num_heads * seq_len + 3 * cfg.intermediate_size + cfg.vocab_size
    return 4 * floats


def scratch_prefill_bytes(cfg: ModelConfig, prompt_len: int) -> int:
    """Estimate: fp32 buffers for prefilling ``prompt_len`` tokens at once (score matrices dominate as it grows)."""
    p, h, d = prompt_len, cfg.hidden_size, cfg.head_dim
    floats = (4 * p * h + p * cfg.num_heads * d + 2 * p * cfg.num_kv_heads * d + cfg.num_heads * p * p
              + 3 * p * cfg.intermediate_size + p * cfg.vocab_size)
    return 4 * floats


def budget(cfg: ModelConfig, seq_lens: tuple[int, ...] = (128, 256), quantize_embeddings: bool = False) -> dict[str, Any]:
    """The per-model figures ``docs/benchmarks`` reports (bytes and MiB)."""
    fp32, int8 = weight_bytes_fp32(cfg), weight_bytes_int8(cfg, quantize_embeddings)
    out: dict[str, Any] = {
        "parameters": count_parameters(cfg), "embedding_parameters": cfg.vocab_size * cfg.hidden_size,
        "fp32_weight_bytes": fp32, "fp32_weight_MiB": round(fp32 / MiB, 3),
        "int8_weight_bytes": int8, "int8_weight_MiB": round(int8 / MiB, 3),
        "int8_quantizes_embeddings": quantize_embeddings, "kv_cache": {}, "runtime_ram_estimate": {},
        "note": "kv_cache is exact; runtime_ram_estimate = fp32 weights + KV cache + scratch (an estimate; measure on device)"}
    for s in seq_lens:
        kv = kv_cache_bytes(cfg, s)
        scratch = max(scratch_decode_bytes(cfg, s), scratch_prefill_bytes(cfg, s))
        out["kv_cache"][str(s)] = {"bytes": kv, "MiB": round(kv / MiB, 3)}
        out["runtime_ram_estimate"][str(s)] = {"fp32_MiB": round((fp32 + kv + scratch) / MiB, 2),
                                                "int8_weights_dequantized_on_the_fly_MiB": round((int8 + kv + scratch) / MiB, 2)}
    return out
