"""Quantize every 2D weight in a real ``TinyMindTransformer`` and measure
what it actually costs in output quality — brief section 20's own
requirement: "Quantization must be benchmarked rather than assumed to be
beneficial."

Only 2D tensors (every ``Linear`` projection, plus the token embedding
table) are quantized; 1D tensors (every ``RMSNorm.weight``) are left at
float32 — standard practice, and also a hard constraint of the schemes
themselves (``Int8Scheme``/``Int4Scheme.quantize()`` reject non-2D input;
see ``tinymind/quantization/__init__.py``). Norm weights are a small
fraction of total parameters and quantizing them risks destabilizing the
normalization math for little size benefit.

This module produces a **reference, dequantize-then-run** quantized model:
the quantized bytes are the real compressed representation (and are what
gets written to a `.tm` file — see ``export_quantized_tm``), but running
inference still happens through the ordinary float32
``TinyMindTransformer`` forward pass, with each quantized tensor
dequantized back to float32 first. This delivery has no low-precision
GEMM kernel (that's a native-runtime concern — ``native/src/model.cpp`` is
still a stub; see ``STATUS.md``), so the *runtime* memory/speed benefit of
quantization isn't realized by this reference path — only the *storage*
benefit is (measured directly below) and, just as importantly, the
*quality* cost is (also measured directly, by comparing real logits before
and after).
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.model import TinyMindTransformer
from tinymind.quantization import Int4Scheme, Int8Scheme, QuantizationReport, QuantizationScheme

_SCHEMES: dict[str, type[QuantizationScheme]] = {"int8": Int8Scheme, "int4": Int4Scheme}


@dataclasses.dataclass
class QuantizedTensor:
    name: str
    quantized_bytes: bytes
    metadata: dict
    original_shape: tuple[int, ...]


@dataclasses.dataclass
class QuantizedModel:
    config: ModelConfig
    scheme_name: str
    quantized: dict[str, QuantizedTensor]   # 2D tensors, compressed
    unquantized: dict[str, np.ndarray]      # 1D tensors (norms), kept at float32


def quantize_model(model: TinyMindTransformer, scheme_name: str = "int8",
                   quantize_embeddings: bool = True) -> QuantizedModel:
    """``quantize_embeddings=False`` keeps the token-embedding table in float32 (it is also the tied output head,
    so its rounding error lands directly on the logits); every other 2-D weight is quantized either way. The
    default stays ``True`` for Phase 3A compatibility; Phase 3B measures both (``benchmarks/quantization_eval.py``)."""
    if scheme_name not in _SCHEMES:
        raise ValueError(f"unknown scheme {scheme_name!r}; available: {sorted(_SCHEMES)}")
    scheme = _SCHEMES[scheme_name]()
    scheme.calibrate([])  # a real, documented no-op for these two schemes — see module docstring above

    quantized: dict[str, QuantizedTensor] = {}
    unquantized: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if param.data.ndim == 2 and (quantize_embeddings or name != "embed_tokens"):
            qbytes, metadata = scheme.quantize(param.data.tobytes(), param.data.shape, "float32")
            quantized[name] = QuantizedTensor(name=name, quantized_bytes=qbytes, metadata=metadata,
                                             original_shape=param.data.shape)
        else:
            unquantized[name] = param.data.copy()

    return QuantizedModel(config=model.config, scheme_name=scheme_name, quantized=quantized,
                          unquantized=unquantized)


def dequantize_to_model(quantized_model: QuantizedModel, *, seed: int | None = None) -> TinyMindTransformer:
    """Reconstruct a runnable ``TinyMindTransformer`` with every quantized
    tensor dequantized back to float32 — see module docstring for exactly
    what this does and doesn't prove about runtime cost."""
    scheme = _SCHEMES[quantized_model.scheme_name]()
    model = TinyMindTransformer(quantized_model.config, seed=seed)  # weights overwritten below
    params = dict(model.named_parameters())

    for name, qtensor in quantized_model.quantized.items():
        raw = scheme.dequantize(qtensor.quantized_bytes, qtensor.metadata)
        params[name].data[...] = np.frombuffer(raw, dtype=np.float32).reshape(qtensor.original_shape)
    for name, array in quantized_model.unquantized.items():
        params[name].data[...] = array

    return model


def report(quantized_model: QuantizedModel, original_model: TinyMindTransformer,
          probe_input_ids: np.ndarray) -> QuantizationReport:
    """A real ``QuantizationReport`` — every field measured, none guessed.
    ``accuracy_delta`` is the mean absolute difference in output
    probability assigned to the *original* model's own top predicted
    token at every position of ``probe_input_ids``, before vs. after
    quantization — a direct, interpretable measure of "how much did
    quantization change what the model would actually output," rather
    than a raw logit-distance number with no intuitive scale.
    """
    scheme_bits = _SCHEMES[quantized_model.scheme_name].bits
    original_size = sum(p.data.nbytes for _name, p in original_model.named_parameters())
    quantized_size = (sum(len(q.quantized_bytes) for q in quantized_model.quantized.values())
                      + sum(a.nbytes for a in quantized_model.unquantized.values()))

    dequantized_model = dequantize_to_model(quantized_model)

    start = time.monotonic()
    original_logits = original_model(probe_input_ids).logits.data
    original_latency = (time.monotonic() - start) * 1000.0

    start = time.monotonic()
    quantized_logits = dequantized_model(probe_input_ids).logits.data
    quantized_latency = (time.monotonic() - start) * 1000.0

    original_probs = _softmax_last_axis(original_logits)
    quantized_probs = _softmax_last_axis(quantized_logits)
    top_token = np.argmax(original_probs, axis=-1)
    b_idx, t_idx = np.indices(top_token.shape)
    prob_at_original_top = original_probs[b_idx, t_idx, top_token]
    prob_at_original_top_after_quant = quantized_probs[b_idx, t_idx, top_token]
    accuracy_delta = float(np.mean(np.abs(prob_at_original_top - prob_at_original_top_after_quant)))

    return QuantizationReport(
        scheme=quantized_model.scheme_name, bits=scheme_bits,
        original_size_bytes=original_size, quantized_size_bytes=quantized_size,
        accuracy_delta=accuracy_delta, latency_delta_ms=quantized_latency - original_latency,
        tokens_per_second=None,  # a single forward pass isn't a generation-throughput measurement;
                                 # see benchmarks/model_baseline.py for that, run separately
    )


def _softmax_last_axis(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


_QUANTIZED_ARCHITECTURE_TAG = "tinymind-transformer-v1-quantized"


def export_quantized_tm(quantized_model: QuantizedModel, path) -> None:
    """Write a quantized model to a `.tm` file. Per-row scales are stored
    as ordinary sibling float32 tensors (``<name>.scale``) rather than as
    format-level metadata — the existing `.tm` reader/writer
    (``tinymind.runtime.format``) already handles an arbitrary named
    tensor perfectly well, so a scale vector doesn't need the format
    itself to change, only this module's mapping into it. For
    ``int4``, the stored tensor's declared ``shape`` is the *logical*
    (unpacked) weight shape for readability (e.g. via ``tinymind inspect``)
    even though the underlying bytes are packed two values per byte — the
    `.tm` format itself never cross-checks byte count against
    ``product(shape)`` (see ``tinymind.runtime.format``), so this is safe;
    ``import_quantized_tm`` below is the only code that needs to know the
    packed layout, via the ``packed_shape`` recorded in this file's JSON
    metadata block instead.
    """
    from tinymind.runtime.format import write_model

    scheme_bits = _SCHEMES[quantized_model.scheme_name].bits
    dtype_tag = quantized_model.scheme_name  # "int8" / "int4" — already valid tinymind.runtime.format dtypes

    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    per_tensor_metadata: dict[str, dict] = {}
    for name, qtensor in quantized_model.quantized.items():
        tensors[name] = (dtype_tag, qtensor.original_shape, qtensor.quantized_bytes)
        tensors[f"{name}.scale"] = ("float32", (len(qtensor.metadata["scales"]),),
                                    np.array(qtensor.metadata["scales"], dtype=np.float32).tobytes())
        per_tensor_metadata[name] = {k: v for k, v in qtensor.metadata.items() if k != "scales"}
    for name, array in quantized_model.unquantized.items():
        tensors[name] = ("float32", array.shape, array.tobytes())

    metadata = {
        "architecture": _QUANTIZED_ARCHITECTURE_TAG,
        "container_kind": "deployment-quantized",
        "model_config": quantized_model.config.to_dict(),
        "quantization_scheme": quantized_model.scheme_name,
        "quantization_bits": scheme_bits,
        "per_tensor_metadata": per_tensor_metadata,
        "quantized_tensor_names": sorted(quantized_model.quantized),
    }
    write_model(path, metadata=metadata, tensors=tensors)


def import_quantized_tm(path) -> QuantizedModel:
    from tinymind.runtime.format import ModelFormatError, read_model

    model_file = read_model(path)
    if model_file.metadata.get("architecture") != _QUANTIZED_ARCHITECTURE_TAG:
        raise ModelFormatError(
            f"this .tm file declares architecture {model_file.metadata.get('architecture')!r}, "
            f"expected {_QUANTIZED_ARCHITECTURE_TAG!r} — it was not exported by "
            "export_quantized_tm(), or is a different format version")

    config = ModelConfig.from_dict(model_file.metadata["model_config"])
    scheme_name = model_file.metadata["quantization_scheme"]
    per_tensor_metadata = model_file.metadata["per_tensor_metadata"]
    quantized_names = set(model_file.metadata["quantized_tensor_names"])

    quantized: dict[str, QuantizedTensor] = {}
    unquantized: dict[str, np.ndarray] = {}
    for name, entry in model_file.tensors.items():
        if name.endswith(".scale"):
            continue  # handled alongside its owning tensor below
        if name in quantized_names:
            scale_entry_bytes = model_file.read_tensor(f"{name}.scale")
            scales = np.frombuffer(scale_entry_bytes, dtype=np.float32).tolist()
            metadata = dict(per_tensor_metadata[name])
            metadata["scales"] = scales
            quantized[name] = QuantizedTensor(name=name, quantized_bytes=model_file.read_tensor(name),
                                             metadata=metadata, original_shape=tuple(entry.shape))
        else:
            unquantized[name] = np.frombuffer(model_file.read_tensor(name), dtype=np.float32).reshape(entry.shape)

    return QuantizedModel(config=config, scheme_name=scheme_name, quantized=quantized, unquantized=unquantized)
