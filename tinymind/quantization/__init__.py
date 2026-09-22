"""Quantization: real, working post-training weight quantization, per the
engineering brief section 20.

Deferred from Phase 1 (needed real weights to quantize — brief section 44)
and implemented now that Phase 3A produced a real, trainable
``TinyMindTransformer``. This is **weight-only, per-output-channel,
symmetric min-max quantization** — not Needle's own "Cactus Quants"
technique (needle-analysis.md section 13: a Hadamard-basis rotation plus a
shared Lloyd-Max codebook, trained through with a straight-through
estimator), which is a legitimate, more sophisticated technique worth
building independently later, but is a training-time (QAT) method; nothing
in this delivery trains with quantization in the loop, so a *post-training*
scheme is what's actually implementable from what exists today. See
``docs/architecture/tinymind-design.md`` section 11 for that original
scoping decision.

**Why "weight-only" and not activation quantization too**: activation
quantization needs representative activation *statistics* — running real
inputs through the model and recording what the intermediate activations
typically look like. This delivery has no real text corpus to draw
representative prompts from (see ``STATUS.md``), so ``calibrate()`` below
is a real, working no-op for these two schemes: a per-row absolute-max
scale is derived directly and freshly from each weight tensor's own values
at ``quantize()`` time, which is a standard, legitimate post-training
quantization approach in its own right (sometimes called "dynamic" or
"round-to-nearest" quantization) — it simply doesn't need a separate
calibration pass the way activation quantization would.
"""
from __future__ import annotations

import abc
import dataclasses

import numpy as np


@dataclasses.dataclass
class QuantizationReport:
    scheme: str
    bits: int
    original_size_bytes: int
    quantized_size_bytes: int
    # accuracy_delta/latency_delta_ms/tokens_per_second stay Optional[float]
    # = None until a benchmark run fills them in for a *specific* model and
    # workload — see tinymind.quantization.benchmark.benchmark_quantization,
    # which populates all three for real against an actual
    # TinyMindTransformer; a QuantizationReport built directly (bypassing
    # that function) has no basis to guess these and should leave them None
    # rather than invent a number.
    accuracy_delta: float | None = None
    latency_delta_ms: float | None = None
    tokens_per_second: float | None = None

    @property
    def compression_ratio(self) -> float:
        return self.original_size_bytes / self.quantized_size_bytes if self.quantized_size_bytes else float("inf")


class QuantizationScheme(abc.ABC):
    """One scheme (int8, int4, int3, int2, AWQ, GPTQ, ...) implements this."""

    name: str
    bits: int

    @abc.abstractmethod
    def calibrate(self, sample_activations: list) -> None:
        """Compute whatever calibration statistics this scheme needs beyond
        what ``quantize()`` derives from the tensor itself. A no-op for
        ``Int8Scheme``/``Int4Scheme`` — see this module's docstring for
        why that's a real, considered answer and not a shortcut."""

    @abc.abstractmethod
    def quantize(self, tensor_bytes: bytes, shape: tuple[int, ...], dtype: str
                ) -> tuple[bytes, dict]:
        """Return (quantized_bytes, extra_metadata) for one tensor."""

    @abc.abstractmethod
    def dequantize(self, quantized_bytes: bytes, metadata: dict) -> bytes:
        """Return float32 bytes, reconstructed from quantized_bytes+metadata."""


def _per_row_absmax_scale(weights: np.ndarray, qmax: int) -> np.ndarray:
    """One scale per row (axis 0 — the output-channel axis for every
    ``Linear`` weight in this codebase, ``[out_features, in_features]``;
    see ``docs/architecture/native-model-contract.md`` section 2). Per-row
    rather than one scale for the whole tensor: different output channels
    of a trained weight matrix routinely have quite different value
    ranges, and a single global scale wastes precision on every row except
    the one with the largest values — a well-known, standard reason
    per-channel quantization beats per-tensor quantization in practice.
    """
    row_absmax = np.abs(weights).max(axis=-1)
    row_absmax = np.where(row_absmax == 0, 1.0, row_absmax)  # an all-zero row would otherwise divide by zero
    return (row_absmax / qmax).astype(np.float32)


class Int8Scheme(QuantizationScheme):
    name, bits = "int8", 8
    _QMAX = 127  # symmetric signed range [-127, 127]; -128 is left unused so +/- are symmetric

    def calibrate(self, sample_activations: list) -> None:
        pass  # see module docstring

    def quantize(self, tensor_bytes: bytes, shape: tuple[int, ...], dtype: str) -> tuple[bytes, dict]:
        if dtype != "float32":
            raise ValueError(f"Int8Scheme.quantize() expects float32 input, got {dtype!r}")
        weights = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(shape)
        if weights.ndim != 2:
            raise ValueError(f"Int8Scheme.quantize() expects a 2D [out, in] weight, got shape {shape}")
        scales = _per_row_absmax_scale(weights, self._QMAX)
        quantized = np.round(weights / scales[:, None]).clip(-self._QMAX, self._QMAX).astype(np.int8)
        metadata = {"scales": scales.tolist(), "shape": list(shape), "bits": self.bits}
        return quantized.tobytes(), metadata

    def dequantize(self, quantized_bytes: bytes, metadata: dict) -> bytes:
        shape = tuple(metadata["shape"])
        scales = np.array(metadata["scales"], dtype=np.float32)
        quantized = np.frombuffer(quantized_bytes, dtype=np.int8).reshape(shape)
        return (quantized.astype(np.float32) * scales[:, None]).tobytes()


def _pack_int4(values: np.ndarray) -> np.ndarray:
    """Two signed 4-bit values (range [-7, 7], stored as an unsigned
    nibble 0-15 via a +8 offset — i.e. excess-8 / offset-binary, decoded by
    subtracting 8 back out in ``_unpack_int4``) packed per output byte:
    low nibble = values[..., 0::2], high nibble = values[..., 1::2].
    ``values`` must have an even number of elements along its last axis —
    callers pad with a zero column first if needed (see ``quantize()``,
    which does exactly that and records the pre-padding width in
    metadata so ``dequantize()`` can slice the padding back off).
    """
    offset = (values.astype(np.int16) + 8).astype(np.uint8)  # 0..15
    low = offset[..., 0::2]
    high = offset[..., 1::2]
    return (low | (high << 4)).astype(np.uint8)


def _unpack_int4(packed: np.ndarray, last_dim: int) -> np.ndarray:
    low = (packed & 0x0F).astype(np.int16) - 8
    high = ((packed >> 4) & 0x0F).astype(np.int16) - 8
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.int16)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out[..., :last_dim]


class Int4Scheme(QuantizationScheme):
    name, bits = "int4", 4
    _QMAX = 7  # symmetric signed range representable in the excess-8 nibble encoding above

    def calibrate(self, sample_activations: list) -> None:
        pass  # see module docstring

    def quantize(self, tensor_bytes: bytes, shape: tuple[int, ...], dtype: str) -> tuple[bytes, dict]:
        if dtype != "float32":
            raise ValueError(f"Int4Scheme.quantize() expects float32 input, got {dtype!r}")
        weights = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(shape)
        if weights.ndim != 2:
            raise ValueError(f"Int4Scheme.quantize() expects a 2D [out, in] weight, got shape {shape}")
        scales = _per_row_absmax_scale(weights, self._QMAX)
        quantized = np.round(weights / scales[:, None]).clip(-self._QMAX, self._QMAX).astype(np.int16)

        original_cols = quantized.shape[-1]
        if original_cols % 2 != 0:
            quantized = np.pad(quantized, ((0, 0), (0, 1)))  # one zero column so packing has an even width
        packed = _pack_int4(quantized)

        metadata = {"scales": scales.tolist(), "shape": list(shape), "bits": self.bits,
                   "packed_shape": list(packed.shape)}
        return packed.tobytes(), metadata

    def dequantize(self, quantized_bytes: bytes, metadata: dict) -> bytes:
        shape = tuple(metadata["shape"])
        packed_shape = tuple(metadata["packed_shape"])
        scales = np.array(metadata["scales"], dtype=np.float32)
        packed = np.frombuffer(quantized_bytes, dtype=np.uint8).reshape(packed_shape)
        unpacked = _unpack_int4(packed, shape[-1])
        return (unpacked.astype(np.float32) * scales[:, None]).tobytes()


# INT3/INT2/AWQ/GPTQ/mixed-precision are further out (brief section 20:
# "Later experiment with...") and are not stubbed individually here to
# avoid exactly the "hundreds of near-empty files" pattern the brief warns
# against (section 60) — Int8Scheme and Int4Scheme above establish the
# pattern every later scheme follows; adding one is implementing
# QuantizationScheme again, not designing a new interface.
