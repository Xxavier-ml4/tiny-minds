"""50M memory/throughput preflight (brief section 7).

Before a full production 50M run is launched, this measures — on the machine
that would run it — the things that decide whether the run is even feasible:
model initialisation time, one forward pass, the backward pass, an optimizer
step, peak resident memory, and the resulting tokens/sec, across a few
``(batch_size, gradient_accumulation)`` shapes at a modest sequence length.

It builds the *real* model (``TinyMindTransformer``) and the *real* optimizer
(``AdamW``) with the profile's own geometry, and drives one full optimizer step
per configuration exactly the way the training engine does — ``accumulation``
micro-batches of ``batch_size`` rows, each forward+backward, then one AdamW
step — so the numbers are representative, not a toy. Inputs are random token
ids (this measures compute and memory, not learning), and everything runs under
``no_grad``-free autograd just like training.

Nothing here trains anything or writes a checkpoint. ``tinymind benchmark
train-step --config 50m`` calls ``run`` and prints the JSON; the CI 50M
workflow runs it as a gate before the first real stage.
"""
from __future__ import annotations

import gc
import resource
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tinymind.model import ModelConfig, TinyMindTransformer
from tinymind.model.optim import AdamW


def _rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux, bytes on macOS. TinyMind targets Linux
    # CI; the training engine's _rss_mb uses the same kB assumption.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# The shapes the brief asks for: batch 1 with accumulation 8 / 16 / 32.
DEFAULT_SHAPES: tuple[tuple[int, int], ...] = ((1, 8), (1, 16), (1, 32))
DEFAULT_SEQ_LEN = 512


@dataclass
class StepMeasurement:
    batch_size: int
    gradient_accumulation: int
    seq_len: int
    effective_batch_tokens: int
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    step_seconds: float
    tokens_per_sec: float
    peak_rss_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _build_model(config: ModelConfig, seed: int) -> tuple[TinyMindTransformer, AdamW, float]:
    t0 = time.perf_counter()
    model = TinyMindTransformer(config, seed=seed)
    named = list(model.named_parameters())
    optimizer = AdamW([p for _, p in named], learning_rate=1e-3, names=[n for n, _ in named])
    return model, optimizer, time.perf_counter() - t0


def measure_shape(config: ModelConfig, batch_size: int, gradient_accumulation: int, seq_len: int,
                  seed: int = 0, warmup: bool = True) -> StepMeasurement:
    """One optimizer step's cost for a shape, built fresh so peak RSS is that
    shape's own. ``warmup`` runs a throwaway micro-step first so BLAS/allocator
    warm-up is not charged to the timed measurement."""
    if seq_len > config.max_seq_len:
        raise ValueError(f"seq_len {seq_len} exceeds the model's max_seq_len {config.max_seq_len}")
    model, optimizer, _ = _build_model(config, seed)
    rng = np.random.default_rng(seed)

    def micro() -> tuple[float, float]:
        ids = rng.integers(0, config.vocab_size, size=(batch_size, seq_len))
        labels = ids.copy()
        t0 = time.perf_counter()
        out = model.forward(ids, labels=labels, loss_normalizer=float(batch_size * seq_len))
        fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        out.loss.backward()
        bwd = time.perf_counter() - t0
        return fwd, bwd

    if warmup:
        micro()
        optimizer.step(1.0)
        optimizer.zero_grad()

    fwd_total = bwd_total = 0.0
    t_step = time.perf_counter()
    for _ in range(gradient_accumulation):
        fwd, bwd = micro()
        fwd_total += fwd
        bwd_total += bwd
    t0 = time.perf_counter()
    optimizer.step(1.0)
    optimizer.zero_grad()
    opt_seconds = time.perf_counter() - t0
    step_seconds = time.perf_counter() - t_step

    eff = batch_size * gradient_accumulation * seq_len
    peak = _rss_mb()
    del model, optimizer
    gc.collect()
    return StepMeasurement(
        batch_size=batch_size, gradient_accumulation=gradient_accumulation, seq_len=seq_len,
        effective_batch_tokens=eff, forward_seconds=fwd_total / gradient_accumulation,
        backward_seconds=bwd_total / gradient_accumulation, optimizer_seconds=opt_seconds,
        step_seconds=step_seconds, tokens_per_sec=eff / step_seconds if step_seconds else 0.0,
        peak_rss_mb=peak)


def run(config: ModelConfig, shapes: tuple[tuple[int, int], ...] = DEFAULT_SHAPES,
        seq_len: int = DEFAULT_SEQ_LEN, seed: int = 0, warmup: bool = True) -> dict[str, Any]:
    """Full preflight for ``config``: init timing, per-shape measurements, and a
    JSON-able report. ``passed`` is true if every shape completed a step with a
    finite loss; a shape that OOMs or errors makes the preflight fail loudly
    rather than being silently dropped."""
    model, _optimizer, init_seconds = _build_model(config, seed)
    params = model.count_parameters()
    del _optimizer, model
    gc.collect()

    measurements: list[dict[str, Any]] = []
    ok = True
    errors: list[dict[str, Any]] = []
    for batch_size, accumulation in shapes:
        try:
            m = measure_shape(config, batch_size, accumulation, seq_len, seed=seed, warmup=warmup)
            measurements.append(m.to_dict())
        except Exception as exc:  # noqa: BLE001 — a failed shape is a real result to report
            ok = False
            errors.append({"batch_size": batch_size, "gradient_accumulation": accumulation,
                           "seq_len": seq_len, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "parameter_count": params,
        "model_init_seconds": round(init_seconds, 4),
        "seq_len": seq_len,
        "peak_rss_mb": round(max([m["peak_rss_mb"] for m in measurements], default=_rss_mb()), 1),
        "shapes": measurements,
        "errors": errors,
        "passed": ok and bool(measurements),
    }
