"""A lightweight baseline benchmark, per the engineering brief section 23:
prefill latency, decode latency, tokens/sec, and peak memory (where
available) at prompt lengths 32/128/512.

This establishes a baseline for the reference architecture on **pure
NumPy, single-threaded, CPU-only** execution — see
``tinymind/model/tensor.py``'s module docstring for why there's no GPU or
compiled-kernel path to benchmark instead in this delivery, and treat every
number this script prints accordingly: it measures this specific reference
implementation in this specific environment, not what TinyMind's
architecture could do with a real (native, quantized, GPU-resident) engine
behind it. Per the brief section 23: "Do not optimize aggressively yet. The
purpose is to establish a baseline."

Run with: ``python -m benchmarks.model_baseline``
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.generation import ModelGenerationConfig, generate_with_cache_ids
from tinymind.model.model import TinyMindTransformer

try:
    import resource
    _HAVE_RESOURCE = True
except ImportError:  # resource is POSIX-only; Windows has no equivalent in the stdlib
    _HAVE_RESOURCE = False


def _peak_rss_mb() -> float | None:
    if not _HAVE_RESOURCE:
        return None
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_kb / 1024.0  # ru_maxrss is KB on Linux


def _measure_prefill(model: TinyMindTransformer, prompt_len: int, repeats: int = 3) -> float:
    input_ids = np.random.randint(0, model.config.vocab_size, size=(1, prompt_len))
    # One untimed warm-up call: excludes one-time costs (e.g. first-call
    # allocation) from the measurement, the standard microbenchmark practice.
    model(input_ids, use_cache=False)
    times = []
    for _ in range(repeats):
        start = time.monotonic()
        model(input_ids, use_cache=False)
        times.append(time.monotonic() - start)
    return min(times) * 1000.0  # best-of-N, ms


def _measure_decode(model: TinyMindTransformer, prompt_len: int, decode_steps: int = 16) -> tuple[float, float]:
    config = ModelGenerationConfig(max_new_tokens=decode_steps, do_sample=False, eos_token_id=None)
    prompt = np.random.randint(0, model.config.vocab_size, size=(1, prompt_len))
    start = time.monotonic()
    generate_with_cache_ids(model, prompt, config)
    elapsed = time.monotonic() - start
    tokens_generated = decode_steps  # eos disabled above, so exactly this many were generated
    return (elapsed / tokens_generated) * 1000.0, tokens_generated / elapsed  # (ms/token, tokens/sec)


def run_baseline(config: ModelConfig, prompt_lengths: list[int] = (32, 128, 512),
                 decode_steps: int = 16, seed: int = 0) -> list[dict]:
    np.random.seed(seed)
    model = TinyMindTransformer(config, seed=seed)
    results = []
    for prompt_len in prompt_lengths:
        if prompt_len > config.max_seq_len:
            continue  # this config physically cannot prefill a prompt this long
        prefill_ms = _measure_prefill(model, prompt_len)
        max_decode_steps = min(decode_steps, config.max_seq_len - prompt_len)
        decode_ms_per_token, tokens_per_sec = (
            _measure_decode(model, prompt_len, max_decode_steps) if max_decode_steps > 0 else (None, None))
        results.append({
            "prompt_len": prompt_len,
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms_per_token": round(decode_ms_per_token, 2) if decode_ms_per_token else None,
            "decode_tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="50m", help="a tinymind config preset name")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--decode-steps", type=int, default=16)
    args = parser.parse_args()

    from tinymind.model.config import load_preset
    config = load_preset(args.preset)

    print(f"TinyMind model baseline — preset={args.preset!r}, "
         f"~{config.approx_param_count / 1e6:.1f}M params (config estimate)")
    print("Backend: pure NumPy, single-process, CPU-only reference implementation "
         "(see this file's module docstring)\n")

    results = run_baseline(config, prompt_lengths=args.prompt_lengths, decode_steps=args.decode_steps)
    for row in results:
        print(f"prompt_len={row['prompt_len']:5d}  prefill={row['prefill_ms']:8.2f} ms  "
             f"decode={row['decode_ms_per_token'] or 0:7.2f} ms/token  "
             f"({row['decode_tokens_per_sec'] or 0:6.2f} tok/s)")

    peak = _peak_rss_mb()
    if peak is not None:
        print(f"\npeak RSS: {peak:.1f} MB (this process, not just the model)")
    else:
        print("\npeak RSS: unavailable on this platform (no `resource` module)")


if __name__ == "__main__":
    main()
