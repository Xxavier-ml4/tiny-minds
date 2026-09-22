"""Benchmark the machine we are on and pick the BLAS thread count.

GitHub-hosted runners vary (vCPU count, CPU model, noisy neighbours), so the
stage workflow runs this first instead of assuming. It:

1. records OS, Python, NumPy, CPU model, visible/allowed cores, total RAM and the
   BLAS NumPy was built against;
2. runs the real training-step benchmark (``train_benchmark.py run``) in a
   fresh subprocess for each candidate thread count, with
   ``OPENBLAS_NUM_THREADS = OMP_NUM_THREADS = MKL_NUM_THREADS = n`` set
   *before* NumPy is imported (the only time they take effect), each measured
   ``--repeats`` times;
3. selects a thread count with ``select_threads``: the fastest median
   throughput, but a higher count must beat the next-lower one by more than
   ``--min-gain`` (default 5 %) — more threads on tiny matrices often just
   add synchronisation cost, and a setting that wins by noise is not "stable";
4. writes ``runner_profile.json`` and, with ``--github-env``, appends the chosen
   variables to ``$GITHUB_ENV`` so later steps (the training run) inherit them.

    python benchmarks/runner_probe.py --profile tiny_mobile --out runner_profile.json [--github-env "$GITHUB_ENV"]

On a 1-vCPU machine the only candidate is 1; the multi-thread behaviour is
therefore *unmeasured in the sandbox this repository was developed in* and is
measured by this script on the runner itself.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def allowed_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux
        return os.cpu_count() or 1


def blas_info() -> dict:
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        return {"name": blas.get("name"), "version": blas.get("version"), "openblas_config": blas.get("openblas configuration")}
    except Exception as exc:  # noqa: BLE001 - informational only
        return {"error": repr(exc)}


def total_ram_mb() -> int | None:
    try:
        with open("/proc/meminfo") as handle:
            return int(next(l for l in handle if l.startswith("MemTotal")).split()[1]) // 1024
    except (OSError, StopIteration):
        return None


def environment() -> dict:
    import numpy as np
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    return {"os": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "cpu": cpu,
            "cpu_count": os.cpu_count(), "allowed_cores": allowed_cores(), "ram_mb": total_ram_mb(), "blas": blas_info(),
            "github_runner": {k: os.environ.get(k) for k in ("RUNNER_OS", "RUNNER_ARCH", "RUNNER_NAME", "ImageOS", "ImageVersion")}}


def candidates(cores: int) -> list[int]:
    out = sorted({1, 2, max(1, cores // 2), cores})
    return [n for n in out if 1 <= n <= max(1, cores)]


def select_threads(results: dict[int, list[float]], min_gain: float = 0.05) -> dict:
    """``results[n]`` = tokens/sec measurements with n threads. Walk up from the
    smallest n; move to a larger n only if its median beats the current choice
    by more than ``min_gain``."""
    medians = {n: statistics.median(v) for n, v in results.items() if v}
    ordered = sorted(medians)
    chosen = ordered[0]
    for n in ordered[1:]:
        if medians[n] > medians[chosen] * (1.0 + min_gain):
            chosen = n
    return {"chosen_threads": chosen, "median_tokens_per_sec": {str(n): round(m) for n, m in medians.items()},
            "rule": f"a higher thread count must beat the next-lower choice by > {min_gain:.0%}"}


def measure_threads(n: int, profile: str, batch: int, seq: int, steps: int) -> float:
    env = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(n)
    out = subprocess.run([sys.executable, str(HERE / "train_benchmark.py"), "run", "--profile", profile, "--batch", str(batch),
                          "--seq", str(seq), "--steps", str(steps)], capture_output=True, text=True, env=env, check=True)
    return float(json.loads(out.stdout)["training"]["tokens_per_sec"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="tiny_mobile")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--out", default="runner_profile.json")
    p.add_argument("--github-env", default=None, help="path of $GITHUB_ENV to append the chosen thread variables to")
    args = p.parse_args()

    env = environment()
    results = {n: [measure_threads(n, args.profile, args.batch, args.seq, args.steps) for _ in range(args.repeats)]
               for n in candidates(env["allowed_cores"])}
    selection = select_threads(results, args.min_gain)
    profile = {"environment": env, "benchmark": {"profile": args.profile, "batch": args.batch, "seq": args.seq,
                                                    "tokens_per_sec_by_threads": {str(n): v for n, v in results.items()}},
               "selection": selection}
    Path(args.out).write_text(json.dumps(profile, indent=2) + "\n")
    n = selection["chosen_threads"]
    lines = [f"{k}={n}" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")]
    if args.github_env:
        with open(args.github_env, "a") as handle:
            handle.write("\n".join(lines) + "\n")
    print(json.dumps(profile["selection"], indent=2))


if __name__ == "__main__":
    main()
