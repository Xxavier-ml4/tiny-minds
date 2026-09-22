"""Measured CPU training throughput. Nothing here is estimated: every number
comes from timing real forward + backward + AdamW steps on this machine.

Single configuration (prints one JSON object)::

    python benchmarks/train_benchmark.py run --profile tiny_mobile --batch 8 --seq 256

Matrices used by ``docs/benchmarks/cpu-training-baseline.md``::

    python benchmarks/train_benchmark.py matrix --kind attention   # MHA vs GQA vs MQA at equal parameter budget
    python benchmarks/train_benchmark.py matrix --kind seqlen      # 64/128/256/512
    python benchmarks/train_benchmark.py matrix --kind batch       # micro-batch size
    python benchmarks/train_benchmark.py matrix --kind profiles    # tiny_debug / tiny_mobile / tiny_mobile_plus

Each configuration runs in a fresh subprocess so peak RSS is that
configuration's own. ``--fused 0`` benchmarks the reference (composed) ops.

Method: ``warmup`` untimed steps, then ``steps`` timed steps of
zero_grad -> forward(+loss) -> backward(retain_graph=False) -> AdamW.step on
random token ids with every token in the loss; the reported time is the
median step. Random ids make step time independent of data, which is the
point of a throughput benchmark; *loss* comparisons need real data and are
done by ``benchmarks/train_quality_sweeps.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tinymind.model import ModelConfig, TinyMindTransformer  # noqa: E402
from tinymind.model import fused  # noqa: E402
from tinymind.model.config import count_parameters, load_preset  # noqa: E402
from tinymind.model.optim import AdamW  # noqa: E402


def load_profile(name: str) -> ModelConfig:
    path = Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml"
    return ModelConfig.from_yaml(path)


def match_budget(cfg: ModelConfig, target: int) -> ModelConfig:
    """Adjust ``intermediate_size`` (multiples of 8) so ``count_parameters`` is as close to ``target`` as possible;
    this is how MHA / GQA / MQA are compared at the same parameter budget."""
    best = cfg
    best_gap = abs(count_parameters(cfg) - target)
    for inter in range(max(8, cfg.intermediate_size // 2), cfg.intermediate_size * 2, 8):
        cand = dataclasses.replace(cfg, intermediate_size=inter)
        gap = abs(count_parameters(cand) - target)
        if gap < best_gap:
            best, best_gap = cand, gap
    return best


def attention_variants(base: ModelConfig) -> dict[str, ModelConfig]:
    """MHA (kv = heads), GQA (kv = heads/2), MQA (kv = 1) with the parameter count of the MHA model."""
    mha = dataclasses.replace(base, num_kv_heads=base.num_heads, attention_type="mha")
    target = count_parameters(mha)
    out = {"mha": mha}
    for name, kv in (("gqa", max(1, base.num_heads // 2)), ("mqa", 1)):
        cand = dataclasses.replace(base, num_kv_heads=kv, attention_type="gqa" if name == "gqa" else "mqa")
        out[name] = match_budget(cand, target)
    return out


def kv_cache_bytes(cfg: ModelConfig, seq_len: int) -> int:
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * seq_len * 4  # K and V, float32


def measure(cfg: ModelConfig, batch: int, seq: int, accum: int = 1, steps: int = 6, warmup: int = 2,
            use_fused: bool = True, seed: int = 0) -> dict:
    with fused.use_fused(use_fused):
        model = TinyMindTransformer(cfg, seed=seed)
        named = list(model.named_parameters())
        opt = AdamW([p for _, p in named], learning_rate=1e-3, names=[n for n, _ in named], decay_min_ndim=2)
        rng = np.random.default_rng(seed)
        ids = rng.integers(4, cfg.vocab_size, size=(batch, seq))
        n_loss = batch * (seq - 1) * accum
        times, split = [], []
        for step in range(warmup + steps):
            t0 = time.perf_counter()
            opt.zero_grad()
            fw = bw = 0.0
            for _ in range(accum):
                a = time.perf_counter()
                out = model(ids, labels=ids, loss_normalizer=float(n_loss))
                b = time.perf_counter()
                out.loss.backward(retain_graph=False)
                c = time.perf_counter()
                fw += b - a
                bw += c - b
                del out
            d = time.perf_counter()
            opt.step(grad_clip_norm=1.0)
            e = time.perf_counter()
            if step >= warmup:
                times.append(e - t0)
                split.append((fw, bw, e - d))
        gc.collect()
    med = statistics.median(times)
    tokens = batch * seq * accum
    fw, bw, op = (statistics.median(x[i] for x in split) for i in range(3))
    return {
        "model": {"parameters": count_parameters(cfg), "hidden": cfg.hidden_size, "layers": cfg.num_layers,
                  "heads": cfg.num_heads, "kv_heads": cfg.num_kv_heads, "intermediate": cfg.intermediate_size,
                  "vocab": cfg.vocab_size, "seq_len": seq, "head_dim": cfg.head_dim},
        "training": {"batch": batch, "accumulation": accum, "effective_batch": batch * accum, "fused_ops": use_fused,
                     "step_seconds_median": round(med, 4), "steps_per_sec": round(1 / med, 3),
                     "seconds_per_100_steps": round(100 * med, 1), "tokens_per_sec": round(tokens / med),
                     "examples_per_sec": round(batch * accum / med, 2),
                     "forward_s": round(fw, 4), "backward_s": round(bw, 4), "optimizer_s": round(op, 4),
                     "step_seconds_min_max": [round(min(times), 4), round(max(times), 4)]},
        "memory": {"parameter_MB_fp32": round(count_parameters(cfg) * 4 / 2**20, 2),
                   "kv_cache_MB_fp32_at_seq": round(kv_cache_bytes(cfg, seq) / 2**20, 3),
                   "peak_rss_MB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)},
    }


def machine_info() -> dict:
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    return {"cpu": cpu, "cores_visible": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(),
            "threads_env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def config_from_args(args) -> ModelConfig:
    if args.profile:
        cfg = load_profile(args.profile)
    else:
        cfg = ModelConfig(hidden_size=args.hidden, num_layers=args.layers, num_heads=args.heads,
                          num_kv_heads=args.kv_heads, intermediate_size=args.intermediate, max_seq_len=max(args.seq, 64),
                          vocab_size=args.vocab)
    if args.kv_heads_override:
        cfg = dataclasses.replace(cfg, num_kv_heads=args.kv_heads_override)
    if args.intermediate_override:
        cfg = dataclasses.replace(cfg, intermediate_size=args.intermediate_override)
    return dataclasses.replace(cfg, max_seq_len=max(cfg.max_seq_len, args.seq))


def cmd_run(args) -> None:
    cfg = config_from_args(args)
    print(json.dumps(measure(cfg, args.batch, args.seq, args.accum, args.steps, args.warmup, bool(args.fused))))


def run_sub(profile: str | None, dims: dict, batch: int, seq: int, accum: int = 1, steps: int = 6, fused_flag: int = 1) -> dict:
    cmd = [sys.executable, __file__, "run", "--batch", str(batch), "--seq", str(seq), "--accum", str(accum),
           "--steps", str(steps), "--fused", str(fused_flag)]
    if profile:
        cmd += ["--profile", profile]
    for key, value in dims.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def cmd_matrix(args) -> None:
    rows: list[dict] = []
    base = load_profile(args.profile)
    if args.kind == "attention":
        for seq in (128, 256):
            for name, cfg in attention_variants(dataclasses.replace(base, max_seq_len=seq)).items():
                r = run_sub(None, dict(hidden=cfg.hidden_size, layers=cfg.num_layers, heads=cfg.num_heads,
                                       kv_heads=cfg.num_kv_heads, intermediate=cfg.intermediate_size, vocab=cfg.vocab_size),
                            args.batch, seq, steps=args.steps)
                r["variant"] = name
                rows.append(r)
    elif args.kind == "seqlen":
        for seq in (64, 128, 256, 512):
            batch = max(1, args.tokens_per_step // seq)  # constant tokens per step so tokens/sec is comparable
            r = run_sub(args.profile, {}, batch, seq, steps=args.steps)
            rows.append(r)
    elif args.kind == "batch":
        for batch in (1, 2, 4, 8, 16):
            rows.append(run_sub(args.profile, {}, batch, args.seq, steps=args.steps))
    elif args.kind == "profiles":
        for name in ("tiny_debug", "tiny_mobile", "tiny_mobile_plus"):
            seq = load_profile(name).max_seq_len
            rows.append({"profile": name, **run_sub(name, {}, args.batch, seq, steps=args.steps)})
    elif args.kind == "fused":
        for seq in (128, 256):
            for flag in (0, 1):
                rows.append(run_sub(args.profile, {}, args.batch, seq, steps=args.steps, fused_flag=flag))
    else:
        raise SystemExit(f"unknown kind {args.kind}")
    result = {"kind": args.kind, "machine": machine_info(), "rows": rows}
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text if not args.out else f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile")
    r.add_argument("--hidden", type=int, default=128)
    r.add_argument("--layers", type=int, default=6)
    r.add_argument("--heads", type=int, default=4)
    r.add_argument("--kv-heads", type=int, default=2)
    r.add_argument("--intermediate", type=int, default=384)
    r.add_argument("--vocab", type=int, default=260)
    r.add_argument("--kv-heads-override", type=int, default=0)
    r.add_argument("--intermediate-override", type=int, default=0)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--seq", type=int, default=256)
    r.add_argument("--accum", type=int, default=1)
    r.add_argument("--steps", type=int, default=6)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--fused", type=int, default=1)
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser("matrix")
    m.add_argument("--kind", required=True, choices=["attention", "seqlen", "batch", "profiles", "fused"])
    m.add_argument("--profile", default="tiny_mobile")
    m.add_argument("--batch", type=int, default=8)
    m.add_argument("--seq", type=int, default=256)
    m.add_argument("--steps", type=int, default=6)
    m.add_argument("--tokens-per-step", type=int, default=2048)
    m.add_argument("--out")
    m.set_defaults(fn=cmd_matrix)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
