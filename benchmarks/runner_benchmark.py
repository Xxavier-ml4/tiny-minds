"""Characterise this machine for training TinyMind — not a training run.

Runs a short, bounded number of real optimizer steps (default 30) through the
production `TrainingEngine` on a tiny slice of the deterministic curriculum,
then reports hardware + throughput + memory + the actual sizes of the
checkpoint files that real training would produce. It is deliberately short:
its job is to answer "how fast is this machine, and how big will a
checkpoint be", not to produce a useful model. `--steps` bounds it; there is
no time budget and no resume, because a benchmark run is not meant to be
continued.

    python benchmarks/runner_benchmark.py --profile tiny_mobile --steps 30 --out runner_benchmark.json

Recorded (all measured, none estimated except `estimated_tokens_per_hour`,
which is `tokens_per_sec × 3600` and is labelled as an estimate):
CPU, RAM, BLAS, current BLAS thread-count env vars, exact parameter count,
batch size, sequence length, steps/sec, tokens/sec, peak RSS, the real
`model.npz`/`optimizer.npz`/checkpoint-directory byte sizes, and
estimated tokens/hour.

Training data: `tinymind.data.curriculum` stage0 (the brief's own
"engineering sanity" stage) at a small `--data-scale`, so the benchmark
exercises the real data pipeline without spending time generating a large
corpus. No validation set is built (`allow_no_validation=True`) — this
benchmark measures training throughput, not model quality; use
`tinymind eval` / the stage workflow for quality.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import runner_probe  # noqa: E402


def run_benchmark(*, profile: str, steps: int, batch: int | None, seq: int | None, seed: int, data_scale: float,
                  work_dir: Path) -> dict:
    from tinymind.data.curriculum import build_stage
    from tinymind.data.render import ChatRenderer
    from tinymind.model.config import ModelConfig, count_parameters
    from tinymind.model.tokenizer import ByteTokenizer
    from tinymind.training.checkpoint import environment_info
    from tinymind.training.config import TrainingConfig
    from tinymind.training.data import DataSource, TokenizedDataset
    from tinymind.training.engine import TrainingEngine

    model_config = ModelConfig.from_yaml(HERE.parent / "configs" / f"{profile}.yaml")
    seq = seq or model_config.max_seq_len
    if seq > model_config.max_seq_len:
        raise ValueError(f"--seq {seq} exceeds {profile}'s max_seq_len {model_config.max_seq_len}")
    batch = batch or 8
    tokenizer = ByteTokenizer()
    renderer = ChatRenderer(tokenizer)

    data = build_stage("stage0", seed=seed, scale=data_scale)
    train_records = data["train"]["sanity"]
    if len(train_records) < batch * steps // 4:  # make sure `steps` real optimizer steps are actually reachable
        data = build_stage("stage0", seed=seed, scale=max(data_scale, (batch * steps) / 4000))
        train_records = data["train"]["sanity"]
    train_ds = TokenizedDataset.from_records(train_records, renderer, seq, overflow="drop", name="bench")

    train_config = TrainingConfig(stage="benchmark", seed=seed, batch_size=batch, max_seq_len=seq, max_steps=steps,
                                  warmup_steps=min(5, max(1, steps // 4)), eval_interval=0, checkpoint_interval=0,
                                  log_interval=0, packing=False)
    engine = TrainingEngine(model_config=model_config, tokenizer=tokenizer, config=train_config,
                            sources=[DataSource("bench", train_ds)], validation=None, output_dir=work_dir,
                            allow_no_validation=True, log=None)

    t0 = time.perf_counter()
    summary = engine.train()
    wall_seconds = time.perf_counter() - t0

    ckpt_dir = Path(summary["checkpoint"])
    file_sizes = {f.name: f.stat().st_size for f in ckpt_dir.iterdir() if f.is_file()}
    checkpoint_total_bytes = sum(file_sizes.values())
    model_weights_bytes = file_sizes.get("model.npz")
    optimizer_state_bytes = file_sizes.get("optimizer.npz")

    tokens_per_sec = summary["tokens_per_second_train_average"]
    steps_completed = summary["final_step"] - summary["initial_step"]
    steps_per_sec = steps_completed / summary["train_seconds_total"] if summary["train_seconds_total"] else None

    env = runner_probe.environment()
    thread_env = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env_info = environment_info()
    return {
        "profile": profile, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": env_info["git_commit"],
        "hardware": {"cpu": env["cpu"], "cpu_count": env["cpu_count"], "allowed_cores": env["allowed_cores"],
                     "ram_mb": env["ram_mb"], "blas": env["blas"], "threads_env": thread_env,
                     "github_runner": env["github_runner"]},
        "model": {"parameters": count_parameters(model_config), "hidden_size": model_config.hidden_size,
                  "num_layers": model_config.num_layers, "num_heads": model_config.num_heads,
                  "num_kv_heads": model_config.num_kv_heads, "vocab_size": model_config.vocab_size},
        "training": {"batch_size": batch, "sequence_length": seq, "steps_requested": steps,
                     "steps_completed": steps_completed, "warmup_steps": train_config.warmup_steps,
                     "wall_seconds": round(wall_seconds, 3), "train_seconds": round(summary["train_seconds_total"], 3),
                     "steps_per_sec": round(steps_per_sec, 4) if steps_per_sec else None,
                     "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else None,
                     "tokens_processed": summary["tokens_processed_total"]},
        "memory": {"peak_rss_mb": summary["peak_rss_mb"]},
        "checkpoint": {"directory": str(ckpt_dir), "file_bytes": file_sizes,
                       "checkpoint_total_bytes": checkpoint_total_bytes,
                       "model_weights_bytes": model_weights_bytes, "optimizer_state_bytes": optimizer_state_bytes},
        "estimated_tokens_per_hour": round(tokens_per_sec * 3600) if tokens_per_sec else None,
        "note": "short bounded benchmark (not a training run): throughput/memory/checkpoint-size characterisation only",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="tiny_mobile")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seq", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-scale", type=float, default=0.05, dest="data_scale")
    p.add_argument("--out", default="runner_benchmark.json")
    p.add_argument("--work-dir", default=None, dest="work_dir")
    args = p.parse_args()
    import tempfile
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="tinymind-runner-bench-"))
    result = run_benchmark(profile=args.profile, steps=args.steps, batch=args.batch, seq=args.seq, seed=args.seed,
                           data_scale=args.data_scale, work_dir=work_dir)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
