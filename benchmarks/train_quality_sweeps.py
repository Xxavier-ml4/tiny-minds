"""Short, measured training runs that decide defaults (brief sections 4, 5, 12).

    python benchmarks/train_quality_sweeps.py lr        --profile tiny_mobile --steps 250 --out docs/benchmarks/sweep-lr.json
    python benchmarks/train_quality_sweeps.py attention --profile tiny_mobile --steps 250 --lr 3e-3 --out ...
    python benchmarks/train_quality_sweeps.py seqlen    --profile tiny_mobile --steps 250 --lr 3e-3 --out ...

Every run trains from the same seed on the same data (the stage-2 curriculum
at reduced scale — tool routing, structured output, dialogue, refusal,
instruction, language) with the production engine (packing, AdamW, cosine
schedule, gradient clipping), then reports: initial and final training loss,
initial and final validation loss, the gradient-norm trajectory (max/mean, how
many steps were clipped), whether it diverged, and tokens/sec.

What these runs can and cannot say: they are *short* (a few hundred steps), so
they rank settings by early progress. A learning rate that wins here can be too
high for a long run; the three-stage plan therefore uses a cosine decay and the
stage gate, and the write-up says so. Timings are only meaningful when nothing
else is running on the machine.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tinymind.data.curriculum import build_stage  # noqa: E402
from tinymind.data.render import ChatRenderer  # noqa: E402
from tinymind.model import ModelConfig  # noqa: E402
from tinymind.model.config import count_parameters  # noqa: E402
from tinymind.model.tokenizer import ByteTokenizer  # noqa: E402
from tinymind.training.config import TrainingConfig  # noqa: E402
from tinymind.training.data import DataSource, DatasetError, TokenizedDataset  # noqa: E402
from tinymind.training.engine import TrainingDivergedError, TrainingEngine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_benchmark import attention_variants, load_profile, machine_info  # noqa: E402


def make_data(scale: float, max_len: int, seed: int = 0, val_max_len: int | None = None):
    tok = ByteTokenizer()
    renderer = ChatRenderer(tok)
    data = build_stage("stage2", seed=seed, scale=scale)
    sources = []
    for name, recs in data["train"].items():
        try:
            sources.append(DataSource(name, TokenizedDataset.from_records(recs, renderer, max_len, overflow="drop", name=name)))
        except DatasetError:  # every example of this source is longer than the context: the source cannot be trained at all
            print(f"  (source {name!r} has no example that fits {max_len} tokens; excluded)", flush=True)
    val = TokenizedDataset.from_records(data["val"], renderer, val_max_len or max_len, overflow="drop", name="val")
    return tok, sources, val, data["mixture"]


def run_short(cfg: ModelConfig, tok, sources, val, *, steps: int, lr: float, batch: int, seq: int, seed: int = 1,
              min_lr_ratio: float = 0.1, warmup: int | None = None, epoch_examples: int = 0) -> dict:
    mixture = {s.name: w for s, w in zip(sources, [1.0] * len(sources))}
    tcfg = TrainingConfig(stage="sweep", seed=seed, learning_rate=lr, min_learning_rate=lr * min_lr_ratio, max_steps=steps,
                          warmup_steps=warmup if warmup is not None else max(5, steps // 15), batch_size=batch,
                          max_seq_len=seq, packing=True, eval_interval=0, checkpoint_interval=0, log_interval=0,
                          overflow="drop", mixture=mixture, epoch_examples=epoch_examples)
    engine = TrainingEngine(model_config=cfg, tokenizer=tok, config=tcfg, sources=sources, validation=val,
                            output_dir=Path("/tmp/sweep-run"), log=None)
    infos: list[dict] = []
    real_step = engine.train_step

    def traced():
        info = real_step()
        infos.append(info)
        return info
    engine.train_step = traced
    v0 = engine.evaluate()
    t0 = time.perf_counter()
    diverged = None
    try:
        for _ in range(steps):
            engine.train_step()
    except TrainingDivergedError as exc:
        diverged = str(exc)
    wall = time.perf_counter() - t0
    v1 = engine.evaluate() if diverged is None else None
    losses = [i["loss"] for i in infos]
    norms = [i["grad_norm"] for i in infos]
    real = sum(i["real_tokens"] for i in infos)
    return {"lr": lr, "steps_completed": len(infos), "diverged": diverged,
            "initial_train_loss": losses[0] if losses else None,
            "final_train_loss_mean_last_10": float(np.mean(losses[-10:])) if losses else None,
            "min_train_loss": float(min(losses)) if losses else None,
            "initial_val_loss": v0["val_loss"], "final_val_loss": v1["val_loss"] if v1 else None,
            "grad_norm": {"mean": float(np.mean(norms)) if norms else None, "max": float(np.max(norms)) if norms else None,
                          "steps_clipped_at_1.0": int(sum(n > 1.0 for n in norms)), "steps": len(norms)},
            "loss_spikes_over_1.5x_running_min": int(sum(1 for i, l in enumerate(losses) if i > 20 and l > 1.5 * min(losses[:i]))),
            "tokens_per_sec": round(real / wall) if wall > 0 else None, "wall_seconds": round(wall, 1)}


def _load_partial(path: str | None) -> dict:
    if path and Path(path).is_file():
        try:
            return json.loads(Path(path).read_text())
        except ValueError:
            pass
    return {}


def _persist(path: str | None, result: dict) -> None:
    if path:
        tmp = Path(path).with_suffix(".partial")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(path)


def cmd_lr(args) -> dict:
    """Runs are persisted one by one and skipped on restart, so an interrupted sweep loses at most one run."""
    cfg = load_profile(args.profile)
    tok, sources, val, _ = make_data(args.data_scale, args.seq)
    result = _load_partial(args.out) or {}
    result.update({"question": "which peak learning rate", "profile": args.profile, "steps": args.steps, "batch": args.batch,
                   "seq": args.seq, "data": f"stage2 curriculum scale {args.data_scale}"})
    rows = result.setdefault("rows_seed1", [])
    repeats = result.setdefault("top2_repeat_seed2", [])
    done = {r["lr"] for r in rows}
    for lr in args.lrs:
        if lr in done:
            continue
        r = run_short(cfg, tok, sources, val, steps=args.steps, lr=lr, batch=args.batch, seq=args.seq, seed=1)
        rows.append(r)
        _persist(args.out, result)
        print(f"lr={lr:g}: val {r['initial_val_loss']:.3f} -> {r['final_val_loss']} train_last10 {r['final_train_loss_mean_last_10']} "
              f"gnorm max {r['grad_norm']['max']:.2f} diverged={r['diverged']} {r['tokens_per_sec']} tok/s", flush=True)
    finite = sorted((r for r in rows if r["final_val_loss"] is not None), key=lambda r: r["final_val_loss"])
    done_rep = {r["lr"] for r in repeats}
    for r in finite[:2]:  # does the ranking survive another seed?
        if r["lr"] in done_rep:
            continue
        rr = run_short(cfg, tok, sources, val, steps=args.steps, lr=r["lr"], batch=args.batch, seq=args.seq, seed=2)
        repeats.append(rr)
        _persist(args.out, result)
        print(f"  repeat seed 2 lr={r['lr']:g}: final val {rr['final_val_loss']}", flush=True)
    return result


def cmd_attention(args) -> dict:
    """MHA / GQA / MQA at equal parameter budget, same data and seed. Persisted per variant; restart skips finished ones."""
    base = load_profile(args.profile)
    tok, sources, val, _ = make_data(args.data_scale, args.seq)
    result = _load_partial(args.out) or {}
    result.update({"question": "MHA vs GQA vs MQA at equal parameter budget (quality)", "profile": args.profile, "steps": args.steps, "lr": args.lr})
    rows = result.setdefault("rows", [])
    done = {r["variant"] for r in rows}
    for name, cfg in attention_variants(dataclasses.replace(base, max_seq_len=args.seq)).items():
        if name in done:
            continue
        r = run_short(cfg, tok, sources, val, steps=args.steps, lr=args.lr, batch=args.batch, seq=args.seq, seed=1)
        r.update({"variant": name, "kv_heads": cfg.num_kv_heads, "intermediate": cfg.intermediate_size, "parameters": count_parameters(cfg)})
        rows.append(r)
        _persist(args.out, result)
        print(f"{name}: kv={cfg.num_kv_heads} params={r['parameters']:,} final val {r['final_val_loss']:.4f} {r['tokens_per_sec']} tok/s", flush=True)
    return result


def cmd_seqlen(args) -> dict:
    """Training context length at a constant number of tokens per step. Persisted per length."""
    cfg = load_profile(args.profile)
    result = _load_partial(args.out) or {}
    result.update({"question": "training sequence length", "profile": args.profile, "steps": args.steps, "lr": args.lr,
                   "tokens_per_step": args.tokens_per_step})
    rows = result.setdefault("rows", [])
    done = {r["train_seq_len"] for r in rows}
    shortest = min(args.seqs)  # common yardstick: validation examples that fit in the shortest context tried
    n_all = sum(len(s.dataset) for s in make_data(args.data_scale, 256)[1])
    for seq in args.seqs:
        if seq in done:
            continue
        tok, sources, val, _ = make_data(args.data_scale, seq, val_max_len=shortest)
        n_kept = sum(len(s.dataset) for s in sources)
        batch = max(1, args.tokens_per_step // seq)
        c = dataclasses.replace(cfg, max_seq_len=max(cfg.max_seq_len, seq))
        r = run_short(c, tok, sources, val, steps=args.steps, lr=args.lr, batch=batch, seq=seq, seed=1)
        r.update({"train_seq_len": seq, "batch": batch, "train_examples_kept": n_kept, "train_examples_total": n_all,
                  "fraction_of_training_examples_dropped_as_too_long": round(1 - n_kept / n_all, 3),
                  "val_examples_in_common_subset": len(val)})
        rows.append(r)
        _persist(args.out, result)
        print(f"T={seq}: kept {n_kept}/{n_all} examples, val loss (<= {shortest} tok subset) {r['final_val_loss']} {r['tokens_per_sec']} tok/s", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("lr", "attention", "seqlen"):
        q = sub.add_parser(name)
        q.add_argument("--profile", default="tiny_mobile")
        q.add_argument("--steps", type=int, default=250)
        q.add_argument("--batch", type=int, default=8)
        q.add_argument("--seq", type=int, default=256)
        q.add_argument("--data-scale", type=float, default=0.08, dest="data_scale")
        q.add_argument("--out")
        q.add_argument("--lr", type=float, default=3e-3)
        q.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
        q.add_argument("--seqs", type=int, nargs="+", default=[64, 128, 256])
        q.add_argument("--tokens-per-step", type=int, default=2048, dest="tokens_per_step")
    args = p.parse_args()
    result = {"lr": cmd_lr, "attention": cmd_attention, "seqlen": cmd_seqlen}[args.cmd](args)
    result["machine"] = machine_info()
    if args.out:
        _persist(args.out, result)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
