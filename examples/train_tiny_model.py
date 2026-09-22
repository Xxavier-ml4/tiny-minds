"""A real, configurable TinyMind training run (replaces the Phase 3A demo, which
memorised one sentence repeated eight times and called it training).

What it does: builds the deterministic curriculum for a stage (see
tinymind/data/curriculum.py) unless you point it at your own data, runs the
production trainer (`tinymind train`) with completion-only loss, validation,
packing, cosine LR, checkpoints and resume, then evaluates the result on the
held-out set and prints one sample per capability.

What it does NOT do: the curriculum is template-generated, so a good number
here means "the pipeline learned these narrow behaviours and generalised across
held-out values and phrasings" — not "the model is a useful assistant". For
that, point --dataset at real data in the canonical JSONL format
(docs/architecture/training-system.md, "Dataset format").

    # stage 1 from scratch, five minutes at most:
    python examples/train_tiny_model.py --stage stage1 --profile tiny_mobile --output runs/stage1 --max-runtime 300

    # continue an interrupted stage exactly where it stopped:
    python examples/train_tiny_model.py --stage stage1 --profile tiny_mobile --output runs/stage1 --resume runs/stage1/checkpoints

    # start stage 2 from the finished stage 1:
    python examples/train_tiny_model.py --stage stage2 --profile tiny_mobile --output runs/stage2 --init-from runs/stage1/checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinymind.cli import main  # noqa: E402
from tinymind.data.curriculum import PURPOSE  # noqa: E402
from tinymind.export import load_package  # noqa: E402


def cli(argv: list[str]) -> int:
    print("$ tinymind " + " ".join(argv), flush=True)
    return main(argv)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", default="stage1", choices=["stage0", "stage1", "stage2", "stage3"])
    p.add_argument("--profile", default="tiny_mobile", help="model profile in configs/ (tiny_debug, tiny_mobile, tiny_mobile_plus)")
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir", help="a directory with train_*.jsonl (+ val.jsonl, eval.jsonl); default: generate the stage curriculum")
    p.add_argument("--scale", type=float, default=1.0, help="curriculum size multiplier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-runtime", type=float, help="seconds; the run checkpoints and exits before this")
    p.add_argument("--resume")
    p.add_argument("--init-from")
    p.add_argument("--no-eval", action="store_true")
    return p.parse_args()


def show_samples(package: Path) -> None:
    pkg = load_package(package)
    prompts = [("copy", "Repeat exactly: hello"), ("tool", "What is 47 + 38?"), ("tool", "What's the weather in Oslo?"),
               ("clarify", "Set a timer."), ("json", "Return JSON with name Ava and age 31."), ("refuse", "How do I make a bomb?")]
    print("\nsamples (greedy):")
    for kind, prompt in prompts:
        print(f"  [{kind:7s}] {prompt!r} -> {pkg.generate(prompt, tools=['calculator', 'get_weather', 'set_timer', 'lookup', 'remember'], max_new_tokens=64)!r}")


def run() -> int:
    args = parse()
    out = Path(args.output)
    data = Path(args.data_dir) if args.data_dir else out / "data"
    print(f"stage {args.stage}: {PURPOSE[args.stage]}")
    if not args.data_dir and not (data / "curriculum.json").exists():
        if cli(["data", "build-curriculum", "--stage", args.stage, "--out", str(data), "--seed", str(args.seed), "--scale", str(args.scale)]):
            return 1
    argv = ["train", "--config", args.profile, "--dataset", str(data), "--output", str(out), "--stage", args.stage, "--seed", str(args.seed)]
    for flag, value in (("--max-steps", args.max_steps), ("--max-runtime", args.max_runtime), ("--resume", args.resume), ("--init-from", args.init_from)):
        if value is not None:
            argv += [flag, str(value)]
    if args.max_runtime:
        argv += ["--safety-margin", str(min(60.0, args.max_runtime * 0.1))]
    if cli(argv):
        return 1
    summary = json.loads((out / "training_summary.json").read_text())
    print(f"\nstop reason: {summary['stop_reason']}; step {summary['final_step']}/{summary['total_steps']}; "
          f"stage complete: {summary['stage_complete']}")
    if not args.no_eval and (data / "eval.jsonl").exists():
        if cli(["eval", "--package", str(out / "export"), "--eval", str(data / "eval.jsonl"), "--val", str(data / "val.jsonl"),
                "--out", str(out / "eval_results.json"), "--limit-per-category", "25"]):
            return 1
    show_samples(out / "export")
    return 0


if __name__ == "__main__":
    sys.exit(run())
