"""Run a chain of training stages the way the GitHub workflow does, on this machine.

Each *job* is a fresh subprocess with its own directory that sees only what a downloaded artifact contains:

    job k of stage S:  copy the incoming bundle -> check_incoming (integrity, identity, mode, gate)
                       -> `tinymind train ... --max-runtime <chunk>` (--resume / --init-from)
                       -> bundle_run -> [stage complete: eval -> re-bundle with eval_results.json]

so a stage that does not fit in one job is continued by the next one, and a completed stage is promoted into the
next through the same verification the workflow performs. State lives on disk (``<out>/<stage>/job*/bundle``);
re-running the command continues where it stopped, at chunk granularity.

    python benchmarks/run_stage_chain.py --profile tiny_mobile --out runs/chain \
        --stage stage0:600:3e-3:3.0 --stage stage1:700:2e-3 --stage stage2:900:1e-3 --stage stage3:350:3e-4 --scale 0.35

``--stage NAME:STEPS:PEAK_LR[:SCALE]`` (``SCALE`` overrides ``--scale`` for that stage). The same ``--seed/--scale/--batch-size`` must be used for every job of a stage
(the trainer refuses a resume otherwise). This is a *harness*, not a benchmark: chunk lengths are wall-clock.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tinymind.ci.stage_io import BundleError, bundle_run, check_incoming  # noqa: E402


def cli(*argv: str, timeout: float = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "tinymind.cli", *argv], cwd=REPO, capture_output=True, text=True, timeout=timeout)


def latest_bundle(stage_dir: Path) -> Path | None:
    jobs = sorted(stage_dir.glob("job*/bundle"), key=lambda p: int(p.parent.name[3:]))
    return jobs[-1] if jobs else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="tiny_mobile")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", action="append", required=True, help="NAME:STEPS:PEAK_LR (repeatable, in order)")
    p.add_argument("--scale", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-seconds", type=float, default=200.0, help="each job's --max-runtime")
    p.add_argument("--safety-margin", type=float, default=15.0)
    p.add_argument("--eval-limit", type=int, default=20, help="examples per category in the stage-end eval")
    p.add_argument("--gate-dir", default=str(REPO / "configs" / "stages"))
    p.add_argument("--skip-gate", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    profile_yaml = REPO / "configs" / f"{args.profile}.yaml"
    log = []

    previous_final: Path | None = None
    for spec in args.stage:
        parts = spec.split(":")
        stage, steps, lr = parts[0], int(parts[1]), float(parts[2])
        scale = float(parts[3]) if len(parts) > 3 else args.scale
        stage_dir = out / stage
        data = out / "data" / stage
        if not (data / "curriculum.json").exists():
            r = cli("data", "build-curriculum", "--stage", stage, "--out", str(data), "--seed", str(args.seed), "--scale", str(scale))
            if r.returncode:
                print(r.stderr)
                return 1
        while True:
            last = latest_bundle(stage_dir)
            manifest = json.loads((last / "bundle_manifest.json").read_text()) if last else None
            if manifest and manifest["stage_complete"]:
                previous_final = last
                print(f"[{stage}] already complete at step {manifest['global_step']}", flush=True)
                break
            job_no = (int(last.parent.name[3:]) + 1) if last else 1
            job = stage_dir / f"job{job_no}"
            if job.exists():
                shutil.rmtree(job)  # an interrupted job left partial files; jobs are disposable, bundles are not
            job.mkdir(parents=True)
            incoming_src = last or previous_final
            argv = ["train", "--config", args.profile, "--dataset", str(data), "--output", str(job / "out"), "--stage", stage,
                    "--seed", str(args.seed), "--max-steps", str(steps), "--learning-rate", str(lr), "--min-learning-rate", str(lr / 10),
                    "--warmup-steps", str(max(10, steps // 20)), "--batch-size", str(args.batch_size), "--eval-interval", "100",
                    "--checkpoint-interval", "100", "--log-interval", "25", "--max-runtime", str(args.chunk_seconds),
                    "--safety-margin", str(args.safety_margin)]
            mode = "fresh"
            if incoming_src is not None:
                local = job / "incoming"
                shutil.copytree(incoming_src, local)  # "download"
                try:
                    decision = check_incoming(local, stage=stage, profile_config=profile_yaml, requested_mode="auto",
                                              gate_dir=args.gate_dir, skip_gate=args.skip_gate)
                except BundleError as exc:
                    print(f"[{stage}] REFUSED incoming artifact: {exc}", flush=True)
                    return 2
                mode = decision["mode"]
                argv += ["--resume" if mode == "resume" else "--init-from", decision["checkpoint"]]
                if decision.get("gate"):
                    print(f"[{stage}] promotion gate: {json.dumps(decision['gate'])[:300]}", flush=True)
            t0 = time.time()
            r = cli(*argv, timeout=args.chunk_seconds + 900)
            (job / "train.log").write_text(r.stderr + "\n" + r.stdout)
            if r.returncode:
                print(f"[{stage}] job{job_no} FAILED ({r.returncode}):\n{r.stderr[-1500:]}", flush=True)
                return 1
            summary = json.loads((job / "out" / "training_summary.json").read_text())
            extra = {}
            if summary["stage_complete"]:
                ev = job / "out" / "eval_results.json"
                r2 = cli("eval", "--package", str(job / "out" / "export"), "--eval", str(data / "eval.jsonl"), "--val", str(data / "val.jsonl"),
                         "--out", str(ev), "--limit-per-category", str(args.eval_limit))
                if r2.returncode:
                    print(r2.stderr[-1500:])
                    return 1
                extra["eval_results.json"] = ev
            bundle = bundle_run(job / "out", job / "bundle", profile=args.profile, stage=stage, run_id=f"local-{stage}-{job_no}", extra_files=extra)
            row = {"stage": stage, "job": job_no, "mode": mode, "step": summary["final_step"], "of": summary["total_steps"], "stop": summary["stop_reason"],
                   "seconds": round(time.time() - t0), "val_loss": (summary["final_validation"] or {}).get("val_loss"),
                   "artifact": bundle["artifact_name"], "stage_complete": summary["stage_complete"]}
            log.append(row)
            print(json.dumps(row), flush=True)
            (out / "chain-log.jsonl").open("a").write(json.dumps(row) + "\n")
            if summary["stage_complete"]:
                previous_final = job / "bundle"
                break
    print("chain complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
