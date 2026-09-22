"""CI sanity check for the training pipeline (about 10-20 s on one core).

Runs the *production* CLI path end to end on a tiny stage-0 curriculum and
checks the properties CI must never lose:

  1. loss falls (validation loss at the end < at the start);
  2. a run interrupted after 15 of 30 steps and resumed in a fresh engine ends
     with weights identical to an uninterrupted 30-step run (exact resume);
  3. the run ends with a verified inference package that loads without any
     training code and answers a prompt through the same template it trained on.

This is an engineering test. It says nothing about model quality; use
examples/train_tiny_model.py for a real (still synthetic-data) training run.

    python examples/smoke_train.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinymind.cli import main  # noqa: E402
from tinymind.export import load_package, verify_package  # noqa: E402


def run(argv: list[str]) -> None:
    code = main(argv)
    if code != 0:
        raise SystemExit(f"FAILED: tinymind {' '.join(argv[:2])} exited with {code}")


def main_smoke() -> None:
    work = Path(tempfile.mkdtemp(prefix="tinymind-smoke-"))
    data = work / "data"
    run(["data", "build-curriculum", "--stage", "stage0", "--out", str(data), "--scale", "0.05"])
    common = ["train", "--config", "tiny_debug", "--dataset", str(data), "--max-steps", "30", "--warmup-steps", "3",
              "--eval-interval", "10", "--checkpoint-interval", "10", "--log-interval", "0", "--seed", "7", "--stage", "stage0"]

    whole = work / "whole"
    run([*common, "--output", str(whole)])
    s_whole = json.loads((whole / "training_summary.json").read_text())
    v0, v1 = s_whole["initial_validation"]["val_loss"], s_whole["final_validation"]["val_loss"]
    if not v1 < 0.7 * v0:
        raise SystemExit(f"FAILED: validation loss {v0:.3f} -> {v1:.3f} did not fall by 30%")

    split = work / "split"
    run([*common, "--output", str(split), "--stop-after-steps", "15"])
    if json.loads((split / "training_summary.json").read_text())["final_step"] != 15:
        raise SystemExit("FAILED: the interrupted run did not stop at step 15")
    run([*common, "--output", str(split), "--resume", str(split / "checkpoints")])

    a, b = load_package(whole / "export"), load_package(split / "export")
    for (name, x), (_, y) in zip(a.model.named_parameters(), b.model.named_parameters()):
        if not np.array_equal(x.data, y.data):
            raise SystemExit(f"FAILED: resumed weights differ from the uninterrupted run at {name}")
    if not verify_package(split / "export").ok:
        raise SystemExit("FAILED: exported package does not verify")
    reply = b.generate("Repeat exactly: abc", max_new_tokens=8)
    print(f"PASS  val loss {v0:.3f} -> {v1:.3f}; 15+15 resume == 30 continuous (bitwise); package verified; "
          f"sample completion {reply!r} (stage-0 sanity model, not a quality claim)")


if __name__ == "__main__":
    main_smoke()
