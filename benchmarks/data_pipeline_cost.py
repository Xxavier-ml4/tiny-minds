"""What the data pipeline costs per training step, next to the model's step time (brief section 13: "data loading").

    python benchmarks/data_pipeline_cost.py --out docs/benchmarks/data-pipeline-cost.json

Measures, on the stage-2 curriculum: rendering + tokenizing examples (examples/s), building the epoch plan, and
materialising one micro-batch (padded and packed), each as a fraction of a tiny_mobile training step.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinymind.data.curriculum import build_stage  # noqa: E402
from tinymind.data.render import ChatRenderer  # noqa: E402
from tinymind.model.tokenizer import ByteTokenizer  # noqa: E402
from tinymind.training.data import DataPlan, DataSource, TokenizedDataset  # noqa: E402


def timed(fn, repeats=5):
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs), out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--step-ms", type=float, default=None, help="training step time to compare against (default: read bench-profiles.json)")
    args = p.parse_args()
    tok = ByteTokenizer()
    renderer = ChatRenderer(tok)
    data = build_stage("stage2", seed=0, scale=0.35)
    records = [r for rs in data["train"].values() for r in rs]
    t_render, ds = timed(lambda: TokenizedDataset.from_records(records, renderer, 256, name="t"), repeats=3)
    sources = [DataSource("all", ds)]
    result = {"examples": len(records), "tokens": ds.stats()["total_tokens"],
              "render_and_tokenize_seconds": round(t_render, 3), "render_examples_per_sec": round(len(records) / t_render),
              "render_tokens_per_sec": round(ds.stats()["total_tokens"] / t_render)}
    for packing in (False, True):
        plan = DataPlan(sources, seed=0, batch_size=8, max_seq_len=256, packing=packing, pad_id=0)
        t_plan, _ = timed(lambda: (setattr(plan, "_cache", None), plan.epoch_rows(1))[1], repeats=3)
        n = plan.num_micro_batches(1)
        t_batch, batch = timed(lambda: plan.micro_batch(1, n // 2), repeats=20)
        key = "packed" if packing else "padded"
        result[key] = {"epoch_plan_seconds": round(t_plan, 4), "micro_batches_per_epoch": n, "micro_batch_ms": round(t_batch * 1e3, 3),
                       "batch_shape": list(batch.shape), "real_tokens": batch.num_real_tokens}
    step_ms = args.step_ms
    if step_ms is None:
        bench = json.loads((Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "bench-profiles.json").read_text())
        step_ms = next(r["training"]["step_seconds_median"] * 1e3 for r in bench["rows"] if r["profile"] == "tiny_mobile")
    result["tiny_mobile_step_ms_for_comparison"] = round(step_ms, 1)
    for key in ("padded", "packed"):
        result[key]["micro_batch_share_of_step"] = f"{result[key]['micro_batch_ms'] / step_ms:.3%}"
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
