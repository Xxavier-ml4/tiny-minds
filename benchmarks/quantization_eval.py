"""FP32 vs INT8 on the same held-out data (brief section 34): size, memory, latency, validation loss and every
capability metric, reported side by side with nothing aggregated.

    python benchmarks/quantization_eval.py --package runs/stage3/export --eval data/eval.jsonl --val data/val.jsonl \
        --out docs/benchmarks/quantization-eval.json [--limit-per-category 40]

Variants: ``fp32`` (the package as trained), ``int8_embed_fp32`` (every Linear int8, the token-embedding / tied
output head kept fp32), ``int8_embed_int8`` (embedding quantized too). The package is copied first; the original is
never modified.

What the numbers mean: the reference runtime dequantizes int8 weights to float32 before computing, so *quality* and
*file size* are real measurements, while *RAM and latency are expected to be unchanged* and are reported as measured
(no speed-up is claimed). ``hypothetical_int8_runtime`` gives what a kernel that keeps weights in int8 would need —
arithmetic from the config, not something this repository implements.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinymind.evaluation.tiny_suite import compare, format_report, run_suite, strip_rows  # noqa: E402
from tinymind.export.budget import budget  # noqa: E402
from tinymind.export.int8 import INT8_FILE, add_int8, load_int8_package  # noqa: E402
from tinymind.export.package import load_package  # noqa: E402
from tinymind.training.data import read_jsonl  # noqa: E402


def evaluate_variant(pkg, eval_records, val_records, limit, max_new_tokens):
    t0 = time.perf_counter()
    res = run_suite(pkg, eval_records, val_records, max_new_tokens=max_new_tokens, limit_per_category=limit)
    res["python_weights_in_memory_bytes"] = int(sum(p.data.nbytes for p in pkg.model.parameters()))
    res["wall_seconds"] = round(time.perf_counter() - t0, 1)
    return strip_rows(res)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--package", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--val")
    p.add_argument("--out")
    p.add_argument("--limit-per-category", type=int, default=40, dest="limit")
    p.add_argument("--max-new-tokens", type=int, default=96, dest="max_new_tokens")
    args = p.parse_args()
    eval_records = read_jsonl(args.eval)
    val_records = read_jsonl(args.val) if args.val else None

    work = Path(tempfile.mkdtemp(prefix="quant-eval-"))
    results: dict = {}
    sizes: dict = {}
    fp32_pkg = load_package(args.package)
    sizes["fp32"] = {"model_file_bytes": (Path(args.package) / "model.tm").stat().st_size}
    results["fp32"] = evaluate_variant(fp32_pkg, eval_records, val_records, args.limit, args.max_new_tokens)
    for name, embed in (("int8_embed_fp32", False), ("int8_embed_int8", True)):
        copy = work / name
        shutil.copytree(args.package, copy)
        entry = add_int8(copy, quantize_embeddings=embed)
        sizes[name] = {"model_file_bytes": entry["size"]}
        results[name] = evaluate_variant(load_int8_package(copy), eval_records, val_records, args.limit, args.max_new_tokens)
        print(f"[{name}] done", file=sys.stderr, flush=True)
    for name in ("int8_embed_fp32", "int8_embed_int8"):
        sizes[name]["ratio_vs_fp32_file"] = round(sizes["fp32"]["model_file_bytes"] / sizes[name]["model_file_bytes"], 3)
    cfg = fp32_pkg.model.config
    out = {
        "package": str(args.package), "eval_examples": len(eval_records), "limit_per_category": args.limit,
        "sizes": sizes,
        "hypothetical_int8_runtime": {"note": "NOT implemented: arithmetic for a runtime that keeps weights in int8",
                                       "embed_fp32": budget(cfg, quantize_embeddings=False), "embed_int8": budget(cfg, quantize_embeddings=True)},
        "variants": results,
        "deltas_vs_fp32": {n: compare(results["fp32"], results[n], ("fp32", n)) for n in ("int8_embed_fp32", "int8_embed_int8")},
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    for name, res in results.items():
        print(f"\n===== {name} =====")
        print(format_report({**res, "_rows": []}))
    print("\nfile sizes:", json.dumps(sizes))


if __name__ == "__main__":
    main()
