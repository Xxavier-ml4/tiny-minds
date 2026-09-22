"""The benchmark matrix of brief section 38, measured for each profile.

    python benchmarks/mobile_matrix.py --out docs/benchmarks/mobile-matrix.json [--profiles tiny_debug tiny_mobile ...]

Per profile (random-initialised weights: latency and memory do not depend on weight values):

MODEL      parameters, hidden, layers, heads, KV heads, vocabulary, sequence length
TRAINING   batch, accumulation, effective batch, tokens/s, steps/s, seconds per 100 steps (real steps, this machine)
INFERENCE  prefill latency and per-token decode latency + tokens/s — the Python reference runtime, and the native
           C++ runtime (native/tests/bench_native.cpp, plain g++ -O2, when g++ is available)
STORAGE    float32 ``.tm`` and int8 ``.tm`` bytes, from real exports of the real files
MEMORY     parameter bytes, KV-cache bytes (exact, seq 128 and 256), estimated runtime RAM (labelled estimate),
           peak resident set of the Python inference process and of the native process (measured)

Everything is measured on the machine this runs on; nothing is a phone number. Multi-thread behaviour is whatever
the BLAS/threads environment of the invoking shell gives.
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

from tinymind.export import export_package, load_package  # noqa: E402
from tinymind.export.budget import budget  # noqa: E402
from tinymind.export.int8 import INT8_FILE, add_int8  # noqa: E402
from tinymind.model import ModelConfig, TinyMindTransformer  # noqa: E402
from tinymind.model.config import count_parameters  # noqa: E402
from tinymind.model.generation import ModelGenerationConfig, generate_with_cache_ids  # noqa: E402
from tinymind.model.tokenizer import ByteTokenizer  # noqa: E402
from train_benchmark import machine_info, measure  # noqa: E402


def python_inference(model: TinyMindTransformer, prompt_len: int, decode_tokens: int, repeats: int = 3) -> dict:
    ids = np.array([[4 + (i * 7 + 3) % 200 for i in range(prompt_len)]])
    cfg1 = ModelGenerationConfig(max_new_tokens=1, do_sample=False, eos_token_id=None)
    cfgn = ModelGenerationConfig(max_new_tokens=1 + decode_tokens, do_sample=False, eos_token_id=None)
    generate_with_cache_ids(model, ids, cfg1)  # warm-up
    t1, tn = [], []
    for _ in range(repeats):
        a = time.perf_counter(); generate_with_cache_ids(model, ids, cfg1); t1.append(time.perf_counter() - a)
        b = time.perf_counter(); generate_with_cache_ids(model, ids, cfgn); tn.append(time.perf_counter() - b)
    prefill = float(np.median(t1))
    per_token = (float(np.median(tn)) - prefill) / decode_tokens
    return {"prompt_len": prompt_len, "prefill_ms_incl_first_token": round(prefill * 1e3, 2), "prefill_tokens_per_sec": round(prompt_len / prefill, 1),
            "decode_ms_per_token": round(per_token * 1e3, 3), "decode_tokens_per_sec": round(1 / per_token, 1)}


def build_native(work: Path) -> Path | None:
    if shutil.which("g++") is None:
        return None
    native = REPO / "native"
    binary = work / "bench_native"
    cmd = ["g++", "-std=c++17", "-O2", "-I", str(native / "include"), "-I", str(native / "src"), str(native / "tests" / "bench_native.cpp"),
           str(native / "src" / "model.cpp"), str(native / "src" / "tensor.cpp"), "-o", str(binary)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("native build failed:", r.stderr[-800:], file=sys.stderr)
        return None
    return binary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", nargs="+", default=["tiny_debug", "tiny_mobile", "tiny_mobile_plus"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--train-steps", type=int, default=5)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--out")
    args = p.parse_args()
    work = Path(tempfile.mkdtemp(prefix="mobile-matrix-"))
    native = build_native(work)
    rows = []
    for name in args.profiles:
        cfg = ModelConfig.from_yaml(REPO / "configs" / f"{name}.yaml")
        train = measure(cfg, args.batch, cfg.max_seq_len, steps=args.train_steps)
        model = TinyMindTransformer(cfg, seed=0)
        pkg_dir = work / name
        export_package(model, ByteTokenizer(), pkg_dir, provenance={"note": "random-init benchmark model"})
        sizes = {"fp32_tm_bytes": (pkg_dir / "model.tm").stat().st_size}
        for label, emb in (("int8_embed_fp32_tm_bytes", False), ("int8_embed_int8_tm_bytes", True)):
            sizes[label] = add_int8(pkg_dir, quantize_embeddings=emb)["size"]
        pkg = load_package(pkg_dir)
        py = python_inference(pkg.model, args.prompt_len, args.decode_tokens)
        row = {"profile": name,
               "model": {"parameters": count_parameters(cfg), "hidden": cfg.hidden_size, "layers": cfg.num_layers, "heads": cfg.num_heads,
                         "kv_heads": cfg.num_kv_heads, "intermediate": cfg.intermediate_size, "vocab": cfg.vocab_size, "seq_len": cfg.max_seq_len},
               "training": train["training"],
               "inference": {"python_reference": py},
               "storage": {**sizes, "fp32_MiB": round(sizes["fp32_tm_bytes"] / 2**20, 3),
                           "int8_embed_fp32_MiB": round(sizes["int8_embed_fp32_tm_bytes"] / 2**20, 3)},
               "memory": {"budget": budget(cfg), "python_training_peak_rss_MB": train["memory"]["peak_rss_MB"]}}
        if native:
            r = subprocess.run([str(native), str(pkg_dir / "model.tm"), str(args.prompt_len), str(args.decode_tokens), "5"], capture_output=True, text=True)
            row["inference"]["native_cpp"] = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-300:]}
        else:
            row["inference"]["native_cpp"] = {"error": "no g++ available"}
        rows.append(row)
        print(f"[{name}] python decode {py['decode_tokens_per_sec']} tok/s; native {row['inference']['native_cpp'].get('decode_tokens_per_sec')}", file=sys.stderr, flush=True)
    result = {"machine": machine_info(), "prompt_len": args.prompt_len, "decode_tokens": args.decode_tokens, "rows": rows,
              "note": "latency measured on the machine above (not a phone); native = plain g++ -O2, no -march=native"}
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
