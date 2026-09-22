"""Render the measured benchmark JSON files in docs/benchmarks/ as markdown tables (so no number in the docs is
typed by hand). Missing files are skipped, not invented.

    python benchmarks/make_benchmark_tables.py > docs/benchmarks/tables.generated.md
    python benchmarks/make_benchmark_tables.py --inject docs/benchmarks/cpu-training-baseline.md   # replaces the block
                                                    # between <!-- BEGIN GENERATED TABLES --> and <!-- END GENERATED TABLES -->
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"


def load(name):
    p = D / name
    return json.loads(p.read_text()) if p.is_file() else None


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    out = []
    m = load("bench-profiles.json")
    if m:
        out += ["### Profiles: training throughput (batch 8, each profile's own context)", "", table(
            ["profile", "parameters", "h × L", "heads / kv", "seq", "step (ms)", "tokens/s", "s per 100 steps", "peak RSS (MB)"],
            [[r["profile"], f"{r['model']['parameters']:,}", f"{r['model']['hidden']} × {r['model']['layers']}", f"{r['model']['heads']} / {r['model']['kv_heads']}",
              r["model"]["seq_len"], round(r["training"]["step_seconds_median"] * 1e3), f"{r['training']['tokens_per_sec']:,}",
              r["training"]["seconds_per_100_steps"], r["memory"]["peak_rss_MB"]] for r in m["rows"]]), ""]
    a, b = load("phase3a-profile-T256.json"), load("phase3b-fused-profile-T256.json")
    a2, b2 = load("phase3a-profile-T128.json"), load("phase3b-fused-profile-T128.json")
    if a and b:
        rows = []
        for T, x, y in ((256, a, b), (128, a2, b2)):
            if x and y:
                rows.append([T, "Phase 3A (composed ops)", x["step_ms_median"]["total"], f"{x['tokens_per_sec']:,}", x["graph"]["nodes"], x["graph"]["retained_MB_data_plus_grad"]])
                rows.append([T, "Phase 3B (fused ops)", y["step_ms_median"]["total"], f"{y['tokens_per_sec']:,}", y["graph"]["nodes"], y["graph"]["retained_MB_data_plus_grad"]])
        out += ["### Profiler runs, tiny_mobile shape, batch 8 (`benchmarks/audit/profile_training_step.py`)", "",
                table(["seq", "implementation", "step (ms)", "tokens/s", "graph nodes", "graph memory (MB)"], rows), ""]
    f = load("bench-fused-vs-reference.json")
    if f:
        out += ["### Fused vs reference ops, tiny_mobile, batch 8 (`train_benchmark.py matrix --kind fused`)", "", table(
            ["seq", "ops", "step (ms)", "tokens/s", "forward / backward (ms)", "peak RSS (MB)"],
            [[r["model"]["seq_len"], "fused" if r["training"]["fused_ops"] else "reference", round(r["training"]["step_seconds_median"] * 1e3),
              f"{r['training']['tokens_per_sec']:,}", f"{r['training']['forward_s'] * 1e3:.0f} / {r['training']['backward_s'] * 1e3:.0f}",
              r["memory"]["peak_rss_MB"]] for r in f["rows"]]), ""]
    at = load("bench-attention-variants.json")
    if at:
        out += ["### MHA vs GQA vs MQA at equal parameter budget (speed)", "", table(
            ["variant", "seq", "kv heads", "intermediate", "parameters", "step (ms) [min–max]", "tokens/s", "KV cache fp32 at seq (MiB)"],
            [[r["variant"].upper(), r["model"]["seq_len"], r["model"]["kv_heads"], r["model"]["intermediate"], f"{r['model']['parameters']:,}",
              f"{r['training']['step_seconds_median'] * 1e3:.0f} [{r['training']['step_seconds_min_max'][0] * 1e3:.0f}–{r['training']['step_seconds_min_max'][1] * 1e3:.0f}]",
              f"{r['training']['tokens_per_sec']:,}", r["memory"]["kv_cache_MB_fp32_at_seq"]] for r in at["rows"]]), ""]
    s = load("bench-seqlen.json")
    if s:
        out += ["### Sequence length at a constant 2048 tokens per step (tiny_mobile)", "", table(
            ["seq", "batch", "step (ms)", "tokens/s", "examples/s", "peak RSS (MB)"],
            [[r["model"]["seq_len"], r["training"]["batch"], round(r["training"]["step_seconds_median"] * 1e3), f"{r['training']['tokens_per_sec']:,}",
              r["training"]["examples_per_sec"], r["memory"]["peak_rss_MB"]] for r in s["rows"]]), ""]
    bt = load("bench-batch.json")
    if bt:
        out += ["### Micro-batch size at seq 128 (tiny_mobile)", "", table(
            ["batch", "tokens/step", "step (ms)", "tokens/s", "peak RSS (MB)"],
            [[r["training"]["batch"], r["training"]["batch"] * r["model"]["seq_len"], round(r["training"]["step_seconds_median"] * 1e3),
              f"{r['training']['tokens_per_sec']:,}", r["memory"]["peak_rss_MB"]] for r in bt["rows"]]), ""]
    lr = load("sweep-lr.json")
    if lr:
        rows = []
        for r in lr["rows_seed1"]:
            g = r["grad_norm"]
            rows.append([f"{r['lr']:g}", f"{r['initial_val_loss']:.3f}", "diverged" if r["final_val_loss"] is None else f"{r['final_val_loss']:.4f}",
                         "—" if r["final_train_loss_mean_last_10"] is None else f"{r['final_train_loss_mean_last_10']:.4f}",
                         f"{g['mean']:.2f} / {g['max']:.2f}", f"{g['steps_clipped_at_1.0']}/{g['steps']}", r["loss_spikes_over_1.5x_running_min"],
                         "no" if not r["diverged"] else "YES", f"{r['tokens_per_sec']:,}"])
        out += [f"### Learning-rate sweep ({lr['profile']}, {lr['steps']} steps, batch {lr['batch']}, seq {lr['seq']}, {lr['data']}; seed 1)", "", table(
            ["peak LR", "initial val loss", "final val loss", "train loss (last 10)", "grad norm mean / max", "steps clipped", "loss spikes", "NaN/Inf", "tokens/s"], rows), ""]
        if lr.get("top2_repeat_seed2"):
            out += ["Repeat with seed 2 of the two best:", "", table(["peak LR", "final val loss (seed 2)"],
                    [[f"{r['lr']:g}", f"{r['final_val_loss']:.4f}"] for r in lr["top2_repeat_seed2"]]), ""]
    q = load("sweep-attention.json")
    if q:
        out += [f"### Attention layout, quality at equal budget ({q['steps']} steps, lr {q['lr']:g})", "", table(
            ["variant", "kv heads", "intermediate", "parameters", "final val loss", "tokens/s"],
            [[r["variant"].upper(), r["kv_heads"], r["intermediate"], f"{r['parameters']:,}", f"{r['final_val_loss']:.4f}", f"{r['tokens_per_sec']:,}"] for r in q["rows"]]), ""]
    q = load("sweep-seqlen.json")
    if q:
        out += [f"### Training context length ({q['steps']} steps, lr {q['lr']:g}, ~{q['tokens_per_step']} tokens per step)", "", table(
            ["train seq", "batch", "training examples kept", "dropped as too long", "val loss (common short subset)", "tokens/s"],
            [[r["train_seq_len"], r["batch"], f"{r['train_examples_kept']}/{r['train_examples_total']}", f"{r['fraction_of_training_examples_dropped_as_too_long']:.1%}",
              f"{r['final_val_loss']:.4f}", f"{r['tokens_per_sec']:,}"] for r in q["rows"]]), ""]
    t = load("tokenizer-compare.json")
    if t:
        out += ["### Tokenizers on the curriculum (held-out text)", "", table(
            ["tokenizer", "embedding params", "model params", "tokens/char", "rendered example tokens mean / p95 / max", "train ex. > 256 tok", "attention proxy vs byte"],
            [[r["tokenizer"], f"{r['embedding_parameters']:,}", f"{r['model_parameters']:,}", r["tokens_per_char_of_message_text"],
              f"{r['rendered_example_tokens']['mean']} / {r['rendered_example_tokens']['p95']} / {r['rendered_example_tokens']['max']}",
              r["train_examples_over_256_tokens"], r["attention_cost_proxy_vs_byte(mean len^2)"]] for r in t["rows"]]), ""]
    pk = load("packing-utilization.json")
    if pk:
        out += ["### Padding utilisation: padded vs packed batches (batch 8, one epoch, context 256)", "", table(
            ["curriculum", "examples", "mean tokens/example", "padded utilisation", "packed utilisation"],
            [[k, f"{v['examples']:,}", v["mean_len"], f"{v['padded']['utilization']:.1%}", f"{v['packed']['utilization']:.1%}"] for k, v in pk.items()]), ""]
    dp = load("data-pipeline-cost.json")
    if dp:
        out += ["### Data pipeline cost (stage-2 curriculum, scale 0.35)", "", table(
            ["quantity", "value"],
            [["render + tokenize", f"{dp['render_examples_per_sec']:,} examples/s ({dp['render_tokens_per_sec']:,} tokens/s)"],
             ["epoch plan (packed)", f"{dp['packed']['epoch_plan_seconds']} s per epoch"],
             ["micro-batch, padded / packed", f"{dp['padded']['micro_batch_ms']} ms / {dp['packed']['micro_batch_ms']} ms"],
             ["share of a tiny_mobile step (395 ms)", f"{dp['padded']['micro_batch_share_of_step']} / {dp['packed']['micro_batch_share_of_step']}"]]), ""]
    mm = load("mobile-matrix.json")
    if mm:
        out += ["### Benchmark matrix (random-initialised weights; latency and memory do not depend on weight values)", "", table(
            ["profile", "parameters", "train tok/s", "s/100 steps", "Python prefill 64 (ms)", "Python decode (ms/tok)", "native prefill 64 (ms)", "native decode (ms/tok)",
             "native RSS above bare process (MiB)", "fp32 .tm (MiB)", "int8 .tm, embed fp32 (MiB)", "KV cache fp32 @128 / @256 (MiB)"],
            [[r["profile"], f"{r['model']['parameters']:,}", f"{r['training']['tokens_per_sec']:,}", r["training"]["seconds_per_100_steps"],
              r["inference"]["python_reference"]["prefill_ms_incl_first_token"], r["inference"]["python_reference"]["decode_ms_per_token"],
              r["inference"]["native_cpp"].get("prefill_full_forward_ms_median"), r["inference"]["native_cpp"].get("decode_ms_per_token_median"),
              round((r["inference"]["native_cpp"]["peak_rss_kb"] - r["inference"]["native_cpp"]["baseline_rss_kb"]) / 1024, 1),
              r["storage"]["fp32_MiB"], r["storage"]["int8_embed_fp32_MiB"],
              f"{r['memory']['budget']['kv_cache']['128']['MiB']} / {r['memory']['budget']['kv_cache']['256']['MiB']}"] for r in mm["rows"]]), ""]
    qe = load("quantization-eval.json")
    if qe:
        v = qe["variants"]
        rows = []
        for n, label in (("fp32", "float32"), ("int8_embed_fp32", "int8, embedding float32"), ("int8_embed_int8", "int8, embedding int8")):
            tb = v[n]["tool_behavior"]
            rows.append([label, f"{qe['sizes'][n]['model_file_bytes']:,}", f"{v[n]['loss']['validation']['loss']:.4f}", f"{v[n]['loss']['eval']['loss']:.4f}",
                         f"{tb['correct_tool_rate']:.3f}", f"{tb['argument_accuracy']:.3f}", f"{tb['false_positive_call_rate']:.3f}",
                         v[n]["timing"]["tokens_per_second_python_runtime"], f"{v[n]['python_weights_in_memory_bytes'] / 2**20:.2f}"])
        out += ["### FP32 vs INT8 on the stage-3 model (same held-out data; `benchmarks/quantization_eval.py`)", "", table(
            ["variant", ".tm bytes", "validation loss", "held-out eval loss", "correct tool", "argument acc.", "false-positive calls", "Python tok/s", "weights in RAM (MiB)"], rows), ""]
    rp = load("runner-profile-sandbox.json")
    if rp:
        e = rp["environment"]
        out += ["### Machine these numbers come from", "", f"{e['cpu']}; visible cores {e['allowed_cores']}; RAM {e['ram_mb']} MB; Python {e['python']}; NumPy {e['numpy']}; "
                f"BLAS {e['blas'].get('name')} {e['blas'].get('version')}; thread selection {rp['selection']['chosen_threads']} "
                f"(measured {rp['selection']['median_tokens_per_sec']} tokens/s).", ""]
    text = "\n".join(out) + "\n"
    if len(sys.argv) > 2 and sys.argv[1] == "--inject":
        target = Path(sys.argv[2])
        doc = target.read_text()
        begin, end = "<!-- BEGIN GENERATED TABLES -->", "<!-- END GENERATED TABLES -->"
        head, _, rest = doc.partition(begin)
        _, _, tail = rest.partition(end)
        if not rest:
            raise SystemExit(f"{target} has no {begin} marker")
        target.write_text(head + begin + "\n" + text + end + tail)
        print(f"injected {len(text.splitlines())} lines into {target}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
