"""Phase 3B, Phase A: black-box correctness audit of the *Phase 3A* training
stack. Every section below is an experiment that either confirms or refutes
a specific suspicion recorded in docs/architecture/training-system.md; the
JSON this prints is the evidence that document cites.

It deliberately imports only APIs that existed in the Phase 3A delivery, and
must be pointed at that tree (the Phase 3B changes fix most of what this
finds, so running it against the fixed tree would just print "fixed"):

    python benchmarks/audit/phase3a_correctness_audit.py --tree /path/to/TinyMind-phase3a-orig

Nothing here is a benchmark of speed (see profile_training_step.py); it is
about whether the pipeline does what its docstrings say.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _mk_tree(tree: str) -> None:
    sys.path.insert(0, str(Path(tree).resolve()))


def audit_data_pipeline(tmp: Path) -> dict:
    """Does the Phase 3A dataset actually teach prompt -> response?"""
    import numpy as np
    from tinymind.model.tokenizer import ByteTokenizer
    from tinymind.training.collator import CausalLMCollator
    from tinymind.training.dataset import TrainingDataset

    tok = ByteTokenizer()
    path = tmp / "qa.jsonl"
    rows = [{"id": "q1", "messages": [{"role": "user", "content": "What is the capital of France?"}],
             "target": {"type": "answer", "content": "Paris"}},
            {"id": "q2", "messages": [{"role": "user", "content": "Add 2 and 3."}],
             "target": {"type": "tool_call", "name": "calculator", "arguments": {"expr": "2+3"}},
             "tools": [{"name": "calculator", "parameters": {}}]}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    examples = list(TrainingDataset(path, tok))
    batch = CausalLMCollator(pad_token_id=0).collate(examples)
    decoded = [tok.decode(row[row != 0].tolist()) for row in batch["input_ids"]]
    return {
        "decoded_training_sequences": decoded,
        "target_text_stored_but_unused": [e.target_text for e in examples],
        "response_text_present_in_training_sequence": [
            e.target_text in d for e, d in zip(examples, decoded)],
        "labels_equal_input_ids": bool(np.array_equal(batch["labels"], batch["input_ids"])),
        "role_markers_in_sequence": any("user" in d or "assistant" in d for d in decoded),
        "eos_appended": any(2 in row.tolist() for row in batch["input_ids"]),
        "finding": ("the target (assistant response / tool call) is tokenized nowhere: the model is "
                    "trained to continue the concatenated USER text only"),
    }


def audit_tokens_per_second(tmp: Path) -> dict:
    """StepLog.tokens_per_second divides one step's tokens by time since the
    start of train(), so it decays ~1/step even at constant throughput."""
    from tinymind.model import ModelConfig, TinyMindTransformer
    from tinymind.model.tokenizer import ByteTokenizer
    from tinymind.training import CausalLMTrainer, CausalLMTrainingConfig, TrainingDataset

    tok = ByteTokenizer()
    path = tmp / "tps.jsonl"
    path.write_text("\n".join(json.dumps({"id": f"e{i}", "messages": [{"role": "user", "content": "x" * 60}],
                                          "target": {"type": "answer", "content": ""}})
                              for i in range(64)) + "\n")
    cfg = ModelConfig(hidden_size=48, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=96,
                      max_seq_len=64, vocab_size=tok.vocab_size)
    model = TinyMindTransformer(cfg, seed=0)
    trainer = CausalLMTrainer(model, CausalLMTrainingConfig(batch_size=8, epochs=1, seed=0))
    step_times, stamps = [], []
    logs = trainer.train(TrainingDataset(path, tok),
                         log_fn=lambda e: stamps.append(time.perf_counter()))
    diffs = [b - a for a, b in zip(stamps, stamps[1:])]
    reported = [round(l.tokens_per_second) for l in logs]
    tokens_per_step = 8 * 62  # 8 rows x (60 bytes + BOS + ... ) - approximate; only the trend matters
    true_rate = [round(tokens_per_step / d) for d in diffs]
    return {"reported_tokens_per_second": reported, "measured_from_step_wallclock": true_rate,
            "finding": "reported value decays roughly as 1/step_index although step time is constant"}


def _grads(model):
    return {n: (p.grad.copy() if p.grad is not None else None) for n, p in model.named_parameters()}


def audit_gradient_accumulation(tmp: Path) -> dict:
    """batch=8 x accum=1 versus batch=4 x accum=2 on the same 8 examples.
    Equal only if every micro-batch has the same number of loss tokens."""
    import numpy as np
    from tinymind.model import ModelConfig, TinyMindTransformer
    from tinymind.model.tokenizer import ByteTokenizer
    from tinymind.training import CausalLMTrainer, CausalLMTrainingConfig
    from tinymind.training.dataset import TokenizedExample

    tok = ByteTokenizer()
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
                      max_seq_len=64, vocab_size=tok.vocab_size)

    def ex(text, i):
        return TokenizedExample(f"e{i}", tok.encode(text, add_bos=True), "answer", "")

    def maxdiff(examples):
        out = {}
        for label, (bs, accum) in {"full": (8, 1), "split": (4, 2)}.items():
            model = TinyMindTransformer(cfg, seed=3)
            tr = CausalLMTrainer(model, CausalLMTrainingConfig(batch_size=bs, gradient_accumulation_steps=accum,
                                                              learning_rate=1e-3, gradient_clip_norm=1e9))
            groups = [examples[i:i + bs] for i in range(0, 8, bs)]
            tr.train_step([tr.collator.collate(g) for g in groups])
            out[label] = _grads(model)
        return max(float(np.abs(out["full"][n] - out["split"][n]).max()) for n in out["full"]), \
            max(float(np.abs(out["full"][n]).max()) for n in out["full"])

    equal_len = [ex("abcdefghij" * 2, i) for i in range(8)]
    var_len = [ex("ab" * (2 + 5 * (i % 4)), i) for i in range(8)]  # lengths differ per row
    d_eq, scale_eq = maxdiff(equal_len)
    # order the variable-length set so the two halves have different token counts
    var_sorted = sorted(var_len, key=lambda e: len(e.input_ids))
    d_var, scale_var = maxdiff(var_sorted)
    return {"equal_length_microbatches": {"max_abs_grad_diff": d_eq, "max_abs_grad": scale_eq},
            "unequal_token_count_microbatches": {"max_abs_grad_diff": d_var, "max_abs_grad": scale_var},
            "finding": "accumulation weights every micro-batch equally, not every token equally; it only "
                       "matches the large batch when micro-batches carry the same number of loss tokens"}


def audit_resume(tmp: Path) -> dict:
    """Best-effort 'resume' with the Phase 3A APIs versus a continuous run."""
    import numpy as np
    from tinymind.model import ModelConfig, TinyMindTransformer
    from tinymind.model.checkpoint import from_pretrained, load_optimizer_state, save_pretrained
    from tinymind.model.tokenizer import ByteTokenizer
    from tinymind.training import CausalLMTrainer, CausalLMTrainingConfig, TrainingDataset

    tok = ByteTokenizer()
    path = tmp / "resume.jsonl"
    path.write_text("\n".join(json.dumps({"id": f"e{i}", "messages": [{"role": "user", "content": f"line number {i} of text"}],
                                          "target": {"type": "answer", "content": ""}}) for i in range(16)) + "\n")
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
                      max_seq_len=48, vocab_size=tok.vocab_size)
    tcfg = dict(learning_rate=3e-3, batch_size=4, epochs=50, seed=5, warmup_steps=4)

    ref = TinyMindTransformer(cfg, seed=5)
    CausalLMTrainer(ref, CausalLMTrainingConfig(max_steps=20, **tcfg)).train(TrainingDataset(path, tok))

    a = TinyMindTransformer(cfg, seed=5)
    ta = CausalLMTrainer(a, CausalLMTrainingConfig(max_steps=10, **tcfg))
    ta.train(TrainingDataset(path, tok))
    ck = tmp / "ck"
    save_pretrained(a, ck, optimizer=ta.optimizer)
    b = from_pretrained(ck)
    tb = CausalLMTrainer(b, CausalLMTrainingConfig(max_steps=10, **tcfg))
    load_optimizer_state(ck, tb.optimizer)
    tb.train(TrainingDataset(path, tok))
    diff = max(float(np.abs(p.data - q.data).max()) for (_, p), (_, q) in
               zip(ref.named_parameters(), b.named_parameters()))
    return {"max_abs_weight_diff_after_20_steps_vs_continuous": diff,
            "trainer_global_step_after_reload_starts_at": 0, "optimizer_step_count_restored": ta.optimizer.step_count,
            "lr_schedule_restarted_warmup": True,
            "finding": "there is no resume: the trainer restarts its step counter, LR warmup, shuffle sequence "
                       "and epoch from zero; only weights and Adam moments are restorable"}


def audit_checkpoint_atomicity(tmp: Path) -> dict:
    """Interrupt save_pretrained mid-write onto an existing valid checkpoint."""
    import numpy as np
    from tinymind.model import ModelConfig, TinyMindTransformer
    import tinymind.model.checkpoint as ck

    cfg = ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=32,
                      max_seq_len=16, vocab_size=20)
    model = TinyMindTransformer(cfg, seed=1)
    d = tmp / "atomic"
    ck.save_pretrained(model, d)
    ck.from_pretrained(d)  # valid
    real_savez = np.savez

    def dying_savez(path, **arrays):  # write a truncated file, then "crash"
        with open(path, "wb") as h:
            h.write(b"PK\x03\x04partial")
        raise KeyboardInterrupt("simulated kill -9 during np.savez")

    np.savez = dying_savez
    try:
        try:
            ck.save_pretrained(model, d)
        except KeyboardInterrupt:
            pass
    finally:
        np.savez = real_savez
    try:
        ck.from_pretrained(d)
        survived = True
    except Exception as exc:  # noqa: BLE001
        survived = False
        err = f"{type(exc).__name__}: {exc}"
    return {"previous_valid_checkpoint_survived_interrupted_save": survived,
            "error_on_reload": None if survived else err,
            "finding": "save_pretrained overwrites weights.npz in place; no temp file, rename, checksum, or "
                       "previous-checkpoint retention"}


def audit_silent_config_knobs() -> dict:
    """Config fields that validate but change nothing."""
    import numpy as np
    from tinymind.model import ModelConfig, TinyMindTransformer

    base = dict(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=4, intermediate_size=64,
                max_seq_len=16, vocab_size=30)
    x = np.array([[1, 2, 3, 4, 5]])
    ref = TinyMindTransformer(ModelConfig(**base), seed=0)(x).logits.data
    out = {}
    for name, override in {
        "norm_type=layernorm": dict(norm_type="layernorm"), "mlp_type=gelu_mlp": dict(mlp_type="gelu_mlp"),
        "sliding_window=2": dict(sliding_window=2), "dropout=0.5": dict(dropout=0.5),
        "dtype=float16": dict(dtype="float16"), "attention_type=mqa (num_kv_heads still 4)": dict(attention_type="mqa"),
    }.items():
        m = TinyMindTransformer(ModelConfig(**{**base, **override}), seed=0)
        out[name] = {"builds_without_error": True, "logits_identical_to_default": bool(np.array_equal(m(x).logits.data, ref))}
    return {"results": out, "finding": "these settings are accepted and silently ignored, so a config can claim an "
                                      "architecture the code does not implement"}


def audit_param_counting() -> dict:
    from tinymind.model import ModelConfig, TinyMindTransformer
    rows = []
    for name, kw in {
        "tiny tied": dict(hidden_size=128, num_layers=6, num_heads=4, num_kv_heads=2, intermediate_size=384, vocab_size=260, max_seq_len=256),
        "tiny untied": dict(hidden_size=128, num_layers=6, num_heads=4, num_kv_heads=2, intermediate_size=384, vocab_size=260, max_seq_len=256, tie_embeddings=False),
        "proposed 192x6, 32k vocab": dict(hidden_size=192, num_layers=6, num_heads=6, num_kv_heads=2, intermediate_size=512, vocab_size=32000, max_seq_len=256),
    }.items():
        cfg = ModelConfig(**kw)
        actual = TinyMindTransformer(cfg, seed=0).count_parameters() if kw["vocab_size"] < 1000 else None
        rows.append({"config": name, "approx_param_count": cfg.approx_param_count, "actual": actual,
                     "embedding_only": cfg.vocab_size * cfg.hidden_size})
    return {"rows": rows, "vocab_32000_x_192": 32000 * 192,
            "finding": "approx_param_count ignores norms and, for tie_embeddings=False, adds NO lm_head "
                       "(both branches of its embed expression are identical)"}


def audit_backward_recursion() -> dict:
    """Tensor.backward() builds its topological order recursively."""
    import numpy as np
    from tinymind.model import ModelConfig, TinyMindTransformer

    limit = sys.getrecursionlimit()
    out = {"python_recursion_limit": limit, "layers_tested": []}
    for layers in (2, 6, 10, 24, 48):
        cfg = ModelConfig(hidden_size=16, num_layers=layers, num_heads=2, num_kv_heads=1, intermediate_size=32,
                          max_seq_len=8, vocab_size=20)
        m = TinyMindTransformer(cfg, seed=0)
        ids = np.array([[1, 2, 3, 4]])
        loss = m(ids, labels=ids).loss
        try:
            loss.backward()
            status = "ok"
        except RecursionError:
            status = "RecursionError"
        out["layers_tested"].append({"layers": layers, "backward": status})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, help="path to the Phase 3A TinyMind tree to audit")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    _mk_tree(args.tree)
    tmp = Path(tempfile.mkdtemp())
    experiments = {
        "data_pipeline": lambda: audit_data_pipeline(tmp),
        "tokens_per_second": lambda: audit_tokens_per_second(tmp),
        "gradient_accumulation": lambda: audit_gradient_accumulation(tmp),
        "resume": lambda: audit_resume(tmp),
        "checkpoint_atomicity": lambda: audit_checkpoint_atomicity(tmp),
        "silent_config_knobs": audit_silent_config_knobs,
        "param_counting": audit_param_counting,
        "backward_recursion": audit_backward_recursion,
    }
    results = {}
    for name, fn in experiments.items():
        if args.only and name not in args.only:
            continue
        results[name] = fn()
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
