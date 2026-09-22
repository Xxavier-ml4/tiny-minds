"""Phase 3B command-line commands (registered into ``tinymind.cli``).

    tinymind train              staged training with resume / init-from / time budget
    tinymind checkpoint-info    what a training checkpoint holds (verifies it first)
    tinymind verify-checkpoint  integrity check + optional identity expectations
    tinymind export             checkpoint -> inference package
    tinymind verify-package     integrity check of an inference package
    tinymind eval               tiny-model capability suite on a package
    tinymind eval-compare       side-by-side of two eval results (FP32 vs INT8, stage vs stage)
    tinymind stage-gate         promotion criteria between stages
    tinymind budget             mobile size budget of a profile
    tinymind data build-curriculum | check-contamination | render

Exit status: 0 success (including a clean stop for the time budget or a signal:
read ``training_summary.json`` ``stage_complete`` to tell), 1 configuration /
data / checkpoint error, 3 training diverged (NaN/Inf; last good checkpoint kept).
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

EXIT_ERROR, EXIT_DIVERGED = 1, 3


def _err(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_ERROR


# ---------------------------------------------------------------------------- shared helpers
def build_tokenizer(spec: dict[str, Any] | None):
    from tinymind.model.tokenizer import tokenizer_from_spec
    if not spec or spec.get("type", "byte") == "byte":
        from tinymind.model.tokenizer import ByteTokenizer
        return ByteTokenizer()
    if spec["type"] == "bpe" and "path" in spec:
        return tokenizer_from_spec(json.loads(Path(spec["path"]).read_text()))
    return tokenizer_from_spec(spec)


def load_profile(config_arg: str):
    """``(ModelConfig, tokenizer spec, training section dict)`` from a profile name, YAML path or legacy preset."""
    import yaml
    from tinymind.model.config import ModelConfig, load_preset

    path = Path(config_arg)
    if not path.is_file():
        candidate = Path(__file__).resolve().parents[1] / "configs" / f"{config_arg}.yaml"
        path = candidate if candidate.is_file() else path
    if path.is_file():
        data = yaml.safe_load(path.read_text()) or {}
        model = ModelConfig.from_dict(data["model"]) if "model" in data else ModelConfig.from_dict(data)
        return model, data.get("tokenizer"), dict(data.get("training", {}))
    return load_preset(config_arg), None, {}


def _parse_mixture(text: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (text or "").split(","):
        if part.strip():
            name, _, weight = part.partition("=")
            out[name.strip()] = float(weight)
    return out


def _collect_training_data(paths: list[str]):
    """``({source name: raw records}, mixture weights from curriculum.json or {}, dir with val/eval or None)``."""
    from tinymind.training.data import read_jsonl

    sources: dict[str, list[dict]] = {}
    weights: dict[str, float] = {}
    data_dir: Path | None = None
    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            data_dir = p
            for f in sorted(p.glob("train_*.jsonl")):
                sources[f.stem[len("train_"):]] = read_jsonl(f)
            if (p / "curriculum.json").is_file():
                weights.update(json.loads((p / "curriculum.json").read_text()).get("mixture", {}))
            if not sources and (p / "train.jsonl").is_file():
                sources["train"] = read_jsonl(p / "train.jsonl")
        elif p.is_file():
            sources[p.stem] = read_jsonl(p)
        else:
            raise FileNotFoundError(f"dataset path {entry!r} does not exist")
    if not sources:
        raise ValueError("no training data found (a directory needs train_*.jsonl files)")
    return sources, weights, data_dir


# ---------------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> int:
    from tinymind.data.contamination import ContaminationError, assert_no_contamination
    from tinymind.data.render import ChatRenderer, ExampleError
    from tinymind.model.config import ModelConfigError
    from tinymind.training.checkpoint import CheckpointError
    from tinymind.training.config import TrainingConfig, TrainingConfigError
    from tinymind.training.data import DataSource, DatasetError, TokenizedDataset, read_jsonl, split_records
    from tinymind.training.engine import TrainingDivergedError, TrainingEngine
    from tinymind.training.exporter import package_exporter

    try:
        model_config, tok_spec, yaml_training = load_profile(args.config)
        tokenizer = build_tokenizer(tok_spec)
        overrides = {k: v for k, v in {
            "stage": args.stage, "epochs": args.epochs, "max_steps": args.max_steps, "seed": args.seed,
            "batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation,
            "learning_rate": args.learning_rate, "min_learning_rate": args.min_learning_rate, "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay, "checkpoint_interval": args.checkpoint_interval, "eval_interval": args.eval_interval,
            "log_interval": args.log_interval, "max_seq_len": args.max_seq_len, "max_runtime_seconds": args.max_runtime,
            "safety_margin_seconds": args.safety_margin, "overflow": args.overflow, "keep_checkpoints": args.keep_checkpoints,
            "epoch_examples": args.epoch_examples}.items() if v is not None}
        if args.packing is not None:
            overrides["packing"] = args.packing == "on"
        sources_raw, curriculum_weights, data_dir = _collect_training_data(args.dataset)
        mixture = _parse_mixture(args.mixture) or (curriculum_weights if set(curriculum_weights) == set(sources_raw) else {})
        if mixture:
            overrides["mixture"] = mixture
        merged = {**yaml_training, **overrides}
        merged.setdefault("max_seq_len", min(256, model_config.max_seq_len))
        if model_config.vocab_size != tokenizer.vocab_size:
            return _err(f"model vocab_size {model_config.vocab_size} does not match the tokenizer's {tokenizer.vocab_size}; "
                        "the config's vocab_size must equal the tokenizer's (byte tokenizer: 260)")
        config = TrainingConfig.from_dict(merged)
        if config.max_seq_len > model_config.max_seq_len:
            return _err(f"training max_seq_len {config.max_seq_len} exceeds the model's max_seq_len {model_config.max_seq_len}")
        renderer = ChatRenderer(tokenizer)

        # validation: explicit file, else <dir>/val.jsonl, else a deterministic split of each source
        val_records: list[dict] | None = None
        if args.validation_dataset:
            val_records = read_jsonl(args.validation_dataset)
        elif data_dir is not None and (data_dir / "val.jsonl").is_file():
            val_records = read_jsonl(data_dir / "val.jsonl")
        elif args.validation_fraction > 0:
            val_records = []
            for name in list(sources_raw):
                sources_raw[name], held = split_records(sources_raw[name], args.validation_fraction, seed=config.seed)
                val_records += held
        if not val_records and not args.allow_no_validation:
            return _err("no validation data: pass --validation-dataset, put val.jsonl next to the training files, "
                        "or use --validation-fraction > 0 (--allow-no-validation is for engineering smoke tests only)")

        eval_path = args.eval_dataset or (str(data_dir / "eval.jsonl") if data_dir and (data_dir / "eval.jsonl").is_file() else None)
        if eval_path:
            # validation data steers checkpoint/LR choices, so eval must be disjoint from it as well as from training
            report = assert_no_contamination([r for rs in sources_raw.values() for r in rs] + list(val_records or []),
                                             read_jsonl(eval_path), allow=args.allow_contamination, what=f"eval set {eval_path}")
            print(f"[data] contamination check vs {eval_path}: {report.summary()['exact_prompt_overlap']} exact / "
                  f"{report.summary()['normalized_prompt_overlap']} normalised prompt overlaps", file=sys.stderr)

        overflow = config.overflow
        sources = [DataSource(name, TokenizedDataset.from_records(recs, renderer, config.max_seq_len, overflow=overflow, name=name))
                   for name, recs in sources_raw.items()]
        validation = TokenizedDataset.from_records(val_records, renderer, config.max_seq_len, overflow=overflow, name="validation") \
            if val_records else None
        for s in sources + ([DataSource("validation", validation)] if validation else []):
            st = s.dataset.stats()
            print(f"[data] {s.name}: {st['num_examples']} examples, {st['total_tokens']} tokens "
                  f"({st['loss_tokens']} with loss), dropped {st['dropped']}", file=sys.stderr)

        engine = TrainingEngine(model_config=model_config, tokenizer=tokenizer, config=config, sources=sources,
                                validation=validation, output_dir=args.output, resume=args.resume, init_from=args.init_from,
                                carry_optimizer=args.carry_optimizer, allow_no_validation=args.allow_no_validation,
                                stop_after_steps=args.stop_after_steps, log=lambda m: print(m, file=sys.stderr, flush=True), exporter=package_exporter)
    except (TrainingConfigError, DatasetError, ExampleError, CheckpointError, ContaminationError, ModelConfigError,
            FileNotFoundError, ValueError, KeyError) as exc:
        return _err(str(exc))

    def on_signal(signum, _frame):  # checkpoint at the next step boundary instead of dying mid-write
        engine.request_stop(f"signal {signum}")
        signal.signal(signum, signal.SIG_DFL)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)

    try:
        summary = engine.train()
    except TrainingDivergedError as exc:
        print(f"error: training diverged: {exc} (last good checkpoint kept under {engine.checkpoint_root})", file=sys.stderr)
        return EXIT_DIVERGED
    except (CheckpointError, DatasetError) as exc:
        return _err(str(exc))
    print(json.dumps({"stage": summary["stage"], "stop_reason": summary["stop_reason"], "stage_complete": summary["stage_complete"],
                      "final_step": summary["final_step"], "total_steps": summary["total_steps"],
                      "final_validation": summary["final_validation"], "checkpoint": summary["checkpoint"],
                      "package": summary["exports"].get("package"),
                      "summary": str(Path(args.output) / "training_summary.json")}, indent=2))
    return 0


# ---------------------------------------------------------------------------- checkpoints
def cmd_checkpoint_info(args: argparse.Namespace) -> int:
    from tinymind.training.checkpoint import CheckpointCorruptError, checkpoint_summary, resolve_checkpoint
    try:
        path, notes = resolve_checkpoint(args.path)
    except CheckpointCorruptError as exc:
        return _err(str(exc))
    info = checkpoint_summary(path)
    info["resolved_from"] = str(args.path)
    info["fallback_notes"] = notes
    print(json.dumps(info, indent=2))
    return 0 if info["valid"] else EXIT_ERROR


def cmd_verify_checkpoint(args: argparse.Namespace) -> int:
    from tinymind.training.checkpoint import CheckpointCorruptError, resolve_checkpoint, sha256_file, verify_checkpoint
    try:
        path, notes = resolve_checkpoint(args.path) if not args.exact else (Path(args.path), [])
    except CheckpointCorruptError as exc:
        return _err(str(exc))
    report = verify_checkpoint(path)
    problems = list(report.errors)
    if report.ok:
        m = report.manifest
        expectations = {"stage": args.expect_stage, "global_step": args.expect_step, "model_config_hash": args.expect_model_hash,
                        "dataset_hash": args.expect_dataset_hash, "training_config_hash": args.expect_config_hash,
                        "tokenizer_hash": args.expect_tokenizer_hash, "stage_complete": None if args.expect_complete is None else args.expect_complete == "true"}
        for key, want in expectations.items():
            if want is not None and m[key] != want:
                problems.append(f"manifest {key} is {m[key]!r}, expected {want!r}")
        if args.expect_manifest_sha256 and sha256_file(path / "manifest.json") != args.expect_manifest_sha256:
            problems.append("manifest.json SHA-256 differs from the expected value")
    if problems:
        print("INVALID " + str(path), file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return EXIT_ERROR
    m = report.manifest
    print(json.dumps({"ok": True, "path": str(path), "stage": m["stage"], "global_step": m["global_step"], "stage_complete": m["stage_complete"],
                      "manifest_sha256": sha256_file(path / "manifest.json"), "model_config_hash": m["model_config_hash"],
                      "dataset_hash": m["dataset_hash"], "training_config_hash": m["training_config_hash"],
                      "tokenizer_hash": m["tokenizer_hash"], "fallback_notes": notes}, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from tinymind.export.package import export_package
    from tinymind.model import TinyMindTransformer
    from tinymind.model.config import ModelConfig
    from tinymind.model.tokenizer import tokenizer_from_spec
    from tinymind.training.checkpoint import CheckpointError, load_checkpoint, resolve_checkpoint, sha256_file
    try:
        path, _ = resolve_checkpoint(args.checkpoint)
        loaded = load_checkpoint(path, verify=False)
        model = TinyMindTransformer(ModelConfig.from_dict(loaded.state["model_config"]), seed=0)
        for name, p in model.named_parameters():
            p.data[...] = loaded.weights[name]
        manifest = export_package(model, tokenizer_from_spec(loaded.state["tokenizer"]), args.output, provenance={
            "stage": loaded.manifest["stage"], "global_step": loaded.manifest["global_step"], "stage_complete": loaded.manifest["stage_complete"],
            "dataset_hash": loaded.manifest["dataset_hash"], "training_config_hash": loaded.manifest["training_config_hash"],
            "checkpoint": path.name, "checkpoint_manifest_sha256": sha256_file(path / "manifest.json"), "git_commit": loaded.manifest.get("git_commit")})
    except (CheckpointError, ValueError, KeyError) as exc:
        return _err(str(exc))
    print(json.dumps({"package": args.output, "parameters": manifest["parameter_count"], "files": manifest["files"]}, indent=2))
    return 0


def cmd_verify_package(args: argparse.Namespace) -> int:
    from tinymind.export.package import verify_package
    report = verify_package(args.path)
    if not report.ok:
        return _err(f"{args.path}: " + "; ".join(report.errors))
    print(json.dumps({"ok": True, "parameters": report.manifest["parameter_count"], "files": report.manifest["files"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------- evaluation
def cmd_eval(args: argparse.Namespace) -> int:
    from tinymind.evaluation.tiny_suite import format_report, run_suite, strip_rows
    from tinymind.export.package import PackageError, load_package
    from tinymind.training.data import read_jsonl
    try:
        pkg = load_package(args.package)
        result = run_suite(pkg, read_jsonl(args.eval), read_jsonl(args.val) if args.val else None, max_new_tokens=args.max_new_tokens,
                           limit_per_category=args.limit_per_category, progress=lambda m: print(m, file=sys.stderr, flush=True))
    except (PackageError, FileNotFoundError, ValueError) as exc:
        return _err(str(exc))
    rows = result["_rows"]
    out = strip_rows(result)
    if args.per_example:
        out["examples"] = [{k: r[k] for k in ("id", "category", "ok", "text", "finish")} for r in rows]
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(format_report(result))
    return 0


def cmd_eval_compare(args: argparse.Namespace) -> int:
    from tinymind.evaluation.tiny_suite import compare
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    rows = compare(a, b, (args.label_a, args.label_b))
    print(json.dumps(rows, indent=2) if args.json else "\n".join(
        f"{r['metric']:60s} {r[args.label_a]:>12.4f} {r[args.label_b]:>12.4f} {r['delta']:>+10.4f}" for r in rows))
    return 0


def cmd_stage_gate(args: argparse.Namespace) -> int:
    from tinymind.training.gate import evaluate_gate
    try:
        criteria = json.loads(Path(args.criteria).read_text())
        verdict = evaluate_gate(criteria, summary=json.loads(Path(args.summary).read_text()) if args.summary else None,
                                eval_results=json.loads(Path(args.eval_results).read_text()) if args.eval_results else None)
    except (OSError, ValueError, KeyError) as exc:
        return _err(str(exc))
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else EXIT_ERROR


def cmd_budget(args: argparse.Namespace) -> int:
    from tinymind.export.budget import budget
    try:
        model_config, _, _ = load_profile(args.config)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    print(json.dumps(budget(model_config, quantize_embeddings=args.quantize_embeddings), indent=2))
    return 0


# ---------------------------------------------------------------------------- data
def cmd_data_build(args: argparse.Namespace) -> int:
    from tinymind.data.curriculum import write_stage
    manifest = write_stage(args.stage, args.out, seed=args.seed, scale=args.scale)
    print(json.dumps({"stage": args.stage, "out": args.out, "mixture": manifest["mixture"],
                      "files": {k: v["examples"] for k, v in manifest["files"].items()},
                      "eval_dropped_for_overlap": manifest["eval_dropped_for_overlap"]}, indent=2))
    return 0


def cmd_data_contamination(args: argparse.Namespace) -> int:
    from tinymind.data.contamination import check_overlap
    from tinymind.training.data import read_jsonl
    train = [r for p in args.train for r in read_jsonl(p)]
    report = check_overlap(train, read_jsonl(args.eval))
    print(json.dumps(report.summary(), indent=2))
    return 0 if (not report.contaminated or args.allow) else EXIT_ERROR


def cmd_data_render(args: argparse.Namespace) -> int:
    from tinymind.data.render import ChatRenderer
    from tinymind.training.data import read_jsonl
    renderer = ChatRenderer(build_tokenizer(None))
    for record in read_jsonl(args.file)[: args.n]:
        runs = renderer.explain(record)
        print(f"# {record.get('id')}  ({sum(len(t) for t, _ in runs)} chars)")
        print("".join(f"[{t}]" if trained else t for t, trained in runs).replace("\n", "\\n\n"))
        print("  ([...] = trained: loss is applied to those tokens only)\n")
    return 0


# ---------------------------------------------------------------------------- registration
def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="staged training: resume, init-from, time budget, validation, packing")
    p.add_argument("--config", required=True, help="profile name (tiny_mobile), YAML path, or legacy preset")
    p.add_argument("--dataset", required=True, nargs="+", help="JSONL file(s) or a curriculum directory (train_*.jsonl)")
    p.add_argument("--validation-dataset")
    p.add_argument("--eval-dataset", help="held-out set to check for contamination against the training data")
    p.add_argument("--output", required=True, help="run directory (checkpoints/, export/, training_summary.json)")
    p.add_argument("--resume", help="checkpoint or checkpoint root: exact continuation of the SAME stage")
    p.add_argument("--init-from", dest="init_from", help="checkpoint whose weights start a NEW stage")
    p.add_argument("--carry-optimizer", action="store_true", dest="carry_optimizer")
    p.add_argument("--stage")
    p.add_argument("--epochs", type=int)
    p.add_argument("--max-steps", type=int, dest="max_steps")
    p.add_argument("--max-runtime", type=float, dest="max_runtime", help="seconds; checkpoint + export + exit before this")
    p.add_argument("--safety-margin", type=float, dest="safety_margin", help="seconds reserved for the final checkpoint/export")
    p.add_argument("--stop-after-steps", type=int, dest="stop_after_steps",
                   help="stop THIS invocation after N optimizer steps (checkpoint + export first); for chunked runs")
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--gradient-accumulation", type=int, dest="gradient_accumulation")
    p.add_argument("--learning-rate", type=float, dest="learning_rate")
    p.add_argument("--min-learning-rate", type=float, dest="min_learning_rate")
    p.add_argument("--warmup-steps", type=int, dest="warmup_steps")
    p.add_argument("--weight-decay", type=float, dest="weight_decay")
    p.add_argument("--checkpoint-interval", type=int, dest="checkpoint_interval")
    p.add_argument("--eval-interval", type=int, dest="eval_interval")
    p.add_argument("--log-interval", type=int, dest="log_interval")
    p.add_argument("--keep-checkpoints", type=int, dest="keep_checkpoints")
    p.add_argument("--max-seq-len", type=int, dest="max_seq_len")
    p.add_argument("--packing", choices=["on", "off"])
    p.add_argument("--overflow", choices=["error", "drop", "truncate"])
    p.add_argument("--mixture", help="source=weight,... (default: curriculum.json weights, else equal)")
    p.add_argument("--epoch-examples", type=int, dest="epoch_examples")
    p.add_argument("--validation-fraction", type=float, default=0.05, dest="validation_fraction")
    p.add_argument("--allow-no-validation", action="store_true", dest="allow_no_validation")
    p.add_argument("--allow-contamination", action="store_true", dest="allow_contamination")
    p.add_argument("--legacy", action="store_true", help="the Phase 3A trainer (trains on prompt text only, no resume) — deprecated")

    p = sub.add_parser("checkpoint-info", help="show what a training checkpoint holds (verifies it first)")
    p.add_argument("path", help="checkpoint directory or a run's checkpoints/ root")
    p.set_defaults(func=cmd_checkpoint_info)

    p = sub.add_parser("verify-checkpoint", help="verify a checkpoint's integrity and (optionally) its identity")
    p.add_argument("path")
    p.add_argument("--exact", action="store_true", help="verify exactly this directory, no fallback if it is a root")
    p.add_argument("--expect-stage")
    p.add_argument("--expect-step", type=int)
    p.add_argument("--expect-model-hash")
    p.add_argument("--expect-dataset-hash")
    p.add_argument("--expect-config-hash")
    p.add_argument("--expect-tokenizer-hash")
    p.add_argument("--expect-manifest-sha256")
    p.add_argument("--expect-complete", choices=["true", "false"])
    p.set_defaults(func=cmd_verify_checkpoint)

    p = sub.add_parser("export", help="write an inference package from a training checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify-package", help="verify an inference package")
    p.add_argument("path")
    p.set_defaults(func=cmd_verify_package)

    p = sub.add_parser("eval", help="run the tiny-model capability suite on an inference package")
    p.add_argument("--package", required=True)
    p.add_argument("--eval", required=True, help="held-out JSONL (eval.jsonl)")
    p.add_argument("--val", help="validation JSONL for loss")
    p.add_argument("--out", help="write machine-readable results here")
    p.add_argument("--max-new-tokens", type=int, default=96, dest="max_new_tokens")
    p.add_argument("--limit-per-category", type=int, dest="limit_per_category")
    p.add_argument("--per-example", action="store_true", dest="per_example")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("eval-compare", help="compare two eval result files metric by metric")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_eval_compare)

    p = sub.add_parser("stage-gate", help="check promotion criteria before the next stage starts")
    p.add_argument("--criteria", required=True)
    p.add_argument("--summary")
    p.add_argument("--eval-results", dest="eval_results")
    p.set_defaults(func=cmd_stage_gate)

    p = sub.add_parser("budget", help="mobile size budget (weights, KV cache, RAM estimate) of a profile")
    p.add_argument("--config", required=True)
    p.add_argument("--quantize-embeddings", action="store_true", dest="quantize_embeddings")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("data", help="curriculum and data tools")
    dsub = p.add_subparsers(dest="data_command", required=True)
    q = dsub.add_parser("build-curriculum", help="write a stage's train/val/eval JSONL files")
    q.add_argument("--stage", required=True, choices=["stage0", "stage1", "stage2", "stage3"])
    q.add_argument("--out", required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--scale", type=float, default=1.0)
    q.set_defaults(func=cmd_data_build)
    q = dsub.add_parser("check-contamination", help="fail if eval prompts occur in the training data")
    q.add_argument("--train", nargs="+", required=True)
    q.add_argument("--eval", required=True)
    q.add_argument("--allow", action="store_true")
    q.set_defaults(func=cmd_data_contamination)
    q = dsub.add_parser("render", help="show the exact training sequence and loss mask of dataset examples")
    q.add_argument("file")
    q.add_argument("-n", type=int, default=3)
    q.set_defaults(func=cmd_data_render)
