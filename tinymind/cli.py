"""``tinymind`` command-line interface, per the engineering brief section 28.

Every subcommand here either does something real against this delivery's
actual code, or — for a command that genuinely still needs something this
delivery doesn't have (``finetune`` needs an adapter/LoRA implementation;
``distill`` needs a configured teacher endpoint; ``convert`` needs an
external source format like safetensors to convert from) — prints a
clear, specific "not implemented in this phase, see STATUS.md" message and
exits non-zero. ``train`` and ``quantize`` were in that bucket through
Phase 1; both are real as of Phase 3A (a trained model and real weights to
quantize now exist) and have real implementations below. Nothing here
silently pretends to succeed; see the brief's own section 45 ("do not
pretend the model is good before testing it") and section 54 ("never hide
errors"), both of which apply just as much to a CLI's own status reporting
as to model quality.
"""
from __future__ import annotations

import argparse
import json
import sys

from tinymind import Model, __version__
from tinymind.model.config import ModelConfig, ModelConfigError, load_preset
from tinymind.runtime.format import ModelFormatError, read_model
from tinymind.tools.builtins import register_builtins
from tinymind.tools.registry import ToolRegistry

_NOT_YET_IMPLEMENTED = {
    "finetune": "needs an adapter/LoRA implementation on top of the real base model (Phase 3A "
               "has the base model; adapter training itself is not implemented)",
    "distill": "needs a configured teacher endpoint (Phase 5)",
    "convert": "needs a source format (e.g. a real .cact or safetensors file) to convert from",
}


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"tinymind {__version__}")
    return 0


def _load_tools_file(path: str) -> ToolRegistry:
    """A --tools file is a JSON array of raw tool schemas (brief section
    28's example: ``tinymind run ... --tools tools.json``). Schema-only
    (no callable) registrations are for *display*/routing purposes — they
    cannot be executed without a Python function bound to them, so
    ``run``/``serve`` report that honestly rather than silently no-op'ing
    on a call to one."""
    registry = ToolRegistry()
    with open(path, "r", encoding="utf-8") as handle:
        schemas = json.load(handle)
    for schema in schemas:
        registry.register(schema, fn=_unbound_tool_stub(schema.get("name", "?")))
    return registry


def _unbound_tool_stub(name: str):
    def _stub(**_kwargs):
        raise RuntimeError(
            f"tool {name!r} was declared via a --tools schema file with no bound Python "
            "function; schema-only tools can be listed and routed to, but not executed "
            "from the CLI — register it in Python via tinymind.tools.ToolRegistry.register() "
            "with fn=<callable> instead")
    return _stub


def _cmd_run(args: argparse.Namespace) -> int:
    registry = ToolRegistry()
    register_builtins(registry)
    if args.tools:
        for entry in _load_tools_file(args.tools).list():
            registry.register(entry.schema.to_dict(), fn=entry.fn, permissions=entry.permissions)

    model = Model(args.model_path, registry=registry)

    if not model._backend.is_real_model:
        print(f"note: {type(model._backend).__name__} is a deterministic stand-in, not a "
             "trained model — see STATUS.md", file=sys.stderr)

    result = model.run(args.prompt)
    if result.text is not None:
        print(result.text)
    elif result.tool_result is not None:
        print(json.dumps(result.tool_result.to_dict(), indent=2))
    else:
        print(f"[{result.mode.value}] {result.reason}", file=sys.stderr)
        return 1
    return 0


def _cmd_tools_list(args: argparse.Namespace) -> int:
    registry = ToolRegistry()
    register_builtins(registry)
    if args.tools:
        for entry in _load_tools_file(args.tools).list():
            registry.register(entry.schema.to_dict(), fn=entry.fn, permissions=entry.permissions)
    for entry in registry.list():
        print(f"{entry.name:20s} [{entry.permissions.describe()}]  {entry.schema.description}")
    return 0


def _cmd_model_info(args: argparse.Namespace) -> int:
    path_arg = args.preset_or_path
    # A real .tm file (Phase 3A): report the model's *exact* parameter
    # count from the actually-instantiated architecture, not the
    # approx_param_count estimate a bare config gives — see
    # tinymind.model.model.TinyMindTransformer.count_parameters().
    if path_arg.endswith(".tm"):
        from tinymind.model.tm_export import TmExportError, import_from_tm
        try:
            model = import_from_tm(path_arg)
        except (ModelFormatError, TmExportError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(model.config.to_dict(), indent=2))
        print(f"\nexact parameter count: {model.count_parameters():,} "
             f"({model.count_parameters() / 1e6:.2f}M)", file=sys.stderr)
        return 0

    try:
        config = load_preset(path_arg)
    except ModelConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(config.to_dict(), indent=2))
    print(f"\nparameter count: {config.count_parameters():,} ({config.count_parameters() / 1e6:.2f}M) — exact "
         f"(tinymind.model.config.count_parameters; equals an instantiated model's count)", file=sys.stderr)
    return 0


def _cmd_tokenizer_train_bpe(args: argparse.Namespace) -> int:
    """Train a byte-level BPE tokenizer from a corpus and write tokenizer.json
    (brief section 2): deterministic, full merge list, spec hash reported."""
    from tinymind.model.bpe_io import save_tokenizer, spec_hash, train_bpe
    try:
        tokenizer = train_bpe(args.input, vocab_size=args.vocab_size)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    digest = save_tokenizer(tokenizer, args.output)
    report = {"output": args.output, "type": "bpe", "vocab_size": tokenizer.vocab_size,
              "merges": len(tokenizer.spec()["merges"]), "spec_sha256": digest}
    print(json.dumps(report, indent=2))
    # The hash is the tokenizer's stored identity (checkpoints/packages compare it).
    print(f"\ntokenizer spec SHA-256: {digest}", file=sys.stderr)
    return 0


def _cmd_benchmark_train_step(args: argparse.Namespace) -> int:
    """50M memory/throughput preflight (brief section 7)."""
    from tinymind.training.benchmark_50m import DEFAULT_SHAPES, run
    try:
        config = load_preset(args.config)
    except ModelConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    shapes = _parse_shapes(args.shapes) if args.shapes else DEFAULT_SHAPES
    result = run(config, shapes=shapes, seq_len=args.seq_len, seed=args.seed, warmup=not args.no_warmup)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        print("\npreflight FAILED: at least one shape did not complete a step (see 'errors')", file=sys.stderr)
    return 0 if result["passed"] else 1


def _parse_shapes(text: str) -> tuple[tuple[int, int], ...]:
    """``"1x8,1x16,1x32"`` -> ((1,8),(1,16),(1,32))."""
    shapes = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        b, _, a = part.partition("x")
        shapes.append((int(b), int(a)))
    if not shapes:
        raise argparse.ArgumentTypeError(f"no shapes parsed from {text!r} (expected e.g. '1x8,1x16')")
    return tuple(shapes)


def _cmd_generate(args: argparse.Namespace) -> int:
    from tinymind.model.backends.transformer import TransformerBackend
    from tinymind.model.tm_export import TmExportError

    backend = TransformerBackend()
    try:
        backend.load(args.model)
    except (ModelFormatError, TmExportError) as exc:
        print(f"error loading {args.model!r}: {exc}", file=sys.stderr)
        return 1

    try:
        result = backend.generate(args.prompt, max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature,
                                  repetition_penalty=args.repetition_penalty,
                                  no_repeat_ngram_size=args.no_repeat_ngram_size,
                                  top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    except ValueError as exc:  # e.g. a non-positive penalty, an invalid top-p
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.text)
    print(f"\n[{result.tokens_generated} tokens, {result.latency_ms:.1f} ms, "
         f"finish_reason={result.finish_reason}]", file=sys.stderr)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Phase 3B staged trainer (``tinymind.cli_training``); ``--legacy`` runs the Phase 3A trainer, which trains
    on the prompt text only and cannot resume (see docs/architecture/training-system.md, part 1)."""
    if getattr(args, "legacy", False):
        print("warning: --legacy runs the Phase 3A trainer: it never sees the assistant response, has no "
              "validation, and cannot resume. Use it only to reproduce old behaviour.", file=sys.stderr)
        return _cmd_train_legacy(args)
    from tinymind import cli_training
    return cli_training.cmd_train(args)


def _cmd_train_legacy(args: argparse.Namespace) -> int:
    args.dataset = args.dataset[0] if isinstance(args.dataset, list) else args.dataset
    args.epochs = 10 if args.epochs is None else args.epochs
    args.batch_size = 8 if args.batch_size is None else args.batch_size
    args.learning_rate = 3e-4 if args.learning_rate is None else args.learning_rate
    args.warmup_steps = 0 if args.warmup_steps is None else args.warmup_steps
    args.seed = 0 if args.seed is None else args.seed
    from tinymind.model.model import TinyMindTransformer
    from tinymind.model.checkpoint import save_pretrained
    from tinymind.model.tokenizer import ByteTokenizer
    from tinymind.training import CausalLMTrainer, CausalLMTrainingConfig, TrainingDataset

    try:
        config = load_preset(args.config)
    except ModelConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tokenizer = ByteTokenizer()
    if config.vocab_size != tokenizer.vocab_size:
        print(f"error: config vocab_size ({config.vocab_size}) does not match the only "
             f"tokenizer this CLI knows how to train with (ByteTokenizer, vocab_size="
             f"{tokenizer.vocab_size}) — edit the config's vocab_size to match", file=sys.stderr)
        return 1

    model = TinyMindTransformer(config, seed=args.seed)
    try:
        dataset = TrainingDataset(args.dataset, tokenizer)
        trainer_config = CausalLMTrainingConfig(
            learning_rate=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs,
            seed=args.seed, warmup_steps=args.warmup_steps)
        trainer = CausalLMTrainer(model, trainer_config)

        def _log(entry):
            print(f"step {entry.step}: loss={entry.loss:.4f} grad_norm={entry.grad_norm:.4f} "
                 f"lr={entry.learning_rate:.2e}", file=sys.stderr)

        logs = trainer.train(dataset, log_fn=_log)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_pretrained(model, args.output, training_meta={"final_loss": logs[-1].loss if logs else None,
                                                       "steps": len(logs)})
    print(f"saved checkpoint to {args.output} (final loss: {logs[-1].loss:.4f})" if logs
         else f"saved checkpoint to {args.output} (no training steps ran — dataset smaller than "
              "one batch?)")
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    import numpy as np

    from tinymind.model.tm_export import TmExportError, import_from_tm
    from tinymind.quantization.model_quantizer import export_quantized_tm, quantize_model, report

    try:
        model = import_from_tm(args.model)
    except (ModelFormatError, TmExportError) as exc:
        print(f"error loading {args.model!r}: {exc}", file=sys.stderr)
        return 1

    qmodel = quantize_model(model, args.scheme)
    probe = np.random.default_rng(0).integers(0, model.config.vocab_size, size=(1, min(32, model.config.max_seq_len)))
    rep = report(qmodel, model, probe)
    export_quantized_tm(qmodel, args.output)

    print(json.dumps({
        "scheme": rep.scheme, "bits": rep.bits,
        "original_size_bytes": rep.original_size_bytes, "quantized_size_bytes": rep.quantized_size_bytes,
        "compression_ratio": round(rep.compression_ratio, 2),
        "accuracy_delta": round(rep.accuracy_delta, 6) if rep.accuracy_delta is not None else None,
    }, indent=2))
    print(f"\nsaved quantized model to {args.output} — note: this is a reference "
         "dequantize-then-run export (no low-precision GEMM kernel exists in this delivery; "
         "see tinymind/quantization/model_quantizer.py's module docstring)", file=sys.stderr)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        model_file = read_model(args.model_path)
    except ModelFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(model_file.metadata, indent=2))
    print(f"\n{len(model_file.tensors)} tensor(s):", file=sys.stderr)
    for name, entry in sorted(model_file.tensors.items()):
        print(f"  {name:30s} {entry.dtype:8s} shape={entry.shape} bytes={entry.nbytes}", file=sys.stderr)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if args.suite == "train-step":
        return _cmd_benchmark_train_step(args)
    if args.suite != "tools":
        print(f"error: only the 'tools' benchmark layer is populated in this delivery "
             f"(see benchmarks/{args.suite}/README.md) — see STATUS.md", file=sys.stderr)
        return 1
    from benchmarks.tools.desk_suite import run as run_desk_suite
    result = run_desk_suite()
    print(result.summary())
    return 0 if result.passed else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    from tinymind.serve import serve
    serve(host=args.host, port=args.port)
    return 0


def _cmd_not_implemented(name: str):
    def handler(_args: argparse.Namespace) -> int:
        print(f"tinymind {name}: not implemented in this delivery — {_NOT_YET_IMPLEMENTED[name]}. "
             "See STATUS.md at the repository root for what's real versus designed-only.",
             file=sys.stderr)
        return 2
    return handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinymind", description=(
        "A mobile-first, tool-aware, structured-output runtime for tiny local reasoning "
        "agents. Phase 1 delivery: see STATUS.md for what's real versus designed-only."))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("version", help="print the tinymind version")
    p.set_defaults(func=_cmd_version)

    p = sub.add_parser("run", help="run a single prompt against a model")
    p.add_argument("model_path", help="path to a .tm model file (any string works against "
                                      "the default EchoBackend — see STATUS.md)")
    p.add_argument("prompt", help="the prompt to run")
    p.add_argument("--tools", help="path to a JSON file of tool schemas", default=None)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("tools", help="tool registry commands")
    tools_sub = p.add_subparsers(dest="tools_command", required=True)
    p2 = tools_sub.add_parser("list", help="list registered tools")
    p2.add_argument("--tools", help="path to a JSON file of additional tool schemas", default=None)
    p2.set_defaults(func=_cmd_tools_list)

    p = sub.add_parser("model", help="model commands")
    model_sub = p.add_subparsers(dest="model_command", required=True)
    p2 = model_sub.add_parser("info", help="show a model's geometry and parameter count "
                                           "(a named preset/config YAML, or a real .tm file)")
    p2.add_argument("preset_or_path", help="a named preset (e.g. '150m'), a path to a config YAML, "
                                           "or a path to a .tm file for an exact parameter count")
    p2.set_defaults(func=_cmd_model_info)

    p = sub.add_parser("tokenizer", help="tokenizer commands")
    tok_sub = p.add_subparsers(dest="tokenizer_command", required=True)
    p2 = tok_sub.add_parser("train-bpe", help="train a byte-level BPE tokenizer from a corpus and write tokenizer.json")
    p2.add_argument("--input", required=True, nargs="+",
                    help="corpus: text/JSONL file(s) or a directory of them")
    p2.add_argument("--vocab-size", type=int, required=True, dest="vocab_size")
    p2.add_argument("--output", required=True, help="path to write tokenizer.json to")
    p2.set_defaults(func=_cmd_tokenizer_train_bpe)

    p = sub.add_parser("generate", help="generate text from a trained .tm model (brief section 30)")
    p.add_argument("--model", required=True, help="path to a .tm model file")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=32, dest="max_new_tokens")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 (default) = deterministic greedy decoding")
    p.add_argument("--no-repeat-ngram-size", type=int, default=0, dest="no_repeat_ngram_size",
                   help="forbid completing any n-gram already in the text (e.g. 3); the targeted fix for loops "
                        "like 'the town of the town of'. 0 (default) = off")
    p.add_argument("--repetition-penalty", type=float, default=1.0, dest="repetition_penalty",
                   help="CTRL-style penalty on tokens already used (1.1-1.2 is typical); blunter than "
                        "--no-repeat-ngram-size. 1.0 (default) = off")
    p.add_argument("--top-k", type=int, default=None, dest="top_k", help="sample only from the k likeliest tokens "
                                                                         "(needs --temperature > 0)")
    p.add_argument("--top-p", type=float, default=None, dest="top_p", help="nucleus sampling cut-off in (0, 1] "
                                                                           "(needs --temperature > 0)")
    p.add_argument("--seed", type=int, default=None, help="sampling seed (needs --temperature > 0)")
    p.set_defaults(func=_cmd_generate)

    from tinymind import cli_training
    cli_training.register(sub)
    sub.choices["train"].set_defaults(func=_cmd_train)

    p = sub.add_parser("quantize", help="quantize a trained .tm model's weights")
    p.add_argument("--model", required=True, help="path to a float32 .tm model file")
    p.add_argument("--scheme", choices=["int8", "int4"], default="int8")
    p.add_argument("--output", required=True, help="path to write the quantized .tm file to")
    p.set_defaults(func=_cmd_quantize)

    p = sub.add_parser("inspect", help="inspect a .tm model file's metadata and tensor directory")
    p.add_argument("model_path")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("benchmark", help="run a benchmark suite, or the 50M train-step memory/throughput preflight")
    p.add_argument("suite", choices=["capability", "tools", "reasoning", "structured", "mobile", "performance", "train-step"])
    p.add_argument("--config", default="50m", help="profile for 'train-step' (default: 50m)")
    p.add_argument("--seq-len", type=int, default=512, dest="seq_len", help="'train-step' sequence length (default: 512)")
    p.add_argument("--shapes", default=None, help="'train-step' shapes, e.g. '1x8,1x16,1x32' (batch x accumulation)")
    p.add_argument("--seed", type=int, default=0, help="'train-step' RNG seed")
    p.add_argument("--no-warmup", action="store_true", dest="no_warmup", help="'train-step': skip the warm-up micro-step")
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("serve", help="start the local HTTP API server")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1 — "
                                                        "pass explicitly to expose beyond localhost)")
    p.add_argument("--port", type=int, default=8420)
    p.set_defaults(func=_cmd_serve)

    for name in _NOT_YET_IMPLEMENTED:
        p = sub.add_parser(name, help=f"(not implemented in this delivery: {_NOT_YET_IMPLEMENTED[name]})")
        p.set_defaults(func=_cmd_not_implemented(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    # The not-yet-implemented subcommands accept arbitrary flags (whatever
    # the eventual real command will want) and always just report status —
    # so parse them before full argparse validation, rather than trying to
    # make argparse accept an open-ended, not-yet-designed flag set for a
    # command that will refuse to run regardless of what's passed.
    if argv and argv[0] in _NOT_YET_IMPLEMENTED:
        return _cmd_not_implemented(argv[0])(argparse.Namespace())

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
