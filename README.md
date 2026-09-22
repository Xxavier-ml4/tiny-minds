# TinyMind

TinyMind is an open-source, mobile-first AI runtime and model stack for
building tiny local reasoning agents with tools, structured outputs,
verification, and adaptive computation.

**Read this first:** [`STATUS.md`](STATUS.md) is the honest, current
accounting of what in this repository is real and tested versus designed
but not yet implemented. This README describes the whole intended system;
`STATUS.md` says which parts of it exist today. When the two seem to
disagree, `STATUS.md` is correct.

## Why tiny models

A model that runs entirely on-device — no network round-trip, no per-token
cost, no data leaving the phone — is a different product than an API call
to a large model, not a worse version of the same product. TinyMind's bet
is that a small model plus a genuinely good runtime (tool retrieval,
deterministic computation for the things determinism is good at, structured
constraints, grounding, verification, adaptive compute) can behave far more
capably than its parameter count alone would suggest. That combination —
not "train a smaller LLM" in isolation — is the actual project; see
`docs/architecture/tinymind-design.md` section 0 and the "most important
design principle" section of the engineering brief this repository was
built from.

## Architecture

TinyMind was designed after a full read of the [Needle 2](https://github.com/cactus-compute/needle)
source tree (Apache-2.0, by Cactus Compute) — not a fork or a rename, an
independent implementation built after documenting what that architecture
gets right and where TinyMind deliberately does something else instead.

- [`docs/architecture/needle-analysis.md`](docs/architecture/needle-analysis.md) —
  the analysis, section by section.
- [`docs/architecture/tinymind-design.md`](docs/architecture/tinymind-design.md) —
  TinyMind's own decisions, each one stated next to the analysis point it
  responds to.
- [`docs/architecture/model-implementation.md`](docs/architecture/model-implementation.md) —
  the real transformer implementation (Phase 3A): architecture, shapes,
  initialization, training/generation flow, and the environment constraint
  (no PyTorch/JAX/TensorFlow available) that shaped it.
- [`docs/architecture/native-model-contract.md`](docs/architecture/native-model-contract.md) —
  the exact numerical contract a future native (C++) implementation must
  match: tensor names, layouts, formulas, nothing left implicit.
- [`docs/legal/licensing.md`](docs/legal/licensing.md) and
  [`NOTICE`](NOTICE) — provenance, in detail.

The short version of what's different: TinyMind has a real non-tool-call
response mode (Needle's every response is a tool call or the empty call
`[]`), a Python-level (not closed-native-binary) structured-output
constraint engine, a named/versioned model file format, multi-session
support from the start, and telemetry that doesn't exist rather than
defaulting on.

## The model

There is now a real, trained-in-this-repository transformer — decoder-only,
GQA attention with RoPE, SwiGLU MLP, RMSNorm, pre-norm residuals (see
`docs/architecture/model-implementation.md`). Because no PyTorch/JAX/
TensorFlow is installable in the environment this was built in (verified
directly — see that doc's section 0), the whole thing — forward pass,
backward pass, and an AdamW optimizer — runs on a small, from-scratch
reverse-mode autodiff engine over plain NumPy arrays
(`tinymind/model/tensor.py`), checked op-by-op against numerical
(finite-difference) gradients.

```python
from tinymind.model import ModelConfig, TinyMindTransformer, AdamW, ByteTokenizer
from tinymind.training import CausalLMTrainer, CausalLMTrainingConfig, TrainingDataset

tokenizer = ByteTokenizer()
model = TinyMindTransformer(ModelConfig(hidden_size=48, num_layers=2, num_heads=4,
                                        num_kv_heads=2, intermediate_size=96,
                                        max_seq_len=64, vocab_size=tokenizer.vocab_size))
trainer = CausalLMTrainer(model, CausalLMTrainingConfig(learning_rate=5e-3, epochs=40))
trainer.train(TrainingDataset("train.jsonl", tokenizer))
```

See [`examples/train_tiny_model.py`](examples/train_tiny_model.py) for the
full runnable script this is drawn from — in this sandbox it takes a
54,192-parameter model from loss 6.20 to 0.003 on a synthetic pattern and
greedy-generates the memorized continuation back exactly. Checkpoint
save/load (`tinymind.model.checkpoint`, no pickle) and `.tm` export/import
(`tinymind.model.tm_export`) both round-trip bit-exact. What this *isn't*:
training at the 100M+ parameter scale the project targets — pure-NumPy,
single-core, GPU-less matmuls make that impractical regardless of
correctness. See `STATUS.md`'s Phase 3A section for the precise line
between what's proven and what isn't.

## The tool system

```python
from tinymind import Model
from tinymind.tools.builtins import calculator, unit_convert

model = Model("models/tinymind-150m-q4.tm", tools=[calculator, unit_convert])
result = model.run("What's the capital of France?")  # -> CHAT, no tool forced
```

Tools are declared from a plain Python function's type hints and docstring,
a `@dataclasses.dataclass`, a Pydantic model, or a raw JSON Schema dict.
Every tool call is validated against its schema before it runs, is
permission-tagged (`READ_ONLY`/`LOCAL_WRITE`/`NETWORK`/`SENSITIVE`/
`DESTRUCTIVE`, with the last two requiring explicit confirmation from the
calling application — never from the model), and is checked for whether its
arguments are actually grounded in the input text rather than invented.
None of this needs a trained model to work — see `examples/custom_tool.py`
for the full pipeline running end to end today. See
[`docs/architecture/tinymind-design.md`](docs/architecture/tinymind-design.md)
sections 2-5.

## Reasoning and adaptive computation

`tinymind.routing.Router` decides a response mode (`CHAT`/`TOOL_CALL`/
`STRUCTURED_OUTPUT`/`PLAN`/`REASON`/`REFUSE`/`ASK_CLARIFICATION`) and a
compute level (`FAST`/`NORMAL`/`DEEP`/`VERIFY`/`ESCALATE`) before any
generation happens, and a small built-in deterministic tool set
(calculator, unit conversion, date arithmetic — `tinymind/tools/
builtins.py`) means arithmetic is computed, not hallucinated. This is
real, rule-based, and tested against every worked example in the
engineering brief's own section 9 (`tests/test_runtime.py`); a
learned router is a drop-in replacement for `Router.route()` later, not a
different interface.

## Mobile / native runtime

Designed for `arm64-v8a` Android as the primary target, offline-first,
telemetry-off. The C ABI (`native/include/tinymind.h`) and the
model-independent native pieces (a tensor buffer, a sampler, KV-cache
memory management, a byte tokenizer) are real, compiled C++ with a passing
test suite. The forward pass through an actual architecture is not — see
`STATUS.md`'s Phase 7 section for exactly what "partial" means here and
why it isn't close to a working on-device model yet. It now has a complete
numerical specification to implement against, though:
[`docs/architecture/native-model-contract.md`](docs/architecture/native-model-contract.md).

## Installation

```sh
git clone <this-repo>
cd TinyMind
pip install -e . --break-system-packages   # or use a venv
```

No network access and no GPU are required for anything in this delivery —
see `CONTRIBUTING.md`.

## Quickstart

```sh
tinymind version
tinymind tools list
tinymind model info 150m
tinymind run models/placeholder.tm "hello there"
tinymind benchmark tools
tinymind serve   # binds 127.0.0.1:8420
```

`tinymind run` and the default `Model()` use `EchoBackend`, a deterministic
non-neural stand-in, unless a real trained `.tm` model is loaded through
`TransformerBackend` — see `tinymind model info <file>.tm` (exact parameter
count from a real model) and `tinymind generate --model <file>.tm --prompt
"..."` for the commands that use one. See
[`examples/basic_usage.py`](examples/basic_usage.py),
[`examples/custom_tool.py`](examples/custom_tool.py), and
[`examples/train_tiny_model.py`](examples/train_tiny_model.py) (a real configurable training run; `examples/smoke_train.py` is the CI sanity check) for runnable
code covering both paths.

## Training

**Phase 3B** — a reliable, resumable, mobile-first training pipeline for a 0.5-5 M-parameter model on CPU.
Design and evidence: [`docs/architecture/training-system.md`](docs/architecture/training-system.md) (audit of
Phase 3A + the implemented design), measurements in [`docs/benchmarks/`](docs/benchmarks/), the staged plan in
[`docs/training/three-stage-plan.md`](docs/training/three-stage-plan.md), results and caveats in
[`docs/training/implementation-report.md`](docs/training/implementation-report.md).

```bash
tinymind data build-curriculum --stage stage1 --out data/stage1          # deterministic, held-out eval split
tinymind train --config tiny_mobile --dataset data/stage1 --output runs/s1 --max-runtime 3600
tinymind train --config tiny_mobile --dataset data/stage1 --output runs/s1 --resume runs/s1/checkpoints   # exact continuation
tinymind train --config tiny_mobile --dataset data/stage2 --output runs/s2 --stage stage2 --init-from runs/s1/checkpoints
tinymind verify-checkpoint runs/s2/checkpoints ; tinymind checkpoint-info runs/s2/checkpoints
tinymind eval --package runs/s2/export --eval data/stage2/eval.jsonl --val data/stage2/val.jsonl --out eval.json
```

What is different from Phase 3A (each item has a test): the assistant response is actually in the training
sequence and is the only thing with loss (completion-only masking, EOS, deterministic prompt template shared with
inference); padded or packed batching with no cross-example leakage; `batch × accumulation` is exactly equivalent
to a bigger batch; warmup + cosine LR; validation loss every N steps; atomic, SHA-256-verified checkpoints holding
weights, AdamW moments, scheduler, data position, RNG, dataset/tokenizer/config identity — and a resume that is
refused (listing every mismatch) rather than silently degraded; a time budget that checkpoints and exports before the
limit; a GitHub Actions workflow that moves a stage between jobs as a verified artifact. The Phase 3A trainer is
still available as `tinymind train --legacy` and is documented as defective.

The profiles are `configs/tiny_debug.yaml` (115 K parameters), `tiny_mobile.yaml` (1.21 M) and
`tiny_mobile_plus.yaml` (3.49 M); `tinymind budget --config tiny_mobile` prints the size / KV-cache budget.
**The bundled curriculum is template-generated**: it shows that the pipeline learns narrow behaviours and
generalises across held-out values and phrasings; it is not evidence of open-domain usefulness.

## Quantization

Post-training INT8 is real as a **storage format** with a reference loader: per-row symmetric int8 + float32 scales
(`tinymind.quantization`), packaged as `model.int8.tm` (`tinymind.export.int8`), compared against float32 on the
same held-out data by `benchmarks/quantization_eval.py` (file size, memory, latency, validation loss and every
capability metric — see the implementation report for the measured numbers). The Python reference dequantizes to
float32 before computing and the native runtime cannot read the int8 file, so **no int8 speed or RAM saving is
claimed**. Int4 is still an interface only.

## Android

Plan only — [`android/README.md`](android/README.md), including a
documented pitfall (a Linux ARM64 binary is not an Android ARM64 binary —
Bionic vs. glibc) surfaced by this project's own field testing, recorded
so a future CI pipeline can guard against it from day one rather than
discovering it on a real device.

## Benchmarks

```sh
tinymind benchmark tools
```

`tinymind.evaluation.AcceptanceSuite` — a frozen, categorized (positive/
missing/irrelevant/negation/invalid/parallel) acceptance-suite runner with
critical-failure gating — is real and demonstrated end to end in
`benchmarks/tools/desk_suite.py` (15 cases, all 6 categories, run against a
rule-based demo predictor that is explicitly *not* a claim about a trained
model's quality — see that file's docstring). `benchmarks/{capability,
reasoning,structured,mobile,performance}/` each explain what they're
waiting on. No performance numbers are published anywhere in this
repository for the simple reason that there is no trained model or native
engine yet to measure — see the brief's own section 51 ("do not make
unsupported performance claims") and section 57 (don't claim to beat
anything without reproducible benchmarks).

## Roadmap

See `STATUS.md` for the authoritative, current-as-of-last-edit version of
this. A real small reference model architecture, forward pass, backward
pass, and training loop now exist (Phase 3A/4 — see "The model" above).
In order from here: distillation from a teacher (Phase 5) → quantization
of the real weights that now exist (Phase 6) → a working native forward
pass behind the existing C ABI, against the numerical contract already
written down (Phase 7) → an Android build (Phase 8) → measurement and
optimization against real numbers (Phase 9).

## Licensing

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). TinyMind is
an independent implementation inspired by Needle 2's architecture; see
[`docs/legal/licensing.md`](docs/legal/licensing.md) for exactly what that
does and doesn't mean.
