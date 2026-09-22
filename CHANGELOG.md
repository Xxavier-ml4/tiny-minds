# Changelog

All notable changes to this project are recorded here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); this project has
not yet made a tagged release, so everything below is unreleased.

## [Unreleased] — Phase 3A

### Added

- `tinymind/model/tensor.py` — a from-scratch reverse-mode autodiff engine
  on plain NumPy arrays (no PyTorch/JAX available in this build
  environment — see that module's docstring), every op checked against
  numerical gradients (`tests/model/test_tensor.py`, 26 tests).
- `tinymind/model/module.py` — a minimal parameter-tracking `Module` base
  class every layer below builds on.
- Real neural network layers, each with a dedicated gradient-checked test
  file: `norm.py` (RMSNorm), `positional.py` (RoPE, vectorized, cached),
  `attention.py` (causal self-attention, GQA/MQA/MHA as one code path),
  `mlp.py` (SwiGLU), `linear.py`, `layers.py` (pre-norm `TransformerBlock`).
- `tinymind/model/model.py` — the full `TinyMindTransformer` (embedding →
  N blocks → final norm → LM head, tied or untied) and `KVCache`.
- `tinymind/model/loss.py` — causal LM loss (next-token shift, padding
  mask support).
- `tinymind/model/generation.py` — greedy (default) and sampling
  generation with KV caching; the brief's own "mandatory" cache-vs-full-
  sequence logit equivalence test passes (`tests/model/test_cache.py`).
- `tinymind/model/optim.py` — AdamW, implemented directly against the new
  autodiff engine.
- `tinymind/model/checkpoint.py` — `save_pretrained`/`from_pretrained`, no
  pickle (`numpy.savez`/`load` with `allow_pickle=False`), independent
  from the `.tm` deployment format.
- `tinymind/model/tm_export.py` — bridges real model weights to the
  existing (further-hardened this phase) `.tm` format.
- `tinymind/model/backends/transformer.py` — `TransformerBackend`, a real
  `ModelBackend` implementation; `EchoBackend` unchanged and still used by
  Phase 1's runtime tests.
- `tinymind/training/causal_lm_trainer.py`, `collator.py` — a real
  supervised causal-LM training loop (gradient accumulation, warmup,
  gradient clipping, periodic checkpointing).
- `docs/architecture/model-implementation.md`,
  `docs/architecture/native-model-contract.md` — the new architecture
  documents this phase's brief required.
- `benchmarks/model_baseline.py` — prefill/decode latency baseline.
- `examples/train_tiny_model.py` — a complete, runnable train → checkpoint
  → `.tm` export → generate demonstration.
- CLI: `tinymind generate --model <file>.tm --prompt "..."`; `tinymind
  model info <file>.tm` now reports an exact parameter count from a real
  loaded model (previously only the config-based approximation).
- `.tm` format hardening: tensors may no longer overlap or point into the
  header/directory region; a maximal 64-bit offset/size pair is confirmed
  not to overflow (`tests/test_format.py`, 4 new tests, all passing
  alongside the unmodified 8 Phase 1 tests); `read_model()` on a missing
  file now raises `ModelFileNotFoundError` (a `ModelFormatError`
  subclass) instead of a raw, uncaught `FileNotFoundError`.
- `ModelConfig` gained `norm_epsilon`, `dropout`, `dtype` (Phase 1's
  fields unchanged; existing configs and tests unaffected).

### Fixed

- A real, reproducible non-determinism bug: `Tensor._prev` was a Python
  `set` of `Tensor` objects, which (since `Tensor` has no custom
  `__hash__`) ordered by memory-address-based identity hash — varying
  between separate process runs and occasionally producing tiny
  (~1e-7) differences in a training run's loss between two runs given the
  same seed. Fixed by making `_prev` a tuple; regression-tested by
  spawning real subprocesses (`tests/model/test_tensor.py::
  TestDeterminism`), confirmed to fail against the old code by temporarily
  reverting the fix.
- `numpy` moved from an unused `[train]` optional extra to a core
  dependency in `pyproject.toml` — it's imported transitively by the
  top-level `tinymind` package now that `tinymind.model.model` exists.

### Not included in this delivery

See `STATUS.md`. In short: training at the 100M+ parameter scale this
project targets (pure NumPy on one CPU core makes that impractical
regardless of correctness), a working native (C++) forward pass,
quantization of the real weights that now exist, distillation, LoRA, QAT,
and an Android build.

## [Unreleased] — Phase 1

### Added

- `docs/architecture/needle-analysis.md` — full architecture analysis of the
  supplied Needle 2 source tree.
- `docs/architecture/tinymind-design.md` — TinyMind's own design decisions,
  written against that analysis.
- `docs/legal/licensing.md`, `NOTICE` — provenance and Apache-2.0 attribution
  documentation.
- Repository skeleton per `docs/architecture/tinymind-design.md` §13.
- `tinymind.config` — YAML/JSON runtime configuration loader.
- `tinymind.model.config` — `ModelConfig` dataclass and named size presets
  (`configs/50m.yaml` … `configs/1b.yaml`).
- `tinymind.model.tokenizer` — `Tokenizer` interface and a working
  byte-level reference implementation (`ByteTokenizer`).
- `tinymind.model.backend` — `ModelBackend` interface and a deterministic
  `EchoBackend` reference implementation for exercising the runtime without
  a trained model.
- `tinymind.tools` — registry, JSON-Schema-derived tool schemas (functions,
  dataclasses, optional Pydantic models), a standalone validator, a
  permissions system, lexical tool retrieval, a safe executor, and a small
  built-in deterministic tool set (calculator, unit conversion, date
  arithmetic).
- `tinymind.runtime.constraints` — a model-independent JSON-Schema validator
  and constrained-decoding state machine.
- `tinymind.runtime.format` — the `.tm` model file format (named, versioned
  tensor directory; bounds-checked reader/writer).
- `tinymind.runtime.grounding` — source-span grounding checks for generated
  arguments.
- `tinymind.runtime.verification` — a `Verifier` interface plus JSON,
  arithmetic, tool-schema, and grounding verifiers.
- `tinymind.confidence` — component-level confidence scoring with an
  explicit `valid_for` field.
- `tinymind.routing` — `ResponseMode`, `ComputeLevel`, and a rule-based
  `Router`.
- `tinymind.memory` — short-term, long-term (SQLite), and retrieval memory
  stores.
- `tinymind.data` — JSONL training-data validation, deduplication, and
  splitting for the format in the engineering brief §22.
- `tinymind.evaluation` — a frozen, categorized acceptance-suite runner
  (inspired by the Needle `environments/` pattern; see
  needle-analysis.md §21) plus one example suite (`desk` — a small-desk
  device-control domain, independently authored).
- `tinymind.cli` — `tinymind run|tools|model|serve|version` against the
  reference runtime.
- `native/include/tinymind.h`, `native/CMakeLists.txt`, `native/src/*.cpp` —
  C ABI header and compilable interface stubs for the future native engine.
- `android/README.md` — deployment plan and the Bionic/glibc pitfall this
  project's own field testing surfaced.
- Test suite under `tests/` (stdlib `unittest`, no install required).

### Not included in this delivery

See `STATUS.md`. In short: no trained model, no compiled native engine, no
Android build, no quantization/distillation/training implementation beyond
interfaces — these are Phase 3 onward per the brief's own phasing.
