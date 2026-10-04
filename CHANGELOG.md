# Changelog

All notable changes to this project are recorded here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); this project has
not yet made a tagged release, so everything below is unreleased.

## [Unreleased] — Dropout, anti-looping diagnostics and decoding controls

Prompted by an external review of fluent-but-looping stage-1 output ("the town of the town of ...").
Each item was checked against the code first; what the review got wrong is recorded here too.

### Added

- **Training-time dropout** (`model.dropout`, previously rejected): attention probabilities, attention output and
  MLP output (GPT-2 placement) in both the fused kernel and the reference path. It is enabled only by passing a
  `dropout_rng` to `forward` — there is no train/eval flag to forget, so evaluation, generation and export are
  dropout-free by construction. The trainer seeds masks from `(seed, step, micro-batch)`, so interrupted and
  resumed runs are bit-identical to uninterrupted ones. Every trainer applies it (the engine, the `--legacy`
  Phase 3A trainer, and the `benchmark train-step` preflight, so measured cost/memory include it); a single
  `dropout_generator` defines the seeding. Default stays `0.0` (see "Not changed").
- **Measuring memorization instead of assuming it**: `[eval]` lines, `val_history` and `metrics.jsonl` carry
  `train_loss_recent` and `generalization_gap`; `memorization_signal` warns when validation loss rises while
  training loss falls on consecutive evaluations; `DataPlan.planned_passes` (per-epoch repeat factor × epochs the
  token budget implies) feeds a start-up `[data] WARNING` and `training_summary.json` → `dataset.planned_passes`.
  All advisory; none can stop a run.
- **Decoding controls**: `ModelGenerationConfig.no_repeat_ngram_size`, input validation, and `tinymind generate`
  flags `--no-repeat-ngram-size`, `--repetition-penalty`, `--top-k`, `--top-p`, `--seed` (the CLI previously exposed
  only `--temperature`). Defaults are unchanged: plain deterministic greedy.
- **Non-gating `generation_decoded` diagnostic** in stage reports (`generation.diagnostic_decoding`, enabled in
  `stage1.objective.json`): the same fixed prompts under a repetition penalty and a 3-gram ban, shown beside the raw
  greedy numbers. No measurement may reference it.
- `docs/training/degeneration-and-memorization.md`; tests `test_dropout.py`, `test_dropout_training.py`,
  `test_decoding_controls.py`, `test_objective_diagnostic.py`.

### Fixed

- `tests/ci/test_stage_io.py::...interrupted_stage` asserted an artifact name by `rsplit("-", 1)`, which broke
  whenever the run id (a git SHA in CI, a fallback outside a checkout) contained a hyphen. It now matches the prefix.
- Tests that asserted `dropout` was rejected now assert it is supported. (`benchmarks/audit/phase3a_correctness_audit.py`
  is a frozen record of the Phase 3A audit and was already out of date for the other knobs; it is left as it was.)

### Not changed, on purpose

- **Cross-document loss masking**: already implemented and tested (`collate_packed` requires `labels[0] == -100`;
  block-diagonal attention via `segment_ids`; RoPE positions restart per segment; see
  `test_packed_loss_equals_sum_of_individual_losses`). Nothing to fix.
- **The stage gate still measures raw greedy output.** A repetition penalty in the gate would let a model that loops
  pass `generation_not_looping`.
- **`configs/50m.yaml` keeps `dropout: 0.0`.** At a budget of a few passes over the data a 50M model is
  training-limited, not data-limited, and dropout slows it; it is also part of the model-config hash, so enabling it
  makes existing checkpoints unresumable. Turn it on at the start of a stage if the new diagnostics show memorization.

## [Unreleased] — Fixes from the first GitHub run of train-50m

### Fixed

- The first real run failed at "Confirm the profile really is the 50M model":
  the step searched `tinymind model info`'s standard output for
  `50,370,624`, but that command prints the count on standard error. The
  step now counts the profile's parameters directly
  (`load_profile(...).count_parameters()`). A test runs the step's own
  script against the real 50m profile.
- When the corpus was too small for the profile's vocabulary, the tokenizer
  step still succeeded. For example, the default hermetic manifest gives a
  1,557-token BPE for a 16,000-token profile. The run then failed later at
  the Train step with a vocabulary mismatch. The step now fails immediately,
  names the fix (a real corpus with `allow_download=true`), and is covered
  by an executed-step test.

### Changed

- `BPETokenizer.train` keeps pair counts up to date incrementally: after
  each merge it recounts only the words that contain the merged pair, and a
  lazily invalidated heap tracks the maximum. It produces the same merges
  as recounting every pair at every step, including tie-breaks and
  overlapping pairs. `tests/model/test_bpe.py` compares the two on several
  corpora, and mutation checks confirm those tests catch deviations.
  - A 16k vocabulary from the 8,000-chunk tokenizer sample now trains in
    about 1 second instead of 20–60 minutes.
  - Every CI job re-derives the tokenizer, so this cost used to come out of
    every job's training budget.
- `fetch_shard` retries a failed http(s) download up to 3 attempts in all,
  with backoff. It retries connection errors, timeouts, HTTP 408/429/5xx,
  and bodies cut short of their `Content-Length`. Before, `http.client`
  accepted a short body silently. Other HTTP errors such as 404 still fail
  at once.
  - The http(s) path had no test. A local-server test now covers the
    redirect, the bearer token reaching the first host only, retries, and
    the refusal of truncated bodies.

## [Unreleased] — Audit of the objective-driven stages and the Stage-1 corpus path

### Fixed

- A checkpoint saved before a stage's first evaluation stored the untrained
  step-0 baseline as the regression reference. A run resumed from it compared
  its first checkpoint with a random model; checkpoints now store exactly the
  in-memory reference, so a resumed run decides like the uninterrupted one.
- `train-50m.yml` ran steps under GitHub's default `bash -e` without
  `pipefail`, so `prepare-corpus … | tee`, `build-curriculum-v2 … | tee` and
  the preflight `… | tee` passed even when the command failed. The workflow
  now sets `defaults: run: shell: bash`.
- The corpus split was seeded with the training seed. A later stage
  dispatched with another seed would get a different split, natural held-out
  text and tokenizer, and init-from would fail. The split now uses the
  corpus's own fixed seed.
- The Stage-1 data requirement counted corpus chunks the trainer drops for
  exceeding `max_seq_len`. `data.natural_train_bytes` now counts only the
  natural text actually trained, with a warning.

### Changed

- Tokenizer training input is `tinymind data tokenizer-sample`: a seeded
  uniform sample (bottom-k by `sha256(seed:id)`) over every shard of the corpus
  TRAIN split plus the synthetic supplements, with a `sample_manifest.json`.
  It replaces `head -n` of each file, which covered only the first shards and
  also fed held-out text into the tokenizer.
- `val_text.jsonl` / `test_text.jsonl` are the same kind of sample across
  every shard, not the first N records, and are identical for every stage
  built from the same corpus.
- `max_records` is spread evenly over a source's shards, so every shard is
  read.
- The trainer's wall-clock budget in `train-50m.yml` subtracts the time spent
  on data preparation and tokenizer training, so a long preparation cannot run
  the job into the hard timeout.
- Corpus provenance (`corpus_manifest.json`: sources, licenses, shard hashes)
  and `curriculum.json` go into `<run>/data_provenance/` and the stage bundle.
  The workflow adds the tokenizer sample manifest.
- `train-50m.yml` passes the optional secrets `HF_TOKEN` / `CORPUS_TOKEN` to
  the data-preparation step only, for private corpus shards (`auth_env`).
- `stage_io check-incoming` writes its routing decision, or the reason for a
  rejection, to the job summary.
- Tests: `tests/ci/test_objective_chain.py` (the whole chain through the real
  command lines), plus regression tests for each fix above.

## [Unreleased] — Objective-driven stages and real pretraining data

### Added

- `tinymind/data/corpus.py`: real pretraining-corpus ingestion. It streams
  sharded local, `file://` and `http(s)` data (network only with
  `--allow-download`), verifies per-shard SHA-256, and supports private
  shards via `auth_env` (the bearer token is never forwarded on redirects or
  recorded). It chunks text; deduplicates across the corpus (repeated
  paragraphs, exact, MinHash near duplicates); splits train/val/test
  deterministically by document group; decontaminates against evaluation
  prompts; and writes a provenance `corpus_manifest.json`.
  CLI: `tinymind data prepare-corpus`.
- `build-curriculum-v2 --corpus`: attaches the corpus as configured by the
  per-stage `corpus` blocks in `configs/curriculum_v2.json`. Stage 1 trains
  80% on natural text by bytes, with the synthetic generators as a
  supplement; later stages replay 10%. It writes natural held-out text
  (`val_text.jsonl`) and a lexicon.
- `tinymind/training/objective.py` and
  `configs/stages_v2/stage1..7.objective.json`: stage objectives measured on
  the live model at every evaluation checkpoint. They cover held-out loss,
  natural-text bits per byte, generation quality on fixed prompts, grammar
  checks, data requirements, retained earlier capabilities, and regression
  against the previous checkpoint and against promotion-time metrics. Every
  measurement must pass.
- Per-checkpoint objective reports (JSON + Markdown with the exact prompts and
  raw generations) and `tinymind objective-report` for the GitHub job summary.
- `tinymind train --continue-stage` (with `--reopen-stage`): same-stage
  continuation with a larger token budget. Also `--no-objective`,
  `--objective-dir`, `--objective-min-tokens` and `--objective-validation`.
- `stage_io`: `continue` routing mode; bundles record `stop_reason`, budget and
  objective state, and carry the objective reports.
- `datasets/v2/samples/` (a small CC0 natural-text sample) and
  `datasets/v2/corpus.manifest.example.json`; `docs/training/objective-driven-stages.md`.

### Changed

- A v2 stage is complete only when its objective is met. Its token budget is
  the minimum training chunk. A stage that uses its budget without meeting the
  objective ends `gate_failed` and stays incomplete. Stages without an
  objective (the v1 pipeline, `--no-objective`) keep budget completion.
- `train-50m.yml` prepares the real corpus and trains until the objective is
  met (`resume` / `continue` / `init-from`). It publishes every checkpoint's
  report to the job summary, uploads the reports as an artifact, and previews
  the promotion gate.
- `datasets/v2/manifest.json` holds the natural-text sample (primary, pinned)
  and marks the synthetic entries as supplemental. `tinymind data
  prepare-external` streams and chunks `local`/`url` entries, including
  sharded and `.gz` ones.

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
