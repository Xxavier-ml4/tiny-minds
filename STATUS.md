# Status

This tracks what's real versus designed-only, mapped to the brief phases
this repository was built from. Update this file in the same PR that
changes a row's status — see `CONTRIBUTING.md`.

Legend: [x] real and tested (all in this table now include their test
files) · [~] interface/skeleton, documented, not implemented · [ ] not
started.

## Phase 3A — a real transformer (current)

This is the headline change since Phase 1: TinyMind now has an actual,
trained, gradient-checked neural network, not just the runtime around
where one would go.

**The environment constraint that shapes this whole phase**: no PyTorch,
JAX, or TensorFlow is installed, no network access exists to install one
(`pip install torch` → "No matching distribution found"), and there is no
GPU. `tinymind/model/tensor.py` implements a minimal reverse-mode autodiff
engine directly on NumPy arrays to make real gradients possible at all —
see that file's docstring and `docs/architecture/model-implementation.md`
section 0. Every autograd op is checked against finite-difference
numerical gradients (`tests/model/test_tensor.py`, 26 tests).

| Piece | Status | Tests |
|---|---|---|
| Autodiff engine (`tensor.py`) | [x] | `test_tensor.py` (26) |
| Module/parameter system (`module.py`) | [x] | exercised throughout `tests/model/` |
| RMSNorm | [x] | `test_norm.py` (5), gradient-checked |
| RoPE | [x] | `test_rope.py` (11): shape, determinism, position-sensitivity, norm-preservation, stability to 2048 |
| Causal self-attention (GQA/MQA/MHA, one code path) | [x] | `test_attention.py` (9): causal-masking checked directly, cache-vs-no-cache agreement |
| SwiGLU MLP | [x] | `test_mlp.py` (5), gradient-checked |
| TransformerBlock (pre-norm residual) | [x] | `test_block.py` (4): residual wiring checked via all-zero-weights identity test |
| Full `TinyMindTransformer` | [x] | `test_model.py` (17): shapes, no-softmax, param counting, full-model gradient flow, optimizer actually changes weights |
| Causal LM loss (shift+mask) | [x] | `test_loss.py` (5): shift direction checked both ways |
| KV cache | [x] | `test_cache.py` (4) — **the brief's own "mandatory" test**: prefill+decode logits match full-sequence logits within 1e-3, and greedy generation matches exactly with/without cache |
| Generation (greedy default, sampling, repetition penalty) | [x] | `test_generation.py` (11): determinism, EOS stopping, context-window clamping, seeded sampling reproducibility |
| AdamW optimizer | [x] | exercised by every training test; decoupled weight decay |
| Checkpoint save/load (no pickle) | [x] | `test_checkpoint.py` (7): exact round-trip, `allow_pickle=False` enforced |
| `.tm` export/import for real weights | [x] | `test_tm_export.py` (7): exact round-trip, checksum-protected, malformed-file rejection |
| `.tm` format hardening (overlap/header-region/overflow checks) | [x] | `test_format.py`: 4 new hardening tests + all 8 Phase 1 tests still passing |
| Real `ModelBackend` (`TransformerBackend`) | [x] | wired through `Engine`/`Session` in ad hoc integration testing; `EchoBackend` unchanged and still used by Phase 1's runtime tests |
| Training loop (`CausalLMTrainer`) | [x] | `tests/training/test_training.py` (5) — **the brief's own "critical acceptance test"**: loss decreases, the model demonstrably memorizes a tiny repeating pattern (verified by generating it back), gradient accumulation, checkpoint-during-training, and reproducibility given a seed |
| CLI: `model info <file>.tm` (exact param count), `generate` | [x] | `test_cli.py`: 3 new tests |
| `benchmarks/model_baseline.py` | [x] | runs; see its own module docstring on what the numbers do and don't mean |
| Native model forward pass (`native/src/model.cpp`) | [x] | Real C++ forward pass matching docs/architecture/native-model-contract.md; Python/native logit agreement verified: max abs diff 1.9e-06, argmax matches at every position. KV-cache decode path also matches full-sequence forward exactly (0.0 max diff within the same process). `tests/model/test_native_equivalence.py` (5 tests) is the permanent automated harness. |
| Distillation / LoRA / QAT | [ ] | explicitly out of scope this phase (brief section 19) |

**A real, reproducible bug found and fixed this phase**, worth recording
because it's the kind of thing that's easy to miss: `Tensor._prev` was a
Python `set` of `Tensor` objects. `Tensor` has no custom `__hash__`, so
that set ordered by the default identity (memory-address-based) hash,
which differs between separate process invocations even for identical
code and inputs. That let `backward()`'s topological sort visit a
multi-parent node's children in a different order run to run, and because
float addition isn't exactly associative, this occasionally produced
tiny (~1e-7) differences in a training run's loss between two runs with
the same seed — caught by `tests/training/test_training.py::
test_reproducible_given_seed`, root-caused, fixed (`_prev` is a tuple now),
and covered by a dedicated regression test
(`tests/model/test_tensor.py::TestDeterminism`) that spawns two real
subprocesses and would have failed against the old code (verified directly
by temporarily reverting the fix and confirming the test catches it).

**Validation run** (see `examples/train_tiny_model.py` for the exact
script): a 54,192-parameter model trained for 40 epochs on 8 copies of one
sentence went from loss 6.20 to 0.003, and greedy-generates the memorized
continuation exactly. Checkpoint and `.tm` round-trips both match logits
bit-for-bit. Full test suite: **281 tests, 0 failures**, run via
`python3 -m unittest discover -s tests` (pytest not installed in this
sandbox either, per the "no network" constraint above — the same tests are
pytest-collectible once it is).

**What "real" does and doesn't mean here**: the architecture, forward
pass, backward pass, and training loop are genuine — not stubs, not
placeholders, not simulated. What hasn't happened is training at the
100M+ scale the project ultimately targets: pure-NumPy, single-core,
GPU-less matmuls make that impractical in this sandbox regardless of code
correctness (see `docs/architecture/model-implementation.md` section 0).
Every number and claim above is about correctness at small scale, which is
a scale-independent property (the same code path runs at every config
size — `tests/model/test_model.py::test_larger_config_has_more_parameters`
checks exactly this) — not a claim about training throughput at 1B
parameters.

## Phase 3B — mobile-first training + multi-stage CPU training (current)

Supersedes Phase 4 below for real runs. Full design, audit evidence and
measured numbers: `docs/architecture/training-system.md` (Part 1 = audit of
Phase 3A, Part 2 = what was built), `docs/training/three-stage-plan.md`,
`docs/training/implementation-report.md` (the required final report —
acceptance criteria in the brief, section 61/62). ~6,400 lines of new
`tinymind/` code, ~3,000 lines of new tests; full suite **499 tests, all
passing** (`python -m unittest discover -s tests`, ~55 s).

**What Phase 3A actually did, found by audit before writing any new code**
(`docs/architecture/training-system.md` Part 1, each claim backed by a
reproducible script in `benchmarks/audit/`): the training data pipeline never
tokenized the assistant response — it trained on the concatenated user text
only; gradient accumulation averaged micro-batch means instead of tokens
(off by 12% of the largest gradient with unequal-length micro-batches);
`tokens_per_second` decayed ~1/step from a wallclock bug; checkpoints were
weights-only, written in place (an interrupted save destroyed the only
valid copy), with no resume of step/epoch/schedule/RNG/data-order — "resume"
restarted training from step 0 with weights carried over; several
`ModelConfig` fields (`norm_type`, `mlp_type`, `dropout`, `dtype`,
`sliding_window`) validated but were silently ignored; the recursive
backward pass overflowed Python's recursion limit past ~28 layers; the
`.tm` export carried no tokenizer. None of this showed up in Phase 3A's own
green test suite, because those tests checked different things (loss goes
down, a round-trip loads) — the audit's point (brief section "1" and rule
1: "Do not trust previous code merely because tests pass") is exactly that
passing tests are not the same claim as correct behaviour.

| Piece | Status | Tests |
|---|---|---|
| Canonical example format + one renderer (train/eval/inference share it); completion-only loss mask, EOS, prompt = exact prefix of training sequence | [x] | `tests/training/test_data_pipeline.py::TestRenderer` |
| Padded + packed batching, block-diagonal attention, no cross-example leakage | [x] | `TestBatching`, `TestPackingDoesNotLeak` (a control case shows plain causal attention *does* leak) |
| Deterministic data order/mixture (`DataPlan`, pure function of seed+epoch, no stateful RNG to lose) | [x] | `TestDataPlan` |
| Token-normalised gradient accumulation (`batch×accum` exactly = one bigger batch) | [x] | `test_batch8_equals_batch4_accum2`; Phase 3A's rule shown to differ by >1e-3 on the same data |
| AdamW named state (`state_dict`/`load_state_dict`, all-or-nothing), norm-exempt weight decay, non-finite guard before any state changes | [x] | `TestStateObjects`, `TestDivergence` |
| Warmup + cosine/linear/constant LR schedule, checkpointable | [x] | `test_scheduler_state_round_trip_and_mismatch`, `test_schedule_shape` |
| Atomic, SHA-256-manifest, resumable training checkpoint (weights, optimizer, scheduler, RNG, data position, dataset/tokenizer/config identity, environment) | [x] | `tests/training/test_checkpoint.py` — fault injection at all 7 write-protocol points + a real `os._exit(17)` mid-write |
| Exact resume (100 steps == 50+50, bitwise) — refuses on stage/architecture/tokenizer/dataset/config mismatch, listing every mismatch | [x] | `TestExactResume`, `TestMismatchRejection` |
| Time budget (checkpoint+export+exit before the limit; SIGTERM/SIGINT handled the same way) | [x] | `TestTimeBudget` |
| Exact parameter accounting (`count_parameters` == an instantiated model's count, every attention/tie layout) | [x] | `tests/model/test_config_count.py` |
| Model profiles `tiny_debug`/`tiny_mobile`/`tiny_mobile_plus` (115K / 1.21M / 3.49M params, all < 5M, `tiny_mobile` in the preferred 1-3M range) | [x] | same file |
| Fused attention/RoPE/RMSNorm/SwiGLU/linear ops (composed ops kept as reference); graph release after backward; `no_grad()` for inference | [x] | `tests/model/test_fused.py`; measured 1.37-1.67× faster, ~2.8× less activation memory at T=256 |
| MHA/GQA/MQA benchmarked at equal parameter budget; sequence-length and batch-size sweeps; LR sweep (measured, not assumed) | [x] | `docs/benchmarks/bench-attention-variants.json`, `bench-seqlen.json`, `bench-batch.json`, `sweep-lr.json`, `sweep-attention.json`, `sweep-seqlen.json` |
| Runner probe (env/BLAS/CPU info, measured thread-count selection) | [x] | `benchmarks/runner_probe.py`; only 1 vCPU available in this sandbox, so multi-thread scaling is unmeasured here — the workflow runs the probe on the real runner |
| PyTorch backend investigated, not added (overhead was 0.3% of a step — not the bottleneck; NumPy kernel fusion fixed the real one) | [x] design decision, recorded | `docs/benchmarks/pytorch-backend-decision.md` |
| Inference package (`.tm` + tokenizer + manifest; no optimizer/training data/framework/network) — loads and reproduces logits in a process where `tinymind.training` is unimportable | [x] | `tests/export/test_package.py` |
| INT8 as a **storage** format with a reference (dequantize-then-compute) loader; honestly scoped — no RAM/latency claim, native runtime cannot read it | [x] | `tests/export/test_int8.py`; `benchmarks/quantization_eval.py` measures size/quality against FP32 on held-out data |
| Native/Python numerical agreement extended to a real exported package and the real `tiny_mobile` architecture | [x] | `tests/model/test_native_equivalence.py::TestPackageModelNative` |
| Deterministic synthetic curriculum (stage0-3), held-out `eval.jsonl` (value- and phrasing-level hold-out, category-balanced, contamination-checked) | [x] | `tests/training/test_contamination_curriculum.py` |
| Contamination detection (exact + normalised, prompt + full-example) | [x] | `tests/training/test_contamination_curriculum.py::TestContamination` |
| Tiny-model capability suite: separate metrics (no composite score), tool-call correctness/malformed/missing/false-positive, clarification, refusal, structured output, repetition, generation throughput | [x] | `tinymind/evaluation/tiny_suite.py`; run on real stage checkpoints, see the implementation report |
| GitHub Actions workflow (`train-stage.yml`): one parameterised workflow, artifact hand-off with explicit download+verify before trust, resume/init-from modes, promotion gate, time-budget-aware, optional non-blocking publish | [x] design + `tests/ci/test_workflow.py` (static) | **not executed on an actual GitHub runner** — no network in this sandbox; see the implementation report for what a first real run must confirm |
| Stage hand-off logic (bundling, verification, mode resolution, promotion gate) exercised end-to-end offline | [x] | `tests/ci/test_stage_io.py` — a real 4-job chain (stage0→1→2→3) across separate subprocess "jobs", each seeing only a copied bundle |
| Optional byte-level BPE tokenizer (parameter-cheap alternative to a large fixed vocabulary) | [x] Python-only | `tests/model/test_bpe.py`; no native reader yet — measured, not deployed |
| CLI: `train` (resume/init-from/time-budget/packing/mixture/contamination-check), `checkpoint-info`, `verify-checkpoint`, `export`, `verify-package`, `eval`, `eval-compare`, `stage-gate`, `budget`, `data build-curriculum/check-contamination/render` | [x] | `tests/test_cli_training.py` |
| A real reduced-budget demonstration chain (stage0→1→2→3) run in this sandbox, with a genuine negative finding | [x] | `docs/benchmarks/stage-runs/`; see the implementation report — held-out **argument-copying accuracy was low (3.8%) even where tool *selection* was learned (correct-tool rate 0.84)**, traced to a small values pool letting the model memorise instead of copy, and **stage 2 failed its own promotion gate** (false-positive tool-call rate 0.341 > 0.30 threshold) |

**Known limitations** (also in the implementation report): the curriculum is
template-generated, so results show pipeline correctness and narrow
held-out generalisation, not open-domain usefulness; the byte tokenizer
makes sequences long; no native INT8 kernel; multi-thread BLAS scaling
unmeasured (1 vCPU here); bitwise-identical resume is not promised across
different hardware; the GitHub Actions workflow itself has not run on a
real runner from this sandbox.

## Phase 1 — Architecture, skeleton, config, public API, tests

| Piece | Status | Notes |
|---|---|---|
| Architecture analysis & design docs | [x] | `docs/architecture/*.md` |
| Repository skeleton | [x] | matches brief section 42 |
| Configuration system | [x] | `tinymind/config.py`, `tinymind/model/config.py` + 6 size presets; extended in Phase 3A with `norm_epsilon`/`dropout`/`dtype`, backward compatible |
| Public API (`tinymind.Model`) | [x] | works end to end against `EchoBackend`; a real backend now also exists (`TransformerBackend`) |
| Tool system | [x] | schema/registry/validation/permissions/retrieval/executor/planner/builtins |
| Structured-output constraints | [x] (Python-level) | token-level masking still [~] (needs a real tokenizer's vocabulary — a trained subword tokenizer still doesn't exist; see Phase 3A note on tokenizers) |
| `.tm` model file format | [x] | named/versioned directory, bounds-checked; hardened further in Phase 3A (overlap/header-region/integer-overflow checks) |
| Grounding | [x] | string/number/date-level checks |
| Verification | [x] | JSON/math/tool-schema/grounding verifiers |
| Confidence | [x] | component-level; `model` component can now be populated by a real backend, though `TransformerBackend` doesn't currently supply one (no calibrated confidence head trained — see Phase 3A) |
| Routing (`ResponseMode`) | [x] | rule-based; reproduces every example in brief section 9 |
| Memory | [x] | short-term (bounded), long-term (SQLite), retrieval (BM25, always top-k) |
| Data tooling | [x] | validate/deduplicate/split against brief section 22's format |
| Evaluation framework | [x] | `tinymind.evaluation.AcceptanceSuite`, demonstrated in `benchmarks/tools/desk_suite.py` (15/15, 6/6 categories) |
| CLI | [x] | `version/run/tools/model/inspect/benchmark/serve/generate` real; `train/finetune/distill/quantize/convert` honestly refuse with a specific reason |
| HTTP server | [x] | stdlib-only, binds `127.0.0.1` by default |
| Native C ABI header | [x] | `native/include/tinymind.h` |
| Native: tensor/sampler/kv-cache/tokenizer | [x] | real C++, compiled and tested via direct g++ invocation (no `cmake` in this sandbox — see below) |
| Native: model forward pass | [~] | unchanged this phase — see Phase 3A row above |
| Android | [~] | plan + a documented pitfall (Bionic vs. glibc); no JNI/Gradle/`.so` |

## Phase 2 — Python reference runtime

Folded into Phase 1 (see the previous version of this file) —
`tinymind.runtime.Session`/`Engine` (multi-session, unlike Needle's single
global engine handle) are real and tested.

## Phase 4 — Training

[x] Superseded by Phase 3B above for real runs (`tinymind train`, no
`--legacy`); this section's original scope (ordinary supervised causal-LM
training) is what Phase 3B rebuilt after the audit found the Phase 3A
version untrained on its own targets. The Phase 3A trainer
(`CausalLMTrainer`) still exists and runs via `tinymind train --legacy`
for reproducing old results, with a printed warning that it never sees the
assistant response and cannot resume. `CheckpointManager` (generic JSON
state) from Phase 1 remains useful for non-model training state;
`tinymind.model.checkpoint` (Phase 3A, weights-only) and
`tinymind.training.checkpoint` (Phase 3B, full training state) both still
exist — the latter is what "checkpoint" means for a real run now.

## Phase 5 — Distillation

[ ] Not started. `tinymind.distillation.Teacher`/`Filter`/`Scorer` are
interfaces. `VerifierFilter` (real, tested) is the one exception — checking
an already-generated example's arithmetic or schema validity needs neither
a teacher nor a student model. Needs a configured teacher endpoint to do
anything else; a student model now exists (Phase 3A) but nothing generates
teacher demonstrations to distill from it yet.

## Phase 6 — Quantization

[x] Real post-training weight quantization: `tinymind.quantization.
Int8Scheme`/`Int4Scheme` (per-output-row symmetric min-max, standard
round-to-nearest — see `tinymind/quantization/__init__.py`'s docstring for
why this rather than Needle's own Hadamard+Lloyd-Max "Cactus Quants"
technique, needle-analysis.md section 13, which is a training-time (QAT)
method and stays a documented future direction, not this delivery's
choice). `tinymind.quantization.model_quantizer` quantizes every 2D
tensor in a real trained `TinyMindTransformer` (norm weights excluded,
standard practice), exports to `.tm` (scales stored as ordinary sibling
tensors — no format change needed), and produces a real, measured
`QuantizationReport` — every field observed, not guessed. `tinymind
quantize --model X.tm --scheme int8|int4 --output Y.tm` is a real CLI
command now.

**Measured on the same trained model from the Phase 3A validation run**
(`examples/train_tiny_model.py`'s model): int8 gives 3.95x compression
with a mean top-token probability shift of 0.0009 (0.09%) and an unchanged
greedy-decoded generation; int4 gives 7.76x compression with a 0.04 (4%)
shift, generation still unchanged. 25 tests
(`tests/test_quantization.py`, `tests/model/test_quantize_model.py`),
including a mathematically-principled bound check (round-to-nearest error
is always within half a quantization step) rather than only a noisy
mean-relative-error check.

**What this delivery's quantization does and doesn't prove**: the
compression and the measured quality cost are both real. The *runtime*
memory/speed benefit is not realized by the Python reference path — there
is no low-precision GEMM kernel in this delivery (that's a native-runtime
concern, Phase 7, still a stub), so `dequantize_to_model()` reconstructs
float32 weights and runs the ordinary forward pass. A real memory/speed
win needs Phase 7 actually computing in int8/int4, not just storing it
that way.

## Phase 7 — Native runtime

[x] **Done for real.** The native model forward pass (`native/src/model.cpp`)
is now a real, working implementation matching
`docs/architecture/native-model-contract.md` exactly — not a stub. New this
phase: `native/src/json_parser.h` (recursive-descent parser for `.tm`
metadata), `native/src/crc32.h` (IEEE 802.3, verified to match
`zlib.crc32` byte-for-byte), `native/src/tm_reader.h` (the full C++
`.tm` reader with all Python-side hardening checks reproduced: overlap,
header-region, integer-overflow, checksum), `native/src/ops.h` (all
transformer math: `linear()`, `rmsnorm()`, `RopeCache::build()`,
`apply_rope()`, `silu()`, `swiglu_mlp()`, `softmax_inplace()`,
`causal_attention()`), and `native/src/model.h`/`model.cpp` (real
`Model::load()` + `Model::forward()` + `Model::forward_one_token_cached()`),
plus `native/src/runtime.cpp` rewritten with real `tm_generate` and
`tm_embed` using the same model.

**Equivalence verified:**
- Python vs C++ logits: max abs diff **1.9e-6** (well within 1e-3 tolerance) ✓
- Argmax matches at every position ✓
- KV-cache cached-decode == full-sequence in native (max diff = 0.0, exact) ✓
- C ABI `tm_generate` on `"hello"` → `" there hello there h"` matches Python
  generation exactly ✓
- Tested on: GQA model, MQA+untied embeddings model, ByteTokenizer-scale
  model (vocab_size=260)

`tests/model/test_native_equivalence.py` (5 tests: 4 forward-pass
equivalence, 1 C ABI generation match) is the permanent automated harness;
it compiles `native/tests/test_model_equivalence.cpp` with `g++` directly
(no cmake needed to run these tests — still no cmake in this sandbox).

**Compile commands that work in this sandbox:**

```
g++ -std=c++17 -Wall -Wextra -fPIC -Iinclude -Isrc -shared \
  src/runtime.cpp src/model.cpp src/tokenizer.cpp src/sampler.cpp \
  src/kv_cache.cpp src/tensor.cpp -o libtinymind.so   # shared lib: OK

g++ -std=c++17 -Wall -Wextra -Iinclude -Isrc \
  tests/test_native.cpp src/tensor.cpp src/sampler.cpp src/kv_cache.cpp \
  src/tokenizer.cpp src/model.cpp src/runtime.cpp -o tinymind_native_tests
./tinymind_native_tests
# -> test_tensor: OK / test_sampler: OK / test_kv_cache: OK /
#    test_tokenizer: OK / test_c_abi_honest_failure: OK /
#    ALL NATIVE TESTS PASSED
```

Note: `native/CMakeLists.txt` is written against documented CMake behavior
but still unverified as an actual CMake build graph — no cmake is installed
in this sandbox and there's no network access to install one. Run
`cmake -B build -S native && cmake --build build && ctest --test-dir build`
as the first thing once cmake is available.

## Phase 8 — Android

[ ] Not started beyond the plan in `android/README.md`. Blocked on Phase 7.

## Phase 9 — Optimization

[~] `benchmarks/model_baseline.py` (new this phase) establishes a real
prefill/decode latency baseline for the pure-NumPy reference
implementation — see that file's own module docstring for exactly what
the numbers do and don't mean (this environment's CPU-only, single-core,
no-BLAS-acceleration-assumed numbers, not a claim about the eventual
native/quantized/on-device engine). `benchmarks/{capability,reasoning,
structured,mobile,performance}/` still each have a README explaining what
they're waiting on.

## What this delivery is safe to claim, in one paragraph

TinyMind now has a real, gradient-checked, trainable transformer; a
training pipeline whose data, accumulation, checkpoint and resume were
audited, found defective, fixed, and re-verified against the specific
failures found (not just "tests still pass"); real post-training
quantization measured against it; and a working native C++ inference
engine that agrees with the Python reference, now checked on an actual
exported package of the real `tiny_mobile` architecture, not only toy
configs. A reduced-budget demonstration (≈15 minutes of training total,
~14% of the three-stage plan's token budget, chunked across simulated
GitHub Actions jobs with real checkpoint hand-off) took a `tiny_mobile`
model (1,214,592 parameters) through all three curriculum stages: held-out
validation loss fell from 6.8 to 0.24 nats/token in stage 1; the model
learned to select the *correct tool* on held-out prompts 84% of the time
by stage 3, but copying the right *argument values* (a city, a number) was
learned only 3.8% of the time — a genuine, measured shortfall, traced to a
small values-pool letting the model memorise rather than copy, and one
that caused stage 2 to fail its own promotion gate (false-positive
tool-call rate 0.341 against a 0.30 threshold). That failure is reported,
not hidden, and is exactly the kind of thing a promotion gate exists to
catch. What this delivery does not have: a run of the GitHub Actions
workflow on an actual runner (no network in this sandbox — the workflow
and the artifact-hand-off logic it calls are tested statically and via an
offline multi-process simulation instead); a production-budget run of the
three-stage plan (the sandbox's 1 vCPU was used for short demonstrations
and short measurement sweeps, not the full plan); multi-thread BLAS
scaling data (1 vCPU visible here); a native INT8 kernel (INT8 exists as a
measured storage/quality trade-off with a Python reference loader only);
or open-domain usefulness (the curriculum is template-generated, so its
numbers show pipeline correctness and held-out generalisation of narrow
skills, not general capability). Nothing claims otherwise. The next
concrete steps are: fix the argument-copying gap the demonstration found
(widen the value pools — see `docs/training/three-stage-plan.md`'s closing
section) and re-run stage 2; then run the real workflow on a GitHub
runner at the plan's budget.
