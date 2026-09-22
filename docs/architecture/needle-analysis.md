# Needle 2 — Architecture Analysis

Internal document. Produced by inspecting the full `needle-main.zip` source tree
(not just the README) before any TinyMind implementation began. Needle 2 is
Apache-2.0 licensed, built by Cactus Compute. This document exists to extract
architectural lessons, not to describe an implementation we intend to copy.

Scope actually inspected: `needle/__init__.py`, `needle/_worker.py`,
`needle/_telemetry.py`, `needle/agent/{tools,fetch}.py`, `needle/model/*.py`,
`needle/environments/*.py`, `needle/playground/server.py`, `needle/cli.py`,
`doc/apis.md`, `doc/finetuning.md`, `doc/environments.md`, `tests/*`,
`pyproject.toml`, `requirements*.txt`, `LICENSE`, `.github/workflows/*.yaml`.
~6,200 lines of Python total — small enough to read in full, which we did.

## 1. Package architecture

`cactus-needle` ships as a thin Python package with one required runtime
dependency (`huggingface_hub`). The actual neural engine is **not** in this
repository: it is a closed, precompiled native binary (`libneedle.{so,dylib,dll}`)
fetched from a Hugging Face model repo on first use and cached at
`~/.cache/cactus-needle/v<gen>/<version>/`. Training-only dependencies
(`jax`, `flax`, `optax`, `sentencepiece`) live behind a `[train]` extra and are
imported lazily, so `import needle` for inference-only use stays lightweight.
Two engine "generations" (2 and 3) are supported side by side, each with its
own repo, version, and cached binary.

**Lesson:** separating "the package you `pip install`" from "the engine binary
you run" lets the runtime stay tiny and lets engine updates ship independently
of the Python API. The cost is that the interesting systems work (the actual
inference kernel, the grammar compiler) is invisible to anyone reading this
source tree — it's a black box behind a C ABI.

## 2. Model architecture

Needle 2 is a **Simple Attention Network (SAN)**, a dense ~45M-parameter
recipe, not a stock Transformer:

- GQA attention (`num_heads` / `num_kv_heads` split) with RoPE.
- A **Hadamard MLP** in place of a standard FFN: a fixed (weightless) Walsh–
  Hadamard transform mixes channels in *n log n* time, with a small number of
  learned per-channel gates riding on top.
- **Engram**: hashed n-gram key/value tables read at specific layers
  (`engram_layers`), giving the model a sparse associative-memory path that
  doesn't go through attention at all.
- **Multi-lane hyper-connections (MHC)**: instead of one residual stream, the
  model carries several parallel lanes, combined by a doubly-stochastic
  routing matrix computed via Sinkhorn iteration.
- Sandwich-normed, gated residuals (`ZCRMSNorm`, a zero-centered RMSNorm
  variant) around both attention and MLP.
- Two small heads sit on top of the shared trunk: a **ContrastiveHead** (tool
  retrieval embeddings) and a **ConfidenceHead** (calibrated accept/escalate
  score), both reading pooled hidden "cells" rather than the full sequence.

**Lesson:** most of the architecture's novelty is in cheap, mostly-weightless
mixing (Hadamard transform, hashed engram lookups) rather than more attention
layers. That's a legitimate way to buy capability at fixed parameter count,
but it is also research-grade and comes with real ablation risk — see
Weaknesses.

## 3. Tokenizer

`SANTokenizer` wraps a SentencePiece model, auto-downloaded from HF if not
present locally. It exposes `encode`/`decode`, `pad_token_id`/`eos_token_id`/
`bos_token_id`/`vocab_size`, and a `__call__` shaped like a HF tokenizer
(`truncation=`, `max_length=`). Non-English text is measurably more expensive:
the fine-tuning doc reports ~1.7x more tokens for Spanish, which taxes both
quality and the 256-token context window.

## 4. Inference path

Python → `ctypes.CDLL` → C ABI (`needle_init`, `needle_load`, `needle_complete`,
`needle_embed`, `needle_reset`) → native engine. Requests and responses cross
the boundary as UTF-8 JSON: the caller passes a fixed-size output buffer,
the native side writes a JSON envelope into it, and a negative return code
plus buffer contents signals an error. There is no streaming across this
boundary — one call, one completed JSON object back.

## 5. Decoding

The *training-side* decode path (`model/decode.py`, used for local
inference/eval, not the shipped engine) is KV-cached autoregressive
generation compiled with JAX: a `lax.scan`-based "rollout" function runs the
per-token loop under `jit`, with causal and packing masks, an optional
sliding KV window, and temperature sampling. This is a completely separate
code path from the native `.cact` engine — it exists for training-time
evaluation, not for the shipped product.

## 6. Grammar / schema constraints

Every declared tool's JSON Schema is compiled into a byte-level grammar at
`needle_init` time, inside the closed native engine. The Python layer never
sees the grammar — it only ever sees the guarantee that `needle_complete`'s
output is schema-valid JSON. This is the single biggest "trust the black box"
point in the whole system: the constraint engine is not something a reader of
this source tree can inspect, test, or reuse.

## 7. Tool calling

Tool calling isn't one mode among several — it is the *entire* response
contract. Every turn returns exactly one JSON envelope with a `type` field
(`"call"` or `"respond"`), a `function_calls` list, and a short unconstrained
`reasoning` string deriving each argument from its source span. A request no
declared tool can serve returns the **empty call `[]`** — there is, by
design, no free-text fallback.

## 8. Tool retrieval

A built-in `ContrastiveHead` embeds every declared tool schema once at
`Needle(...)` construction and the query once per turn; above five tools,
only the top-5 by embedding score enter the grammar for that turn — an
unselected tool is *unreachable*, not merely deprioritized. `tool_index_path`
persists the embeddings to disk keyed by a fingerprint over the schema set
and model version, so a matching fingerprint skips re-embedding.

## 9. Confidence

`confidence` is `min(calibrated_head_score, decode_probability_of_call_tokens)`
— an explicit AND of two independent signals, so a confident-sounding but
low-probability call, or vice versa, still gets a low score. Calibration is
scoped tightly: it holds for the shipped **base** model only. Fine-tuning
updates the trunk but not the confidence head, so a `Needle(weights=...)`
agent reports `confidence: None` and warns once at construction — a real,
documented gap the project is honest about rather than papering over
(the finetuning doc also flags non-English confidence as unreliable even on
the base model).

## 10. Structured extraction

Not a separate feature. `extract()` builds a one-tool `Needle` agent where the
"tool" is the target schema, forces the call, and returns `arguments` (or a
constructed Pydantic instance). Because the grammar admits exactly one call
shape with one tool declared, schema conformance is a consequence of the tool
system rather than a bolt-on validator.

## 11. Fine-tuning

LoRA on the frozen base, rank 16 / alpha 32 by default, targeting the five
attention projections in every layer. Training is plain JAX/Flax/Optax
(CPU/CUDA/Metal), quantization-*aware* by default (`--qat-bits auto` trains
through the checkpoint's declared Cactus-Quants scheme so the adapter matches
what `build` will export). Optional synthetic-data generation/augmentation
goes through an OpenRouter-compatible chat endpoint with a worker pool. A
held-out validation split (10% default) reports loss every epoch. The docs
are unusually candid about failure modes: step-count math on small datasets
(200 examples at batch 16 is 13 steps/epoch — three default epochs barely
moves a rank-16 adapter), a historical NaN-on-CPU bug, and an old
confidence-gating bug that made tuned models refuse everything.

## 12. LoRA

`lora_target_paths` walks the parameter tree for the attention projection
leaves, `init_lora` builds low-rank A/B pairs, `merge_lora` folds them back
into the base weights **at export time**. There is no adapter-swap-at-runtime
story here — a tuned model is a fully merged, fully quantized `.cact`, not a
base model plus a hot-loadable adapter file.

## 13. Quantization

"Cactus Quants" (CQ): weights are rotated onto a fixed Hadamard basis, then
quantized to 2/3/4-bit against a **shared, precomputed Lloyd-Max codebook per
bit-width** (`cb2`/`cb3`/`cb4`), so mixed-precision export needs only one
small codebook blob rather than per-tensor scale/zero-point pairs.
Quantization-aware training uses a straight-through estimator
(`cq_ste_params`), and a `bits_map` lets different layers/tensors ship at
different bit-widths in one archive. Activations and the KV-cache default to
int8. This is meaningfully more sophisticated than a naive per-tensor
min-max INT4 scheme.

## 14. Model export

`export.py` performs the actual packing: tensors are extracted from the
parameter pytree, quantized per the (possibly mixed) bit-width map, pre-
**transposed** to `[out, in]` so each output row is contiguous along the
reduction axis a GEMV/GEMM kernel streams over, and laid out **layer-major**
(embedding, then every tensor of layer 0, layer 1, ..., final norm) so a
kernel's working set for one layer is one contiguous region.

## 15. The `.cact` model format

A single self-contained binary: a **fixed 120-byte header** (30 packed
`u32`/`f32` fields) that encodes the *entire* architecture geometry
(vocab, d_model, heads, layer count, engram geometry, Hadamard size, MHC
lanes, RoPE theta, KV window/bit-width, ...), followed by the concatenated
quantization codebooks, then a **nameless / positional** tensor directory
(dtype, shape, offset, size, group, bits — but no tensor *name*; position in
a fixed canonical order is the identity), then 64-byte-aligned tensor blobs.
The tokenizer itself is embedded as one more (raw) tensor. Because the header
carries full geometry, **one engine binary can load any configuration of the
architecture family** — nothing about layer count or width is hard-coded
into the loader.

A 4-byte magic tag at the front of the file (`_weight_generation` in
`needle/__init__.py`) dispatches Needle-2 vs Needle-3 archives to their
respective engines; an unrecognized tag raises immediately rather than being
fed to the wrong engine.

## 16. Native engine interaction

The engine is a **closed binary**, not source in this repository. Python
loads it via `ctypes.CDLL`, sets `argtypes`/`restype` explicitly for five
functions, and calls through. Per-platform builds (`agent/fetch.py`) cover an
unusually wide matrix — macOS/Linux/Windows arm64+x86, several Android ABIs,
iOS/tvOS/watchOS, WebAssembly (including a signed WASI/WIT **component**
published to GHCR with Sigstore keyless signing) — all fetched from the same
HF repo and dispatched by a platform tag string.

## 17. Worker process design

There is exactly **one process-global native engine handle per generation**
(`_lib_handles`, `_active` — module-level dicts in `needle/__init__.py`), and
only one `Needle` instance per generation may be "active" (bound into the
engine) at a time; binding a second base-model instance silently displaces
the first. Needle's actual answer to this limitation is **not** to fix the
global state — it's to route every *tuned*-weights agent through a dedicated
child process (`FineTuneWorker` in `_worker.py`) that loads its own copy of
the engine and speaks a length-prefixed JSON protocol over stdin/stdout. That
gives each tuned agent an independent engine/KV-cache/conversation, at the
cost of one subprocess (and its full memory footprint) per active tuned
agent. Multiple *base*-model sessions are not independently supported at all.

## 18. Android deployment

There is no Android-specific code in this repository — no JNI, no Gradle, no
CMake. Android is just two more entries (`android-arm64`, `android-armv7`,
`android-riscv64`) in the same platform-tag table used for every other OS,
fetched the same way as a desktop build. This works, but it means "Android
support" has never had to solve anything Android-specific (permissions,
`.aab` packaging, thermal/battery integration, a JNI surface) inside this
source tree — that work, if it exists, lives entirely in the closed engine
and whatever app wraps it.

## 19. HTTP server

`needle playground` starts a stdlib-only `http.server.ThreadingHTTPServer` on
`127.0.0.1:7860` by default, serving a small static UI plus JSON endpoints
for completion and a background-thread "fine-tune on these tools" flow that
runs the full generate→train→export pipeline and streams log lines back to
the browser. It is explicitly a local dev/demo surface, not a documented
production API (no auth, no OpenAI-compatible route, no concurrency
story beyond one lock around one `Engine` object).

## 20. CLI

`argparse`-based, one subparser per verb (`run`, `finetune`, `generate-data`,
`build`, `download`, `fetch`, `playground`). Notably, real engineering effort
went into **suppressing noisy native/XLA log spam** at the file-descriptor
level (a background thread reads a piped copy of fd 2, drops matched
`LOG(ERROR)` blocks from the Triton autotuner, and forwards everything else)
— a sign this project has actually been run enough in anger to hit that
problem and fix it properly rather than telling users to redirect stderr.

## 21. Testing strategy

Plain `pytest`, tests at the repo root (not inside the package). Anything
that needs the fetched native engine is behind a `skipif`
(`conftest.py:_engine_available`) so CI and offline contributors degrade
gracefully instead of failing on missing network access. A `tiny_checkpoint`
fixture builds a genuinely tiny **real** 2-layer, d_model=64 model via
`SimpleAttentionNetwork.init(...)` for fast, deterministic tests — not a
mock. Separately, `needle/environments/` ships six hand-curated tool
surfaces (smart home, media player, productivity, wearable, kitchen
appliance, data capture), each with a **frozen 32-case acceptance suite**
across six categories (`positive`, `missing`, `irrelevant`, `negation`,
`invalid`, `parallel`), with `missing`/`negation`/`invalid` cases marked
`critical` so a single critical regression fails the suite regardless of the
aggregate pass rate. This is a real system-benchmark layer, distinct from
unit tests, and it's a genuinely good design.

## 22. Strengths

- Tool calling as a first-class, unavoidable response contract (not
  something bolted onto a chat model) — everything downstream (grounding,
  confidence, retrieval) is simpler because of this choice.
- Grounding is enforced, not advisory: `run()` in strict mode will not
  execute a call with an ungrounded argument; it feeds back an error instead.
- Confidence is an explicit two-signal AND, and the project is honest in its
  own docs about exactly when it stops being valid (fine-tuned weights,
  non-English input).
- The `.cact` format's geometry-in-header design means one engine binary
  handles the whole architecture family, not one binary per shape.
- The frozen, categorized acceptance suites in `environments/` are a strong
  benchmark pattern: they separate "does it pick the right tool" from "does
  it correctly refuse", and gate on critical-category regressions.
- Honest, specific documentation of failure modes (loss-curve reading,
  step-count arithmetic on small datasets, a historical NaN bug, non-English
  confidence) rather than only describing the happy path.
- Real engineering polish in unglamorous places: XLA log filtering, offline
  install instructions, `HF_HUB_OFFLINE` guidance for air-gapped devices.

## 23. Weaknesses

- **No free-text response mode.** Every input either matches a declared tool
  or returns `[]`. "What's the capital of France?" with no matching tool is
  a refusal, not an answer — a real product limitation the brief's own field
  testing already ran into.
- **Global, singleton native engine state.** Only one base-model `Needle`
  instance can be bound per process; a second silently displaces the first.
  The fix for tuned models (a subprocess per agent) is a workaround, not a
  solution, and doesn't extend to running two base-model sessions at once.
- **The grammar/constraint engine is entirely opaque.** It's the single
  component every other guarantee (schema validity, retrieval-narrowed
  grammar) rests on, and it's compiled C++ we never see.
- **The `.cact` tensor directory is positional/nameless.** Robust for one
  frozen architecture family; fragile the moment the architecture needs to
  add, reorder, or optionally omit a tensor — there's no name to key on
  during a migration.
- **LoRA adapters are merge-only.** No hot-swappable adapter story; every
  tuned model is a fully separate exported archive, which is heavier for a
  product that wants many small task-specific adapters over one base.
- **Confidence calibration is narrow and silently degrades.** Non-English
  input and any fine-tuning both invalidate it, and a consumer who doesn't
  read the fine-tuning doc closely could easily ship on an uncalibrated
  score without knowing it.
- **No deterministic-tool offload.** Arithmetic, date math, and unit
  conversion all go through the neural model as a "tool call" the model
  itself must still get right end-to-end — there's no built-in guaranteed-
  correct executor for the common deterministic cases.
- **Everything-is-a-tool-call forces awkward shapes onto plain questions.**
  Because there is no other output type, even "acknowledge and continue"
  turns have to be modeled as calls or empty calls.
- **Telemetry defaults to on.** Anonymous and disclosed, but opt-out rather
  than opt-in, and it happens on package import via a background thread.

## 24. Architectural opportunities for TinyMind

- Add a genuine non-tool response mode (`CHAT`/`REASON`) instead of forcing
  every turn through the tool-call/empty-call binary — see brief §7.
  Reference: point 7 and 23 above.
- Make the constraint/grammar engine a first-class, inspectable Python-level
  component (even if a compiled backend is swapped in later) instead of a
  black box behind a C ABI — see §6.
  Reference: point 6 and 23 above.
- Design a session/runtime object model where state (engine handle, KV
  cache, tool registry, conversation) is owned per-session from the start,
  so running N sessions concurrently is the normal case, not something that
  needs a subprocess workaround — see §17.
- Prefer a **named**, versioned tensor directory in the model file format
  (still fixed-header, still one binary per architecture family) so an
  optional or renamed tensor doesn't break every archive ever exported — see
  §15.
- Build a real deterministic-tool layer (calculator, date/unit conversion,
  JSON/schema validation) that the router can prefer over asking the neural
  model to "be" a calculator — see §23.
- Keep confidence, but make its scope and validity machine-checkable (e.g.
  a calibration-domain tag on the model file) rather than something a
  consumer has to remember from the docs — see §9 and §23.
- Keep the frozen, categorized acceptance-suite pattern from
  `environments/` — it is one of the best ideas in this codebase — but make
  it a core library feature (`tinymind.evaluation`) rather than a per-example
  convention repeated six times.
- Telemetry off by default, explicit opt-in, per the brief's own §35 — this
  is a values choice, not a technical one, and Needle's own choice (on by
  default) is one we deliberately do not carry over.
