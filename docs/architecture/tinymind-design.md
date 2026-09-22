# TinyMind — Design Document

Companion to `needle-analysis.md`. Where that document records what Needle 2
does and why, this one records what TinyMind does instead, and why —
concept by concept, with an explicit note on what is real, running code in
this delivery versus what is designed but not yet built.

TinyMind is a new, independently-implemented project. Ideas are inspired by
Needle where the analysis above found something worth keeping; no Needle
code, branding, or naming is reused. See `docs/legal/licensing.md`.

## 0. What "done" means for this delivery

The master brief that scoped this project is explicit that the first
deliverable should be an architecture review and a Phase 1 skeleton, not a
finished system — model training, a compiled native engine, and an Android
build are multi-week efforts in their own right and are *designed* here, not
faked. Concretely, in this delivery:

- **Real, tested, runnable Python**: configuration system, tool subsystem
  (registry/schema/validation/permissions/retrieval/executor), JSON-Schema
  structured-output constraints, a rule-based router and confidence
  estimator, in-memory + SQLite memory stores, dataset validation, and a CLI
  that runs all of the above end to end against a small deterministic
  `EchoBackend` model stand-in.
- **Designed, documented, stubbed with clear `NotImplementedError`s, not
  claimed as working**: the trained neural model itself, the native C++
  engine, the Android build, quantization, distillation, and the training
  pipeline. Each has a real interface and, where useful, a header or config
  schema, but no fabricated benchmark numbers or "it runs on-device" claims.

`STATUS.md` at the repository root tracks this split per-subsystem so it
never has to be reconstructed from memory later.

## 1. Response modes: the main departure from Needle

Needle's biggest limitation (analysis §7, §23) is that every turn is either a
tool call or the empty call `[]]` — there is no way to just answer a
question. TinyMind's runtime defines an explicit `ResponseMode` enum —
`CHAT`, `TOOL_CALL`, `STRUCTURED_OUTPUT`, `PLAN`, `REASON`, `REFUSE`,
`ASK_CLARIFICATION` — and a `ComputePolicy`/`Router` decide which mode and
how much computation a request gets *before* generation, not as a side
effect of what tools happen to be declared. "What's the capital of France?"
with a calculator tool declared should get `CHAT`, not a forced tool
call — this is a behavioral requirement, not just an API nicety, and it's
implemented and tested in this delivery (`tinymind/routing/router.py`) using
deterministic rules; a learned router is future work, tracked in `STATUS.md`.

## 2. Tools: keep the shape, replace the implementation

Needle's schema-from-function-signature approach (analysis §3 of the brief's
own tool section, and needle-analysis.md §11 "Fine-tuning" data format) is
worth keeping — inferring JSON Schema from type hints, docstrings, and
`Annotated[...]` constraints is genuinely convenient. TinyMind reimplements
this independently in `tinymind/tools/schema.py` with its own `Field` type
and its own doc-parsing, and adds what Needle's Python layer doesn't expose:

- **Permissions** (`tinymind/tools/permissions.py`): every registered tool
  declares a capability tag — `READ_ONLY`, `LOCAL_WRITE`, `NETWORK`,
  `SENSITIVE`, `DESTRUCTIVE` — and `DESTRUCTIVE`/`SENSITIVE` tools require an
  explicit `requires_confirmation=True` acknowledged by the caller before
  `executor.py` will run them. The model never gets a path to arbitrary code
  execution — the executor only ever calls a registered, schema-validated
  tool function.
- **Validation** (`validation.py`) is a real, standalone JSON-Schema-subset
  validator (type, required, enum, pattern, min/max, items, nested objects,
  `additionalProperties`) that does not depend on a neural grammar existing
  at all — it can validate any dict against any schema, which is what makes
  it reusable for structured output generally, not just tool arguments.
- **Retrieval** (`retrieval.py`) starts where Needle's does (analysis §8) —
  narrow a large tool catalogue down before it reaches the model — but the
  first implementation here is lexical (token-overlap / BM25-style scoring)
  rather than a learned embedding head, because there is no trained model in
  this delivery to produce embeddings from. The `ToolRetriever` interface is
  designed so a learned-embedding implementation is a drop-in replacement
  later, not a rewrite.
- **Grounding** stays a first-class, separate concern rather than folding
  into confidence: `tinymind/runtime/grounding.py` checks whether a
  generated argument's value actually appears (verbatim or as a recognized
  transformation) in the source text, in the spirit of Needle's temporal
  grounding check (needle-analysis.md §9) but generalized to any field, not
  only dates.

## 3. Structured output as a core, model-independent feature

Needle's constrained decoding lives entirely inside its closed native
engine (needle-analysis.md §6) — a real strength for guarantees, a real
weakness for anyone who wants to understand or reuse it. TinyMind's
constraint engine (`tinymind/runtime/constraints/`) is Python, tested, and
usable with **no model at all**: `json_schema.py` validates or repairs a
candidate JSON object against a schema; `state_machine.py` defines the
token-acceptance state machine interface a real constrained decoder would
drive; `tokenizer_constraints.py` is the seam where a real tokenizer's
vocabulary would be compiled into per-step allowed-token masks once a
trained model exists. This delivery ships the schema validator and the state
machine fully working and tested against hand-built and tool-generated
schemas; the tokenizer-level integration is an interface with no
implementation yet, because it has nothing to constrain without a trained
model and tokenizer.

## 4. Confidence: keep the shape, make components explicit

Needle's confidence is `min(head_score, decode_probability)` — clean, but
reported as a single opaque number a consumer has to know the caveats for
from documentation (needle-analysis.md §9, §23). TinyMind's
`tinymind/confidence/` returns the components, not just the minimum:

```json
{
  "confidence": 0.82,
  "confidence_components": {
    "model": null,
    "tool_selection": 0.94,
    "arguments": 0.89,
    "grounding": 1.0,
    "verification": null
  },
  "valid_for": ["tool_selection", "arguments", "grounding"]
}
```

`valid_for` names which components are actually meaningful for *this*
response — mirroring the real, documented gap Needle has around fine-tuned
weights and non-English input (needle-analysis.md §9), but making it a field
the caller can branch on instead of a fact they have to remember. The
`model` component (a calibrated head's score) and `verification` component
(a second model/tool pass) are `null` in this delivery because there is no
trained model — everything currently populated (`tool_selection`,
`arguments`, `grounding`) is computed deterministically from the tool
registry, the validator, and the grounding checker, and is real, not a
placeholder.

## 5. Hybrid reasoning and the router

Needle asks the neural model to *be* the calculator (needle-analysis.md
§23). TinyMind's router (`tinymind/routing/router.py`) inspects a request
and decides a `ComputeLevel` (`FAST`/`NORMAL`/`DEEP`/`VERIFY`/`ESCALATE`)
and, independently, whether a deterministic tool can serve it before any
generation happens. A small built-in deterministic tool set
(`tinymind/tools/builtins.py`) — calculator, unit conversion, date
arithmetic — is registered by default specifically so arithmetic never has
to be a hallucination risk. This is real, tested code: `router.py` and
`builtins.py` do not depend on a trained model.

## 6. Memory

`tinymind/memory/` separates `short_term.py` (bounded conversation buffer),
`long_term.py` (a small SQLite-backed store, since the stdlib gives us that
for free and it survives a process restart), and `retrieval.py` (lexical
relevance scoring over stored entries — same rationale and same
future-upgrade seam as tool retrieval, §2 above). Every stored entry carries
`content`, `timestamp`, `source`, `confidence`, and `scope`, and retrieval is
always top-k, never "inject everything" — this was an explicit brief
requirement and it is enforced in code (`retrieval.py` has no "return all"
path).

## 7. Model / runtime separation

`ModelBackend` (`tinymind/model/backend.py`) is an abstract interface:
`load`, `generate`, `stream`, `reset`, `embed`. This delivery ships exactly
one concrete backend, `EchoBackend` (`tinymind/model/backends/echo.py`),
which is explicitly and loudly documented as a deterministic stand-in for
testing the runtime, tool loop, router, and CLI end to end — it does not
generate language-model output and must never be described as "the
TinyMind model." Its purpose is to let every other subsystem be exercised by
real tests without needing a trained checkpoint, exactly the same role
Needle's own `tiny_checkpoint` pytest fixture plays for its test suite
(needle-analysis.md §21) — except Needle could build that fixture because it
already has a working `SimpleAttentionNetwork`; TinyMind does not yet, so the
stand-in lives at the `ModelBackend` boundary instead. A real
`TinyMindTransformer` backend is future work (Phase 3 of the brief) and is
not implemented here.

## 8. Configuration and model sizing

`tinymind/model/config.py` defines a `ModelConfig` dataclass
(`hidden_size`, `num_layers`, `num_heads`, `intermediate_size`,
`max_seq_len`, `vocab_size`, plus explicit fields for the architectural
experiments called out in the brief — GQA head count, RoPE theta, norm type,
MLP type — each with a safe default and none hard-coded into any consumer).
`configs/*.yaml` gives named presets (`50m.yaml` through `1b.yaml`) that only
set the handful of fields that actually change size; nothing in the loader
assumes a specific size.

## 9. Model file format: `.tm`

A new format, not a reproduction of `.cact`. Kept: the single fixed-size
binary header carrying full architecture geometry, so one loader handles any
size in the config family (needle-analysis.md §15, a genuine strength).
Changed: the tensor directory is **named and versioned**
(`tinymind/runtime/format.py`: magic `b"TM01"`, then a length-prefixed JSON
metadata block, then a directory of `{name, dtype, shape, offset, size,
checksum}` records, then aligned tensor blobs) — trading a few hundred bytes
of directory size for the ability to add, rename, or omit a tensor across
format versions without breaking every previously-exported file, which was
flagged as a concrete fragility in the Needle format (needle-analysis.md
§23). All integer fields are bounds-checked against the actual file size
before any tensor is read — no `assert`, explicit exceptions (`ModelFormatError`
and subclasses) per the brief's §31 and §54. This delivery implements and
tests the header/directory reader and writer against synthetic (non-neural)
tensors; writing real trained weights is naturally gated on a real model
existing.

## 10. Native runtime and Android

Designed, not built. `native/include/tinymind.h` specifies the same shape of
C ABI Needle uses (opaque context handle, `tm_create`/`tm_load_model`/
`tm_generate`/`tm_reset`/`tm_free`) because that shape is a genuinely good
fit for a stable cross-language boundary, independently arrived at and
independently named. `native/CMakeLists.txt` and the `.cpp` files under
`native/src/` are real, compilable-shaped stubs that raise "not implemented"
at the one function each currently calls — they exist so the build system
and header contract are exercised now, not to claim a working inference
engine. `android/README.md` documents the intended ABI matrix
(`arm64-v8a` primary) and explicitly calls out the Bionic-vs-glibc pitfall
this project's own testing against Needle surfaced — that a Linux ARM64
binary is not an Android ARM64 binary — as a named failure mode to guard
against in CI once there is a real build (`android/README.md`, "Known
pitfall"). No `.so`, no `.apk`, no claim of an Android build in this
delivery: there is no trained model or compiled engine yet for one to run.

## 11. Quantization, distillation, training

All three get real module skeletons (`tinymind/quantization/`,
`tinymind/distillation/`, `tinymind/training/`) with function/class
signatures matching the brief's §20/§21/§23, docstrings describing intended
behavior, and `NotImplementedError` bodies — deliberately *not* padded out
with fake logic that would pass a shallow read as "implemented." Needle's
own quantization (Cactus Quants: Hadamard rotation + shared Lloyd-Max
codebooks, needle-analysis.md §13) is a legitimate technique worth
reimplementing independently once there are real weights to quantize; that
work is scoped for Phase 6, after Phase 3–4 produce something to quantize.

## 12. Security posture carried over from day one

Never `eval`/`exec`/`shell=True` on model- or tool-generated content
(enforced structurally: `executor.py` only ever calls a function object
already present in the registry, never a string). The CLI's `serve` command
binds `127.0.0.1` unless `--host` is passed explicitly. Model-format loading
is bounds-checked (§9 above). Telemetry does not exist in this codebase at
all, and if it is ever added it defaults off — this is a deliberate
departure from Needle's opt-out default (needle-analysis.md §23), not an
oversight.

## 13. Repository layout

See the repository root for the actual tree; it follows the brief's §42
structure with one deliberate deviation — directories with no real content
in this delivery (`native/tests/`, `distillation/`, etc.) are still created
because they're referenced by path elsewhere in the docs and configs, but
each contains at minimum a real `__init__.py` or `README.md` stating its
Phase-1 status rather than being silently empty, per the brief's own
instruction not to scatter empty placeholder files without saying so.
