# Model implementation — Phase 3A

Companion to `docs/architecture/tinymind-design.md` (Phase 1) and
`docs/architecture/native-model-contract.md` (the Python/native numerical
contract, split out separately per the brief's own request). This document
covers what changed to move from "runtime prototype" to "a real, trained,
gradient-checked transformer": the architecture itself, why it's built the
way it is, and the one big environmental constraint that shaped every
decision below.

## 0. The environment constraint that shapes everything here

Verified directly before writing any code: this sandbox has NumPy and SciPy
installed, but no PyTorch, JAX, or TensorFlow, and no network access to
install one (`pip install torch` reports "No matching distribution found").
No GPU is present. `nproc` reports one CPU core.

The brief this phase was built from assumes a framework provides
tensor autodiff. It doesn't exist here, so `tinymind/model/tensor.py`
implements a minimal reverse-mode autodiff engine directly on NumPy arrays
— see that file's module docstring for the full reasoning. Every operation
in it is checked against finite-difference numerical gradients
(`tests/model/test_tensor.py`), which is the actual reason this phase can
honestly claim "gradients flow through the entire model" rather than
asserting it from the formulas looking right.

**Consequence for scale**: the architecture (below) is fully
configuration-driven and instantiates correctly at every size from 50M to
1B (`tests/model/test_model.py::test_larger_config_has_more_parameters`),
but *training* at those sizes on single-threaded, GPU-less NumPy is not
practical — a matmul-heavy forward+backward pass over a 1B-parameter model
would take an impractically long time per step in this environment. The
synthetic-overfit acceptance test (`tests/training/test_training.py`,
brief section 20) deliberately runs at a genuinely tiny scale
(hidden_size 16-32, 1-2 layers) specifically to prove *correctness* of the
gradient flow, which is a scale-independent property — not to demonstrate
practical training throughput at the target parameter range. See
`STATUS.md` for exactly what is and isn't claimed as a result of this.

## 1. Chosen architecture

Decoder-only causal transformer, Llama/GPT-NeoX-family shape:

```
token ids
  -> embedding lookup                         tinymind/model/model.py
  -> N x [ RMSNorm -> GQA attention (RoPE) -> +residual
           RMSNorm -> SwiGLU MLP           -> +residual ]   tinymind/model/layers.py
  -> final RMSNorm
  -> LM head (tied or untied)
  -> logits [B, T, vocab_size]
```

No softmax inside `forward()` — logits are returned raw (`tests/model/
test_model.py::test_no_softmax_applied_logits_not_bounded_0_1` checks this
isn't accidentally true anyway). No tool/runtime logic anywhere in
`tinymind/model/` — `tests/model/test_block.py::
test_no_tool_or_runtime_logic_imported` checks this by grepping the actual
source for `tinymind.tools`/`tinymind.runtime` imports.

## 2. Configuration

Extended the existing `tinymind.model.config.ModelConfig`
(`docs/architecture/tinymind-design.md` section 8) rather than
duplicating it, per the brief's own instruction to reuse existing
interfaces. Added: `norm_epsilon` (float, default `1e-6`), `dropout`
(float, default `0.0` — accepted for forward compatibility; not yet wired
into any layer's forward pass, since a 20-50M-parameter model trained on a
handful of synthetic examples has no overfitting problem for dropout to
solve — wiring it in is a one-line change to `SwiGLUMLP`/attention's
forward when a real dataset makes it relevant), and `dtype` (string,
default `"float32"`, validated against a fixed set — the autograd engine
only actually computes in float32; see section 0 above and
`tinymind/model/config.py`'s field docstring).

The brief's suggested field names (`num_attention_heads`,
`max_position_embeddings`, `tie_word_embeddings`) map onto the existing
`num_heads`, `max_seq_len`, `tie_embeddings` — kept as-is rather than
renamed, per "preserve the public API unless there is a compelling
compatibility reason to change it": renaming would have broken all 153
Phase 1 tests for a purely cosmetic gain.

## 3. Tensor shapes

| Tensor | Shape |
|---|---|
| `input_ids` | `[B, T]` |
| token embedding table | `[vocab_size, hidden_size]` |
| hidden states (between blocks) | `[B, T, hidden_size]` |
| Q (per layer) | `[B, num_heads, T, head_dim]` |
| K, V (per layer) | `[B, num_kv_heads, T, head_dim]`, repeated to `num_heads` for GQA |
| attention scores | `[B, num_heads, T, T_kv]` |
| attention output (pre-`o_proj`) | `[B, T, num_heads * head_dim]` |
| MLP gate/up projections | `[B, T, intermediate_size]` |
| logits | `[B, T, vocab_size]` |

`head_dim = hidden_size // num_heads` (`ModelConfig.head_dim`, unchanged
from Phase 1).

## 4. Parameter-count calculation

`TinyMindTransformer.count_parameters()` (and `Module.count_parameters()`
generally, in `tinymind/model/module.py`) sums `.data.size` over every
registered parameter — an exact count from the actually-instantiated
model, not the `ModelConfig.approx_param_count` estimate Phase 1 shipped
(which stays useful for sizing a preset before instantiating anything, and
is now clearly labeled "approx" in the CLI's `model info` output for a
config path versus "exact" for a real `.tm` file — see `tinymind/cli.py`).

Verified stable and monotonic in the config's size knobs
(`tests/model/test_model.py::TestParameterCounting`).

## 5. Initialization

Every `Linear` (`tinymind/model/linear.py`) initializes its weight from
`Normal(0, 1/sqrt(in_features))` — the standard "scaled" initialization
(GPT-2/nanoGPT convention): keeps the variance of a matmul's output roughly
constant across layers of different width at initialization, without
needing a fan-in/fan-out-aware scheme like Xavier/He (which are designed
around activation functions this architecture doesn't use in the same way).
The token embedding table uses the same `1/sqrt(hidden_size)` scale.
`RMSNorm`'s weight initializes to all-ones (the identity scale — brief's
own implicit expectation, matched here explicitly).

A `seed` argument threads through `TinyMindTransformer.__init__` into one
shared `numpy.random.Generator`, passed down to every sub-layer — not
`numpy.random.default_rng()` called freshly per layer (which would each
seed from OS entropy and make the whole model's initialization
non-reproducible even with the same top-level seed). See
`tests/model/test_model.py::test_deterministic_given_seed`.

## 6. Attention design

`tinymind/model/attention.py`. Q/K/V via three `Linear` projections
(no bias, matching current Llama-family convention), reshaped into heads,
RoPE applied to Q and K (not V), GQA via `_repeat_kv` (a no-op when
`num_kv_heads == num_heads`, so MHA/GQA/MQA are one code path — see
`ModelConfig.num_kv_heads`'s docstring, unchanged from Phase 1). Causal
masking is an additive mask (`-1e9`, not `-inf`, avoiding a `0 * inf = nan`
edge case on a fully-masked row) applied before softmax, with the mask's
diagonal offset computed from how much of the KV cache is already filled
— so a mask built during a cached decode step still masks exactly the
right positions relative to the sequence's true (not just this call's)
length. Verified directly: `tests/model/test_attention.py::
TestCausalMasking` confirms changing a later token's input never changes
an earlier position's output.

## 7. MLP design

SwiGLU (`tinymind/model/mlp.py`): `down(SiLU(gate(x)) * up(x))`, three
`Linear` projections, no bias — see the brief section 7's own formula,
implemented literally.

## 8. Normalization

RMSNorm (`tinymind/model/norm.py`): `x / sqrt(mean(x^2, axis=-1) + eps) *
weight`. No mean-subtraction (that's LayerNorm — `ModelConfig.norm_type`
still accepts `"layernorm"` as a value for forward compatibility, but only
`"rmsnorm"` has an implementation in this delivery; requesting
`"layernorm"` doesn't currently error at config-construction time, since
the value itself is valid, but nothing in `tinymind/model/layers.py`
branches on it yet — see `STATUS.md`). Every RMSNorm gradient is checked
against a numerical gradient (`tests/model/test_norm.py::
test_gradient_matches_numerical`), not just unit-tested for shape.

## 9. Positional encoding

RoPE (`tinymind/model/positional.py`), GPT-NeoX/Llama "rotate-half"
convention, frequencies precomputed once per `(head_dim, max_seq_len,
theta)` and cached — never recomputed per token, and no Python loop over
individual positions (brief section 5's explicit requirement). Verified:
shape correctness, determinism, position-sensitivity (two positions given
the same vector rotate differently), norm preservation (RoPE is a
rotation — it must not change vector length), and numerical stability out
to a 2048-length cache (`tests/model/test_rope.py`).

## 10. Tokenizer interface

Unchanged interface from Phase 1 (`tinymind.model.tokenizer.Tokenizer`,
`ByteTokenizer`) — still the right call for this phase: a trained subword
tokenizer needs a text corpus and a training run neither of which changed
this phase. `TinyMindTransformer.check_tokenizer_compatibility(tokenizer)`
is new: raises if a tokenizer's `vocab_size` exceeds the model's, per the
brief's explicit request for "tokenizer/model vocabulary compatibility
validation" — checked in `TransformerBackend.load()` automatically, so a
mismatched tokenizer/model pairing fails at load time, not on the first
out-of-range token id deep in a forward pass.

## 11. Checkpoint format

`tinymind/model/checkpoint.py`: a directory of `config.json` +
`tokenizer_meta.json` + optional `training_meta.json` + `weights.npz`
(+ optional `optimizer_state.npz`). No pickle — `numpy.savez`/`numpy.load`
with `allow_pickle=False` explicitly passed (NumPy's own `.npy` binary
layout per array; pickle is a NumPy-internal option only ever invoked for
`dtype=object` arrays, which this code never creates — see that module's
docstring). Deliberately independent from the `.tm` deployment format
(`tinymind/model/tm_export.py` bridges the two): a checkpoint needs to hold
optimizer state and full-precision weights for resuming training; a `.tm`
file is for shipping a finished model, and conflating the two seemed like
building the wrong abstraction rather than a shortcut.

Round-trip verified exactly (`tests/model/test_checkpoint.py`): parameters
and logits match bit-for-bit after save/load, not just "close."

## 12. Generation flow

`tinymind/model/generation.py`. Prefill (the whole prompt in one forward
call) followed by incremental decode (one new token per call), sharing one
`KVCache` (`tinymind/model/model.py`). Greedy by default
(`ModelGenerationConfig.do_sample=False` — brief section 14's explicit
"do not make sampling the default"); temperature/top-k/top-p/repetition
penalty available when `do_sample=True`. `max_new_tokens` is clamped to
what the model's `max_seq_len` can actually hold given the prompt length
(a physical limit, treated as an ordinary stopping condition like EOS, not
an error); a prompt that's already too long to prefill at all still raises
clearly, since that genuinely can't be serviced at any `max_new_tokens`
value. KV-cache-vs-full-sequence logit equivalence is checked directly and
is the one test explicitly marked mandatory by the brief (section 13) —
see `tests/model/test_cache.py`.

## 13. Training flow

`tinymind/training/causal_lm_trainer.py`. JSONL -> `TrainingDataset`
(tokenizes with the real `ByteTokenizer`) -> `CausalLMCollator` (pads to a
common length within a batch, builds the attention mask) -> `AdamW`
(`tinymind/model/optim.py`, decoupled weight decay, implemented directly
against `Tensor.data`/`Tensor.grad`) -> gradient accumulation over
`gradient_accumulation_steps` micro-batches -> gradient-norm clipping ->
optimizer step -> periodic checkpointing. Linear warmup over
`warmup_steps`, then constant learning rate (no decay schedule beyond
that in this delivery). Distillation, LoRA, and QAT are explicitly out of
scope for this phase (brief section 19) and remain interfaces only in
`tinymind.distillation`/`tinymind.quantization` — see `STATUS.md`.

## 14. Python/native boundary

Covered in full in `docs/architecture/native-model-contract.md`. In brief:
nothing changed about `native/include/tinymind.h`'s public shape this
phase; `native/src/model.cpp` still throws unconditionally (a C++ forward
pass wasn't attempted this phase — brief section 24 explicitly permits
this: "Do not implement a full native transformer in this phase"). What
this phase adds is the *contract* a future native implementation must
match to reproduce this Python model's numbers exactly.

## 15. Future scaling strategy

The architecture itself needs no changes to reach 1B parameters — every
test in `tests/model/` runs against `ModelConfig` objects of varying size,
and `configs/50m.yaml` through `configs/1b.yaml` (Phase 1) already
instantiate correctly with this real implementation
(`tests/model/test_model.py::test_larger_config_has_more_parameters`
exercises exactly this). What *does* need to change before training at
those sizes is practical, not architectural: a real autodiff/compute
backend (the brief's original assumption — PyTorch/JAX — once network
access or a pre-installed framework is available) and, eventually, the
native runtime actually implementing a forward pass
(`native/src/model.cpp`), which is where on-device inference at any of
these sizes would actually need to run per this project's mobile-first
goal (`docs/architecture/tinymind-design.md` section 0).
