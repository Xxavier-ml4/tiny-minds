# Native/Python numerical contract

This document is the thing a future `native/src/model.cpp` implementation
must match, exactly, to reproduce this Python model's output. Nothing here
is left implicit — per the brief's own instruction (section 25: "Do not
leave these implicit"). Every value below is taken directly from the
current Python implementation (`tinymind/model/`), not aspirational.

Status: the C ABI shape (`native/include/tinymind.h`) and the
model-independent native pieces (`tensor.h`, `sampler.h`, `kv_cache.h`,
`tokenizer.h` — see Phase 1's `STATUS.md`) were already real and tested
before this phase; `native/src/model.cpp` is still an interface stub that
throws unconditionally. This document is what would need to be true of a
real implementation there, written now so that work doesn't have to
reverse-engineer the Python side's exact numerics from scratch later.

## 1. Tensor names

`named_parameters()` (`tinymind/model/module.py`) produces dotted names by
walking attribute assignment order. For a config with `num_layers = N`,
the full name list is:

```
embed_tokens                                   [vocab_size, hidden_size]
block_0.attn_norm.weight                       [hidden_size]
block_0.attention.q_proj.weight                [num_heads*head_dim, hidden_size]
block_0.attention.k_proj.weight                [num_kv_heads*head_dim, hidden_size]
block_0.attention.v_proj.weight                [num_kv_heads*head_dim, hidden_size]
block_0.attention.o_proj.weight                [hidden_size, num_heads*head_dim]
block_0.mlp_norm.weight                        [hidden_size]
block_0.mlp.gate_proj.weight                   [intermediate_size, hidden_size]
block_0.mlp.up_proj.weight                     [intermediate_size, hidden_size]
block_0.mlp.down_proj.weight                   [hidden_size, intermediate_size]
... (block_1 .. block_{N-1}, same shape)
final_norm.weight                              [hidden_size]
lm_head.weight                                 [vocab_size, hidden_size]   (only if tie_embeddings=False)
```

`embed_tokens` (not `embed_tokens.weight`) because it's a bare `Tensor`
attribute on `TinyMindTransformer`, not a `Linear`/`RMSNorm` submodule with
its own `.weight`. No `Linear` in this model has a bias tensor (all
`bias=False` — see `tinymind/model/linear.py`), so no `*.bias` names ever
appear. These are exactly the names `tinymind/model/tm_export.py` writes
into a `.tm` file's tensor directory and the names
`tinymind/model/checkpoint.py` uses as `.npz` array keys — both are
generated from `named_parameters()` directly, not hand-maintained lists,
so they cannot drift from this table.

## 2. Matrix multiplication convention

Every `Linear` computes `y = x @ W^T` (`tinymind/model/linear.py`):
`W` is stored as `[out_features, in_features]`, and the forward pass
transposes it before multiplying. **A native implementation must either
transpose `W` once at load time or use a GEMM call with the transpose flag
set on the weight operand** — the stored orientation is `[out, in]`, not
`[in, out]`.

Row-major (C order) throughout — this is a NumPy implementation, and NumPy
arrays are row-major by default; nothing in this codebase requests
Fortran-order (`order="F"`) storage anywhere.

## 3. Attention layout

Q/K/V start as `[B, T, hidden_size]` (the projection's output), then
`.reshape(B, T, num_heads_or_kv_heads, head_dim).transpose(0, 2, 1, 3)` to
`[B, num_heads, T, head_dim]` — heads is axis 1, sequence is axis 2. This
is the standard "heads before sequence" layout (matches the common
Llama/GPT-NeoX reference implementations), **not** "sequence before
heads." GQA repetition (`_repeat_kv` in `tinymind/model/attention.py`)
expands `[B, num_kv_heads, T, head_dim]` to `[B, num_heads, T, head_dim]`
by repeating each kv head contiguously `n_rep = num_heads // num_kv_heads`
times — head `i` of the expanded tensor reads from kv head `i // n_rep`,
**not** `i % num_kv_heads` (i.e. `[kv0, kv0, kv1, kv1]` for `n_rep=2`, not
`[kv0, kv1, kv0, kv1]`).

Attention scores: `(Q @ K^T) * scale`, `scale = 1 / sqrt(head_dim)`
(standard scaled dot-product attention scaling — computed from `head_dim`,
never from `hidden_size` or a config field). Causal mask: additive,
`-1e9` (not IEEE `-inf`) added to disallowed positions before softmax —
using a large finite negative rather than `-inf` avoids a `0 * inf = nan`
edge case if a row were ever fully masked; a native implementation should
match the finite value or use its own numerically-equivalent masking
scheme, but must not use `-inf` if it also does the `0 * mask_bias`-style
arithmetic anywhere `-inf` would produce `nan`.

## 4. RoPE representation

"Rotate-half" convention (GPT-NeoX/Llama style, **not** the interleaved
"rotate-every-two" convention some other implementations use — the two are
not numerically interchangeable):

```
inv_freq[i] = 1 / theta^(2i / head_dim),  i = 0 .. head_dim/2 - 1
freqs[pos, i] = pos * inv_freq[i]
cos_cache, sin_cache = cos(concat([freqs, freqs], axis=-1)), sin(concat([freqs, freqs], axis=-1))
   # shape [max_seq_len, head_dim] — each half of the last axis is a copy of the other half

rotate_half(x) = concat([-x[..., head_dim/2:], x[..., :head_dim/2]], axis=-1)
x_rope = x * cos_cache[positions] + rotate_half(x) * sin_cache[positions]
```

Applied identically to Q and K, never to V. `theta` is
`ModelConfig.rope_theta` (default `10000.0`). Frequencies are computed in
float64 internally (`tinymind/model/positional.py`,
`precompute_rope_cache`) and only cast to float32 for the final cos/sin
cache — a native implementation computing `inv_freq` directly in float32
may accumulate more rounding error over a long `max_seq_len`; matching
Python exactly means computing the frequency table in double precision
first.

## 5. KV cache layout

Python (`tinymind/model/model.py:KVCache`): one array of shape
`[num_layers, batch_size, num_kv_heads, max_seq_len, head_dim]` for keys,
one for values — **layer is the outermost (slowest-varying) axis**, not
innermost. `update(layer_idx, k, v)` writes at
`[layer_idx, :, :, current_length:current_length+t_new, :]`; the shared
`length` counter advances once per full model forward pass (every layer
sees the same `length` throughout one call), not once per layer — see
`native/src/kv_cache.h`'s own docstring, which independently documents the
equivalent contract for the (also-stubbed) native KV cache and should be
kept in sync with this section if either changes.

## 6. Normalization formula

RMSNorm (`tinymind/model/norm.py`), exactly:

```
variance = mean(x^2, axis=-1, keepdims=True)
normalized = x * rsqrt(variance + eps)
output = normalized * weight
```

`eps` is added **inside** the square root (`rsqrt(variance + eps)`), not
added to `x` or applied after the multiply. `eps` is `ModelConfig.
norm_epsilon`, default `1e-6`. No mean-subtraction (this is not LayerNorm).

## 7. Activation formula

SiLU/Swish, used inside SwiGLU's gate: `silu(x) = x * sigmoid(x)`,
`sigmoid(x) = 1 / (1 + exp(-x))`. Full MLP:
`down_proj(silu(gate_proj(x)) * up_proj(x))` — the gate is silu'd, the up
projection is not, and the elementwise product happens before `down_proj`,
not after.

## 8. Residual order (pre-norm)

```
x = x + attention(rmsnorm_attn(x))
x = x + mlp(rmsnorm_mlp(x))
```

Normalize-then-sublayer-then-add, i.e. **pre-norm**, not post-norm
(`x = rmsnorm(x + attention(x))`). Two separate `RMSNorm` instances per
block (`attn_norm`, `mlp_norm`), each with its own learned weight — not
one shared norm reused twice.

## 9. Softmax behavior

Standard, numerically-stabilized softmax (max-subtraction before `exp`):
`softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`, applied along
the last axis for both attention weights and (internally, for the loss)
cross-entropy. No temperature scaling inside the model's own
`forward()` — temperature is only ever applied at the generation/sampling
layer (`tinymind/model/generation.py`), never inside the transformer.

## 10. Embedding / LM head orientation

Token embedding table: `[vocab_size, hidden_size]`, indexed by token id on
axis 0 (a standard embedding-lookup gather). When `tie_embeddings=True`,
the LM head reuses this exact table, transposed: `logits = hidden @
embed_tokens^T`, giving `[B, T, vocab_size]` — there is no separate
`lm_head.weight` tensor in a tied-embeddings model (see the tensor name
table in section 1: `lm_head.weight` only exists when
`tie_embeddings=False`). When untied, `lm_head` is an ordinary `Linear`
with the same `[out, in]` = `[vocab_size, hidden_size]` orientation as
every other `Linear` in this model.

## 11. Tokenizer IDs

`tinymind.model.tokenizer.ByteTokenizer` (the only tokenizer implemented
in this delivery): ids 0-3 are reserved specials (`PAD=0`, `BOS=1`,
`EOS=2`, `UNK=3`), ids 4-259 are raw byte values `0x00`-`0xFF` offset by
`+4`. `vocab_size = 260`. These are **not** configurable per-instance —
they're fixed constants on the class (`ByteTokenizer.PAD` etc.) — so a
native tokenizer implementation matching this one can hard-code them
rather than reading them from a config. A future trained subword
tokenizer would need its own id table recorded in this document (or a
successor document) before a native implementation could match it; no
such tokenizer exists in this delivery (see `STATUS.md`).

## 12. What's deliberately NOT specified here

Anything about *how* a native implementation computes these results
(SIMD strategy, threading, memory layout for cache locality, fused
kernels) is intentionally out of scope for this document — per
`native/include/tinymind.h`'s own design note, the ABI (and, by extension,
this numerical contract) specifies observable behavior, not
implementation. A native `model.cpp` is free to fuse RMSNorm+matmul, use a
transposed weight cache, or restructure the KV cache's memory layout
internally, as long as the formulas in sections 3-10 above produce
numerically equivalent results (within ordinary floating-point tolerance,
not bit-exact — see `tests/model/test_cache.py`'s tolerance,
`1e-3`, as the precedent for what "equivalent" means in this codebase).

---

## Phase 3B additions (nothing above changed)

**Tensor names, shapes, dtypes, layout, RoPE, attention, KV-cache layout:** unchanged from the sections
above. The single source of truth for names/shapes/order is now also available without instantiating a model:
`tinymind.model.config.parameter_shapes(config)`, and `count_parameters(config)` is the exact parameter count
(tests: `tests/model/test_config_count.py`). The order equals `named_parameters()`; a training checkpoint's
`model.npz` uses the same names, so `.tm` export is a rename-free copy of the trained float32 arrays.

**Extra `.tm` metadata (ignored by the native reader):** besides `architecture`, `container_kind` and
`model_config`, a package's `model.tm` also carries `tokenizer` (the tokenizer spec), `renderer` (prompt template
identity: `tinymind-chat-v1` + tokenizer hash), `provenance` (stage, step, dataset/config hashes, git commit,
checkpoint manifest hash) and `package_format_version`. The native loader reads only the first three
keys; `tests/model/test_native_equivalence.py::TestPackageModelNative` loads a package's `model.tm` natively and
compares logits and the greedy token at every position.

**Tokenizer format:** a JSON spec `{"type": "byte", "version": 1, "vocab_size": 260, "pad": 0, "bos": 1, "eos": 2,
"unk": 3, ...}`; the byte tokenizer maps token `4 + b` to byte `b`. The native runtime implements only this
tokenizer. A learned tokenizer's vocabulary would live in `tokenizer.json`; the native side has no reader for it.

**Prompt template:** `tinymind-chat-v1` (see `docs/architecture/training-system.md` §8). The native runtime does
not render prompts — the host application must produce the ids exactly as `ChatRenderer.render_prompt` does
(`[BOS]` + `tools: ...\n` + `system:\n...\n` + `user:\n...\n` + `assistant:\n`), and treat EOS (id 2) as end of turn.

**Not part of the deployed model:** segment ids / block-diagonal attention exist only in training (packing); a
deployed sequence is one segment with the causal mask, which is what the native forward implements.

**INT8:** `model.int8.tm` (per-row symmetric int8 + float32 scales, embedding optionally float32) is written by
`tinymind.export.int8` and read by the Python reference (dequantize-to-float32). The native runtime **cannot read
it**; deploying int8 natively needs an int8 kernel and a loader that this repository does not have.
