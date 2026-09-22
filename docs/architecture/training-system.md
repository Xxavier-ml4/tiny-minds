# TinyMind training system — Phase 3B audit and design

Status of this document: **Part 1 (audit) was written before any Phase 3B code
changed.** Part 2 onward is the design that was implemented; every claim in it
points at a test or a reproducible command. Evidence for Part 1 is in
`docs/benchmarks/phase3a-audit-results.json` and
`docs/benchmarks/phase3a-profile-T{128,256}.json`, produced by the scripts in
`benchmarks/audit/` run against the untouched Phase 3A tree:

```
python benchmarks/audit/phase3a_correctness_audit.py --tree <phase3a tree>
python benchmarks/audit/profile_training_step.py     --tree <phase3a tree> --seq 256
```

Baseline before touching anything: the Phase 3A suite is green
(`python3 -m unittest discover -s tests`: **312 tests, OK**, ~20 s; STATUS.md
and the CI comment still say 281 — documentation drift, not a failure).

---

# Part 1 — Audit of the Phase 3A implementation

## 1. CURRENT IMPLEMENTATION

How the pieces actually work (read from the code, then confirmed by the
experiments in the JSON above):

| Question | Answer in Phase 3A |
|---|---|
| Parameters | `Tensor(requires_grad=True)` attributes on `Module`s; `named_parameters()` walks attribute-assignment order (`embed_tokens`, `block_i.*`, `final_norm.weight`, optional `lm_head.weight`). float32 only. |
| Gradients | `Tensor.grad` (NumPy array, `None` until touched). `backward()` builds a topological order with a **recursive** DFS, then calls each node's closure. Nothing is freed after backward. |
| Optimizer state | `AdamW._m`, `AdamW._v` (lists indexed by parameter *position*), `step_count`. No `state_dict`. Weight decay is applied to every parameter, including RMSNorm weights and the embedding. |
| RNG | Model init: `np.random.default_rng(seed)`. Training: `random.seed` + `np.random.seed` at the start of `train()`, then `random.shuffle` on the **global** Python RNG once per epoch. The model consumes no randomness at train time (dropout is not implemented). |
| Checkpoints | `save_pretrained`: `config.json`, `tokenizer_meta.json`, optional `training_meta.json`, `weights.npz`, optional `optimizer_state.npz` (`m_i`/`v_i` by index). Written in place. |
| `.tm` | `TM01` container: JSON metadata + named tensor directory + per-tensor CRC32. `export_to_tm` writes float32 tensors named after `named_parameters()`, architecture tag `tinymind-transformer-v1`, and the model config. **No tokenizer, no format/renderer identity, no whole-file digest.** |
| Tokenizer | `ByteTokenizer` (260 ids: PAD/BOS/EOS/UNK + 256 bytes). It is the only tokenizer; the CLI hard-codes it. |
| Model | Pre-norm decoder: embedding → N × (RMSNorm, GQA attention with RoPE, RMSNorm, SwiGLU) → RMSNorm → tied or untied head. |
| Backend | `TransformerBackend.generate(prompt)` encodes the **raw prompt** with BOS. There is no chat template anywhere in inference. |
| Native | C++ forward pass reads only `tinymind-transformer-v1` FP32 `.tm`; byte tokenizer only. It has no int8 path. |

## 2. ACTUAL LIMITATIONS (each one measured)

1. **The data pipeline does not train prompt → response.**
   `TrainingDataset._tokenize` joins the `messages` contents with spaces,
   encodes them with a BOS and nothing else. The `target` (answer or tool
   call) is stored in `target_text` and never tokenized. The collator sets
   `labels = input_ids`. Experiment `data_pipeline`: the decoded training
   sequences are `"What is the capital of France?"` and `"Add 2 and 3."` —
   `"Paris"` and the calculator call appear nowhere; there are no role
   markers and no EOS (so the model can never learn to stop). The
   Phase 3A demo (`examples/train_tiny_model.py`) puts a sentence in a user
   message with an empty target and repeats it 8 times: it demonstrates
   memorising one string, not instruction following.
2. **There is no resume.** Experiment `resume`: restoring weights + Adam
   moments and calling `train()` again restarts the step counter (so LR
   warmup restarts), the epoch and the shuffle sequence; after 10 + 10 steps
   the weights differ from a continuous 20-step run by 1.8e-2 (max abs).
3. **Gradient accumulation weights micro-batches, not tokens.** Loss is a
   mean over each micro-batch's valid tokens, then scaled by
   `1/len(micro_batches)`. Experiment `gradient_accumulation`: identical to
   the large batch to 1.2e-7 when micro-batches have equal token counts, but
   off by **0.128 (max abs grad, whose largest entry is 1.10)** when they
   do not. The existing test only checks that both runs' losses go down.
4. **`tokens_per_second` is wrong.** It divides one step's tokens by the time
   since the start of `train()` (and counts padding). Experiment
   `tokens_per_second`: it reads 20 488 → 3 102 over 8 steps while the true
   rate stays ≈ 22–33 k tokens/s.
5. **Config knobs are silently ignored.** Experiment `silent_config_knobs`:
   `norm_type="layernorm"`, `mlp_type="gelu_mlp"`, `sliding_window=2`,
   `dropout=0.5`, `dtype="float16"` and `attention_type="mqa"` (with 4 KV
   heads) all build and produce **bit-identical logits** to the default.
   A config can therefore claim an architecture that does not exist, which
   defeats "never silently change architecture".
6. **Parameter counting is approximate and partly wrong.**
   `approx_param_count` omits norm weights (−0.14 %) and, with
   `tie_embeddings=False`, adds no `lm_head` at all (−2.8 %; both branches
   of its `embed` expression are identical).
7. **No validation loop, no LR decay, no non-finite handling, no time
   budget.** Warmup then constant LR; a NaN loss would be trained on.
8. **Quantization is evaluated on random token ids.** `tinymind quantize`
   reports the mean top-probability shift on `rng.integers(...)` input, and
   it quantizes the embedding table too. It says nothing about validation
   loss or capability.

## 3. PERFORMANCE BOTTLENECKS (measured, 1 vCPU Xeon @ 2.1 GHz, 4 GB)

Profile of one training step, 1.21 M parameters (h=128, L=6, 4 heads / 2 KV
heads, I=384, byte vocab), batch 8 × 256 tokens, Phase 3A code, medians:

| Quantity | Value |
|---|---|
| Step time | 562 ms (forward 222, backward 329, optimizer 11) → **≈ 3 600 tokens/s** |
| Same model, T = 128 | 245 ms → ≈ 4 200 tokens/s |
| Graph nodes per step | 508 |
| Python overhead of those nodes | **≈ 1.8 ms (0.33 %)** — 2.4 µs forward + 1.2 µs backward per node, measured with a 1-element micro-benchmark |
| Memory retained by one step's graph | **887 MB** (data + grad arrays), most of it `[B,H,T,T]` attention intermediates |
| Forward time by module (inclusive, instrumented run: 277.5 ms total) | attention 184.9 ms (67 %), SwiGLU MLP 70.0 ms (25 %), RMSNorm 13.9 ms (5 %) |
| Backward time by category (instrumented) | attention core 156.2 ms, MLP linear 89.4, MLP elementwise 41.0, attention linear 36.6, norm 24.0 |
| Backward time by op (instrumented) | matmul 195.9 ms, elementwise `mul` 57.1, softmax 37.6, **getitem 34.5** (RoPE slices going through `np.add.at`), sigmoid 16.3 |

Conclusions the numbers force:

* Python/autodiff **graph-construction overhead is not the bottleneck** (0.33 %).
  Replacing the engine or moving to another framework to "reduce overhead"
  would optimise the wrong thing.
* Time is in NumPy kernels, and specifically in the **attention core**: attention
  is 67 % of the instrumented forward time and 54 % of the backward time (its
  score/softmax/context matmuls are ≈ 25 % of the FLOPs), because each layer
  materialises ~8 full `[B,H,T,T]` arrays (scores, scale, mask-add, softmax,
  and their gradients) plus a K/V head-repeat copy, and the weight-gradient
  matmuls go through a batched GEMM followed by an un-broadcast sum.
* The same structure explains the memory: 887 MB for a 1.2 M-parameter model
  is an activation problem, not a parameter problem.
* Multi-thread behaviour cannot be measured here (`nproc` = 1). That is a
  measurement to be taken by the runner probe on the GitHub runner.

## 4. CHECKPOINT RISKS

* **In-place, non-atomic writes.** Experiment `checkpoint_atomicity`: an
  interrupted `save_pretrained` over an existing valid checkpoint leaves a
  truncated `weights.npz`; reload raises `BadZipFile`. The only good copy is
  destroyed by the failure it exists to survive.
* No integrity data (no hashes, sizes, manifest), so a bit-flipped or
  half-uploaded artifact loads as far as NumPy lets it.
* Optimizer moments are stored by parameter *position*; a reordered or
  resized model would load them into the wrong tensors without complaint
  (`load_optimizer_state` does not check shapes).
* No scheduler, step, epoch, data cursor, RNG, dataset identity, tokenizer
  identity or training-config record, so nothing can be validated on resume.
* `CheckpointManager` (generic JSON) is unrelated to model state and cannot
  hold tensors.

## 5. DATA PIPELINE RISKS

* Prompt/response training is absent (§2.1); labels are not masked to the
  completion; no EOS; no roles.
* `TrainingDataset` documents itself as streaming but the trainer does
  `list(dataset)`; shuffling uses the global `random` state.
* No validation split is consumed; no held-out set; no contamination check.
  `tinymind.data.deduplicate` hashes only the message text, not the target.
* Padded batches only, so short instruction examples waste most of a
  256-token row; there is no packing and no way to mask attention across
  packed examples (the attention layer accepts only a causal mask).
* Synthetic data risk: nothing in Phase 3A distinguishes "memorised a
  template" from "learned a behaviour".

## 6. MOBILE RISKS

* The parameter budget is dominated by the vocabulary if a 32 K tokenizer is
  used: 32 000 × 192 = **6 144 000** parameters for the embedding alone.
  The byte tokenizer costs 260 × h (33 K at h=128) but makes sequences long.
* The `.tm` file carries no tokenizer, so "load the model without the
  training environment" also needs a hidden assumption (ByteTokenizer).
* The native runtime cannot read the quantized `.tm`; INT8 exists only as
  store-then-dequantize in Python. Any INT8 claim must say so.
* Inference in Python builds a full autodiff graph even with no backward
  (parameters have `requires_grad=True`), which inflates memory and latency
  for evaluation runs.
* KV-cache and runtime RAM are not reported anywhere.

## 7. PROPOSED CHANGES (and where each one landed)

| # | Change | Rule/section it serves | Implemented in |
|---|---|---|---|
| 1 | Canonical example format + deterministic renderer shared by train/eval/inference; completion-only label masking; EOS | Spec §6–7 | `tinymind/training/render.py` |
| 2 | Padded and packed batching with block-diagonal attention and per-segment positions | §9 | `tinymind/training/batching.py`, `attention.py` |
| 3 | Deterministic epoch plan with a `(seed, epoch, cursor)` position and configurable mixtures | §22, §28 | `tinymind/training/data.py` |
| 4 | Token-normalised loss so `batch×accum` is exactly equivalent to a larger batch | §16 | `tensor.cross_entropy(normalizer=)`, engine |
| 5 | Full `TrainingConfig`, warmup + cosine scheduler, non-finite guard, validation loop | §11, §29 | `training/config.py`, `schedule.py`, `engine.py` |
| 6 | Atomic, manifest-verified, resumable checkpoint (weights, optimizer by name, scheduler, RNG, data position, identities, configs) | §17–22, §25 | `training/checkpoint.py` |
| 7 | Time budget with a safety margin: checkpoint + export + clean exit | §26 | `engine.py` |
| 8 | `count_parameters(config)` exact; reject config knobs the code ignores | §3, rule 7 | `model/config.py`, `model/model.py` |
| 9 | Fused attention / RoPE / RMSNorm / SwiGLU / linear ops with the composed ops kept as the reference; graph freed after backward; `no_grad` | §13–15 | `model/fused.py`, `tensor.py` |
| 10 | Model profiles `tiny_debug/tiny_mobile/tiny_mobile_plus` | §3, §49 | `configs/*.yaml` |
| 11 | Mobile package (`.tm` + tokenizer + manifest with SHA-256) and FP32 vs INT8 comparison on a real eval set | §33–35 | `tinymind/export/` |
| 12 | Stage workflow with artifact hand-off and verification | §23–25 | `.github/workflows/train-stage.yml` |

What is deliberately **not** changed: the NumPy autodiff engine stays the
reference implementation (the audit does not show it is incorrect — its
gradients pass finite-difference tests — only that it is memory-hungry in
attention); no second framework is introduced; distributed training is out of
scope; the `.tm` container keeps its layout.

---

# Part 2 — The design as implemented

Every statement below names the test or command that demonstrates it. Test
modules: `tests/training/` (data, engine, checkpoint), `tests/export/`,
`tests/ci/`, `tests/test_cli_training.py`.

## 8. Dataset format and the exact training sequence

One JSON object per line, three accepted shapes (`tinymind/data/render.py`):

| shape | example | becomes |
|---|---|---|
| `text` | `{"id":"a","text":"The cat sat."}` | `[BOS]The cat sat.[EOS]`, loss on everything except BOS |
| `chat` | `{"id":"b","messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}],"tools":["calculator"]}` | below |
| `legacy` | Phase 3A shape: `messages` + `target{type: answer/tool_call/multi_tool/structured/clarification/refusal}` | the target is appended as the final assistant message |

Rendered `chat` sequence (template `tinymind-chat-v1`; `[X]` = special token,
**bold** = loss):

```
[BOS] tools: calculator, get_weather\n            (only if tools are declared)   no loss
      system:\n<text>\n                                 (only if present)          no loss
      user:\n<prompt>\n                                                            no loss
      assistant:\n                                                                 no loss
      <response>[EOS]                                                             **LOSS**
      tool:\n<tool result>\n                                                       no loss
      assistant:\n<final answer>[EOS]                                             **LOSS**
```

A tool call is ordinary assistant text with a fixed shape,
`{"name":"calculator","arguments":{"expr":"47+38"}}` (name first; argument keys
sorted). The model's output is untrusted text until the runtime parses and
validates it; nothing in the trainer or evaluator executes it.

* Labels follow the convention the model already had: same length as
  `input_ids`, shifted inside the loss, `-100` = no loss, `labels[0]` always
  `-100`. `tests/training/test_data_pipeline.py::TestRenderer` pins the exact
  loss positions, that the response is in the sequence (the Phase 3A failure), that
  `render_prompt` is a strict prefix of the training sequence (a model is prompted
  as it was trained) and that empty completions are **rejected** rather than
  silently trained on.
* Inference uses the same renderer: `InferencePackage.generate` and
  `TransformerBackend` (when a package/`.tm` carries a template) call
  `render_prompt`; benchmark generation goes through `InferencePackage`.
* Examples longer than `max_seq_len` are an error unless `overflow=drop|truncate`
  is chosen explicitly; drops are counted and printed.

## 9. Batching, packing, data order

* **Padded**: one example per row, right padded; padding needs no attention mask
  (causality) and carries label `-100`.
* **Packed**: examples concatenated greedily in plan order into rows of
  `max_seq_len`; `segment_ids` make attention block-diagonal causal and restart
  RoPE positions per example; every example's first label is `-100`, so nothing
  is trained to predict across a boundary. Tests: `TestPackingDoesNotLeak`
  (changing another example leaves a segment's logits bit-identical; a packed
  example equals the same example alone to 2e-5; the packed loss equals the sum of
  the individual losses; a control without `segment_ids` *does* leak).
* **Data order** is a pure function `(seed, epoch) -> batches` (`DataPlan`,
  algorithm `tinymind-dataplan-v1`, documented in `tinymind/training/data.py`):
  per-source permutations seeded by `SeedSequence([seed, epoch, source, 1])`, an
  exact largest-remainder mixture quota per epoch, then a final shuffle. The
  position is `(epoch, micro-batch cursor)`; there is no stateful RNG to lose.
  `drop_last` applies when fewer than `gradient_accumulation_steps` micro-batches
  remain in an epoch.
* Mixtures (`TrainingConfig.mixture`, or `mixture` in `curriculum.json`) are
  exact per-epoch quotas; `repeat_factors()` reports how often each source repeats
  (printed in `training_summary.json`) so an over-repeated small source is visible.

## 10. One optimizer step

```
N = loss tokens in all `gradient_accumulation_steps` micro-batches of this step
for each micro-batch: loss = sum(token losses) / N ; backward
clip global grad norm ; AdamW(lr = schedule(step)) ; step += 1
```

Dividing by the *step's* token count makes `batch 8 × accum 1` and `batch 4 ×
accum 2` produce the same gradient (`test_batch8_equals_batch4_accum2`, to 1e-5
on real gradients; the Phase 3A rule is shown to differ by > 1e-3 on the same
data in `test_accumulation_is_token_weighted_not_micro_batch_weighted`).
AdamW state is addressable by parameter name; weight decay skips 1-D norm gains;
a non-finite gradient raises **before** any weight or moment changes.
The schedule (`training/schedule.py`) is linear warmup then cosine (or linear /
constant) to `min_learning_rate`.

## 11. Checkpoints, resume, promotion

`tinymind/training/checkpoint.py` (module docstring has the full protocol).

* A checkpoint is a directory: `model.npz`, `optimizer.npz` (moments by name),
  `state.json` (model config, tokenizer + renderer spec, dataset identity,
  training config + its compat subset, optimizer hyper-parameters and step count,
  scheduler, progress = step / epoch / data cursor / token counters, gradient
  accumulation state (always empty — checkpoints are taken at step boundaries),
  RNG state, architecture id, environment incl. git commit) and `manifest.json`
  written last with SHA-256 + size of every file and the identity hashes.
  No pickle anywhere (`allow_pickle=False`; a test feeds an object array).
* **Atomic**: build in `.tmp-*` with fsync → manifest last → rename → pointers
  (`previous.json` then `latest.json`, each temp + `os.replace`) → prune. Tests
  inject a failure at all seven points and a real `os._exit(17)` mid-write; the
  previous checkpoint always loads, leftovers are swept on the next save.
* `verify_checkpoint` re-hashes every file, loads arrays without pickle, checks
  every weight's name/shape/dtype/finiteness against the stored config, the optimizer
  arrays against the weights, and state/manifest agreement. A weights-only
  directory (`save_pretrained`) is *not* a checkpoint and fails verification.
* **`--resume`** = exact continuation of the *same* stage: everything above is
  restored, and it is refused — listing **every** mismatch — if the architecture,
  tokenizer, prompt template, dataset (content hash + mixture), training-config
  trajectory fields, or stage differ, or if the stage is already complete. Only
  `eval_interval`, `checkpoint_interval`, `log_interval`, `keep_checkpoints`,
  `max_runtime_seconds`, `safety_margin_seconds`, `epochs` may differ.
* **`--init-from`** = start a *new* stage from a checkpoint's weights (fresh
  optimizer unless `--carry-optimizer`); requires identical architecture,
  tokenizer and template; records `parent` (checkpoint, manifest SHA-256, stage,
  step, run id) in every later checkpoint; `cumulative_steps` continues.
* "Exact" precisely: the *state* restored equals the state saved, so the next step
  computes the same function of the same inputs. Bitwise-identical *trajectories*
  (100 steps == 50 + 50) hold when the numerical environment is the same:
  `test_split_run_matches_continuous_run` (weights, both Adam moments, step,
  schedule, RNG, losses; also mid-epoch, packed + accumulated + multi-epoch) and
  the cross-process CLI test pass on the development machine. A resume on a
  *different CPU or BLAS thread count* (GitHub may schedule the next job on
  different hardware) restores the same state but may round float sums
  differently, so bitwise equality with a hypothetical single-machine run is not
  promised there. The trainer consumes no random numbers after initialisation
  (no dropout; data order is derived from `(seed, epoch)`); the RNG stream is still
  saved and restored so a future stochastic component stays resumable
  (`test_rng_state_survives_json_round_trip`).

## 12. Time budget

`max_runtime_seconds` and `safety_margin_seconds`: before every step the engine
stops if `remaining < margin + 1.5 × (average step) + (average checkpoint time)`;
then it saves a verified checkpoint, exports the inference package and writes
`training_summary.json`. SIGTERM/SIGINT (CLI) request the same clean stop at
the next step boundary. `--stop-after-steps N` bounds one invocation for chunked
runs. Tests: `TestTimeBudget` (fake clock), `test_time_budget_stop_then_resume_in_a_new_process…`.

## 13. Inference package (what ships)

`model.tm` (float32 weights + metadata incl. tokenizer, template, provenance),
`tokenizer.json`, `package.json` (SHA-256 of both files, config hash, parameter
count). Contains no optimizer state or training data; loads in a process where
`tinymind.training` is un-importable and reproduces the logits bit-for-bit
(`test_loads_in_a_process_where_training_code_cannot_be_imported`). The manifest
guards against corruption and inconsistency; it is not a signature and does not
defend against someone who rewrites the files *and* the manifest.
`model.int8.tm` (optional) is a size/quality experiment, see
`tinymind/export/int8.py`: the reference runtime dequantizes it to float32 and the
native runtime cannot read it, so **no INT8 speed or RAM claim is made**.

## 14. GitHub Actions hand-off

`.github/workflows/train-stage.yml` + `tinymind/ci/stage_io.py`. A job's
filesystem is gone when it ends, so a stage crosses jobs as an artifact ("stage
bundle": latest + previous checkpoint, inference package, summary, manifest with
run id / stage / step / model, tokenizer, dataset and config hashes / checkpoint
manifest hash / per-file SHA-256). The receiving job downloads it explicitly
(`actions/download-artifact@v4` with `run-id` + `github-token`; `actions: read`),
then `check-incoming` verifies files, checkpoint integrity, bundle-vs-checkpoint
identity, the model config against the profile about to train, an optionally pinned
manifest hash, the resume/promotion mode rules, and the parent stage's promotion
gate — before any training. `tests/ci/test_stage_io.py` runs stage1 (interrupted)
→ stage1 (resumed) → stage2 → stage3 through this code, each "job" in its own
directory, and refuses corrupted, mismatched, unfinished and un-gated artifacts.
`tests/ci/test_workflow.py` checks the YAML statically. **The workflow itself has
not been executed on a GitHub runner from this sandbox** (no network); see
`docs/training/implementation-report.md` for what a first real run must confirm.

## 15. Known limitations

* The curriculum is template-generated; it measures pipeline behaviour and
  held-out generalisation of narrow skills, not open-domain usefulness.
* A byte tokenizer makes sequences long (≈ 1 token per character); a learned
  subword tokenizer would shorten them but the native runtime has only the byte
  tokenizer.
* No native INT8 path; the native runtime has no generation driver beyond the
  forward and single-token cached step used by the tests and benchmark.
* Multi-thread BLAS scaling and PyTorch installation cost were not measured
  (1-vCPU sandbox, no network); the runner probe measures the former on the runner.
* Bitwise resume across different hardware is not promised (§11).
