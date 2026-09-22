# TinyMind Phase 3B — implementation report

Every number below is measured in this repository and has a script or test
that reproduces it (paths given throughout); nothing is invented. The
development sandbox has 1 vCPU and no network access — every place that
limits what could be measured says so explicitly rather than filling the gap
with an assumption.

## 1. Audit findings

Full detail: `docs/architecture/training-system.md` Part 1, evidence in
`docs/benchmarks/phase3a-audit-results.json` and
`phase3a-profile-T{128,256}.json`, produced by
`benchmarks/audit/phase3a_correctness_audit.py` and
`profile_training_step.py` run against the untouched Phase 3A delivery.

The Phase 3A test suite was green (312 tests) before any change. The audit
found, independent of that suite:

1. **The data pipeline never trained on the assistant's response.**
   `TrainingDataset._tokenize` joined and encoded the `messages` text and set
   `labels = input_ids`; `target` (the answer or tool call) was stored in
   `target_text` and never tokenized. Decoded training sequences contained
   the user's question with no trace of the answer, no role markers, and no
   EOS.
2. **No resume.** Restoring weights and Adam moments and calling `train()`
   again restarted the step counter (so LR warmup restarted), the epoch and
   the shuffle order; 10+10 steps differed from a continuous 20 by 1.8e-2
   (max abs weight).
3. **Gradient accumulation averaged micro-batch means, not tokens** — exact
   only when every micro-batch has the same token count; off by 0.128 (of a
   1.10 max-abs gradient) when they do not.
4. **`tokens_per_second` divided one step's tokens by time since `train()`
   started**, decaying roughly 1/step (20,488 → 3,102 over 8 steps at
   constant true throughput).
5. **Silently ignored config fields**: `norm_type`, `mlp_type`, `dropout`,
   `dtype`, and (with 4 KV heads still set) `attention_type="mqa"` all built
   without error and produced bit-identical logits to the default.
6. **Parameter counting was approximate and, for untied embeddings, wrong**
   (`approx_param_count` added no `lm_head` at all: −2.8%).
7. **Checkpoints were weights-only, written in place, non-atomic.** An
   interrupted `save_pretrained` over a valid checkpoint left a truncated
   `weights.npz`; reload raised `BadZipFile` — the only good copy was
   destroyed by the failure it should have survived.
8. **Recursive `backward()` overflowed Python's recursion limit** past
   roughly 28 transformer layers.
9. No validation loop, no LR decay beyond warmup, no non-finite handling, no
   time budget, no tokenizer recorded in `.tm`.

None of this was visible from "tests pass" — the existing tests checked
different properties (loss decreases on a toy input, a save/load round-trips
identically). That gap between *passes its tests* and *does what it claims*
is exactly what the audit was for.

## 2. Changes made

`docs/architecture/training-system.md` Part 2 has the full design; summary:

* One canonical example format and one renderer
  (`tinymind/data/render.py`) used by training, evaluation and inference
  alike, with completion-only loss masking, EOS, and `render_prompt` a
  provable prefix of the training sequence.
* Padded and packed batching with block-diagonal attention and per-example
  RoPE positions for packing (`tinymind/training/batching.py`), proven not
  to leak between examples.
* A deterministic data-order/mixture plan, `(seed, epoch) → batches`, with no
  stateful RNG to lose (`tinymind/training/data.py`).
* Token-normalised gradient accumulation, so `batch × accumulation` is
  numerically the same optimizer update as one larger batch
  (`training/engine.py`, `model/tensor.py`'s `cross_entropy(normalizer=)`).
* AdamW with named, checkpointable state, weight-decay exemption for norm
  gains, and a non-finite-gradient guard that raises before touching any
  state (`model/optim.py`).
* A warmup→cosine/linear/constant LR schedule with its own checkpointable
  state (`training/schedule.py`).
* Atomic, SHA-256-manifested, fully resumable training checkpoints
  (`training/checkpoint.py`) — see section 8.
* A time budget that checkpoints, exports and exits cleanly before a limit,
  and the same for SIGTERM/SIGINT (`training/engine.py`, `cli_training.py`).
* Exact parameter counting matching an instantiated model in every tested
  attention/tie layout (`model/config.py`).
* Fused attention/RoPE/RMSNorm/SwiGLU/linear ops, with the original composed
  ops kept and tested as the reference (`model/fused.py`).
* An inference package format (`tinymind/export/`) with no optimizer state,
  training data, framework, or network dependency, loadable where
  `tinymind.training` cannot even be imported.
* A GitHub Actions workflow and the artifact-verification logic it calls
  (`tinymind/ci/`, `.github/workflows/train-stage.yml`).
* A deterministic synthetic curriculum with held-out, contamination-checked
  evaluation data (`tinymind/data/curriculum.py`, `contamination.py`) and a
  capability suite that reports metrics separately, never as one composite
  score (`tinymind/evaluation/`).
* An optional byte-level BPE tokenizer (`tinymind/model/bpe.py`) and a
  post-training INT8 package variant (`tinymind/export/int8.py`), both
  honestly scoped (see sections 5, 16).

What was **not** changed: the NumPy autodiff engine is still the reference
implementation (the audit found it memory-hungry in attention, not
incorrect); no second framework was introduced; the `.tm` container's layout
is unchanged (only its metadata grew); no distributed training.

## 3. Final model configuration

Three profiles, each its own YAML in `configs/`, exact counts from
`tinymind.model.config.count_parameters` and checked against an instantiated
model (`tests/model/test_config_count.py`):

| profile | hidden | layers | heads / KV | intermediate | context | parameters |
|---|---|---|---|---|---|---|
| `tiny_debug` | 64 | 2 | 4 / 2 | 192 | 128 | 115,264 |
| `tiny_mobile` | 128 | 6 | 4 / 2 (GQA) | 384 | 256 | **1,214,592** |
| `tiny_mobile_plus` | 192 | 8 | 6 / 2 (GQA) | 576 | 256 | 3,493,824 |

All under the 5M ceiling; `tiny_mobile` is inside the preferred 1-3M range.
Norm: RMSNorm. MLP: SwiGLU. Positions: RoPE (θ = 10,000). Attention: GQA
(head *i* reads KV head `i // (heads / kv_heads)`), causal, block-diagonal
under packing. Embeddings tied. `tiny_mobile.yaml` in full is quoted in
`docs/architecture/training-system.md`. The vocabulary is counted as part of
the parameter budget throughout: a naive 32,000-token vocabulary at
hidden=192 alone costs 6,144,000 parameters — more than the whole `tiny_mobile`
model — which is why the byte tokenizer is the default (see section 5).

## 4. Tokenizer configuration

Default: `ByteTokenizer` — 4 special ids (PAD 0, BOS 1, EOS 2, UNK 3) + 256
byte values = 260. Cost: `260 × hidden_size` parameters (33,280 for
`tiny_mobile`), tied to the output head. Spec: `{"type":"byte","version":1,
"vocab_size":260,...}` (`Tokenizer.spec()` / `spec_hash()`), stored in every
checkpoint and inference package and checked on resume.

Optional: a byte-level BPE tokenizer (`tinymind/model/bpe.py`), trained with
`BPETokenizer.train(texts, vocab_size)`, deterministic given its corpus.
Measured (not assumed) against the byte tokenizer on the stage-2 curriculum's
held-out text — `benchmarks/tokenizer_compare.py`; **the native runtime has
no BPE reader**, so it is a measured alternative, not a deployment path yet.

## 5. Dataset format

One JSON object per line; `docs/architecture/training-system.md` section 8
has the full grammar and the exact rendered token sequence. Summary: a `text`
example is `[BOS]text[EOS]` with loss everywhere but BOS; a `chat` example
renders `tools:`/`system:`/`user:`/`assistant:` sections with loss **only**
on assistant content and its EOS; a `legacy` (Phase 3A-shaped, `target`
field) example is normalised into the same `chat` form. A tool call is
literal assistant text, `{"name":"calculator","arguments":{"expr":"47+38"}}`
— never executed by the trainer or evaluator; the model's output is treated
as untrusted text throughout.

## 6. Training algorithm

One optimizer step (`training/engine.py::TrainingEngine.train_step`):

```
N = total loss tokens across `gradient_accumulation_steps` micro-batches of `batch_size` rows
for each micro-batch:  loss = sum(per-token loss) / N  ;  backward(retain_graph=False)
if any non-finite:                        raise BEFORE any state changes
clip global gradient norm to `gradient_clip_norm`
AdamW step at the scheduled LR (warmup then cosine/linear/constant to `min_learning_rate`)
```

`test_batch8_equals_batch4_accum2` shows this is exact (≤1e-5 max-abs
gradient) regardless of how a fixed set of rows is split into micro-batches;
`test_accumulation_is_token_weighted_not_micro_batch_weighted` reproduces the
Phase 3A rule on the same data and shows it differs by >1e-3.

## 7. Checkpoint format

One directory per checkpoint: `model.npz` (float32 weights by parameter
name), `optimizer.npz` (`m.<name>`/`v.<name>`), `state.json` (model config,
tokenizer + renderer spec, dataset identity, training config + its
resume-relevant subset, optimizer hyperparameters + step count, scheduler
state, progress — step/epoch/data cursor/token counters, RNG state,
architecture id, environment incl. git commit), `manifest.json` written last
with SHA-256 + size of the three files above and every identity hash. No
pickle (`allow_pickle=False`; a test feeds an object array and confirms it is
rejected). `latest.json`/`previous.json` point at verified checkpoints only.

Write protocol: build in a temp directory with `fsync` after each file,
manifest written last, `fsync` the directory, atomic `rename`, then update
`previous.json` then `latest.json` (each via temp-file + `os.replace`), then
prune beyond `keep_checkpoints`. `tests/training/test_checkpoint.py` injects
a failure at every one of these seven points, plus a real `os._exit(17)`
mid-write in a subprocess: in every case the previous valid checkpoint is
still loadable and stale temp directories are swept on the next save.

## 8. Resume guarantees

`--resume` (same stage, exact continuation): restores weights, optimizer
moments *and* step count, scheduler state, RNG, and the exact `(epoch,
cursor)` data position; refused — listing **every** mismatch — if
architecture, tokenizer, prompt template, dataset identity, the
resume-relevant training config, or stage differ, or if the stage is already
complete. `--init-from` (new stage): weights only (optimizer optionally
carried with `--carry-optimizer`), requires identical architecture/tokenizer/
template, records exact lineage (parent checkpoint, its manifest SHA-256,
stage, step) in every later checkpoint.

**"Exact" measured**: `test_split_run_matches_continuous_run` — 100
continuous steps vs. 50+checkpoint+reload+50, on this machine — weights, both
Adam moments, step count, schedule state, RNG state and losses are
**bitwise** identical; also true mid-epoch, and with packing + accumulation
+ multiple epochs. The cross-process CLI test
(`test_time_budget_stop_then_resume_in_a_new_process_equals_one_uninterrupted_run`)
confirms this across separate OS processes, not just separate objects.
**Scope of that claim**: it holds on identical hardware/BLAS/thread count;
resuming on different hardware (which the GitHub workflow may do between
jobs) restores the same *state*, so the next step computes the same function
of the same inputs, but float summation order can differ across
hardware/thread counts, so bitwise equality with a hypothetical
single-machine run is not promised there — this is stated, not tested here,
since only one machine was available.

## 9. CPU benchmark results

Full detail and every table: `docs/benchmarks/cpu-training-baseline.md`.
Headline finding: **Python/autodiff overhead is 0.3% of a training step**
(measured directly, not inferred), so it was not the target of optimisation.
The actual cost is attention's memory traffic (≈55% of a step for ≈25% of
the FLOPs, from unfused intermediate arrays); fusing attention/RoPE/norm/MLP
inside NumPy — while keeping the composed implementation as the tested
reference — measured **1.37-1.67× faster** and **~2.3-2.8× less retained
activation memory** at the `tiny_mobile` shape, T ∈ {128, 256}
(`docs/benchmarks/bench-fused-vs-reference.json`). Full per-profile numbers,
sequence-length and batch-size sweeps: `docs/benchmarks/tables.generated.md`
(regenerate with `python benchmarks/make_benchmark_tables.py`).

## 10. GQA/MHA/MQA comparison

Speed, equal parameter budget, `tiny_mobile` shape
(`docs/benchmarks/bench-attention-variants.json`):

| variant | seq 256 step (ms) | tokens/s | KV cache (MiB) |
|---|---|---|---|
| MHA | 424 | 4,825 | 1.500 |
| GQA | 397 | 5,156 | 0.750 |
| MQA | 386 | 5,306 | 0.375 |

Quality, 250 steps from scratch at equal budget, lr 3e-3
(`docs/benchmarks/sweep-attention.json`):

| variant | final val loss | tokens/s |
|---|---|---|
| MHA | 0.4627 | 3,333 |
| GQA | 0.4627 | 3,140 |
| MQA | 0.4777 | 4,270 |

GQA matched MHA's validation loss exactly while MQA's was measurably worse;
the GQA/MQA speed gap is within run-to-run noise (overlapping min–max
ranges in the throughput sweep). **GQA (2 KV heads) is the default** —
it gets nearly all of MQA's cache saving without MQA's quality cost.

## 11. Sequence-length benchmark

Throughput at constant ≈2048 tokens/step, `tiny_mobile`
(`docs/benchmarks/bench-seqlen.json`):

| seq | tokens/s |
|---|---|
| 64 | 6,626 |
| 128 | 6,119 |
| 256 | 5,205 |
| 512 | 3,996 |

Quality/coverage trade-off, 250 steps, lr 3e-3 (`sweep-seqlen.json`): at
seq=64 only 223/2400 training examples fit (90.7% dropped as too long) and
validation loss (on a subset short enough to compare fairly) is worst
(2.588); at seq=256 every example fits (0% dropped) and validation loss is
1.623. **256 was chosen because it is the shortest length that drops none of
the curriculum's examples**, not for raw throughput.

## 12. Training throughput

`docs/benchmarks/bench-profiles.json`, batch 8, each profile's own context:

| profile | tokens/s | seconds / 100 steps |
|---|---|---|
| `tiny_debug` | 36,964 | 2.8 |
| `tiny_mobile` | 5,178 | 39.6 |
| `tiny_mobile_plus` | 2,462 | 83.2 |

Re-measured cleanly (idle machine, no other process running) to check this:
5,256 tok/s, consistent with the 5,178 above. `docs/benchmarks/mobile-matrix.json`'s
training row for the same shape and batch reports 2,606 tok/s — about half —
because that script's matrix ran while another benchmark or training job was
active on this sandbox's single vCPU; it is a contended measurement, not a
second data point to average in. **Every wall-clock number in this delivery
was measured on a single shared vCPU and some were taken while other
background jobs were running**; treat the ratios and rankings between
settings (GQA vs MHA, fused vs reference, batch/seq sweeps — all measured
with each row's alternatives run back to back) as the reliable signal, and
any single absolute tokens/s figure as accurate to roughly a factor of 2 on
this machine, not a precise number a production estimate should be based on.

## 13. Validation results

**A reduced-budget demonstration** (not the three-stage plan's budget — see
`docs/training/three-stage-plan.md` for what is plan vs. demonstration): all
four curriculum stages, chunked across simulated GitHub Actions jobs
(`benchmarks/run_stage_chain.py`, each job a fresh subprocess seeing only a
copied bundle), `tiny_mobile`, curriculum scale 0.35, ≈15 minutes of training
in total (`docs/benchmarks/stage-runs/chain-log.jsonl`):

| stage | steps | peak LR | val loss start → end | train time |
|---|---|---|---|---|
| stage0 (sanity) | 600 | 3e-3 | 6.15 → 0.016 | 208 s |
| stage1 | 700 | 2e-3 | 6.84 → 0.238 | 243 s |
| stage2 | 900 | 1e-3 | 0.54 → 0.190 | 305 s |
| stage3 | 350 | 3e-4 | 0.21 → 0.177 | 118 s |

Held-out capability suite at stage 3 end, every category, n=18-20 each except
`no_tool` (n=9) (`stage-runs/stage3-eval.json`):

| category | accuracy | category | accuracy |
|---|---|---|---|
| copy | 0.85 | tool_lookup (tool name only) | 0.85 |
| instruction | 0.60 | tool_memory (tool name only) | 1.00 |
| factual_qa | 0.00 | tool_arithmetic (exact args) | 0.00 |
| structured | 0.00 | tool_timer (exact args) | 0.15 |
| clarification | 0.00 | tool_weather (exact args) | 0.00 |
| clarification_followup | 0.00 | tool_result (uses the tool's answer) | 0.15 |
| context_retention | 0.00 | refusal | 0.10 |
| no_tool (didn't call a tool when none was needed) | 0.00 | | |

Tool behaviour, separately (never combined into one score):

| metric | value |
|---|---|
| correct-tool rate (right tool name, 120 tool-needing prompts) | 0.842 |
| **argument accuracy** (right tool AND right arguments) | **0.038** |
| wrong-tool rate | 0.058 |
| malformed-call rate | 0.025 |
| false-positive call rate (called a tool on the 167 prompts needing none) | 0.353 |

**Read plainly, this is a weak result for a ~15-minute, ~14%-of-plan
demonstration, and it is reported as one.** Two different things are visible,
and they should not be blurred together:

1. **Tool-name selection and copying** work: correct-tool rate 0.84,
   `tool_memory`/`tool_lookup` (scored on tool name only) 1.00/0.85, `copy`
   0.85. A control run (stage 0, whose targets are random strings, no tool
   routing involved) copies held-out words correctly 85% of the time after
   only 600 steps — matching `copy`'s stage-3 number almost exactly, which is
   itself informative: copying is learned early and cheaply.
2. **Almost everything that requires producing the *right free-text content*
   is still at or near zero**: `factual_qa`, `structured`, `clarification`,
   `clarification_followup`, `context_retention`, `no_tool` and `refusal`
   (0.10) all score at or near 0, and tool **argument** accuracy is 3.8% even
   though tool *name* accuracy is 84%. This pattern holds across the whole
   chain, not just at stage 3 —
   `stage-runs/stage{1,2}-eval.json` show the same free-text categories near
   zero at every stage, while `copy` is 0.90/0.85 and tool-name accuracy
   climbs (0.72 → 0.64 → 0.84) across stages 1→2→3.

The most likely explanation, supported by the contrast above but **not a
tested fix**: stages 1-3 draw tool arguments and many free-text answers from
small closed pools (≈30 cities, ≈28 words, a handful of fixed refusal/
clarification phrasings), so at this training budget the model has enough
signal to learn *which* tool or *that a pattern exists* but not enough
distinct examples per pool value to learn to *produce the specific right
content* — consistent with copying (an open-ended, position-based skill)
being learned fastest and value-specific or answer-specific skills being
slowest. The training budget itself is also a real factor here — a
production run uses the numbers in `three-stage-plan.md`'s stage table
(≈9.7-13M tokens/stage), and this demonstration used ≈14% of that combined
across all three stages — so *some* of this weakness is very plausibly a
budget effect that a full-budget run would partly close, not only a data
design flaw; this delivery cannot distinguish the two without running the
full budget.

**Consequently, stage 2 failed its own promotion gate**: false-positive call
rate 0.341 exceeds the 0.30 threshold in `configs/stages/stage2.gate.json`
(`stage-runs/gate-verdicts.json`); the demonstration chain used `--skip-gate`
to continue and exercise the rest of the machinery, and stage 3 did not
repair the false-positive rate (0.353, slightly worse). **A real chain, gate
enforced, would have stopped after stage 1** pending a fix. Two concrete next
steps, neither implemented here: (a) widen the value pools (or add
pseudo-word fillers) so copying is the only route to a correct argument, and
(b) run at the plan's actual token budget before concluding the free-text
categories are a data problem rather than a budget one.

## 14. `.tm` artifact size

`tiny_mobile`, stage-3 model, actual file
(`docs/benchmarks/stage-runs/stage3-summary.json`, `exports`):

| | bytes | MiB |
|---|---|---|
| `model.tm` (float32) | 4,863,136 | 4.64 |
| inference package total (+ `tokenizer.json` 161 B, `package.json` 2,304 B) | 4,865,601 | 4.64 |

Matches the exact formula `4 × parameter_count` (`tinymind.export.budget`),
checked against the real weight arrays in `tests/export/test_budget.py`.

## 15. INT8 results

`benchmarks/quantization_eval.py` on the stage-3 model, real held-out data
(`docs/benchmarks/quantization-eval.json`):

| variant | `.tm` bytes | eval loss | correct-tool rate | argument acc. | false-positive rate | Python tokens/s |
|---|---|---|---|---|---|---|
| float32 | 4,863,136 | 0.4832 | 0.837 | 0.025 | 0.316 | 743.8 |
| int8, embedding float32 | 1,360,960 (**3.57× smaller**) | 0.4829 | 0.833 | 0.025 | 0.319 | 738.2 |
| int8, embedding int8 | 1,262,272 (**3.85× smaller**) | 0.4830 | 0.833 | 0.025 | 0.312 | 737.4 |

File size shrinks 3.6-3.9×; every quality metric is within noise of float32
(no capability regresses beyond what looks like run-to-run scoring noise).
**Honestly scoped**: the Python reference dequantizes int8 to float32 before
computing, so RAM and latency here are unchanged by construction — the
"Python tokens/s" and "weights in RAM" columns confirm this rather than
claiming a speed-up. The native runtime cannot read `model.int8.tm` at all.
No int8 speed or RAM claim is made anywhere in this delivery.

## 16. GitHub Actions design

`.github/workflows/train-stage.yml` (one parameterised `workflow_dispatch`
workflow for all four stages) + `tinymind/ci/stage_io.py` (the logic it
calls). A job's filesystem is gone when it ends, so a stage crosses jobs
*only* as a GitHub artifact ("stage bundle": checkpoint(s), inference
package, summary, a manifest with run id / stage / step / every identity
hash / per-file SHA-256). The receiving job downloads it explicitly
(`actions/download-artifact@v4` with `run-id` + token, needing only
`actions: read`), then `check-incoming` verifies file integrity, checkpoint
integrity, bundle-vs-checkpoint identity agreement, the model config against
the profile about to train, an optionally pinned manifest hash, the
resume-vs-promotion mode rules, and the parent stage's promotion gate —
**before any training starts**. The trainer owns its own time budget
(`runtime` subcommand computes `max_hours` minus a 15-minute reserve for
setup/eval/upload; the workflow's `timeout-minutes: 355` is only a backstop
under GitHub's 6-hour kill). Publishing to a GitHub Release / Hugging Face is
optional (`inputs.publish`), needs no secret to be skipped, and
`continue-on-error: true` so it can never fail the training job.

**Tested**: `tests/ci/test_stage_io.py` runs a real 4-job chain
(stage0→interrupted-stage1→resumed-stage1→stage2→stage3) through this exact
code, each "job" a separate subprocess seeing only a copied bundle, and
separately proves every refusal (corrupted download, wrong architecture,
tampered manifest, failed gate, wrong mode). `tests/ci/test_workflow.py`
checks the YAML statically (every input used, no unquoted interpolation into
shell, every referenced command exists, artifact steps present with the
right options, no embedded credentials, minimal permissions).
**Not tested: an actual run on a GitHub-hosted runner** — this sandbox has no
network access. What a first real run needs to additionally confirm: the
`workflow_dispatch` UI accepts the inputs as intended; `download-artifact`
with a `run-id` from a different run of the same repository actually
succeeds under `actions: read`; multi-core BLAS thread selection on the real
runner; wall-clock training throughput at the runner's real (possibly
throttled/shared) CPU.

## 17. Known limitations

* The curriculum is template-generated; results show pipeline correctness and
  held-out generalisation of narrow, synthetic skills — not open-domain
  usefulness.
* **Argument accuracy is weak (3.8%) even where tool-name selection is
  strong (84%), and most free-text categories (factual_qa, structured,
  clarification, context_retention, refusal) are still at or near 0%
  accuracy at the end of the demonstration** — see section 13. Partly
  attributed to small argument-value pools inviting memorisation over
  copying, partly plausibly just the training budget (≈14% of the plan);
  this delivery cannot separate the two without a full-budget run. Not fixed
  here.
* The byte tokenizer makes sequences long; BPE is implemented and measured
  but has no native reader.
* No native INT8 kernel; INT8 is a measured storage/quality trade-off only.
* Multi-thread BLAS scaling is unmeasured (1 vCPU in this sandbox).
* Bitwise-identical resume across different hardware/thread counts is not
  promised (state equality is; float-sum order is not).
* The GitHub Actions workflow has not executed on an actual runner.
* The three-stage plan's production token budget was not run — only a ~14%,
  reduced-budget demonstration, at batch 8 instead of the plan's 16.

## 18-20. Stage commands

Full arguments in `docs/training/three-stage-plan.md`; canonical form (a real
run passes `--max-runtime` computed by `python -m tinymind.ci.stage_io
runtime --max-hours N`, and normally runs through the workflow rather than by
hand):

```bash
# Stage 1 (from scratch)
tinymind data build-curriculum --stage stage1 --out data/stage1
tinymind train --config tiny_mobile --dataset data/stage1 --output runs/s1 \
  --stage stage1 --max-steps 3000 --learning-rate 2e-3 --min-learning-rate 2e-4 \
  --warmup-steps 150 --batch-size 16 --max-runtime <seconds> --safety-margin 300
# ... if the time budget was hit, continue the SAME stage exactly:
tinymind train --config tiny_mobile --dataset data/stage1 --output runs/s1 \
  --resume runs/s1/checkpoints --max-runtime <seconds>

# Stage 2 (promoted from the COMPLETE, gated stage 1)
tinymind stage-gate --criteria configs/stages/stage1.gate.json --summary runs/s1/training_summary.json
tinymind data build-curriculum --stage stage2 --out data/stage2
tinymind train --config tiny_mobile --dataset data/stage2 --output runs/s2 \
  --stage stage2 --init-from runs/s1/checkpoints \
  --max-steps 3600 --learning-rate 1e-3 --min-learning-rate 1e-4 --warmup-steps 150 --batch-size 16

# Stage 3 (promoted from the COMPLETE, gated stage 2)
tinymind stage-gate --criteria configs/stages/stage2.gate.json --summary runs/s2/training_summary.json \
  --eval-results runs/s2/eval_results.json
tinymind data build-curriculum --stage stage3 --out data/stage3
tinymind train --config tiny_mobile --dataset data/stage3 --output runs/s3 \
  --stage stage3 --init-from runs/s2/checkpoints \
  --max-steps 1800 --learning-rate 3e-4 --min-learning-rate 3e-5 --warmup-steps 50 --batch-size 16

# Held-out evaluation and mobile export, any stage:
tinymind eval --package runs/s3/export --eval data/stage3/eval.jsonl --val data/stage3/val.jsonl --out runs/s3/eval_results.json
tinymind verify-package runs/s3/export
```

Through the workflow: `workflow_dispatch` with `stage=stage1`,
`resume_artifact` empty, then re-run with `resume_artifact=<bundle name>` to
continue or to promote (`resume_mode=auto` picks resume vs. init-from from
the artifact's own state).
