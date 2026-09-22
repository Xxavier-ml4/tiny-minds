# Three-stage training plan (profile `tiny_mobile`, 1,214,592 parameters)

Two things are described here and kept apart on purpose:

* **The plan** — what each stage is for and the settings a real run on a GitHub runner should use. Sizes of the data
  are exact (`docs/benchmarks/curriculum-sizes.json`); step counts and durations are *arithmetic from measured
  throughput* and are estimates until `runner_profile.json` from a real run says otherwise.
* **What was actually run** in the development sandbox (1 vCPU): a *reduced-budget demonstration* of the same chain,
  to prove the machinery (chunked jobs, resume, promotion) and to get honest held-out numbers. Its results are in
  `docs/benchmarks/stage-runs/` and in the implementation report. **It is not the plan's budget and its quality
  numbers must not be read as the plan's outcome.**

Every threshold below is a *measurable policy*, not a prediction of quality. No expected accuracy is claimed.

## Common to all stages

| | |
|---|---|
| model | `configs/tiny_mobile.yaml`: hidden 128, 6 layers, 4 heads / 2 KV heads (GQA), SwiGLU 384, byte tokenizer (vocab 260), 1,214,592 parameters |
| sequence length | 256 (the model's context). Every curriculum example fits (max 227 rendered tokens); `overflow=error`, so a longer example stops the run instead of being silently cut |
| batching | packed rows (block-diagonal attention, per-example positions), batch 16 rows, accumulation 1; padding utilisation on this data 77 % packed vs 68 % padded (`docs/benchmarks/packing-utilization.json`) |
| optimiser | AdamW β=(0.9, 0.999), decoupled weight decay 0.01 (norm gains exempt), gradient clip 1.0, linear warmup then cosine to 10 % of peak |
| loss | completion-only (assistant text + EOS); the prompt, tool results and padding carry no loss |
| checkpoint policy | every 250 steps and at exit; `keep_checkpoints=3`; atomic + SHA-256 manifest; the exit checkpoint is always written (stage end, time budget or SIGTERM) |
| validation | `val.jsonl` (disjoint from training prompts, same distribution) every 250 steps and at exit: token-mean loss over the loss mask. **Held-out `eval.jsonl` is never used to choose anything during training**; it runs once at stage end (`tinymind eval`) |
| time budget | `max_hours` (≤ 5.5) minus 15 min setup/upload; the trainer stops itself with a checkpoint 5 min early |
| artifact | `tinymind-tiny_mobile-<stage>-step<NNNNNN>-<sha7>` (bundle: latest + previous checkpoint, inference package, summary, eval results, manifest with hashes) and `…-model` (the inference package alone) |
| contamination | `tinymind train` checks eval vs (train ∪ val) — exact and normalised prompt overlap — and refuses to start on any overlap |

## Stages

Tokens per epoch are exact for the curriculum at `data_scale=1.0`; "≈ tokens" = epochs × tokens per epoch; "≈ steps" =
tokens ÷ real tokens per step (measured in the demonstration: 1.4–1.6 k per 8-row step, so ≈ 2.8–3.2 k per 16-row
step); "≈ time" = tokens ÷ 4.5 k tokens/s (the sandbox's measured training rate, 4.1–5.4 k). Runner speed is unknown.

| | Stage 1 — basic language and instruction | Stage 2 — capability specialisation | Stage 3 — refinement |
|---|---|---|---|
| purpose | token statistics, short instruction following, response format; first exposure to tools and dialogue | tool routing (which tool, which arguments, when *not* to call one), structured output, clarification, refusal, context retention | extra paraphrases and more no-tool negatives; lower LR; reduce false tool calls and malformed calls |
| data (`--dataset`) | curriculum `stage1`: 18,485 examples, **1.62 M tokens**/epoch | `stage2`: 24,722 examples, **3.24 M** tokens/epoch | `stage3`: 23,722 examples, **3.20 M** tokens/epoch |
| mixture (exact per-epoch quotas) | language 0.30 · instruction 0.45 · tools 0.15 · dialogue 0.10 | tools 0.35 · dialogue 0.20 · instruction 0.17 · language 0.10 · structured 0.10 · safety 0.08 | tools 0.35 · dialogue 0.20 · instruction 0.15 · language 0.10 · structured 0.10 · safety 0.10 |
| validation set | 850 examples | 1,150 | 1,100 |
| start | from scratch (`--init-from` absent) | `--init-from` the **complete, gated** stage-1 checkpoint | `--init-from` the complete stage-2 checkpoint |
| epochs / ≈ steps / ≈ tokens | 6 / ≈ 3,100 / ≈ 9.7 M | 4 / ≈ 4,600 / ≈ 13 M | 2 / ≈ 2,300 / ≈ 6.4 M |
| ≈ time at 4.5 k tok/s | ≈ 36 min | ≈ 48 min | ≈ 24 min |
| peak LR → floor | 2e-3 → 2e-4 | 1e-3 → 1e-4 | 3e-4 → 3e-5 |
| warmup | 150 steps | 150 | 50 |
| held-out capability set | same `eval.jsonl` for every stage (657–666 examples, category-balanced, held-out values and phrasings) | | |
| promotion gate (`configs/stages/*.gate.json`) | `stage1.gate.json` | `stage2.gate.json` | — (final) |

Why these learning rates: the sweep in `docs/benchmarks/sweep-lr.json` (250 steps from scratch, cosine to 10 %) ranked
peak LR 3e-3 < 1e-3 < 3e-4 < 1e-2 < 1e-4 in validation loss (best to worst), stable across two seeds for the top two, no
NaN/Inf at any tested value. The plan uses 2e-3 from scratch (a step below the best short-run value, because long runs
tolerate less than 250-step ones), and lower peaks for the continuation stages (a common heuristic; **not swept
separately**). The number of epochs is a design choice, not a measured optimum.

## Promotion criteria (measurable; provisional policy set before the demonstration run)

A stage is promoted only if the workflow's `check-incoming` step passes **and** the gate passes:

1. checkpoint integrity: every file's SHA-256, arrays load, shapes match the stored config, all finite;
2. identity: bundle manifest = checkpoint manifest (stage, step, model/tokenizer/dataset/config hashes, manifest hash);
3. the model config equals the profile about to train; optionally the pinned manifest SHA-256 equals the operator's input;
4. `stage_complete` is true (a stage cannot be promoted while unfinished);
5. the stage's gate file:
   * `stage1.gate.json`: validation loss ≤ 1.5 nats/token and held-out `copy` accuracy ≥ 0.20;
   * `stage2.gate.json`: validation loss ≤ 1.0, correct-tool rate ≥ 0.30, false-positive tool-call rate ≤ 0.30.

These are floors that separate "learned something" from "did not", chosen before the demonstration. They should be
tightened after the first full-budget run on a runner.

## What the reduced-budget demonstration actually was

Run with `benchmarks/run_stage_chain.py` (each "job" a fresh process that sees only a downloaded-artifact-shaped
bundle; ~103 s of training per job; `--skip-gate`, so the machinery could be exercised even where a real chain
would have stopped for promotion). Unlike the plan, stage 1 here was chained with `--init-from` the completed
stage-0 (sanity) checkpoint rather than started from scratch — stage 0 is the brief's own "engineering sanity"
stage (section 27), and initialising stage 1 from it additionally exercised the promotion path one stage earlier
than the plan requires; it does not change the plan's own stage-1 row above, which still starts from scratch for a
real run, since stage 0's data is synthetic sanity data with nothing worth carrying forward. Actual settings and
results, all from `docs/benchmarks/stage-runs/` (`chain-log.jsonl`, `stageN-summary.json`, `stageN-eval.json`):

| stage | data scale | jobs | steps | peak LR | real tokens | train time | val loss (start → end) |
|---|---|---|---|---|---|---|---|
| stage0 (sanity, not a plan stage) | 3.0 | 3 | 600 | 3e-3 | 1.12 M | 208 s | 6.149 → 0.016 |
| stage1 | 0.35 | 3 | 700 | 2e-3 | 1.11 M | 243 s | 6.841 → 0.238 |
| stage2 | 0.35 | 4 | 900 | 1e-3 | 1.27 M | 305 s | 0.540 → 0.190 |
| stage3 | 0.35 | 2 | 350 | 3e-4 | 0.49 M | 118 s | 0.208 → 0.177 |

That is 3.99 M training tokens and 14.6 minutes of training in total — about 14 % of the plan's token budget — with
batch 8 instead of 16. The gates, evaluated afterwards on the recorded results (`tinymind stage-gate`, saved in
`stage-runs/gate-verdicts.json`): **stage 1 passes; stage 2 fails** (false-positive call rate 0.341 > 0.30), so under
the plan the demonstration would not have been promoted into stage 3. The chain used `--skip-gate` to exercise the
rest of the machinery, and stage 3 did not repair it (0.353).

A separate, earlier full 700-step stage-1 run trained from true scratch (no stage-0 warm start) at peak LR 3e-3
instead of 2e-3 reached final validation loss 0.381 versus this chain's 0.238 (`stage1-from-scratch-summary.json`).
The two runs differ in more than the learning rate (one starts from stage 0's weights, the other from
initialisation), so this is corroborating evidence that the higher rate is worse over a full run — consistent with
the 250-step sweep's ranking — and not a controlled second data point.

## The gap this exposes, and the first change to make before spending runner time

On held-out examples the demonstration model has learned the *formats* (well-formed tool-call JSON with the right tool
name: correct-tool rate 0.84 and malformed-call rate 0.025 after stage 3) but not *copying argument values* it has not
memorised: e.g. prompt "…33 plus 95" → `{"expr":"35+51"}`; "weather for Hanoi" → city "Havana"; "JSON array of: rope, stone"
→ `["stone","salt","stone","book"]`; argument accuracy is 3.8 %. The stage-0 model, whose values are random strings,
copies held-out words 85 % of the time after 600 steps, which points at the cause: in stages 1-3 the values come from
small pools (32 training cities, 28 words), so memorising the pool beats learning to copy. That is a *hypothesis
supported by that contrast, not a tested fix*. The change to make first — before more steps — is to draw argument
values (cities, topics, names, list items) from an open pool (random pseudo-words alongside the real ones) so that only
copying works, then re-run stage 2 and compare the held-out argument accuracy. It is not implemented in this delivery.
