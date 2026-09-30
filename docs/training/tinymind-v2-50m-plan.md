# TinyMind v2 — 50M capability-focused training plan (profile `50m`, 50,370,624 parameters)

This document describes the v2 training system: a reproducible, seven-stage
capability curriculum for a ~50M-parameter model, built *alongside* the
existing v1 three-stage pipeline (`docs/training/three-stage-plan.md`), which
is unchanged. Read `STATUS.md` for what is proven versus designed-only — this
plan is the design and the contracts; it does not claim a full run has
happened.

## Why a second pipeline

v1 trains a 1.2M/3.49M model to be a predictable small assistant. v2 keeps that
intact and adds a larger model and a curriculum aimed at *capabilities*
(language, reasoning, mathematics, knowledge, instruction-following, dialogue,
tools, safety) rather than a single quality number. Everything is a pure
function of `(stage, seed, scale)` plus two configuration files, so any stage
is reproducible and any policy knob is editable without touching code.

## The model (brief section 1)

`configs/50m.yaml` (`profile: 50m`) fixes the geometry:

| field | value |
|---|---|
| hidden_size | 576 |
| num_layers | 12 |
| num_heads | 6 |
| num_kv_heads | 2 (grouped-query attention) |
| intermediate_size | 1472 (SwiGLU) |
| vocab_size | 16000 |
| max_seq_len | 1024 (RoPE — no learned position table) |
| tie_embeddings | true |
| norm / activation | RMSNorm / SwiGLU |

`tinymind model info 50m` reports **exactly 50,370,624** parameters. The count
is: token embedding 16000×576 = 9,216,000; twelve transformer blocks at
3,429,504 each = 41,154,048; final RMSNorm 576; total 50,370,624. Because
positions are RoPE, `max_seq_len` does not change the count (verified in
`tests/model/test_50m_profile.py`).

Note on runtime: the model is pure NumPy (custom autograd in
`tinymind/model/tensor.py`; no PyTorch/JAX/TF in this environment). It really
runs here — construction ~1s, and one optimizer step at a small shape completes
in seconds (see the preflight below) — but it is CPU-only and slow relative to
a GPU framework. The full 345M-token curriculum is intended for a runner, not
an interactive session.

## The tokenizer (brief section 2)

v2 uses a first-class byte-level BPE tokenizer (`tinymind/model/bpe.py`,
16,000 vocab). It is:

* **trained from a corpus** with `tinymind tokenizer train-bpe --input <files/dir>
  --vocab-size 16000 --output tokenizers/v2-16k.json` (`tinymind/model/bpe_io.py`);
* **deterministic** — the same corpus yields byte-identical `tokenizer.json`
  and the same SHA-256 spec hash;
* **self-contained** — the tokenizer reconstructs solely from `tokenizer.json`;
* **pinned into artifacts** — checkpoints and exported packages already store
  the tokenizer spec and compare its hash, so a model can never be paired with
  the wrong tokenizer.

`configs/50m.yaml` references the tokenizer by a path resolved relative to the
YAML file, so training is independent of the working directory. Verified in
`tests/model/test_bpe_tokenizer_v2.py`.

## The curriculum (brief sections 3–5)

Seven stages, each teaching one capability then *replaying* earlier ones so
specialisation does not erase them:

| stage | capability | token budget | learning rate | replay |
|---|---|---|---|---|
| stage1 | language + grammar | 60M | 6e-4 | 0% |
| stage2 | reasoning + mathematics | 60M | 5e-4 | 15% language |
| stage3 | knowledge + comprehension | 60M | 5e-4 | 25% (reasoning/language/dialogue) |
| stage4 | instruction + dialogue | 45M | 4e-4 | 25% (reasoning/language/knowledge) |
| stage5 | tool use | 45M | 4e-4 | 25% (reasoning/language/dialogue/knowledge) |
| stage6 | safety + robustness | 25M | 3e-4 | 30% (language/reasoning/tools/instruction) |
| stage7 | integration + refinement | 50M | 3e-4 | 15% (all prior) |

Total budget: **345,000,000 tokens**. All of this is data, in
`configs/curriculum_v2.json` — replay fractions, per-stage budgets, learning
rates, and the primary/replay source weights. `tinymind data curriculum-v2-info
--stage <n>` prints a stage's effective mixture and budget; the mixture always
sums to 1.0 (primary normalised to `1 − replay_fraction`, replay to
`replay_fraction`).

Data is written with `tinymind data build-curriculum-v2 --stage <n> --out
<dir>`, producing `train_<source>.jsonl`, `val.jsonl`, `test.jsonl` (and an
`eval.jsonl` copy for the existing eval/gate tooling) plus a `curriculum.json`
recording the mixture, budget, categories and file hashes. The on-disk layout
matches v1 so the training data path is unchanged.

### Mathematics is verified, not plausible (brief section 4)

Every reasoning/mathematics example is generated programmatically and its final
answer computed with Python's standard library (`fractions`, `math`). The
generated answer follows a `#### <answer>` marker (scored by the `final_answer`
scorer, `tinymind/evaluation/scoring.py`). `build_stage` re-derives each
example's answer independently and drops any that do not match, so the training
set cannot fill with confident-looking wrong answers. Verified in
`tests/training/test_curriculum_v2.py` (`test_all_math_examples_are_verified`)
and `tests/training/test_stage_transitions_v2.py`
(`test_recompute_detects_a_tampered_answer`).

### Held-out tests are independent (brief section 11)

Each stage's `test.jsonl` is generated from *held-out buckets*: numerical
combinations, sentence templates, passages, tool argument values, conversation
structures, and safety wordings whose hash falls in a held-out bucket never
appear in `train`/`val`. `build_stage` additionally drops (and counts) any test
example whose prompt collides — exactly or after normalisation — with a
training prompt, and the existing contamination checker
(`tinymind/data/contamination.py`) runs over the result. Verified in
`tests/training/test_curriculum_v2.py` and `test_stage_transitions_v2.py`.

### External / real corpora (brief section 13)

Stage 1 (language) and Stage 3 (knowledge) are designed to also consume a real
text/knowledge corpus, supplied through `datasets/v2/manifest.json` and
prepared with `tinymind data prepare-external --manifest datasets/v2/manifest.json
--out <dir>` (`tinymind/data/external.py`). The repository commits **no** large
corpora: `synthetic` entries are generated deterministically at preparation
time; `local` entries point at a corpus you place on disk and are verified
against a recorded SHA-256; `url` entries require an explicit network-enabled
run (`--allow-download`) and are refused otherwise, so CI stays hermetic and
never silently produces an empty corpus. To add a real corpus, drop a JSONL
(records with a `text` field) under `datasets/v2/corpora/`, add a `local` entry
with its `sha256`, and it becomes active. The prepared manifest records each
entry's `type`, so a synthetic placeholder is never mistaken for real data.

## Token-budget training (brief sections 6, 8)

A stage is defined by a **token budget**, not a step count. `tinymind train
--config 50m --target-tokens <N>` derives the step horizon as
`ceil(target_tokens / (batch × accumulation × seq_len))`. The training summary
gains a `token_accounting` block (total training tokens, loss tokens,
tokens/sec, optimizer steps, effective batch tokens, stage completion
percent). The budget is *runtime-only*: it is excluded from the run's
compatibility hash, which is taken over the derived step horizon — so a resume
whose budget differs is still correctly refused, without double-counting.
Sequence packing and token-normalised gradient accumulation (loss summed over
all real target tokens across microbatches, divided once) are unchanged from
v1's engine (`tinymind/training/engine.py`). Verified in
`tests/training/test_token_budget.py`.

**Revised: the budget is a minimum, not the finish line.** A v2 stage now
completes only when its *objective* is met (`configs/stages_v2/<stage>.objective.json`,
measured on the live model at every evaluation checkpoint). A stage that uses
its whole budget without meeting the objective stays incomplete
(`stop_reason: gate_failed`) and is continued with `--continue-stage` and a
larger budget. Stage 1 trains primarily on a real corpus (`data prepare-corpus`,
`build-curriculum-v2 --corpus`). Every checkpoint publishes a report with the
raw generations. See [objective-driven-stages.md](objective-driven-stages.md).

## Memory / throughput preflight (brief section 7)

Before a full run, `tinymind benchmark train-step --config 50m` builds the real
model and optimizer and measures, per `(batch, accumulation)` shape at a modest
sequence length: model init time, forward, backward, optimizer-step seconds,
peak RSS, and tokens/sec. It reports the exact parameter count and a `passed`
flag; a shape that cannot complete a step is reported as an error rather than
silently dropped, and the preflight fails. The 50M workflow runs this as a gate
before the first stage. Verified in `tests/training/test_benchmark_50m.py`.

## Evaluation (brief section 10)

The existing capability suite (`tinymind/evaluation/tiny_suite.py`) already
reports **each category separately** with no composite quality score, plus a
separate tool-behaviour block. v2 adds `tinymind/evaluation/suite_v2.py`, which
attaches an explicit eleven-category report — language, grammar, reasoning,
math, knowledge, comprehension, instruction, dialogue, tools, safety,
integration — where each category rolls up its raw sub-categories with
example-count weighting, **preserving every individual metric** and computing no
overall score. It also lists which categories the eval set actually covered
(`present`/`missing`). Verified in `tests/test_suite_v2.py`.

## Promotion gates (brief section 9)

Stages 1–6 each have a machine-checkable gate
(`configs/stages/stageN.gate.json`) applied to the *completed parent* before
the next stage starts. Gates reference real metric paths the suite emits
(`eval.capabilities.<category>.accuracy`, `eval.tool_behavior.<rate>`,
`summary.final_validation.val_loss`) and include **regression floors** on prior
capabilities, so a stage that improves its new skill while destroying an old one
does not pass. A missing metric fails its check. Stage 7 is final and has no
gate. Thresholds are floors set *before* the run, not predictions of quality;
tighten them after the first full-budget run. The gate evaluator
(`tinymind/training/gate.py`) is unchanged. Verified in
`tests/training/test_stage_gates_v2.py`.

## Checkpoint / artifact identity (brief section 14)

Unchanged from v1 and reused: checkpoints pin the model config and tokenizer
spec; a completed stage is bundled into a verified artifact
(`tinymind/ci/stage_io.py`); the next stage verifies checksums, checkpoint
integrity, identity, optional pinned manifest hash, and the parent gate *before
trusting it*, then `init-from`s the parent (fresh optimizer state) while
recording the parent's stage and manifest hash as lineage. A same-stage
continuation `resume`s instead (optimizer state carried, trajectory
reproducible). Demonstrated end-to-end on the fast profile in the session that
built this (token-budget stop → resume to completion → init-from into the next
stage with the parent recorded).

## CI (brief section 12)

`.github/workflows/train-50m.yml` runs any stage on a GitHub-hosted runner,
mirroring `train-stage.yml`: it confirms the profile is the 50M model, runs the
preflight, prepares external + v2 curriculum data, reads the stage's token
budget and learning rate from `configs/curriculum_v2.json`, trains to the token
budget (stopping cleanly on its own wall-clock budget first, checkpointing and
exporting before exit), evaluates the held-out suite when the stage completes,
bundles and verifies the artifact, and hands it to the next stage. Inputs are
routed through the environment (never interpolated into shell text). Structure
verified in `tests/ci/test_train_50m_workflow.py`.

## Commands, end to end

```
# 0. (once) train the 16k BPE tokenizer on a real corpus
tinymind tokenizer train-bpe --input corpora/ --vocab-size 16000 --output tokenizers/v2-16k.json

# 1. confirm the model
tinymind model info 50m                       # -> 50,370,624 parameters

# 2. preflight the runner
tinymind benchmark train-step --config 50m

# 3. per stage: prepare the real corpus and the curriculum, then train until the objective is met
tinymind data prepare-external --manifest datasets/v2/corpus.manifest.example.json --out data/external --allow-download
tinymind data prepare-corpus   --manifest datasets/v2/corpus.manifest.example.json --out data/corpus --allow-download
tinymind data build-curriculum-v2 --stage stage1 --out data --corpus data/corpus
tinymind data curriculum-v2-info --stage stage1        # budget + mixture
tinymind train --config 50m --dataset data --output out/stage1 \
    --stage stage1 --max-steps 0 --target-tokens 60000000
tinymind objective-report --run out/stage1              # every checkpoint, raw generations
# objective not met at the budget (stop_reason gate_failed)? continue the SAME stage with a larger budget:
tinymind train --config 50m --dataset data --output out/stage1b --stage stage1 --max-steps 0 \
    --target-tokens 75000000 --continue-stage out/stage1/checkpoints

# 4. evaluate the held-out suite (per-category, no composite)
tinymind eval --package out/stage1/export --eval data/eval.jsonl --val data/val.jsonl --out eval.json

# 5. gate, then init-from into the next stage (CI does this via the artifact)
```

## Limitations (read `STATUS.md` for the full accounting)

* No full 345M-token 50M run has been executed here; this environment has no
  GPU, no deep-learning framework, and no network, so a production run belongs
  on a runner. What is proven here: the exact parameter count, that the model
  builds and does forward+backward+optimizer steps, the preflight numbers, the
  token-budget/resume/init-from machinery, curriculum determinism/verification/
  contamination control, and gate rejection — all via direct execution and
  tests.
* The synthetic generators demonstrate *capabilities and generalisation across
  held-out values*, not broad world knowledge. Real language/knowledge breadth
  depends on supplying a real corpus through the dataset manifest.
* Gate thresholds are provisional floors, to be tightened after a real run.
* The native (C++) inference runtime targets the v1 byte tokenizer; running the
  16k BPE tokenizer inside the native runtime is not yet wired.
