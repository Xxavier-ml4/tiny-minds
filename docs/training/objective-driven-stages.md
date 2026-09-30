# Objective-driven stages and real pretraining data

This document describes how a TinyMind v2 training stage decides it is done,
where Stage 1's language data comes from, and what every evaluation checkpoint
publishes. Nothing here depends on model size: the 50M, 100M, 300M, 500M and
1B profiles use the same code and differ only in thresholds, prompts and data.

## A stage completes when its objective is met

Before this change a stage was complete when its token budget was used up.
Now:

* The **token budget** (`configs/curriculum_v2.json` `target_tokens`) is the
  *minimum* training chunk and the learning-rate horizon. It is never a
  promotion criterion on its own.
* At every evaluation checkpoint the trainer measures the stage **objective**
  on the live model (`tinymind/training/objective.py`). Its thresholds are in
  `configs/stages_v2/<stage>.objective.json`.
* The objective is **met** only when all of these hold: the minimum budget is
  trained, **every** measurement passes, the retained earlier capabilities
  pass, and nothing regressed against the previous checkpoint. No single number
  (validation loss included) can promote a stage.
* `training_summary.json` `stage_complete` is true only when the objective was
  met. The checkpoint manifest records the same.

Stop reasons (`stop_reason`):

| reason | meaning | stage | what runs next |
|---|---|---|---|
| `gate_passed` | the objective was met (possibly before the horizon) | complete | the next stage, `--init-from` (after its promotion gate) |
| `gate_failed` | the whole budget was used and the objective is not met | **incomplete** | the same stage, `--continue-stage` with a larger budget |
| `time_budget`, `step_limit`, `signal N` | the job stopped early | incomplete | the same stage, `--resume` (exact continuation) |
| `diverged` | NaN/Inf; the last good checkpoint is kept | incomplete | investigate |
| `complete` | legacy: no objective configured (v1 pipeline, `--no-objective`) | complete at the budget | unchanged behaviour |

### Three ways to continue a checkpoint

| flag | when | what it does |
|---|---|---|
| `--resume` | same stage, budget left | Bitwise-exact continuation within the same horizon. It refuses a checkpoint whose budget is used up and points to `--continue-stage`. |
| `--continue-stage` | same stage, objective not met at its budget | Loads weights, optimizer moments, RNG and data position. Continues into a **larger** horizon (a fresh schedule segment resumed at the same step). It enforces every identity check of `--resume` except the horizon length. It refuses another stage's checkpoint, a horizon that does not grow, and a complete stage. |
| `--continue-stage --reopen-stage` | same stage, objective met but its promotion gate failed | As above, for a complete checkpoint. It trains the whole extension before the stage may complete again. |
| `--init-from` | the next stage | Weights from a complete stage. It refuses an incomplete parent (objective not met, or stopped early); `--allow-incomplete-parent` overrides this for engineering runs only. |

In GitHub Actions, `python -m tinymind.ci.stage_io check-incoming` picks the
mode from the incoming bundle, which records `stop_reason`, `budget_exhausted`,
`total_steps`, `budget_tokens` and the objective state:

* `resume`: incomplete, with budget left.
* `continue`: incomplete with the budget used, or complete with the stage's
  own promotion gate failing (reopen).
* `init-from`: a different stage from a complete parent whose promotion gate
  passes. An incomplete parent is rejected with the reason.

The `train-50m` workflow applies the mode: for `continue` it trains
`parent budget + continue_extend_tokens` (default: a quarter of the stage
budget), and for `resume` it keeps the parent's horizon.

## What is measured

| section | measurements | notes |
|---|---|---|
| `loss` | `val_loss`, `val_ppl` on the stage's validation split; `text_val_loss`, `text_val_ppl`, **`text_val_bpb`** on natural held-out text | Bits per byte (summed loss / (ln 2 · UTF-8 bytes of the predicted text)) does not depend on the tokenizer, so one threshold means the same thing for a byte-level model and a 16k-BPE model. |
| `generation` | non-empty rate, mean repetition (char 6-grams), looping rate, distinct-1/2, mean characters/tokens, EOS rate | Greedy continuations of the objective's **fixed prompts**. |
| `grammar` | word-like tokens, **known words** (training-corpus lexicon), letters/punctuation share, sentence starts capitalised after a terminator, space after punctuation, word/sentence length | Reliable automatic checks, not a grammar judge. A model that repeats "the the the" passes the word checks and fails the repetition, looping and diversity checks, which is why every check must pass. |
| `data` | `natural_train_bytes` of the training data | Decided before training: a run whose data can never satisfy the objective is refused up front. |

Stage 1 (`stage1.objective.json`) is the language gate. It has 13
measurements across all four sections and 8 fixed English prompts, and it
requires at least 20 MB of natural training text. Its validation-loss ceiling
mirrors `stage1.gate.json`, so the stage cannot complete and then fail its own
promotion gate on the same number. (A test enforces this for every stage.)

Stages 2–7 keep a validation-loss ceiling (mirroring their promotion gates), a
natural-text bits-per-byte ceiling, and degradation guards on generation.
Their capability floors (math, knowledge, tools, safety, ...) stay in the
promotion gates, which run on the held-out test split once a stage is complete.

### Regression checks

* **Same stage:** each checkpoint is compared with the previous one: loss,
  natural-text bits per byte, repetition, diversity, word-likeness, known
  words. Tolerances come from the `regression` block. A comparison survives
  `--resume` and `--continue-stage`, because the last report is stored in
  the checkpoint. A fresh stage is not compared with its random
  initialisation.
* **Later stages:** `"retain": ["stage1"]` re-measures stage 1's generation and
  grammar checks with stage 1's prompts and thresholds on the later stage's
  live model. They must still pass. They are also compared with the previous
  checkpoint and with stage 1's metrics **at promotion**, which catches slow
  drift. After `--init-from`, the parent's final report is the first
  reference. Natural held-out text is compared across stages when the
  held-out split is identical.

## What every checkpoint publishes

For every evaluation (plus a baseline before training in a stage), the trainer
writes `objective_reports/step-NNNNNNNN.json` and `.md`, `latest.*`, and a
`history.jsonl` row. The Markdown report contains:

* step, budget tokens, real tokens, and the minimum before promotion;
* validation loss and perplexity, and natural-text loss, perplexity and bits
  per byte, each with its change against the previous checkpoint;
* generation statistics and grammar metrics;
* every objective measurement with its threshold and result, and the reasons
  when the objective is not met;
* retained capabilities, with their own raw generations;
* the regression comparison;
* **the exact fixed prompts and the raw generations, verbatim** (fenced blocks;
  control characters shown as `\xNN`; the JSON holds the exact text). Raw
  output is never replaced by a score.

`tinymind objective-report --run out --github-summary` puts a table of every
checkpoint, the newest report in full, and earlier reports in collapsible
sections into the GitHub job summary, within the 1 MiB limit. The workflow
also uploads `out/objective_reports/` as an artifact, and the stage bundle
carries the reports as well.

## Stage-1 data: a real corpus, synthetic as a supplement

`tinymind data prepare-corpus --manifest <manifest> --out data/corpus [--allow-download]`
(`tinymind/data/corpus.py`) reads the manifest's `local`/`url` entries:

* **Sharded streaming.** Local paths, `file://`, and `http(s)://` (the last
  only with `--allow-download`). `.gz` shards are decompressed. Each shard is
  hashed while streaming.
* **Checksums.** A shard is verified against `sha256` when one is given. The
  observed hash is always recorded, so it can be pinned afterwards.
* **Private data.** `auth_env` names an environment variable holding a bearer
  token. The token is sent only to the shard's own host, not on redirects, and
  is never logged or recorded. URLs are recorded without user-info or query
  strings. In `train-50m.yml`, the repository secrets `HF_TOKEN` and
  `CORPUS_TOKEN` are passed to the data-preparation step only (never to
  training), so use one of those two names as `auth_env`.
* **Chunking.** Paragraphs are packed into chunks of about `chunk_chars`
  characters. Keep `chunk_chars × tokens-per-char + 2 ≤ max_seq_len`: longer
  chunks are dropped when the trainer renders them. The objective's
  `data.natural_train_bytes` counts only the natural text that survives, and
  the trainer prints a warning naming how many chunks were dropped.
* **Size caps.** `max_records` caps a source's chunks and is spread evenly
  over its shards (at most ceil(max_records / shards) from each), so a capped
  multi-shard corpus draws from every shard.
* **Deduplication across the whole corpus.** It removes repeated paragraphs,
  exact duplicate chunks (normalised) and near duplicates (MinHash LSH over
  5-word shingles; `--dedup near|exact|none`).
* **Deterministic train/val/test.** Assignment is by document group (a JSONL
  document, or `group_chunks` consecutive chunks of a text shard), ranked by a
  seeded hash. A group never spans splits. Cross-split overlap is verified.
  The split seed belongs to the corpus (default 0), not to the training run:
  the workflow never passes the training seed. Every stage's job therefore
  derives the same split, the same natural held-out text and the same
  tokenizer.
* **Decontamination.** `--decontaminate eval.jsonl ...` drops chunks sharing
  an 8-gram with evaluation prompts.
* **Provenance.** `corpus_manifest.json` records the dataset-manifest hash,
  the hash of the tokenizer used for counting, each source's license, redacted
  URLs and per-shard hashes, token/byte counts per split, the natural fraction,
  and the dedup and contamination reports. It travels with the stage data. The
  trainer copies it and `curriculum.json` into `<run>/data_provenance/`, and
  the stage bundle carries them, hashed, together with the tokenizer sample
  manifest. A trained model can therefore be traced to its data and licenses.

**Tokenizer input.** `tinymind data tokenizer-sample --corpus data/corpus
--external data/external --out data/tokenizer_input` writes the BPE training
input. It is:

* **Deterministic and uniform:** each record is ranked by `sha256(seed:id)`
  and the lowest ranks are kept (a streaming bottom-k), so the sample does not
  depend on file or shard order.
* **Drawn from every shard:** the natural sample comes from every shard and
  source of the corpus's TRAIN split, never its held-out splits. A separate
  sample comes from the synthetic supplements.
* **Recorded:** `sample_manifest.json` records the input hashes and counts.

It is never "the first N records": on a 6-shard test corpus, the first 10,000
lines covered only 2 shards. The seed is fixed (0), so every stage's job
derives the same tokenizer.

Pure-Python BPE training costs roughly 1.8 µs per unique word per merge. The
16k vocabulary needs about 15,700 merges, so keep the sample modest (the
default is 8,000 corpus chunks, about 8 MB). For a real run, train the
tokenizer once and commit it to the profile's tokenizer path: the workflow
then skips this step. The trainer's wall-clock budget already subtracts the
time a job spends on data preparation and tokenizer training.

`tinymind data build-curriculum-v2 --stage stage1 --out data --corpus data/corpus`
adds the corpus to a stage, as the stage's `corpus` block in
`configs/curriculum_v2.json` says:

* **Stage 1** takes `byte_share: 0.8`, so 80% of its training *text* is
  natural corpus. The synthetic generators are the supplement.
* **Stages 2–7** replay 10%.

The build writes these files:

* `train_corpus.jsonl`, decontaminated against the stage's validation and
  test prompts;
* `val_text.jsonl` / `test_text.jsonl`, natural held-out text for the
  objective: a seeded uniform sample across every shard of the corpus's
  held-out splits, identical for every stage built from the same corpus;
* `lexicon.txt`, for the known-word check;
* `corpus_manifest.json`, the corpus provenance;
* a `corpus` block in `curriculum.json`.

The default `datasets/v2/manifest.json` is hermetic. Its only natural text is
a small committed CC0 sample, which the Stage-1 objective rejects as too small.
`datasets/v2/corpus.manifest.example.json` shows a real configuration: public
C4 shards, a pinned local shard, and how to add a private shard with
`auth_env`. Prepared and downloaded corpora are ignored by git.

## Other model sizes

The mechanism has no size-specific code. To run a 500M or 1B profile:

1. Copy `configs/stages_v2/*.objective.json` to, for example,
   `configs/objectives/1b/` and adjust the thresholds. Bits per byte and the
   generation and grammar floors are directly comparable across sizes; only
   the targets should move.
2. Point the run at them with `--objective-dir configs/objectives/1b`, or put
   `objective_dir: objectives/1b` in the profile YAML.
3. List more shards in the dataset manifest (and use `--dedup exact` for a
   corpus deduplicated upstream).

The objective loads automatically only for v2 curriculum data, a profile with
`objective_dir`, or an explicit `--objective-dir`, so the v1 pipeline
(`train-stage.yml`) is unchanged. `--no-objective` restores budget completion
for engineering runs.

## Tests

* `tests/training/test_corpus.py`: manifest ingestion, checksums, network
  refusal, credentials never recorded, dedup, deterministic splits,
  decontamination, tamper detection, curriculum attachment.
* `tests/training/test_objective.py`: measurements, the all-must-pass verdict,
  the minimum budget, regression detection (same stage, retained, at
  promotion, natural text across stages), verbatim rendering, and
  consistency of the shipped configs with the promotion gates.
* `tests/training/test_objective_stages.py`: incomplete on failure,
  completion on success, same-stage continuation and its refusals, reopen,
  regression blocking completion, retained capabilities across stages, the
  same flow at two model sizes.
* `tests/ci/test_objective_routing.py`: bundle facts, `continue` routing,
  next-stage rejection before the parent passes and acceptance after, reopen
  when a promotion gate fails, and the routing decision in the job summary.
* `tests/ci/test_objective_chain.py`: the whole chain through the real
  command lines, job by job, as the workflow runs it.
* `tests/training/test_objective_cli.py` and
  `tests/ci/test_train_50m_workflow.py`: the command line end to end, and the
  workflow wiring.
