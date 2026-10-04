# Fluent but looping output: diagnosis and remedies

**Symptom.** A checkpoint writes grammatical English but repeats itself ("the town of the town of the town of ...").
Three different things produce this, and they need different fixes, so measure before changing anything.

| Cause | How it shows up | What helps |
|---|---|---|
| **Greedy-decoding degeneration** (the model is fine or merely young) | train loss ≈ val loss, both still falling; the loop disappears with `--no-repeat-ngram-size 3` or sampling | decoding controls; more training |
| **Memorization** (data-limited: the same examples seen too many times) | `[data] WARNING` at start-up; the train/val gap keeps widening while val loss rises (`[eval] WARNING ... memorizing`) | more data, a smaller token budget, a lower weight on the repeated source, then dropout |
| **A broken model** | even the decoding-mitigated generations are garbage | look at the loss curve, LR and data, not at decoding |

Greedy decoding of small language models degenerates into repetition even when they are healthy (Holtzman et al.,
2019, *The Curious Case of Neural Text Degeneration*). Repeating training data many times is the other classic cause
(Muennighoff et al., 2023, *Scaling Data-Constrained Language Models*: roughly 4 passes are nearly as good as fresh
data; returns collapse after about 16). Dropout addresses only the second.

## Is it too early to judge?

Effective batch is `batch_size × grad_accum × max_seq_len` tokens per optimizer step: 1 × 16 × 1024 = 16,384 for
`configs/50m.yaml`. Step 400 is therefore about 6.6M tokens, each seen at most once, and the learning-rate warm-up has
only just finished. A model cannot have memorized text it has barely seen; loops at that point are the ordinary
behaviour of a young model under greedy decoding. Stage 1's 60M-token budget is about 3,660 steps.

## What the trainer now measures

* **Start-up, `[data] WARNING`**: `DataPlan.planned_passes` is *per-epoch repeat factor × number of epochs the token
  budget implies*, per source. Any source above `MAX_RECOMMENDED_PASSES` (4) is reported. The per-epoch
  `repeat_factors` alone hides the dangerous case: a token-budget stage keeps starting epochs, and an up-weighted
  small synthetic source multiplies on top. Advisory only; it never stops a run. The values are in
  `training_summary.json` → `dataset.planned_passes` and `dataset.warnings`.
* **Every evaluation**: the `[eval]` log line and the `val_history` / `metrics.jsonl` rows now carry
  `train_loss_recent` (mean of the last up to 20 step losses) and `generalization_gap = val_loss − train_loss_recent`.
  The absolute gap includes a distribution difference (training mixes in easy synthetic text), so read the *trend*.
  With dropout on, `train_loss_recent` is measured with dropout active, which understates the gap slightly.
* **`[eval] WARNING ... memorizing`** (`memorization_signal`): at each of the last 2 evaluations validation loss rose
  *and* training loss fell. A single noisy evaluation does not trigger it.
* **Stage reports**: stages whose objective sets `generation.diagnostic_decoding` (stage 1 does) also generate from the
  same fixed prompts with seeded sampling (temperature 0.7, top-p 0.9) and a 3-gram ban, reported as
  `metrics.generation_decoded` beside the raw numbers. **It is never gated**; see below.

## Why the stage gate stays on raw greedy output

Putting a repetition penalty into the gate would make the gate measure the *penalty*, not the model: a checkpoint that
loops under plain greedy decoding would pass `generation_not_looping`. The objective's point is that "the stage is
done" means the model itself no longer loops. So the gated measurements always use plain greedy decoding
(`generate_continuations` defaults), and no measurement may reference `generation_decoded` (the objective loader
rejects it). Read the two side by side: healthy decoded text with a looping raw decode means a decoding problem;
bad decoded text too means a model problem.

## Where decoding is set in the CI run

The GitHub workflow sets no decoding. Everything the stage report generates is decided in
`configs/stages_v2/<stage>.objective.json`, block `generation`, executed by `tinymind/training/objective.py`:

| key | what it controls | default |
|---|---|---|
| *(none)* | the **gated** measurements | plain greedy decoding |
| `diagnostic_decoding` | the extra, **never-gated** `generation_decoded` section | off (stage 1 sets it) |
| `gated_decoding` | the decoding the **gated** measurements use | off = greedy |

Both blocks take any of `temperature` (> 0 = seeded sampling), `top_p`, `top_k`, `seed` (default 0; prompt *i* uses
`seed + i`, so a measurement is reproducible), `repetition_penalty` and `no_repeat_ngram_size`. `top_p`/`top_k`/`seed`
need a temperature. Stage 1 currently ships
`"diagnostic_decoding": {"temperature": 0.7, "top_p": 0.9, "no_repeat_ngram_size": 3, "seed": 0}`.

Opting in to `gated_decoding` changes what "the stage is done" means, so the report names the decoding in its headings
(`fixed prompts, seeded sampling, temperature 0.7, ...`). If you do, prefer sampling **without** an n-gram ban, e.g.
`"gated_decoding": {"temperature": 0.7, "top_p": 0.9, "seed": 0}`: with a ban the looping and distinct-2 numbers are
close to guaranteed by construction, so they would stop measuring the model. The thresholds in the objective were set
for greedy output, so expect to recalibrate them.

## Decoding controls (for using a model, not for gating it)

```
tinymind generate --model m.tm --prompt "The sun rose over the" \
    --no-repeat-ngram-size 3            # forbid completing any 3-gram already in the text
tinymind generate ... --repetition-penalty 1.15      # CTRL-style; blunter, also penalizes "the"/"of"
tinymind generate ... --temperature 0.8 --top-p 0.9 --top-k 50 --seed 1   # sampling
```

Defaults are unchanged: with no flags, `generate` is deterministic greedy. `no_repeat_ngram_size` is the targeted
tool for phrase loops; a repetition penalty acts on every previously used token, so large values damage fluency.

## Dropout

`model.dropout` (default 0.0) is implemented. Where it applies, per transformer block: the attention probabilities
(after the softmax), the attention output, and the MLP output, i.e. the GPT-2 placement. It is inverted dropout
(survivors are scaled by `1/(1-rate)`), in both the fused kernel the trainer uses and the reference path; the two draw
identical masks from the same generator, and `tests/model/test_dropout.py` checks that they agree on outputs and on
every parameter gradient.

Design points worth knowing:

* **No train/eval flag.** Dropout runs only when a `dropout_rng` is passed to `forward`; the trainer does so for
  training micro-batches, nothing else does. Evaluation, generation, export and the native runtime are therefore
  dropout-free by construction, and it cannot be combined with a KV cache.
* **Resume is exact.** Masks are seeded from `(seed, optimizer step, micro-batch)`, not from consumed generator state,
  so a run interrupted and resumed ends bit-identical to an uninterrupted one
  (`tests/training/test_dropout_training.py`).
* **Not on by default for the 50M profile.** It slows learning when the model is training-limited, and it is part of
  the model-config hash, so a checkpoint resumes only under the rate it was trained with. Enable it (0.05–0.1) at the
  start of a stage if the diagnostics above show memorization. More data is the better first fix.
* Inference is unaffected: `.tm` export and the C++ runtime need no change.
