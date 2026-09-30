"""The Phase 3B training engine.

One optimizer step, precisely::

    plan  -> `gradient_accumulation_steps` micro-batches (each `batch_size` rows)
    N     =  total number of loss tokens across those micro-batches
    for each micro-batch:   loss_mb = sum(token losses) / N   ;  backward
    clip the global gradient norm, AdamW update at the scheduled LR

Dividing every micro-batch by the *whole step's* token count ``N`` (not by its
own) makes the accumulated gradient identical to the gradient of one batch of
``batch_size x accumulation`` rows however the tokens are split between
micro-batches — the Phase 3A version averaged micro-batch means and was off by
12 % of the largest gradient when micro-batches had unequal token counts.
(``tests/training/test_engine.py::test_batch8_equals_batch4_accum2``.)

Checkpoints are taken only after a completed optimizer step, so the gradient
buffer is empty by construction; resume reproduces the exact continuation
(``test_split_run_matches_continuous_run``: 100 steps == 50 + 50, bitwise).

Stopping: the step horizon reached, the time budget, or an external request
(SIGTERM/SIGINT in the CLI). In every case the engine writes a verified
checkpoint, exports the inference package, writes ``training_summary.json``
and returns — it never relies on being killed by a wall-clock timeout.

Stage completion. Without an objective (legacy / v1 pipeline) a stage is
complete when its step horizon is exhausted. With a stage objective
(:mod:`tinymind.training.objective`) the stage is complete only when the
objective is MET: the token budget is the *minimum* training chunk (and the LR
horizon), and the objective is evaluated at every evaluation checkpoint. Stop
reasons: ``gate_passed`` (objective met; the stage may end before its
horizon), ``gate_failed`` (horizon reached, objective not met: the stage stays
INCOMPLETE), ``time_budget``, ``step_limit``, ``signal N``, ``diverged``.
Three ways to continue a checkpoint, all with the same identity protections:
``--resume`` (exact continuation inside the same horizon), ``--continue-stage``
(same stage, LARGER budget, for an objective still failing at its budget, or —
with ``reopen`` — for a stage whose promotion gate failed) and ``--init-from``
(the next stage, from a completed one).
"""
from __future__ import annotations

import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from tinymind.model import ModelConfig, TinyMindTransformer
from tinymind.model.optim import AdamW
from tinymind.model.tensor import no_grad
from tinymind.model.tokenizer import Tokenizer
from tinymind.training import checkpoint as ckpt
from tinymind.training.config import TrainingConfig, TrainingConfigError
from tinymind.training.data import DataPlan, DataSource, DatasetError, TokenizedDataset, sequential_batches
from tinymind.training.objective import StageObjective, capability_snapshot, history_row, render_markdown
from tinymind.training.render import IGNORE_INDEX, ChatRenderer
from tinymind.training.schedule import LRSchedule

_RNG_STREAM = 0x7A11


class TrainingDivergedError(FloatingPointError):
    """Loss or gradients became NaN/Inf. The offending step was not applied and
    no checkpoint was written for it; the last good checkpoint is untouched."""


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class TrainingEngine:
    def __init__(self, *, model_config: ModelConfig, tokenizer: Tokenizer, config: TrainingConfig,
                 sources: Sequence[DataSource], validation: TokenizedDataset | None,
                 output_dir: str | Path, resume: str | Path | None = None, init_from: str | Path | None = None,
                 continue_from: str | Path | None = None, reopen: bool = False, allow_incomplete_parent: bool = False,
                 carry_optimizer: bool = False, allow_no_validation: bool = False, stop_after_steps: int | None = None,
                 objective: StageObjective | None = None, objective_min_tokens: int | None = None,
                 objective_validation: TokenizedDataset | None = None, objective_data: dict[str, Any] | None = None,
                 objective_lexicon: "set[str] | frozenset[str] | None" = None,
                 clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] | None = print,
                 fault_hook: Callable[[str], None] | None = None,
                 exporter: Callable[["TrainingEngine", Path], dict[str, Any]] | None = None) -> None:
        if sum(bool(x) for x in (resume, init_from, continue_from)) > 1:
            raise TrainingConfigError("--resume, --init-from and --continue-stage are different operations; pass only one")
        if reopen and not continue_from:
            raise TrainingConfigError("reopen applies only to --continue-stage")
        config.validate()
        model_config.require_supported()
        if model_config.vocab_size != tokenizer.vocab_size:
            raise TrainingConfigError(f"model vocab_size {model_config.vocab_size} != tokenizer vocab_size "
                                      f"{tokenizer.vocab_size}: a byte tokenizer needs 260, a subword one its own size")
        if config.max_seq_len > model_config.max_seq_len:
            raise TrainingConfigError(f"training max_seq_len {config.max_seq_len} exceeds the model's "
                                      f"{model_config.max_seq_len}")
        if validation is None and not allow_no_validation:
            raise TrainingConfigError("every stage needs a validation set (pass one, or allow_no_validation=True "
                                      "for an engineering smoke test)")
        self.model_config, self.tokenizer, self.config = model_config, tokenizer, config
        self.renderer = ChatRenderer(tokenizer)
        self.output_dir = Path(output_dir)
        self.checkpoint_root = self.output_dir / "checkpoints"
        self.clock, self._log_fn, self.fault_hook, self.exporter = clock, log, fault_hook, exporter
        self.stop_after_steps = stop_after_steps  # run at most this many steps in THIS invocation (chunked runs)
        self.validation = validation

        # ---- data ----------------------------------------------------------
        mixture = dict(config.mixture)
        if mixture:
            names = {s.name for s in sources}
            if set(mixture) != names:
                raise TrainingConfigError(f"mixture keys {sorted(mixture)} must match the data sources {sorted(names)}")
            sources = [DataSource(s.name, s.dataset, float(mixture[s.name])) for s in sources]
        spec = self.renderer.spec()
        for s in list(sources) + ([DataSource("validation", validation)] if validation else []) + \
                ([DataSource("objective_validation", objective_validation)] if objective_validation else []):
            if s.dataset.renderer.spec() != spec:
                raise TrainingConfigError(f"dataset {s.name!r} was rendered with a different tokenizer/template")
        self.plan = DataPlan(sources, seed=config.seed, batch_size=config.batch_size, max_seq_len=config.max_seq_len,
                             packing=config.packing, pad_id=tokenizer.pad_token_id,
                             epoch_examples=config.epoch_examples or None)
        accum = config.gradient_accumulation_steps
        if self.plan.steps_in_epoch(0, accum) < 1:
            raise DatasetError(f"the data yields {self.plan.num_micro_batches(0)} micro-batch(es) of {config.batch_size} "
                               f"row(s) per epoch, fewer than gradient_accumulation_steps={accum}")
        # Effective batch tokens: the tokens in one optimizer step's worth of rows.
        # With packing every row is filled to max_seq_len, so this is the token
        # throughput the stage token budget (v2) is measured against.
        self.effective_batch_tokens = config.batch_size * accum * config.max_seq_len
        self.stage_target_tokens = int(config.target_tokens) or None
        if config.max_steps > 0:
            self.total_steps = config.max_steps
        elif config.target_tokens > 0:
            # Token-budget training (brief section 6): derive a deterministic step
            # horizon from the budget so the LR schedule has an end and the stage
            # stops when it has seen ~target_tokens. Identical on resume because it
            # is a pure function of the (identity-pinned) config, and it flows into
            # the trajectory hash through total_steps.
            self.total_steps = max(1, math.ceil(config.target_tokens / self.effective_batch_tokens))
        else:
            self.total_steps = sum(self.plan.steps_in_epoch(e, accum) for e in range(config.epochs))
            if self.total_steps < 1:
                raise DatasetError("epochs x steps-per-epoch is 0")

        # ---- model / optimizer / schedule --------------------------------------
        self.model = TinyMindTransformer(model_config, seed=config.seed)
        named = list(self.model.named_parameters())
        self.optimizer = AdamW([p for _, p in named], learning_rate=config.learning_rate,
                               betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay,
                               names=[n for n, _ in named],
                               decay_min_ndim=2 if config.weight_decay_exclude_norms else 0)
        self.schedule = LRSchedule(config.scheduler, config.learning_rate, config.min_learning_rate,
                                   config.warmup_steps, self.total_steps)
        self.rng = np.random.default_rng(np.random.SeedSequence([config.seed, _RNG_STREAM]))

        # ---- progress ---------------------------------------------------------------
        self.step = 0
        self.epoch = 0
        self.cursor = 0
        self.cumulative_steps = 0
        self.tokens_processed = 0
        self.loss_tokens_processed = 0
        self.examples_processed = 0
        self.train_seconds = 0.0
        self.parent: dict[str, Any] | None = None
        self.resumed_from: str | None = None
        self.load_notes: list[str] = []
        self.initial_step = 0
        self._stop_reason: str | None = None
        self._ema_step = 0.0
        self._ema_ckpt = 0.0
        self.recent_losses: list[float] = []
        self.first_train_loss: float | None = None
        self.val_history: list[dict[str, float]] = []
        self.initial_validation: dict[str, float] | None = None
        self.last_checkpoint: Path | None = None
        self.exports: dict[str, Any] = {}
        self._compat = config.compat_dict(self.total_steps)

        # ---- stage objective (objective-driven completion) ----------------------------------
        # The token budget (the step horizon) is the MINIMUM training chunk: the objective may
        # promote only once objective_min_tokens budget tokens (steps x effective batch tokens,
        # the unit the budget is defined in) are trained. Default: the stage's own budget.
        self.objective = objective
        self.objective_validation = objective_validation
        self.objective_data = dict(objective_data or {})
        self.objective_lexicon = objective_lexicon
        self._explicit_min_tokens = objective_min_tokens is not None
        self.objective_min_tokens = int(objective_min_tokens) if objective_min_tokens is not None \
            else int(self.stage_target_tokens or self.effective_batch_tokens * self.total_steps)
        self.objective_reports_dir = self.output_dir / "objective_reports"
        self._last_objective_report: dict[str, Any] | None = None   # newest report (this run)
        self._prev_objective_report: dict[str, Any] | None = None   # what the next report compares with
        self._capability_reference: dict[str, Any] = {}              # retained stages' metrics at promotion
        self._objective_met = False
        self._stage_complete = False
        self._text_bytes = self._target_bytes(objective_validation) if objective_validation is not None else 0
        if objective is not None:
            failing = objective.data_failures(self.objective_data)
            if failing:
                raise TrainingConfigError(
                    f"the {config.stage} objective can never be met with this training data: " + "; ".join(
                        f"{c['metric']}={c['value']!r} (needs {c['requirement']})" for c in failing)
                    + ". Attach the data the objective requires (see the dataset manifest), or run without the "
                      "objective (--no-objective) for an engineering run.")

        if resume:
            self._restore_for_resume(Path(resume))
        elif continue_from:
            self._continue_from(Path(continue_from), carry_optimizer, reopen)
        elif init_from:
            self._init_from(Path(init_from), carry_optimizer, allow_incomplete_parent)

    # ------------------------------------------------------------------------
    def log(self, message: str) -> None:
        if self._log_fn:
            self._log_fn(message)

    def request_stop(self, reason: str) -> None:
        """Ask the loop to checkpoint and exit at the next step boundary
        (used by the CLI's SIGTERM/SIGINT handlers)."""
        self._stop_reason = self._stop_reason or reason

    # ---- restoring -------------------------------------------------------------------
    def _identity(self) -> dict[str, Any]:
        return dict(stage=self.config.stage, model_config=self.model_config.to_dict(),
                    tokenizer_spec=self.tokenizer.spec(), renderer_spec=self.renderer.spec(),
                    dataset_hash=self.plan.dataset_hash(), dataset_identity=self.plan.identity(),
                    training_compat=self._compat)

    def _apply_weights(self, loaded: ckpt.LoadedCheckpoint) -> None:
        params = dict(self.model.named_parameters())
        for name, arr in loaded.weights.items():  # verified: same names/shapes as the model
            if name not in params or params[name].data.shape != arr.shape:
                raise ckpt.CheckpointCorruptError(f"weight {name} does not fit this model")
        for name, arr in loaded.weights.items():
            params[name].data[...] = arr

    def _restore_for_resume(self, path: Path) -> None:
        resolved, notes = ckpt.resolve_checkpoint(path)
        self.load_notes = notes
        for note in notes:
            self.log(f"[resume] WARNING: {note}")
        loaded = ckpt.load_checkpoint(resolved, verify=False)  # resolve_checkpoint verified it
        ckpt.validate_resume(loaded, **self._identity())  # raises before anything is modified
        self.optimizer.load_state_dict(loaded.optimizer_state)
        self.schedule.load_state_dict(loaded.state["scheduler"])
        self._apply_weights(loaded)
        self.rng = ckpt.rng_from_json(loaded.state["rng"])
        p = loaded.state["progress"]
        self.step, self.epoch, self.cursor = int(p["global_step"]), int(p["epoch"]), int(p["cursor"])
        self.cumulative_steps = int(p["cumulative_steps"])
        self.tokens_processed = int(p["tokens_processed"])
        self.loss_tokens_processed = int(p["loss_tokens_processed"])
        self.examples_processed = int(p["examples_processed"])
        self.recent_losses = list(p.get("recent_losses", []))
        self.first_train_loss = p.get("first_train_loss")
        self.train_seconds = float(loaded.state.get("metrics", {}).get("train_seconds", 0.0))
        self.val_history = list(loaded.state.get("metrics", {}).get("val_history", []))
        self.initial_validation = loaded.state.get("metrics", {}).get("initial_validation")
        self._restore_objective_state(loaded.state.get("metrics", {}))
        self.parent = loaded.manifest.get("parent")
        self.initial_step = self.step
        self.resumed_from = str(resolved)
        if self.schedule.last_step != self.step:
            raise ckpt.CheckpointCorruptError("scheduler step and global step disagree")
        if self.step >= self.total_steps:
            # validate_resume above refused a COMPLETED checkpoint; this one is incomplete but its whole
            # horizon is used up: the objective was not met at the token budget. Nothing is left to do in
            # this horizon, and an exact resume cannot lengthen it.
            raise ckpt.ResumeMismatchError(
                f"cannot resume {resolved.name}: step {self.step} has used the whole token budget "
                f"(total_steps={self.total_steps}) but the stage objective was not met. Continue the SAME stage with "
                "a larger budget: --continue-stage with a larger --target-tokens/--max-steps.")
        self.log(f"[resume] continuing {self.config.stage} from {resolved.name} "
                 f"(step {self.step}/{self.total_steps}, epoch {self.epoch}, cursor {self.cursor})")

    def _restore_objective_state(self, metrics: dict[str, Any]) -> None:
        """Carry the objective's memory across runs: the last report (what the
        next checkpoint is compared with for regression), the retained stages'
        promotion-time reference, and the minimum budget the stage started with."""
        self._prev_objective_report = metrics.get("objective_report")
        self._capability_reference = dict(metrics.get("capability_reference") or {})
        stored_min = metrics.get("objective_min_tokens")
        if stored_min is not None and not self._explicit_min_tokens:
            self.objective_min_tokens = int(stored_min)

    def _init_from(self, path: Path, carry_optimizer: bool, allow_incomplete_parent: bool = False) -> None:
        resolved, notes = ckpt.resolve_checkpoint(path)
        for note in notes:
            self.log(f"[init-from] WARNING: {note}")
        loaded = ckpt.load_checkpoint(resolved, verify=False)
        ckpt.validate_init_from(loaded, model_config=self.model_config.to_dict(), tokenizer_spec=self.tokenizer.spec(),
                                renderer_spec=self.renderer.spec())
        if not loaded.manifest.get("stage_complete"):
            # A stage is promoted only once it is complete (for a stage with an objective: the objective was met).
            if not allow_incomplete_parent:
                raise ckpt.ResumeMismatchError(
                    f"cannot start {self.config.stage!r} from {resolved.name}: stage {loaded.manifest['stage']!r} is "
                    "not complete (its objective is not met, or it stopped early). Resume or --continue-stage it "
                    "first (--allow-incomplete-parent overrides this for engineering runs only).")
            self.log(f"[init-from] WARNING: parent stage {loaded.manifest['stage']!r} is NOT complete "
                     "(--allow-incomplete-parent)")
        self._apply_weights(loaded)
        if carry_optimizer:
            state = loaded.optimizer_state
            state["hyper"] = self.optimizer.hyperparameters()  # new stage may retune betas/decay; moments carry over
            state["lr"] = self.optimizer.lr
            self.optimizer.load_state_dict(state)
        prior = loaded.state["progress"]
        self.cumulative_steps = int(prior["cumulative_steps"])
        # The parent's last objective report is what this stage's first checkpoint is compared with
        # (retained capabilities); the parent's own capabilities at promotion join the reference set.
        parent_metrics = loaded.state.get("metrics", {})
        self._capability_reference = dict(parent_metrics.get("capability_reference") or {})
        parent_report = parent_metrics.get("objective_report")
        if parent_report:
            self._prev_objective_report = parent_report
            if loaded.manifest.get("stage_complete") and parent_report.get("objective_met"):
                self._capability_reference.setdefault(parent_report["stage"], capability_snapshot(parent_report))
        self.parent = {"checkpoint": resolved.name, "stage": loaded.manifest["stage"],
                       "global_step": loaded.manifest["global_step"],
                       "manifest_sha256": ckpt.sha256_file(resolved / "manifest.json"),
                       "model_config_hash": loaded.manifest["model_config_hash"],
                       "dataset_hash": loaded.manifest["dataset_hash"],
                       "run_id": os.environ.get("GITHUB_RUN_ID"), "optimizer_carried": bool(carry_optimizer)}
        self.log(f"[init-from] new stage {self.config.stage!r} starts from {loaded.manifest['stage']!r} "
                 f"step {loaded.manifest['global_step']} ({'optimizer carried' if carry_optimizer else 'fresh optimizer'})")

    def _continue_from(self, path: Path, carry_optimizer: bool, reopen: bool) -> None:
        """Same-stage continuation with a LARGER token budget.

        ``--resume`` is an exact continuation inside a fixed horizon and refuses a
        longer one; this loads the weights, progress, RNG, data position and
        (with ``carry_optimizer``) the optimizer moments of a same-stage
        checkpoint and continues into a larger ``total_steps`` horizon with a
        fresh schedule segment resumed at the same step. Used when the objective
        was not met at the budget (the checkpoint is incomplete) or, with
        ``reopen``, when the stage objective was met but its promotion gate
        failed (the checkpoint is complete). It refuses another stage's
        checkpoint (that is ``--init-from``), a complete checkpoint without
        ``reopen``, and a horizon that does not grow. Every identity check of
        ``--resume`` except the horizon length is enforced."""
        resolved, notes = ckpt.resolve_checkpoint(path)
        self.load_notes = notes
        for note in notes:
            self.log(f"[continue] WARNING: {note}")
        loaded = ckpt.load_checkpoint(resolved, verify=False)
        if loaded.manifest["stage"] != self.config.stage:
            raise ckpt.ResumeMismatchError(
                f"cannot continue: checkpoint is stage {loaded.manifest['stage']!r}, this run is {self.config.stage!r} "
                "(--init-from starts a new stage)")
        complete = bool(loaded.manifest.get("stage_complete"))
        if complete and not reopen:
            raise ckpt.ResumeMismatchError(
                "cannot continue: the checkpoint's stage is already COMPLETE (its objective was met). Start the next "
                "stage with --init-from; reopen this stage only if its promotion gate failed (--reopen-stage).")
        ckpt.validate_resume(loaded, **self._identity(), skip=("stage_complete", "total_steps"))
        previous_total = int(loaded.state["scheduler"].get("total_steps", loaded.state["progress"].get("total_steps", 0)))
        stored_step = int(loaded.state["progress"]["global_step"])
        if self.total_steps <= max(previous_total, stored_step):
            raise ckpt.ResumeMismatchError(
                f"cannot continue: the new horizon total_steps={self.total_steps} does not extend the checkpoint's "
                f"{previous_total} (step {stored_step}); --continue-stage needs a LARGER --target-tokens/--max-steps "
                "(use --resume to finish the same horizon)")
        if carry_optimizer:
            self.optimizer.load_state_dict(loaded.optimizer_state)
        self._apply_weights(loaded)
        self.rng = ckpt.rng_from_json(loaded.state["rng"])
        p = loaded.state["progress"]
        self.step, self.epoch, self.cursor = int(p["global_step"]), int(p["epoch"]), int(p["cursor"])
        self.cumulative_steps = int(p["cumulative_steps"])
        self.tokens_processed = int(p["tokens_processed"])
        self.loss_tokens_processed = int(p["loss_tokens_processed"])
        self.examples_processed = int(p["examples_processed"])
        self.recent_losses = list(p.get("recent_losses", []))
        self.first_train_loss = p.get("first_train_loss")
        metrics = loaded.state.get("metrics", {})
        self.train_seconds = float(metrics.get("train_seconds", 0.0))
        self.val_history = list(metrics.get("val_history", []))
        self.initial_validation = metrics.get("initial_validation")
        self._restore_objective_state(metrics)
        if reopen and not self._explicit_min_tokens:
            # The objective was already met; a reopened stage trains the whole extension before it may
            # complete again (otherwise it would re-pass at the first evaluation and change nothing).
            self.objective_min_tokens = self.effective_batch_tokens * self.total_steps
        self.schedule.last_step = self.step  # fresh, longer schedule resumed at the same step
        self.initial_step = self.step
        self.resumed_from = str(resolved)
        self.parent = {"checkpoint": resolved.name, "stage": loaded.manifest["stage"],
                       "global_step": loaded.manifest["global_step"],
                       "manifest_sha256": ckpt.sha256_file(resolved / "manifest.json"),
                       "model_config_hash": loaded.manifest["model_config_hash"],
                       "dataset_hash": loaded.manifest["dataset_hash"],
                       "continuation": True, "reopened": bool(reopen), "previous_total_steps": previous_total,
                       "run_id": os.environ.get("GITHUB_RUN_ID"), "optimizer_carried": bool(carry_optimizer)}
        self.log(f"[continue] {'reopening' if reopen else 'extending'} {self.config.stage} from {resolved.name} "
                 f"(step {self.step}): horizon {previous_total} -> {self.total_steps} steps "
                 f"({'optimizer carried' if carry_optimizer else 'fresh optimizer'})")

    # ---- evaluation --------------------------------------------------------------------
    def _loss_over(self, dataset: TokenizedDataset, limit: int | None) -> tuple[float, int]:
        total, tokens = 0.0, 0
        with no_grad():
            for b in sequential_batches(dataset, self.config.batch_size, self.config.max_seq_len,
                                        self.tokenizer.pad_token_id, packing=self.config.packing, limit=limit):
                out = self.model(b.input_ids, labels=b.labels, segment_ids=b.segment_ids, loss_normalizer=1.0)
                total += float(out.loss.item())
                tokens += b.num_loss_tokens
        if tokens == 0:
            raise DatasetError(f"{dataset.name} has no loss tokens")
        return total, tokens

    def evaluate(self) -> dict[str, float]:
        if self.validation is None:
            return {}
        total, tokens = self._loss_over(self.validation, self.config.eval_batches or None)
        loss = total / tokens
        return {"val_loss": loss, "val_ppl": math.exp(min(loss, 30.0)), "val_tokens": tokens}

    def _target_bytes(self, dataset: TokenizedDataset) -> int:
        """UTF-8 bytes of the text every loss token predicts (special tokens decode
        to nothing), so a summed loss converts to bits per byte — a measure that
        does not depend on the tokenizer."""
        total = 0
        for ex in dataset.examples:
            targets = [int(t) for t in ex.labels[1:] if t != IGNORE_INDEX]
            total += len(self.tokenizer.decode(targets).encode("utf-8"))
        return total

    def evaluate_text(self) -> dict[str, float]:
        """Held-out loss on the natural-text objective set (all of it, every
        time), with perplexity and bits per byte."""
        if self.objective_validation is None:
            return {}
        total, tokens = self._loss_over(self.objective_validation, None)
        loss = total / tokens
        out = {"text_val_loss": loss, "text_val_ppl": math.exp(min(loss, 30.0)), "text_val_tokens": tokens}
        if self._text_bytes:
            out["text_val_bytes"] = self._text_bytes
            out["text_val_bpb"] = total / (math.log(2.0) * self._text_bytes)
        return out

    # ---- stage objective -----------------------------------------------------------------
    @property
    def budget_tokens(self) -> int:
        """Training budget consumed, in the unit the stage token budget is
        defined in (optimizer steps x effective batch tokens)."""
        return self.step * self.effective_batch_tokens

    def run_objective(self, held_out: dict[str, Any], *, baseline: bool = False) -> dict[str, Any] | None:
        """Evaluate the stage objective on the live model at this checkpoint,
        write its report (JSON + Markdown with the raw generations), and update
        whether the objective is met. None when the stage has no objective."""
        if self.objective is None:
            return None
        held = {**held_out, **self.evaluate_text()}
        report = self.objective.evaluate_checkpoint(
            model=self.model, tokenizer=self.tokenizer, held_out=held, step=self.step, tokens=self.budget_tokens,
            real_tokens=self.tokens_processed, min_tokens=self.objective_min_tokens, data=self.objective_data,
            previous=self._prev_objective_report, reference=self._capability_reference, baseline=baseline,
            lexicon=self.objective_lexicon, log=self._log_fn)
        self._write_objective_report(report, previous=self._prev_objective_report)
        self._prev_objective_report = report
        self._last_objective_report = report
        self._objective_met = bool(report["objective_met"])
        return report

    def _write_objective_report(self, report: dict[str, Any], previous: dict[str, Any] | None) -> None:
        d = self.objective_reports_dir
        d.mkdir(parents=True, exist_ok=True)
        stem = f"step-{report['step']:08d}"
        md = render_markdown(report, previous=previous)
        _atomic_json(d / f"{stem}.json", report)
        (d / f"{stem}.md").write_text(md, encoding="utf-8")
        _atomic_json(d / "latest.json", report)
        (d / "latest.md").write_text(md, encoding="utf-8")
        with (d / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_row(report)) + "\n")

    # ---- one optimizer step ---------------------------------------------------------------
    def _micro_batches(self):
        accum = self.config.gradient_accumulation_steps
        while self.cursor + accum > self.plan.num_micro_batches(self.epoch):
            self.epoch += 1  # drop_last: the leftover < accum micro-batches of an epoch are skipped
            self.cursor = 0
            if self.plan.num_micro_batches(self.epoch) < accum:
                raise DatasetError("an epoch yields fewer micro-batches than gradient_accumulation_steps")
        return [self.plan.micro_batch(self.epoch, self.cursor + j) for j in range(accum)]

    def train_step(self) -> dict[str, float]:
        micro = self._micro_batches()
        n_loss = sum(b.num_loss_tokens for b in micro)
        if n_loss == 0:
            raise DatasetError("a whole optimizer step has no loss tokens (check masking)")
        self.optimizer.lr = self.schedule.current_lr()
        self.optimizer.zero_grad()
        loss_sum = 0.0
        for b in micro:
            out = self.model(b.input_ids, labels=b.labels, segment_ids=b.segment_ids, loss_normalizer=float(n_loss))
            loss_sum += float(out.loss.item())
            out.loss.backward(retain_graph=False)
            del out
        if not math.isfinite(loss_sum):
            raise TrainingDivergedError(f"loss is {loss_sum} at step {self.step + 1}; not applied")
        try:
            grad_norm = self.optimizer.step(grad_clip_norm=self.config.gradient_clip_norm)
        except FloatingPointError as exc:
            raise TrainingDivergedError(str(exc)) from exc
        self.step += 1
        self.cumulative_steps += 1
        self.cursor += len(micro)
        self.schedule.advance()
        real = sum(b.num_real_tokens for b in micro)
        self.tokens_processed += real
        self.loss_tokens_processed += n_loss
        self.examples_processed += sum(b.num_examples for b in micro)
        self.recent_losses = (self.recent_losses + [loss_sum])[-20:]
        if self.first_train_loss is None:
            self.first_train_loss = loss_sum
        return {"loss": loss_sum, "grad_norm": grad_norm, "lr": self.optimizer.lr, "real_tokens": real,
                "padded_tokens": sum(b.padded_tokens for b in micro), "loss_tokens": n_loss}

    # ---- checkpoint / export -------------------------------------------------------------------
    def _snapshot(self, complete: bool) -> ckpt.Snapshot:
        state = self.optimizer.state_dict()
        return ckpt.Snapshot(
            stage=self.config.stage, stage_complete=complete,
            weights={n: p.data for n, p in self.model.named_parameters()},
            optimizer_state=state, scheduler_state=self.schedule.state_dict(),
            progress={"global_step": self.step, "epoch": self.epoch, "cursor": self.cursor,
                      "cumulative_steps": self.cumulative_steps, "total_steps": self.total_steps,
                      "tokens_processed": self.tokens_processed, "loss_tokens_processed": self.loss_tokens_processed,
                      "examples_processed": self.examples_processed, "recent_losses": self.recent_losses,
                      "first_train_loss": self.first_train_loss},
            rng_state=ckpt.rng_state_to_json(self.rng), model_config=self.model_config.to_dict(),
            tokenizer_spec=self.tokenizer.spec(), renderer_spec=self.renderer.spec(),
            dataset={"dataset_hash": self.plan.dataset_hash(), "identity": self.plan.identity(),
                     "validation_hash": self.validation.content_hash if self.validation else None},
            training_config=self.config.to_dict(), training_config_compat=self._compat, parent=self.parent,
            metrics={"train_seconds": self.train_seconds, "val_history": self.val_history,
                     "initial_validation": self.initial_validation,
                     # objective memory: the report the NEXT checkpoint is compared with (also across jobs),
                     # retained stages' promotion-time metrics, the minimum budget. This is exactly the
                     # in-memory regression reference, so a resumed run decides like the uninterrupted one:
                     # a fresh stage's untrained baseline is never stored as a reference.
                     "objective_report": self._prev_objective_report,
                     "capability_reference": self._capability_reference,
                     "objective_min_tokens": self.objective_min_tokens if self.objective is not None else None})

    def save_checkpoint(self, complete: bool = False) -> Path:
        t0 = time.perf_counter()
        path = ckpt.save_checkpoint(self.checkpoint_root, self._snapshot(complete),
                                    keep_last=self.config.keep_checkpoints, fault_hook=self.fault_hook)
        dt = time.perf_counter() - t0
        self._ema_ckpt = dt if self._ema_ckpt == 0 else 0.7 * self._ema_ckpt + 0.3 * dt
        self.last_checkpoint = path
        return path

    def _time_is_up(self, t0: float) -> bool:
        limit = self.config.max_runtime_seconds
        if limit <= 0:
            return False
        remaining = limit - (self.clock() - t0)
        return remaining < self.config.safety_margin_seconds + 1.5 * self._ema_step + self._ema_ckpt

    # ---- the loop ------------------------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        if self.step >= self.total_steps:
            raise ckpt.ResumeMismatchError(f"step {self.step} >= total_steps {self.total_steps}: this stage is already complete")
        t_wall = time.perf_counter()
        t_budget = self.clock()
        stop = None
        if self.initial_validation is None and self.step == 0 and self.validation is not None:
            self.initial_validation = self.evaluate()
            self.log(f"[eval] step 0: val_loss {self.initial_validation['val_loss']:.4f}")
        if self.objective is not None and self.step == 0 and self._last_objective_report is None:
            # Baseline report before any training in this stage: informational, never promotes. After
            # --init-from it is the parent stage's model and becomes the first regression reference; a
            # randomly initialised model is no reference at all (its byte noise is "diverse" and
            # "non-repetitive"), so a fresh stage starts comparing from its first trained checkpoint.
            self.run_objective(self.initial_validation or {}, baseline=True)
            if self.parent is None:
                self._prev_objective_report = None
        metrics_file = (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8")
        diverged: BaseException | None = None
        try:
            while self.step < self.total_steps:
                if self._stop_reason:
                    stop = self._stop_reason
                    break
                if self._time_is_up(t_budget):
                    stop = "time_budget"
                    break
                if self.stop_after_steps and self.step - self.initial_step >= self.stop_after_steps:
                    stop = "step_limit"
                    break
                t0 = time.perf_counter()
                info = self.train_step()
                dt = time.perf_counter() - t0
                self.train_seconds += dt
                self._ema_step = dt if self._ema_step == 0 else 0.8 * self._ema_step + 0.2 * dt
                if cfg.log_interval and (self.step % cfg.log_interval == 0 or self.step == 1):
                    row = {"step": self.step, "epoch": self.epoch, "loss": round(info["loss"], 5),
                           "lr": info["lr"], "grad_norm": round(info["grad_norm"], 4), "step_seconds": round(dt, 4),
                           "tokens_per_sec": round(info["real_tokens"] / dt, 1),
                           "padding_fraction": round(1 - info["real_tokens"] / info["padded_tokens"], 3)}
                    metrics_file.write(json.dumps(row) + "\n")
                    metrics_file.flush()
                    self.log(f"[train] step {self.step}/{self.total_steps} loss {info['loss']:.4f} lr {info['lr']:.2e} "
                             f"gnorm {info['grad_norm']:.2f} {row['tokens_per_sec']:.0f} tok/s")
                if cfg.eval_interval and self.step % cfg.eval_interval == 0 and self.validation is not None:
                    ev = self.evaluate()
                    self.val_history.append({"step": self.step, **ev})
                    metrics_file.write(json.dumps({"step": self.step, **ev}) + "\n")
                    self.log(f"[eval] step {self.step}: val_loss {ev['val_loss']:.4f} ppl {ev['val_ppl']:.2f}")
                    if self.objective is not None:
                        report = self.run_objective(ev)
                        metrics_file.write(json.dumps({"step": self.step, "objective": history_row(report)}) + "\n")
                        metrics_file.flush()
                        if self._objective_met:
                            stop = "gate_passed"  # objective met after the minimum budget: the stage is done
                            break
                if cfg.checkpoint_interval and self.step % cfg.checkpoint_interval == 0 and self.step < self.total_steps:
                    self.save_checkpoint(complete=False)
            else:
                stop = "complete"
        except TrainingDivergedError as exc:
            diverged = exc
            stop = "diverged"
        finally:
            metrics_file.close()

        if diverged is None:
            final_eval = self.evaluate() if self.validation is not None else {}
            if final_eval and (not self.val_history or self.val_history[-1]["step"] != self.step):
                self.val_history.append({"step": self.step, **final_eval})
            if self.objective is None:
                complete = self.step >= self.total_steps  # legacy: the budget horizon completes the stage
            else:
                # Objective-driven completion: decide on a measurement of THIS checkpoint.
                last = self._last_objective_report
                if stop != "gate_passed" and (last is None or last["step"] != self.step or last.get("baseline")):
                    self.run_objective(final_eval)
                complete = self._objective_met
                if complete:
                    stop = "gate_passed"
                elif stop == "complete":
                    stop = "gate_failed"  # the whole budget was used and the objective is not met: INCOMPLETE
                # time_budget / step_limit / signal keep their reason; the stage is incomplete either way
            self._stage_complete = complete
            self.save_checkpoint(complete=complete)
            self.log(f"[done] {stop}: step {self.step}/{self.total_steps}, stage "
                     f"{'COMPLETE' if complete else 'incomplete'}, checkpoint {self.last_checkpoint.name}")
            if self.exporter:
                self.exports = self.exporter(self, self.output_dir)
        else:
            final_eval = {}
        wall = time.perf_counter() - t_wall
        summary = self._summary(stop or "unknown", final_eval, wall)
        _atomic_json(self.output_dir / "training_summary.json", summary)
        if diverged is not None:
            raise diverged
        return summary

    # ---- summary ---------------------------------------------------------------------------------------------
    def _summary(self, stop_reason: str, final_eval: dict[str, float], wall: float) -> dict[str, Any]:
        steps_this_run = self.step - self.initial_step
        tokens_this_run = None
        env = ckpt.environment_info()
        recent = self.recent_losses[-10:]
        report = self._last_objective_report
        return {
            "stage": self.config.stage, "stop_reason": stop_reason, "stage_complete": self._stage_complete,
            # Objective-driven completion: stage_complete above is True only when the objective was met
            # (or, for a stage without an objective, when its budget horizon was used up).
            "objective": {
                "configured": self.objective is not None,
                "objective_met": self._objective_met if self.objective is not None else None,
                "min_tokens": self.objective_min_tokens if self.objective is not None else None,
                "budget_tokens_trained": self.budget_tokens,
                "report_step": report["step"] if report else None,
                "verdict": report["verdict"] if report else None,
                "regression": report["regression"] if report else None,
                "reports_dir": str(self.objective_reports_dir) if report else None,
            },
            "model_config": self.model_config.to_dict(), "parameter_count": self.model_config.count_parameters(),
            "tokenizer": self.tokenizer.spec(), "renderer": self.renderer.spec(),
            "dataset": {"dataset_hash": self.plan.dataset_hash(), "identity": self.plan.identity(),
                        "validation_hash": self.validation.content_hash if self.validation else None,
                        "repeat_factors": self.plan.repeat_factors()},
            "training_config": self.config.to_dict(), "total_steps": self.total_steps,
            "initial_step": self.initial_step, "final_step": self.step, "steps_this_run": steps_this_run,
            "epoch": self.epoch, "cumulative_steps": self.cumulative_steps,
            "initial_train_loss": self.first_train_loss,
            "final_train_loss_mean_last_10_steps": float(np.mean(recent)) if recent else None,
            "initial_validation": self.initial_validation, "final_validation": final_eval or None,
            "validation_history": self.val_history,
            "tokens_processed_total": self.tokens_processed, "loss_tokens_processed_total": self.loss_tokens_processed,
            "examples_processed_total": self.examples_processed,
            # v2 token-budget accounting (brief section 6): every quantity the trainer
            # is asked to report, grouped so nothing above changes shape. A step here
            # is one optimizer step (gradient_accumulation_steps micro-batches).
            "token_accounting": {
                "total_training_tokens": self.tokens_processed,
                "loss_tokens": self.loss_tokens_processed,
                "tokens_per_sec": round(self.tokens_processed / self.train_seconds, 1) if self.train_seconds else None,
                "examples_per_sec": round(self.examples_processed / self.train_seconds, 2) if self.train_seconds else None,
                "optimizer_steps": self.step,
                "effective_batch_tokens": self.effective_batch_tokens,
                "epoch": self.epoch,
                "stage_target_tokens": self.stage_target_tokens,
                "stage_target_steps": self.total_steps,
                "stage_completion_percent": round(100.0 * self.step / self.total_steps, 2) if self.total_steps else None,
            },
            "train_seconds_total": round(self.train_seconds, 3), "wall_clock_seconds_this_run": round(wall, 3),
            "tokens_per_second_train_average": round(self.tokens_processed / self.train_seconds, 1) if self.train_seconds else None,
            "peak_rss_mb": round(_rss_mb(), 1),
            "checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "resumed_from": self.resumed_from, "parent": self.parent, "resume_notes": self.load_notes,
            "exports": self.exports, "git_commit": env["git_commit"], "environment": env,
        }


def _atomic_json(path: Path, obj: Any) -> None:
    ckpt._replace_json(path, obj)
