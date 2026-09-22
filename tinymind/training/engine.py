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

Stopping: ``max_steps`` reached (stage complete), the time budget, or an
external request (SIGTERM/SIGINT in the CLI). In every case the engine writes
a verified checkpoint, exports the inference package, writes
``training_summary.json`` and returns — it never relies on being killed by a
wall-clock timeout.
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
from tinymind.training.render import ChatRenderer
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
                 carry_optimizer: bool = False, allow_no_validation: bool = False, stop_after_steps: int | None = None,
                 clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] | None = print,
                 fault_hook: Callable[[str], None] | None = None,
                 exporter: Callable[["TrainingEngine", Path], dict[str, Any]] | None = None) -> None:
        if resume and init_from:
            raise TrainingConfigError("--resume and --init-from are different operations; pass only one")
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
        for s in list(sources) + ([DataSource("validation", validation)] if validation else []):
            if s.dataset.renderer.spec() != spec:
                raise TrainingConfigError(f"dataset {s.name!r} was rendered with a different tokenizer/template")
        self.plan = DataPlan(sources, seed=config.seed, batch_size=config.batch_size, max_seq_len=config.max_seq_len,
                             packing=config.packing, pad_id=tokenizer.pad_token_id,
                             epoch_examples=config.epoch_examples or None)
        accum = config.gradient_accumulation_steps
        if self.plan.steps_in_epoch(0, accum) < 1:
            raise DatasetError(f"the data yields {self.plan.num_micro_batches(0)} micro-batch(es) of {config.batch_size} "
                               f"row(s) per epoch, fewer than gradient_accumulation_steps={accum}")
        if config.max_steps > 0:
            self.total_steps = config.max_steps
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

        if resume:
            self._restore_for_resume(Path(resume))
        elif init_from:
            self._init_from(Path(init_from), carry_optimizer)

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
        self.parent = loaded.manifest.get("parent")
        self.initial_step = self.step
        self.resumed_from = str(resolved)
        if self.schedule.last_step != self.step:
            raise ckpt.CheckpointCorruptError("scheduler step and global step disagree")
        self.log(f"[resume] continuing {self.config.stage} from {resolved.name} "
                 f"(step {self.step}/{self.total_steps}, epoch {self.epoch}, cursor {self.cursor})")

    def _init_from(self, path: Path, carry_optimizer: bool) -> None:
        resolved, notes = ckpt.resolve_checkpoint(path)
        for note in notes:
            self.log(f"[init-from] WARNING: {note}")
        loaded = ckpt.load_checkpoint(resolved, verify=False)
        ckpt.validate_init_from(loaded, model_config=self.model_config.to_dict(), tokenizer_spec=self.tokenizer.spec(),
                                renderer_spec=self.renderer.spec())
        self._apply_weights(loaded)
        if carry_optimizer:
            state = loaded.optimizer_state
            state["hyper"] = self.optimizer.hyperparameters()  # new stage may retune betas/decay; moments carry over
            state["lr"] = self.optimizer.lr
            self.optimizer.load_state_dict(state)
        prior = loaded.state["progress"]
        self.cumulative_steps = int(prior["cumulative_steps"])
        self.parent = {"checkpoint": resolved.name, "stage": loaded.manifest["stage"],
                       "global_step": loaded.manifest["global_step"],
                       "manifest_sha256": ckpt.sha256_file(resolved / "manifest.json"),
                       "model_config_hash": loaded.manifest["model_config_hash"],
                       "dataset_hash": loaded.manifest["dataset_hash"],
                       "run_id": os.environ.get("GITHUB_RUN_ID"), "optimizer_carried": bool(carry_optimizer)}
        self.log(f"[init-from] new stage {self.config.stage!r} starts from {loaded.manifest['stage']!r} "
                 f"step {loaded.manifest['global_step']} ({'optimizer carried' if carry_optimizer else 'fresh optimizer'})")

    # ---- evaluation --------------------------------------------------------------------
    def evaluate(self) -> dict[str, float]:
        if self.validation is None:
            return {}
        total, tokens = 0.0, 0
        limit = self.config.eval_batches or None
        with no_grad():
            for b in sequential_batches(self.validation, self.config.batch_size, self.config.max_seq_len,
                                        self.tokenizer.pad_token_id, packing=self.config.packing, limit=limit):
                out = self.model(b.input_ids, labels=b.labels, segment_ids=b.segment_ids, loss_normalizer=1.0)
                total += float(out.loss.item())
                tokens += b.num_loss_tokens
        if tokens == 0:
            raise DatasetError("validation set has no loss tokens")
        loss = total / tokens
        return {"val_loss": loss, "val_ppl": math.exp(min(loss, 30.0)), "val_tokens": tokens}

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
                     "initial_validation": self.initial_validation})

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
            complete = self.step >= self.total_steps
            self.save_checkpoint(complete=complete)
            self.log(f"[done] {stop}: step {self.step}/{self.total_steps}, checkpoint {self.last_checkpoint.name}")
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
        return {
            "stage": self.config.stage, "stop_reason": stop_reason, "stage_complete": self.step >= self.total_steps,
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
            "train_seconds_total": round(self.train_seconds, 3), "wall_clock_seconds_this_run": round(wall, 3),
            "tokens_per_second_train_average": round(self.tokens_processed / self.train_seconds, 1) if self.train_seconds else None,
            "peak_rss_mb": round(_rss_mb(), 1),
            "checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "resumed_from": self.resumed_from, "parent": self.parent, "resume_notes": self.load_notes,
            "exports": self.exports, "git_commit": env["git_commit"], "environment": env,
        }


def _atomic_json(path: Path, obj: Any) -> None:
    ckpt._replace_json(path, obj)
