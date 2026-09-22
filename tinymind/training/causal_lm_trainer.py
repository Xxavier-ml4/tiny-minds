"""``CausalLMTrainer``: the real training loop — JSONL -> tokenizer ->
dataset -> batching -> model -> causal LM loss -> backward -> AdamW ->
checkpoint, per the engineering brief section 19. This is the concrete
implementation the abstract ``Trainer``/``Collator``/``Evaluator`` in
``tinymind/training/__init__.py`` were waiting on a real model to make
possible — see that module's docstring.

Deliberately does not implement distillation, LoRA, or QAT (brief section
19: "First make ordinary supervised causal-LM training work" — those are
explicitly later work, tracked in ``STATUS.md``).
"""
from __future__ import annotations

import dataclasses
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from tinymind.model.checkpoint import save_pretrained
from tinymind.model.model import TinyMindTransformer
from tinymind.model.optim import AdamW
from tinymind.training.collator import CausalLMCollator
from tinymind.training.dataset import TrainingDataset


@dataclasses.dataclass
class CausalLMTrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int | None = None
    epochs: int = 1
    warmup_steps: int = 0
    gradient_clip_norm: float = 1.0
    seed: int = 0
    checkpoint_interval: int | None = None
    checkpoint_dir: str | None = None
    log_every: int = 1


@dataclasses.dataclass
class StepLog:
    step: int
    loss: float
    grad_norm: float
    learning_rate: float
    tokens_per_second: float
    elapsed_seconds: float


class CausalLMTrainer:
    def __init__(self, model: TinyMindTransformer, config: CausalLMTrainingConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = AdamW(model.parameters(), learning_rate=config.learning_rate,
                               weight_decay=config.weight_decay)
        self.collator = CausalLMCollator(pad_token_id=model.config.pad_token_id,
                                         max_length=model.config.max_seq_len)
        self.global_step = 0
        self._start_time: float | None = None

    def _lr_for_step(self, step: int) -> float:
        if self.config.warmup_steps > 0 and step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / self.config.warmup_steps
        return self.config.learning_rate

    def train_step(self, micro_batches: list[dict[str, np.ndarray]]) -> StepLog:
        """Runs ``gradient_accumulation_steps`` micro-batches (each already
        collated), accumulating gradients across all of them before one
        optimizer step — standard gradient accumulation: loss is scaled by
        ``1/len(micro_batches)`` so the accumulated gradient matches what a
        single batch of the full effective size would have produced."""
        self.optimizer.zero_grad()
        self.optimizer.lr = self._lr_for_step(self.global_step)

        total_loss = 0.0
        total_tokens = 0
        scale = 1.0 / len(micro_batches)
        for batch in micro_batches:
            out = self.model(batch["input_ids"], attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"])
            (out.loss * scale).backward()
            total_loss += out.loss.item()
            total_tokens += int(batch["input_ids"].size)

        grad_norm = self.optimizer.step(grad_clip_norm=self.config.gradient_clip_norm)
        self.global_step += 1

        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        return StepLog(step=self.global_step, loss=total_loss / len(micro_batches), grad_norm=grad_norm,
                      learning_rate=self.optimizer.lr,
                      tokens_per_second=total_tokens / max(elapsed, 1e-9) if self._start_time else 0.0,
                      elapsed_seconds=elapsed)

    def train(self, dataset: TrainingDataset, *, log_fn: Callable[[StepLog], None] | None = None
             ) -> list[StepLog]:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._start_time = time.monotonic()
        log_fn = log_fn or (lambda entry: None)

        logs: list[StepLog] = []
        examples = list(dataset)
        if not examples:
            raise ValueError("training dataset is empty")

        checkpoint_dir = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir else None

        for epoch in range(self.config.epochs):
            random.shuffle(examples)
            batches = [examples[i:i + self.config.batch_size]
                      for i in range(0, len(examples), self.config.batch_size)]
            micro_batch_group: list[dict] = []

            for batch_examples in batches:
                collated = self.collator.collate(batch_examples)
                micro_batch_group.append(collated)
                if len(micro_batch_group) < self.config.gradient_accumulation_steps:
                    continue

                step_log = self.train_step(micro_batch_group)
                micro_batch_group = []
                logs.append(step_log)
                if step_log.step % self.config.log_every == 0:
                    log_fn(step_log)

                if (checkpoint_dir is not None and self.config.checkpoint_interval
                        and step_log.step % self.config.checkpoint_interval == 0):
                    save_pretrained(self.model, checkpoint_dir / f"step_{step_log.step}",
                                    optimizer=self.optimizer,
                                    training_meta={"step": step_log.step, "loss": step_log.loss,
                                                  "epoch": epoch})

                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    return logs

            if micro_batch_group:  # a partial accumulation window at epoch end
                step_log = self.train_step(micro_batch_group)
                logs.append(step_log)
                log_fn(step_log)
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    return logs

        return logs
