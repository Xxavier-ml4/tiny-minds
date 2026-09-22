"""Training pipeline, per the engineering brief section 23.

**Phase 3B**: the training system used for real runs is ``tinymind.training.engine.TrainingEngine`` (with
``config.TrainingConfig``, ``data``, ``checkpoint``, ``schedule``, ``gate``); ``tinymind train`` drives it. The
Phase 3A ``CausalLMTrainer`` below is kept for reproducibility of old results (``tinymind train --legacy``) but has
known defects (it never trains on the assistant response, weights gradient-accumulation micro-batches instead of
tokens, reports a wrong ``tokens_per_second``, and cannot resume) — see docs/architecture/training-system.md, Part 1.

**Updated in Phase 3A**: now that a real model exists
(``tinymind.model.model.TinyMindTransformer``), ``CausalLMTrainer``
(``causal_lm_trainer.py``) and ``CausalLMCollator`` (``collator.py``) are
real, working implementations — supervised causal-LM training end to end,
with gradient accumulation, warmup, gradient clipping, and checkpointing.
The abstract ``Trainer``/``Collator``/``Evaluator`` below predate that (see
their docstrings, unchanged) and remain as the general interface for a
*different* model backend to implement later; ``CausalLMTrainer`` does not
subclass them; it was more direct to write against
``TinyMindTransformer``'s actual forward signature than to force-fit the
pre-model-existing abstraction. Distillation, LoRA, and QAT are still not
implemented here — brief section 19: "First make ordinary supervised
causal-LM training work" — see ``STATUS.md``.
"""
from __future__ import annotations

import abc
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from tinymind.training.causal_lm_trainer import CausalLMTrainer, CausalLMTrainingConfig, StepLog
from tinymind.training.collator import CausalLMCollator
from tinymind.training.config import TrainingConfig
from tinymind.training.dataset import TokenizedExample, TrainingDataset

__all__ = [
    "TrainingDataset", "TokenizedExample",
    "CheckpointManager", "CheckpointMetadata",
    "Trainer", "AbstractTrainingConfig", "TrainingConfig", "Collator", "Evaluator", "DistributedConfig",
    "CausalLMTrainer", "CausalLMTrainingConfig", "StepLog", "CausalLMCollator",
]


@dataclasses.dataclass
class CheckpointMetadata:
    step: int
    created_at: float
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


class CheckpointManager:
    """Save/load named checkpoints under a directory. Each checkpoint is a
    ``metadata.json`` plus a ``state.json`` — plain JSON, not a tensor
    format, because this manager doesn't know what a "tensor" is yet (no
    model architecture is wired to it); once ``tinymind.model.model``
    (Phase 3) exists, its state dict is still just nested
    dicts/lists/numbers to this class, so this does not need to change,
    only what's passed to ``save()`` does.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, state: dict[str, Any], step: int, extra: dict[str, Any] | None = None) -> Path:
        checkpoint_dir = self._directory / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata = CheckpointMetadata(step=step, created_at=time.time(), extra=extra or {})
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(dataclasses.asdict(metadata), indent=2), encoding="utf-8")
        (checkpoint_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return checkpoint_dir

    def load(self, name: str) -> tuple[dict[str, Any], CheckpointMetadata]:
        checkpoint_dir = self._directory / name
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"no checkpoint named {name!r} in {self._directory}")
        state = json.loads((checkpoint_dir / "state.json").read_text(encoding="utf-8"))
        metadata_dict = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
        return state, CheckpointMetadata(**metadata_dict)

    def list_checkpoints(self) -> list[str]:
        return sorted(p.name for p in self._directory.iterdir() if p.is_dir())

    def latest(self) -> str | None:
        names = self.list_checkpoints()
        if not names:
            return None
        with_steps = []
        for name in names:
            try:
                _, metadata = self.load(name)
                with_steps.append((metadata.step, name))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return max(with_steps)[1] if with_steps else None


@dataclasses.dataclass
class AbstractTrainingConfig:
    """Config of the *abstract* ``Trainer`` placeholder below. (Named ``TrainingConfig`` until Phase 3B; that name
    now belongs to the real configuration, ``tinymind.training.config.TrainingConfig``, which the staged
    ``TrainingEngine`` uses.)"""
    learning_rate: float = 3e-4
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0


class Collator(abc.ABC):
    """Batch a list of ``TokenizedExample`` into padded tensors. Needs a
    real model's expected input layout (does it want labels shifted by
    one? masked how?) to implement meaningfully."""

    @abc.abstractmethod
    def collate(self, examples: list[TokenizedExample]) -> dict[str, Any]:
        raise NotImplementedError("Collator needs a model architecture to collate a batch for — see STATUS.md")


class Trainer(abc.ABC):
    """The training loop: forward pass, loss, backward pass, optimizer
    step. Needs ``tinymind.model.model`` (Phase 3, not in this delivery)."""

    def __init__(self, config: AbstractTrainingConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError(
            "Trainer.train_step() needs a real model to compute a forward/backward pass "
            "through; none exists in this delivery — see docs/architecture/"
            "tinymind-design.md section 7 and STATUS.md")

    @abc.abstractmethod
    def evaluate(self, dataset: TrainingDataset) -> dict[str, float]:
        raise NotImplementedError("Trainer.evaluate() needs a real model — see STATUS.md")


class Evaluator(abc.ABC):
    """Held-out-set evaluation during training (loss curves, per-epoch
    metrics) — needs a real model for the same reason ``Trainer`` does."""

    @abc.abstractmethod
    def evaluate(self, model: Any, dataset: TrainingDataset) -> dict[str, float]:
        raise NotImplementedError("Evaluator needs a real model — see STATUS.md")


@dataclasses.dataclass
class DistributedConfig:
    """Multi-device training configuration. Declared now so
    ``TrainingConfig``/``Trainer`` call sites don't need to change shape
    later; not implemented because there is no single-device training loop
    to distribute yet (brief's own phasing: distributed comes after
    single-device training works, not before)."""
    num_devices: int = 1
    strategy: str = "data_parallel"  # the only strategy named here; not implemented
