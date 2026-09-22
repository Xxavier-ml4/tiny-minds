"""Training configuration: every knob the brief (section 11) requires, in one
strictly-validated dataclass. Nothing is hard-coded in a script; a run's
configuration is saved in its checkpoints and compared on resume.

``compat_dict`` is the subset that determines the *trajectory* (data order,
optimizer arithmetic, schedule). Resume requires it to be identical. The rest
(``eval_interval``, ``checkpoint_interval``, ``max_runtime_seconds``, ...) may
change between the jobs of a multi-job stage without changing the result.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class TrainingConfigError(ValueError):
    pass


_RUNTIME_ONLY = ("stage", "eval_interval", "eval_batches", "checkpoint_interval", "keep_checkpoints",
                 "log_interval", "max_runtime_seconds", "safety_margin_seconds", "nonfinite_policy", "epochs")


@dataclasses.dataclass
class TrainingConfig:
    stage: str = "stage0"
    seed: int = 0
    # optimisation
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    scheduler: str = "cosine"
    warmup_steps: int = 20
    weight_decay: float = 0.01
    weight_decay_exclude_norms: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    gradient_clip_norm: float = 1.0
    # batching
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 256
    packing: bool = True
    overflow: str = "error"
    # length: max_steps > 0 wins; otherwise ``epochs`` full epochs
    max_steps: int = 0
    epochs: int = 1
    # data mixture: {source name: weight}; empty = every source equally weighted
    mixture: dict[str, float] = dataclasses.field(default_factory=dict)
    epoch_examples: int = 0
    # bookkeeping cadence and safety
    eval_interval: int = 100
    eval_batches: int = 0          # 0 = the whole validation set
    checkpoint_interval: int = 100
    keep_checkpoints: int = 3
    log_interval: int = 10
    max_runtime_seconds: float = 0.0   # 0 = unlimited
    safety_margin_seconds: float = 300.0
    nonfinite_policy: str = "error"

    def __post_init__(self) -> None:
        self.validate()  # an invalid configuration cannot be constructed

    # ---- validation -------------------------------------------------------
    def validate(self) -> None:
        problems = []
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            problems.append("gradient_accumulation_steps must be >= 1")
        if self.max_seq_len < 8:
            problems.append("max_seq_len must be >= 8")
        if self.max_steps < 0 or self.epochs < 0 or (self.max_steps == 0 and self.epochs == 0):
            problems.append("need max_steps > 0 or epochs > 0")
        if not (0.0 < self.learning_rate) or not (0.0 <= self.min_learning_rate <= self.learning_rate):
            problems.append("need 0 <= min_learning_rate <= learning_rate, learning_rate > 0")
        if self.scheduler not in ("cosine", "linear", "constant"):
            problems.append("scheduler must be cosine|linear|constant")
        if self.warmup_steps < 0:
            problems.append("warmup_steps must be >= 0")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            problems.append("weight_decay >= 0 and gradient_clip_norm > 0 required")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            problems.append("betas must be in [0, 1)")
        if self.overflow not in ("error", "drop", "truncate"):
            problems.append("overflow must be error|drop|truncate")
        if self.nonfinite_policy != "error":
            problems.append("nonfinite_policy: only 'error' is implemented (a skipped step would hide divergence)")
        if self.eval_interval < 0 or self.checkpoint_interval < 0 or self.log_interval < 0 or self.keep_checkpoints < 2:
            problems.append("eval/checkpoint/log intervals must be >= 0 and keep_checkpoints >= 2")
        if self.max_runtime_seconds < 0 or self.safety_margin_seconds < 0:
            problems.append("max_runtime_seconds and safety_margin_seconds must be >= 0")
        if any(w < 0 for w in self.mixture.values()):
            problems.append("mixture weights must be >= 0")
        if problems:
            raise TrainingConfigError("; ".join(problems))

    # ---- (de)serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise TrainingConfigError(f"unknown training config field(s) {unknown}; known: {sorted(known)}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, section: str | None = "training") -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if section and section in data:
            data = data[section]
        return cls.from_dict(data)

    def replace(self, **changes: Any) -> "TrainingConfig":
        cfg = dataclasses.replace(self, **changes)
        cfg.validate()
        return cfg

    # ---- identity -------------------------------------------------------------
    def compat_dict(self, total_steps: int) -> dict[str, Any]:
        d = {k: v for k, v in self.to_dict().items() if k not in _RUNTIME_ONLY and k != "max_steps"}
        d["total_steps"] = int(total_steps)
        return d

    def compat_hash(self, total_steps: int) -> str:
        blob = json.dumps(self.compat_dict(total_steps), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()
