"""Learning-rate schedules with checkpointable state.

``lr_at(step)`` is a pure function of the number of optimizer steps already
completed (0 for the first update), so a resumed run reproduces the LR
sequence exactly; ``state_dict``/``load_state_dict`` still exist so a
checkpoint records — and resume validates — *which* schedule was running:
loading refuses a schedule whose kind, peak/floor LR, warmup or length differ.

    warmup:  lr = base * (step + 1) / warmup_steps           (step < warmup_steps)
    cosine:  lr = min + (base - min) * (1 + cos(pi * p)) / 2 (p = progress through the decay)
    linear:  lr = base + (min - base) * p
    constant: lr = base

where ``p = clip((step - warmup) / (total - warmup), 0, 1)``. The last update
(``step = total - 1``) therefore uses an LR just above ``min``, not exactly it.
"""
from __future__ import annotations

import math
from typing import Any

KINDS = ("cosine", "linear", "constant")


class ScheduleStateError(ValueError):
    pass


class LRSchedule:
    def __init__(self, kind: str, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> None:
        if kind not in KINDS:
            raise ValueError(f"scheduler must be one of {KINDS}, got {kind!r}")
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if not (0 <= min_lr <= base_lr):
            raise ValueError(f"need 0 <= min_learning_rate ({min_lr}) <= learning_rate ({base_lr})")
        if not (0 <= warmup_steps < total_steps):
            raise ValueError(f"need 0 <= warmup_steps ({warmup_steps}) < total steps ({total_steps})")
        self.kind, self.base_lr, self.min_lr = kind, float(base_lr), float(min_lr)
        self.warmup_steps, self.total_steps = int(warmup_steps), int(total_steps)
        self.last_step = 0  # optimizer steps completed

    def lr_at(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if self.kind == "constant":
            return self.base_lr
        span = max(1, self.total_steps - self.warmup_steps)
        p = min(1.0, max(0.0, (step - self.warmup_steps) / span))
        if self.kind == "linear":
            return self.base_lr + (self.min_lr - self.base_lr) * p
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * p))

    def current_lr(self) -> float:
        return self.lr_at(self.last_step)

    def advance(self) -> None:
        self.last_step += 1

    def config(self) -> dict[str, Any]:
        return {"kind": self.kind, "base_lr": self.base_lr, "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}

    def state_dict(self) -> dict[str, Any]:
        return {**self.config(), "last_step": self.last_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stored = {k: state.get(k) for k in self.config()}
        if stored != self.config():
            raise ScheduleStateError(f"schedule differs from the checkpoint's: stored {stored}, current {self.config()}")
        last = state.get("last_step")
        if not isinstance(last, int) or not 0 <= last <= self.total_steps:
            raise ScheduleStateError(f"invalid last_step {last!r}")
        self.last_step = last
