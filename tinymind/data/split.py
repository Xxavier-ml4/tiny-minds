"""Train/validation/test splitting, per the engineering brief section 25.

Deterministic given a seed (so re-running a pipeline reproduces the same
split), and stratifiable by an arbitrary key function — for training data
shaped like ``tinymind.data.validate``'s format, splitting by
``target.type`` (so "tool_call" and "answer" examples each get a
proportional share of train/val/test) is usually what's wanted, and is the
default when a ``stratify_key`` is given.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class DatasetSplit:
    train: list
    validation: list
    test: list

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)}


def split_dataset(examples: list[T], *, train: float = 0.8, validation: float = 0.1,
                  test: float = 0.1, seed: int = 0,
                  stratify_key: Callable[[T], str] | None = None) -> DatasetSplit:
    total = train + validation + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train + validation + test must sum to 1.0, got {total}")
    if not examples:
        return DatasetSplit(train=[], validation=[], test=[])

    rng = random.Random(seed)

    if stratify_key is None:
        groups = {"_all": list(examples)}
    else:
        groups: dict[str, list[T]] = {}
        for example in examples:
            groups.setdefault(stratify_key(example), []).append(example)

    out_train: list[T] = []
    out_val: list[T] = []
    out_test: list[T] = []
    for _key, group in groups.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train)
        n_val = round(n * validation)
        out_train.extend(shuffled[:n_train])
        out_val.extend(shuffled[n_train:n_train + n_val])
        out_test.extend(shuffled[n_train + n_val:])

    return DatasetSplit(train=out_train, validation=out_val, test=out_test)
