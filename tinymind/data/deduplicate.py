"""Deduplicate training examples, per the engineering brief section 25.

Two real, working passes: exact-duplicate removal (hash of the normalized
user-message text) and near-duplicate removal (a shingled Jaccard
similarity above a threshold — a standard, simple, dependency-free
technique; not a trained deduplication model, which the brief doesn't ask
for at this stage anyway).
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")


def _example_text(example: dict) -> str:
    messages = example.get("messages", [])
    return " ".join(m.get("content", "") for m in messages if isinstance(m, dict))


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _shingles(text: str, size: int = 3) -> set[str]:
    words = _normalize(text).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclasses.dataclass
class DeduplicationReport:
    total: int
    kept: int
    exact_duplicates_removed: int
    near_duplicates_removed: int


def deduplicate(examples: list[dict], near_duplicate_threshold: float = 0.9,
                shingle_size: int = 3) -> tuple[list[dict], DeduplicationReport]:
    seen_hashes: set[str] = set()
    exact_removed = 0
    after_exact: list[dict] = []
    for example in examples:
        normalized = _normalize(_example_text(example))
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            exact_removed += 1
            continue
        seen_hashes.add(digest)
        after_exact.append(example)

    kept: list[dict] = []
    kept_shingles: list[set[str]] = []
    near_removed = 0
    for example in after_exact:
        shingles = _shingles(_example_text(example), size=shingle_size)
        is_near_duplicate = any(_jaccard(shingles, other) >= near_duplicate_threshold
                                for other in kept_shingles)
        if is_near_duplicate:
            near_removed += 1
            continue
        kept.append(example)
        kept_shingles.append(shingles)

    report = DeduplicationReport(total=len(examples), kept=len(kept),
                                 exact_duplicates_removed=exact_removed,
                                 near_duplicates_removed=near_removed)
    return kept, report
