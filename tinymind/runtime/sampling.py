"""Sampling strategies over a plain probability/logit vector.

Pure math — no dependency on any ``ModelBackend``, tokenizer, or trained
model, which is exactly why this one piece of "generation internals" is
real and tested in this delivery rather than deferred: given *a* logits
vector (from anywhere — a real model's output layer, or a hand-built test
fixture), these functions are the actual, correct implementations any real
backend would call, not a stand-in.
"""
from __future__ import annotations

import math
import random


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    peak = max(logits)
    exp = [math.exp(x - peak) for x in logits]  # subtract max for numerical stability
    total = sum(exp)
    return [x / total for x in exp] if total > 0 else [1.0 / len(logits)] * len(logits)


def apply_temperature(logits: list[float], temperature: float) -> list[float]:
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0:
        return list(logits)  # caller should use greedy/argmax at temperature 0, not sample
    return [x / temperature for x in logits]


def top_k_filter(probs: list[float], k: int) -> list[float]:
    """Zero out every probability except the top ``k``, renormalize the rest."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= len(probs):
        return list(probs)
    threshold = sorted(probs, reverse=True)[k - 1]
    filtered = [p if p >= threshold else 0.0 for p in probs]
    total = sum(filtered)
    return [p / total for p in filtered] if total > 0 else filtered


def top_p_filter(probs: list[float], p: float) -> list[float]:
    """Nucleus sampling: keep the smallest set of highest-probability
    outcomes whose cumulative probability is >= ``p``, renormalize."""
    if not (0 < p <= 1):
        raise ValueError(f"p must be in (0, 1], got {p}")
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    cumulative = 0.0
    keep = set()
    for i in order:
        if cumulative >= p and keep:
            break
        keep.add(i)
        cumulative += probs[i]
    filtered = [prob if i in keep else 0.0 for i, prob in enumerate(probs)]
    total = sum(filtered)
    return [x / total for x in filtered] if total > 0 else filtered


def argmax(values: list[float]) -> int:
    if not values:
        raise ValueError("argmax of an empty sequence")
    best_index = 0
    best_value = values[0]
    for i, v in enumerate(values):
        if v > best_value:
            best_index, best_value = i, v
    return best_index


def sample(probs: list[float], rng: random.Random | None = None) -> int:
    """Sample an index from a (already-normalized) probability distribution."""
    rng = rng or random.Random()
    if not probs:
        raise ValueError("sample() from an empty distribution")
    total = sum(probs)
    if total <= 0:
        raise ValueError("sample() from a distribution that sums to <= 0")
    target = rng.random() * total
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if cumulative >= target:
            return i
    return len(probs) - 1  # floating-point fallback


def select_token(logits: list[float], *, temperature: float = 0.0, top_k: int | None = None,
                 top_p: float | None = None, rng: random.Random | None = None) -> int:
    """The full pipeline: temperature -> softmax -> top-k -> top-p -> sample
    (or argmax at temperature 0, the standard "greedy decoding" convention).
    """
    if temperature == 0:
        return argmax(logits)
    scaled = apply_temperature(logits, temperature)
    probs = softmax(scaled)
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    return sample(probs, rng=rng)
