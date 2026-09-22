"""``ModelBackend``: the seam between the runtime and any neural model.

Per docs/architecture/tinymind-design.md section 7 and section 48 of the
engineering brief, the runtime must not care how the underlying model works.
Everything in ``tinymind.runtime`` talks to a ``ModelBackend``, never to a
concrete model class, so a real trained backend (Phase 3+) or a native-engine
backend (Phase 7+) is a new class implementing this interface, not a
rewrite of the runtime.
"""
from __future__ import annotations

import abc
import dataclasses
import time
from typing import Iterator


@dataclasses.dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    finish_reason: str  # "stop" | "length" | "error"
    latency_ms: float


class ModelBackend(abc.ABC):
    """Interface every TinyMind model backend implements."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from ``path``. Idempotent-safe to call once."""

    @abc.abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> GenerationResult: ...

    @abc.abstractmethod
    def stream(self, prompt: str, max_new_tokens: int = 256,
               temperature: float = 0.0) -> Iterator[str]:
        """Yield text chunks as they're produced."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear any per-session state (KV cache, conversation), keep weights."""

    @abc.abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abc.abstractmethod
    def is_real_model(self) -> bool:
        """False for reference/testing backends (e.g. ``EchoBackend``).

        Every response built on top of a backend where this is False must
        be treated as synthetic by anything that reports confidence or
        benchmark numbers to a person — see ``tinymind.confidence``, which
        reads this flag to null out the ``model`` confidence component
        rather than inventing a number.
        """


def _timed(fn):
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = fn(*args, **kwargs)
        result.latency_ms = (time.monotonic() - start) * 1000.0
        return result
    return wrapper
