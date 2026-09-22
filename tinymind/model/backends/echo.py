"""``EchoBackend``: a deterministic, non-neural ``ModelBackend``.

This is not a scaled-down language model and must never be described as
one. Its only job is to give every other subsystem (session management, the
tool loop, the CLI, the router) something real to run against end to end
before a trained checkpoint exists. It is used throughout ``tests/`` for
exactly that reason.

Two things it does are genuinely non-trivial rather than fully fake:

1. ``embed()`` uses the hashing trick (feature hashing over whitespace
   tokens into a fixed-width, L2-normalized vector) — a real, if crude,
   embedding technique, not a random or zero vector. Two texts that share
   words land closer together in cosine distance than two that don't,
   which is enough for ``tinymind.tools.retrieval`` to have something
   meaningful to test lexical-vs-embedding retrieval against.
2. ``generate()`` is a deterministic function of its input, not random,
   so tests that assert exact output stay stable across runs and Python
   versions.
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Iterator

from tinymind.model.backend import GenerationResult, ModelBackend

_EMBED_DIM = 64


class EchoBackend(ModelBackend):
    """Deterministic stand-in backend. See module docstring."""

    def __init__(self) -> None:
        self._loaded = False
        self._loaded_path: str | None = None

    def load(self, path: str) -> None:
        # There are no real weights to read; this only records that
        # `load()` was called, so callers that check `is_loaded` behave the
        # same way they would against a real backend.
        self._loaded = True
        self._loaded_path = path

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_real_model(self) -> bool:
        return False

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> GenerationResult:
        start = time.monotonic()
        words = prompt.split()
        capped = words[: max(1, max_new_tokens // 4)]  # rough words-per-token stand-in
        finish_reason = "length" if len(capped) < len(words) else "stop"
        text = "[echo] " + " ".join(capped)
        latency_ms = (time.monotonic() - start) * 1000.0
        return GenerationResult(text=text, tokens_generated=len(capped),
                                finish_reason=finish_reason, latency_ms=latency_ms)

    def stream(self, prompt: str, max_new_tokens: int = 256,
               temperature: float = 0.0) -> Iterator[str]:
        result = self.generate(prompt, max_new_tokens, temperature)
        for chunk in result.text.split(" "):
            yield chunk + " "

    def reset(self) -> None:
        pass  # no session state to clear in this backend

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * _EMBED_DIM
        tokens = text.lower().split()
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % _EMBED_DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
