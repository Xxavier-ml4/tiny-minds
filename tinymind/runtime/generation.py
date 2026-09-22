"""``GenerationConfig``: the parameters that shape one ``generate()`` call,
independent of any specific ``ModelBackend``.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    stop_sequences: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive if set, got {self.top_k}")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1] if set, got {self.top_p}")

    def truncate_at_stop_sequence(self, text: str) -> str:
        """Cut ``text`` at the earliest occurrence of any configured stop
        sequence. A real, useful, backend-independent utility — any
        ``ModelBackend`` implementation can call this after generating,
        rather than reimplementing stop-sequence handling itself."""
        earliest = len(text)
        for stop in self.stop_sequences:
            if not stop:
                continue
            index = text.find(stop)
            if index != -1:
                earliest = min(earliest, index)
        return text[:earliest]
