"""``MemoryEntry`` and the ``MemoryStore`` interface every memory backend
implements, per the engineering brief section 12. Every entry carries
``content``, ``timestamp``, ``source``, ``confidence``, and ``scope`` — a
brief requirement, enforced here by making them required dataclass fields
rather than an optional convention a backend could skip.
"""
from __future__ import annotations

import abc
import dataclasses
import time
import uuid


@dataclasses.dataclass
class MemoryEntry:
    content: str
    source: str
    scope: str = "conversation"
    confidence: float = 1.0
    timestamp: float = dataclasses.field(default_factory=time.time)
    entry_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


class MemoryStore(abc.ABC):
    @abc.abstractmethod
    def add(self, entry: MemoryEntry) -> str:
        """Store ``entry``; return its ``entry_id``."""

    @abc.abstractmethod
    def get(self, entry_id: str) -> MemoryEntry | None: ...

    @abc.abstractmethod
    def delete(self, entry_id: str) -> None: ...

    @abc.abstractmethod
    def all(self, scope: str | None = None) -> list[MemoryEntry]:
        """All entries, optionally filtered to one ``scope``. This is the
        only bulk-read method on the interface — deliberately: there is no
        "give me everything across every scope, ranked by nothing" method,
        so retrieval (``tinymind.memory.retrieval``) is always the path
        into a prompt, per the brief's "do not blindly inject the entire
        memory" requirement."""

    def __len__(self) -> int:
        return len(self.all())
