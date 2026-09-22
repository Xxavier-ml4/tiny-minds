"""Bounded, in-process conversation memory. Does not survive a process
restart — that's ``tinymind.memory.long_term``'s job.
"""
from __future__ import annotations

import collections

from tinymind.memory.store import MemoryEntry, MemoryStore


class ShortTermMemory(MemoryStore):
    def __init__(self, max_entries: int = 50) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._max_entries = max_entries
        self._entries: collections.OrderedDict[str, MemoryEntry] = collections.OrderedDict()

    def add(self, entry: MemoryEntry) -> str:
        self._entries[entry.entry_id] = entry
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)  # drop oldest
        return entry.entry_id

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    def all(self, scope: str | None = None) -> list[MemoryEntry]:
        entries = list(self._entries.values())
        return [e for e in entries if e.scope == scope] if scope else entries

    def clear(self) -> None:
        self._entries.clear()
