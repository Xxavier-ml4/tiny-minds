"""Persistent memory: a small SQLite-backed store, using only the standard
library (no vector database, no external service — appropriate for the
mobile/low-RAM targets this whole project is aimed at). Survives a process
restart, unlike ``tinymind.memory.short_term.ShortTermMemory``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from tinymind.memory.store import MemoryEntry, MemoryStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_entries (
    entry_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    scope TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_entries_scope ON memory_entries(scope);
"""


class LongTermMemory(MemoryStore):
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, entry: MemoryEntry) -> str:
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_entries "
            "(entry_id, content, source, scope, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (entry.entry_id, entry.content, entry.source, entry.scope,
             entry.confidence, entry.timestamp))
        self._conn.commit()
        return entry.entry_id

    def get(self, entry_id: str) -> MemoryEntry | None:
        row = self._conn.execute(
            "SELECT entry_id, content, source, scope, confidence, timestamp "
            "FROM memory_entries WHERE entry_id = ?", (entry_id,)).fetchone()
        return _row_to_entry(row) if row else None

    def delete(self, entry_id: str) -> None:
        self._conn.execute("DELETE FROM memory_entries WHERE entry_id = ?", (entry_id,))
        self._conn.commit()

    def all(self, scope: str | None = None) -> list[MemoryEntry]:
        if scope is not None:
            rows = self._conn.execute(
                "SELECT entry_id, content, source, scope, confidence, timestamp "
                "FROM memory_entries WHERE scope = ? ORDER BY timestamp", (scope,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT entry_id, content, source, scope, confidence, timestamp "
                "FROM memory_entries ORDER BY timestamp").fetchall()
        return [_row_to_entry(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "LongTermMemory":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _row_to_entry(row: tuple) -> MemoryEntry:
    entry_id, content, source, scope, confidence, timestamp = row
    return MemoryEntry(entry_id=entry_id, content=content, source=source, scope=scope,
                       confidence=confidence, timestamp=timestamp)
