"""Retrieve the top-k relevant memory entries for a query.

The engineering brief section 12 is explicit: "Do not blindly inject the
entire memory into every prompt. Retrieve only relevant memories." This
module is the only sanctioned path from a ``MemoryStore`` to "what goes in
the prompt" — ``MemoryStore.all()`` exists for administration (listing,
export), not for prompt-building, and nothing in this codebase calls
``all()`` and hands the result to a model.
"""
from __future__ import annotations

from tinymind._text_ranking import bm25_rank
from tinymind.memory.store import MemoryEntry, MemoryStore


def retrieve_relevant(query: str, store: MemoryStore, k: int = 5,
                      scope: str | None = None,
                      min_confidence: float = 0.0) -> list[MemoryEntry]:
    """Rank ``store``'s entries (optionally restricted to ``scope`` and a
    minimum confidence) against ``query`` with BM25 and return the top
    ``k``, highest-scoring first. Entries with zero lexical overlap with
    the query are still ranked (BM25 gives them score 0.0) but ``k`` will
    typically exclude them anyway; callers that want "no result rather than
    an arbitrary one when nothing matches" should check
    ``retrieve_relevant_scored`` directly and filter on score.
    """
    return [entry for entry, _score in
            retrieve_relevant_scored(query, store, k, scope, min_confidence)]


def retrieve_relevant_scored(query: str, store: MemoryStore, k: int = 5,
                             scope: str | None = None,
                             min_confidence: float = 0.0) -> list[tuple[MemoryEntry, float]]:
    entries = [e for e in store.all(scope=scope) if e.confidence >= min_confidence]
    if not entries:
        return []
    ranked = bm25_rank(query, [e.content for e in entries])
    return [(entries[i], score) for i, score in ranked[:k]]
