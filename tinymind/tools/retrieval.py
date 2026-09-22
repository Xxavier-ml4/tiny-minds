"""Tool retrieval: narrow a large catalogue to the top-k tools for a query.

Needle's built-in contrastive embedding head does this well
(needle-analysis.md section 8) but needs a trained model. TinyMind's
``ToolRetriever`` interface is deliberately implementation-agnostic — the
signature is the contract, not the scoring method — so an embedding-based
implementation (once ``ModelBackend.embed`` is backed by a real model) is a
second class implementing the same interface, not a rewrite of anything that
calls a retriever. ``LexicalRetriever`` below is a real, working, tested
BM25 implementation (see ``tinymind._text_ranking``, shared with
``tinymind.memory.retrieval`` so the algorithm exists in exactly one place)
usable today with zero trained model. ``EmbeddingRetriever`` is also real:
it works against any ``ModelBackend``, including today's ``EchoBackend``
(see its module docstring for why that's meaningful, not just a stub).

Every implementation always ranks the *entire* catalogue, even when it's
smaller than ``k`` — ``k`` only ever truncates the final result, it never
skips scoring — because callers (``tinymind.routing.router.Router`` in
particular) rely on ``retrieve_scored()``'s ordering and scores being
meaningful even for a small catalogue, not just a large one.
"""
from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING

from tinymind._text_ranking import bm25_rank
from tinymind.tools.registry import RegisteredTool

if TYPE_CHECKING:
    from tinymind.model.backend import ModelBackend


def _tool_text(tool: RegisteredTool) -> str:
    parts = [tool.schema.name, tool.schema.description or ""]
    for prop_name, prop_schema in tool.schema.parameters.get("properties", {}).items():
        parts.append(prop_name)
        if isinstance(prop_schema, dict) and prop_schema.get("description"):
            parts.append(prop_schema["description"])
    return " ".join(parts)


class ToolRetriever(abc.ABC):
    @abc.abstractmethod
    def retrieve_scored(self, query: str, tools: list[RegisteredTool],
                        k: int = 8) -> list[tuple[RegisteredTool, float]]:
        """Return up to ``k`` ``(tool, score)`` pairs, highest score first,
        ranked over the *entire* ``tools`` list regardless of how it
        compares to ``k``."""

    def retrieve(self, query: str, tools: list[RegisteredTool], k: int = 8) -> list[RegisteredTool]:
        return [tool for tool, _score in self.retrieve_scored(query, tools, k)]


class LexicalRetriever(ToolRetriever):
    """Okapi BM25 (``tinymind._text_ranking.bm25_rank``) over tool name +
    description + parameter names/descriptions.

    No index persistence — recomputed per call, which is fine at the
    catalogue sizes ``retrieve()`` is meant for (tens to low hundreds of
    tools; see the brief section 5 for the intended pipeline this sits at
    the start of).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def retrieve_scored(self, query: str, tools: list[RegisteredTool],
                        k: int = 8) -> list[tuple[RegisteredTool, float]]:
        if not tools:
            return []
        documents = [_tool_text(t) for t in tools]
        ranked = bm25_rank(query, documents, k1=self.k1, b=self.b)
        return [(tools[i], score) for i, score in ranked[:k]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


class EmbeddingRetriever(ToolRetriever):
    """Embed the query and every tool with a ``ModelBackend`` and rank by
    cosine similarity. Works today against ``EchoBackend``'s hashing-trick
    embeddings (see tinymind/model/backends/echo.py); swaps to real
    semantic retrieval automatically once a trained backend's ``embed()``
    is backed by real representations — no change needed here."""

    def __init__(self, backend: "ModelBackend") -> None:
        self._backend = backend

    def retrieve_scored(self, query: str, tools: list[RegisteredTool],
                        k: int = 8) -> list[tuple[RegisteredTool, float]]:
        if not tools:
            return []
        query_vec = self._backend.embed(query)
        scored = [(tool, _cosine(query_vec, self._backend.embed(_tool_text(tool))))
                  for tool in tools]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
