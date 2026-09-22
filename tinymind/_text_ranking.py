"""Internal (underscore-prefixed: not part of the public API) shared BM25
ranking used by both ``tinymind.tools.retrieval`` and
``tinymind.memory.retrieval`` — one implementation of the algorithm, two
call sites, rather than the same scoring loop copied twice. Standard,
textbook Okapi BM25; nothing here is specific to tools or to memory.
"""
from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Standard IR practice: strip common function words before scoring, so two
# texts that only share words like "the"/"is"/"to" don't register as a
# spurious match. See tinymind/tools/retrieval.py and
# tinymind/memory/retrieval.py for the concrete cases this matters for.
_STOPWORDS = frozenset("""
a an the of to in on at for with and or is are was were be been being
this that these those it its as by from into your you i my me we our
what which who whom how do does did can could should would will
""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]


def bm25_rank(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75
             ) -> list[tuple[int, float]]:
    """Rank ``documents`` (by index into the list) against ``query``,
    highest score first. A document with zero term overlap with the query
    scores exactly ``0.0`` — callers can use ``score > 0.0`` as a "was
    there any real lexical signal at all" check, which several callers in
    this codebase do (see ``tinymind.routing.router.Router``)."""
    query_terms = tokenize(query)
    if not documents:
        return []
    if not query_terms:
        return [(i, 0.0) for i in range(len(documents))]

    docs = [tokenize(d) for d in documents]
    doc_lengths = [len(d) for d in docs]
    avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    n_docs = len(docs)

    doc_freq: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scores = []
    for doc, doc_len in zip(docs, doc_lengths):
        term_counts: dict[str, int] = {}
        for term in doc:
            term_counts[term] = term_counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            freq = term_counts.get(term, 0)
            if freq == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / (avg_len or 1))
            score += idf * (freq * (k1 + 1)) / (denom or 1)
        scores.append(score)

    ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    return [(i, scores[i]) for i in ranked]
