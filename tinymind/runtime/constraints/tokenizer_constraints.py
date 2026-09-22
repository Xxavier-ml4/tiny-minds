"""The seam where a real tokenizer's vocabulary gets compiled into per-step
allowed-token masks for constrained generation.

This is genuinely not implementable without a trained model and a matching
tokenizer: computing "which of the ~32,000 vocabulary entries are legal
next" from a ``Grammar``/``ConstraintStateMachine`` requires enumerating (or
efficiently indexing) the tokenizer's vocabulary against the grammar, which
is meaningless work against ``tinymind.model.tokenizer.ByteTokenizer`` (256
byte-tokens; masking them is nearly free and not representative of the real
problem) and impossible without a trained subword tokenizer, which does not
exist in this delivery (see ``tinymind/model/tokenizer.py`` module
docstring and ``STATUS.md``).

What's here is the interface a real implementation must satisfy, and the
one piece of it that's genuinely tokenizer-independent and real: caching.
Compiling a mask for a large vocabulary against a fixed grammar is exactly
the kind of thing worth computing once and reusing across generation steps
that share the same grammar state — ``MaskCache`` is a real, tested LRU-ish
cache keyed on ``(grammar id, state key)``, ready for a real implementation
to use once one exists.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinymind.model.tokenizer import Tokenizer
    from tinymind.runtime.constraints.grammar import Grammar


class TokenizerConstraint(abc.ABC):
    """Interface: compile a token-id allow-mask from a grammar's current
    state, for a specific tokenizer's vocabulary."""

    @abc.abstractmethod
    def allowed_token_ids(self, grammar: "Grammar", state_key: str) -> set[int]:
        """Return the set of token ids legal as the *next* token, given the
        grammar and an opaque state key describing how much has been
        generated so far (e.g. the text generated in this field)."""
        raise NotImplementedError(
            "needs a real tokenizer vocabulary to enumerate/index against; "
            "see this module's docstring")


class MaskCache:
    """A small LRU cache from ``(grammar_id, state_key)`` to a computed
    allow-mask. Tokenizer-independent, real, and tested
    (``tests/test_tokenizer_constraints.py``) even though nothing populates
    it with a real mask yet."""

    def __init__(self, max_entries: int = 4096) -> None:
        self._max_entries = max_entries
        self._store: dict[tuple[int, str], set[int]] = {}
        self._order: list[tuple[int, str]] = []

    def get(self, grammar_id: int, state_key: str) -> set[int] | None:
        key = (grammar_id, state_key)
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None

    def put(self, grammar_id: int, state_key: str, mask: set[int]) -> None:
        key = (grammar_id, state_key)
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self._max_entries:
            oldest = self._order.pop(0)
            del self._store[oldest]
        self._store[key] = mask
        self._order.append(key)

    def __len__(self) -> int:
        return len(self._store)
