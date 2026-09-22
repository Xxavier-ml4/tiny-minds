"""The token-acceptance state machine interface a real constrained decoder
would drive: at each generation step, ask "which tokens are legal next?"
and narrow the model's output distribution to that set before sampling.

``ConstraintStateMachine`` is the interface a real implementation must
satisfy; it is not itself a claim that constrained decoding is implemented
end-to-end in this delivery — that needs a real tokenizer vocabulary and a
real model's logits to mask, neither of which exist yet (see
``tokenizer_constraints.py`` and ``docs/architecture/tinymind-design.md``
section 3).

``JsonPrefixStateMachine`` below is real and useful today: given text fed
incrementally, it answers "is this still a valid prefix of *some* complete
JSON document" versus "this is already broken; no continuation fixes it."
Rather than hand-rolling a character-by-character JSON tokenizer (fragile —
numbers, string escapes, and literal keywords each have their own partial-
match rules), it re-parses the accumulated buffer with the standard
library's own ``json`` module on every ``feed()`` call and classifies the
resulting ``JSONDecodeError`` by its position: an error exactly at the end
of the buffer means "ran out of input, still extendable"; an error before
the end means "there's already content that cannot be made valid by
appending more." Two literal-scanning quirks in the stdlib parser need a
small special case each (see ``_classify``): an unterminated string reports
the position of its *opening* quote, not the end of input, and a partial
literal keyword (``"tru"`` as a prefix of ``true``) reports "Expecting
value" at the position that keyword starts, not the end either. Both are
handled explicitly and are covered by ``tests/test_state_machine.py``.

One known, inherent limitation, not specific to this approach: a bare
top-level scalar (a lone number, not inside an object/array) is
ambiguous about whether it has finished growing — ``"12"`` is already
complete, valid JSON on its own, even though a real token stream might
still be about to emit ``".5"``. Resolving that needs an explicit
end-of-generation signal from the caller, not something inferable from the
text alone; this is a property of streaming numeric literals in general; it
is not particular to this state machine's implementation approach.
"""
from __future__ import annotations

import abc
import json


class ConstraintStateMachine(abc.ABC):
    """Interface: track whether a growing output is still on-schema."""

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def feed(self, chunk: str) -> None:
        """Advance the machine's state by one chunk of text (may be one
        character, one token's text, or more)."""

    @abc.abstractmethod
    def is_valid_so_far(self) -> bool:
        """False means the output is *already* unrecoverable — no
        continuation could make it valid. True means either already
        complete-and-valid, or still extendable into something valid."""

    @abc.abstractmethod
    def is_complete(self) -> bool:
        """True once the fed text is a complete, valid instance."""


_JSON_LITERALS = ("true", "false", "null")


class JsonPrefixStateMachine(ConstraintStateMachine):
    """See module docstring for the approach and its one known limitation."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._complete = False
        self._broken = False

    def feed(self, chunk: str) -> None:
        if self._broken:
            return
        self._buffer += chunk
        self._complete, self._broken = self._classify(self._buffer)

    def is_valid_so_far(self) -> bool:
        return not self._broken

    def is_complete(self) -> bool:
        return self._complete

    @staticmethod
    def _classify(buffer: str) -> tuple[bool, bool]:
        """Returns (complete, broken)."""
        stripped = buffer.strip()
        if not stripped:
            return False, False  # empty/whitespace-only: incomplete, not broken
        try:
            json.loads(buffer)
            return True, False
        except json.JSONDecodeError as exc:
            if exc.msg.startswith("Unterminated string starting at"):
                return False, False  # mid-string: always extendable
            if exc.msg == "Expecting value":
                tail = buffer[exc.pos:].strip()
                if tail and any(lit.startswith(tail) for lit in _JSON_LITERALS):
                    return False, False  # partial "tru", "fals", "nul", ...
            if exc.pos >= len(stripped):
                return False, False  # ran out of input right at the boundary
            return False, True  # leftover content that's already inconsistent
