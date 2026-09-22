"""A ``Grammar`` is anything that can answer "is this text a complete match"
and "could more text still make it match" — the same two questions
``ConstraintStateMachine`` answers for JSON specifically, generalized to
non-JSON constrained outputs (an enum of allowed strings, a fixed pattern
like a phone number or ID format).

``RegexGrammar`` and ``ChoiceGrammar`` are real, working implementations —
useful today for constraining, say, a single enum-valued argument.
Compiling a *full* JSON Schema (with nested objects, typed properties, and
per-field grammars) into one ``Grammar`` the way Needle's native engine does
at ``needle_init`` time (needle-analysis.md section 6) is real, substantial
work — it means turning a schema into a grammar that also knows about JSON
structural syntax *and* per-field type/range constraints simultaneously.
``JsonSchemaGrammar`` below documents that interface and raises
``NotImplementedError``; the structural half of that problem (valid JSON
nesting) is already solved and tested in
``tinymind.runtime.constraints.state_machine.JsonPrefixStateMachine``, and
full-value validation once a candidate is complete is already solved in
``tinymind.runtime.constraints.json_schema`` — what's missing is fusing the
two into one incremental, per-token grammar, which needs a real tokenizer
to be worth building (a token boundary rarely lines up with a JSON
structural boundary, and getting that fusion right without a real
tokenizer to test against would be guesswork).
"""
from __future__ import annotations

import abc
import re


class Grammar(abc.ABC):
    @abc.abstractmethod
    def matches(self, text: str) -> bool:
        """True iff ``text`` is a complete, valid match."""

    @abc.abstractmethod
    def is_valid_prefix(self, text: str) -> bool:
        """True iff some completion of ``text`` could still match."""


_PROBE_CHARS = ("5", "A", "a", "-", " ")  # digit, upper, lower, punctuation, space
_MAX_PROBE_EXTRA = 12


class RegexGrammar(Grammar):
    """Wraps a compiled regex.

    Python's stdlib ``re`` has no built-in notion of "partial match" (the
    third-party ``regex`` package adds one via ``partial=True``; it is not
    available in this environment — see ``STATUS.md``). ``is_valid_prefix``
    therefore uses a bounded, honest heuristic rather than a general
    regex-to-NFA prefix engine: beyond checking whether ``text`` is already
    a complete match, it probes whether padding ``text`` with 0 to
    ``_MAX_PROBE_EXTRA`` copies of a single character drawn from a small
    fixed probe set (a digit, an uppercase letter, a lowercase letter, a
    punctuation mark, a space) produces a complete match. That correctly
    handles the common fixed-format case this class is meant for — a
    pattern like ``[A-Z]{2}\\d{4}`` where the *rest* of a match is
    homogeneous (all digits, or all letters) — but will miss a pattern that
    can only be completed by a *mix* of different character kinds in a
    specific order that the probe set doesn't happen to hit; it will not
    raise, it will just (correctly, conservatively) return ``False`` for
    the parts of the language it doesn't probe. See
    ``tests/test_grammar.py`` for the cases this is and isn't expected to
    catch.
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = re.compile(pattern)

    def matches(self, text: str) -> bool:
        return self._pattern.fullmatch(text) is not None

    def is_valid_prefix(self, text: str) -> bool:
        if self._pattern.fullmatch(text):
            return True
        for extra_len in range(1, _MAX_PROBE_EXTRA + 1):
            for probe_char in _PROBE_CHARS:
                if self._pattern.fullmatch(text + probe_char * extra_len):
                    return True
        return False


class ChoiceGrammar(Grammar):
    """Accepts exactly one of a fixed set of literal strings — the grammar
    an enum-valued field needs."""

    def __init__(self, choices: list[str]) -> None:
        if not choices:
            raise ValueError("ChoiceGrammar needs at least one choice")
        self._choices = list(choices)

    def matches(self, text: str) -> bool:
        return text in self._choices

    def is_valid_prefix(self, text: str) -> bool:
        return any(choice.startswith(text) for choice in self._choices)


class JsonSchemaGrammar(Grammar):
    """Interface only in this delivery — see module docstring for exactly
    what's missing and why (a real tokenizer to test token-boundary fusion
    against)."""

    def __init__(self, schema: dict) -> None:
        self._schema = schema

    def matches(self, text: str) -> bool:
        raise NotImplementedError(
            "JsonSchemaGrammar needs a real tokenizer to fuse structural JSON "
            "validity (already implemented: state_machine.JsonPrefixStateMachine) "
            "with per-field type/range checks (already implemented: "
            "json_schema.validate_structured_output) at the token level. Today, "
            "validate a *complete* candidate with "
            "tinymind.runtime.constraints.json_schema.parse_structured_output "
            "instead of asking this grammar mid-generation.")

    def is_valid_prefix(self, text: str) -> bool:
        raise NotImplementedError(self.matches.__doc__ or "")
