"""Structured-output validation against a JSON Schema.

Needle compiles a byte-level grammar from a schema inside its closed native
engine (needle-analysis.md section 6) — a real guarantee, but not something
a reader of that source tree can inspect or test. TinyMind's version here is
plain Python and testable with no model at all: it reuses
``tinymind.tools.validation`` (the same validator tool arguments go through
— one validator, two call sites, see that module's docstring) and adds the
structured-output-specific pieces: parsing a candidate string as JSON, and
a repair pass for the handful of malformed-JSON shapes a token-by-token
generator is likely to produce (a trailing comma, a missing closing
brace/bracket). This is real, useful validation; it is not the same thing
as *guaranteeing* well-formed output the way constraining generation at the
token level would — that guarantee needs ``tokenizer_constraints.py``, which
is an interface only in this delivery (see its module docstring for why).
"""
from __future__ import annotations

import dataclasses
import json

from tinymind.tools.validation import ValidationResult, validate


@dataclasses.dataclass
class StructuredOutputResult:
    valid: bool
    value: object | None
    errors: list[str]
    repaired: bool = False


def _try_repair(text: str) -> str | None:
    """Fix the small set of malformed-JSON shapes worth guessing at:
    a trailing comma before a closing brace/bracket, or a missing closing
    brace/bracket at the end. Returns None if no repair was attempted."""
    stripped = text.strip()
    if not stripped:
        return None
    repaired = stripped
    # Drop a trailing comma directly before a closing brace or bracket.
    import re
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Balance unclosed braces/brackets by appending closers in the order
    # their openers appeared, ignoring braces/brackets inside string
    # literals (bracket_depth tracking with a simple in-string flag).
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in repaired:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if stack:
        repaired = repaired + "".join(reversed(stack))
    return repaired if repaired != stripped else None


def parse_structured_output(text: str, schema: dict, *, attempt_repair: bool = True
                            ) -> StructuredOutputResult:
    """Parse ``text`` as JSON and validate it against ``schema``.

    On a JSON parse failure, tries one repair pass (see ``_try_repair``)
    when ``attempt_repair`` is True, and reports ``repaired=True`` if that
    repair is what let parsing succeed — callers that care about strict
    provenance (was this exactly what the model produced, or patched
    afterward) can branch on that flag.
    """
    for attempt_text, repaired in ((text, False), (_try_repair(text), True)):
        if attempt_text is None or (repaired and not attempt_repair):
            continue
        try:
            value = json.loads(attempt_text)
        except json.JSONDecodeError as exc:
            if not repaired:
                continue  # fall through to the repair attempt
            return StructuredOutputResult(valid=False, value=None,
                                          errors=[f"invalid JSON even after repair: {exc}"])
            continue
        result: ValidationResult = validate(value, schema)
        return StructuredOutputResult(valid=result.valid, value=value if result.valid else value,
                                      errors=[str(e) for e in result.errors], repaired=repaired)
    return StructuredOutputResult(valid=False, value=None, errors=["could not parse as JSON"])


def validate_structured_output(value: object, schema: dict) -> StructuredOutputResult:
    """Validate an already-parsed Python value (skip the JSON-text step) —
    useful when a backend already returns a Python object, not a string."""
    result = validate(value, schema)
    return StructuredOutputResult(valid=result.valid, value=value,
                                  errors=[str(e) for e in result.errors])
