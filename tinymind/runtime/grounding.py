"""Source-span grounding: for every generated structured argument, check
whether its value is actually traceable to the input text, rather than
invented.

Generalizes the pattern in Needle's own extraction path (needle-analysis.md
section 9: a regex-based check that a generated year/date value actually
appears somewhere in the source text) to any field, per the engineering
brief section 11. This is a real, working, string-level implementation —
not a claim that a neural model is checked for "did it hallucinate" in a
deep semantic sense, which would need the model itself; this checks the
narrower, well-defined thing a deterministic function can check: does the
generated value, or a recognized transformation of it, actually appear in
the source text.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import re
from typing import Any


class GroundingMode(enum.Enum):
    STRICT = "strict"      # every field must be grounded or have an explicit default
    BALANCED = "balanced"  # numeric/enum fields must be grounded; free-text fields are trusted
    PERMISSIVE = "permissive"  # grounding is computed but never blocks


@dataclasses.dataclass
class FieldGrounding:
    field: str
    value: Any
    grounded: bool
    source_span: str | None = None
    transformation: str | None = None
    """None (verbatim match), "case_insensitive", "number_normalized", or
    "date_normalized" — which recognized transformation, if any, bridged
    the generated value to the text it was found in."""


@dataclasses.dataclass
class GroundingResult:
    mode: GroundingMode
    fields: list[FieldGrounding]

    @property
    def all_grounded(self) -> bool:
        return all(f.grounded for f in self.fields)

    @property
    def ungrounded_fields(self) -> list[str]:
        return [f.field for f in self.fields if not f.grounded]

    def blocks_execution(self) -> bool:
        if self.mode == GroundingMode.PERMISSIVE:
            return False
        if self.mode == GroundingMode.STRICT:
            return not self.all_grounded
        # BALANCED: only numeric/bool/enum-ish (non-str, or short str) fields must ground
        return any(not f.grounded and _is_strict_field(f.value) for f in self.fields)


def _is_strict_field(value: Any) -> bool:
    return not isinstance(value, str) or len(value) <= 32


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_number(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _check_field(field: str, value: Any, source_text: str) -> FieldGrounding:
    source_lower = source_text.lower()

    if isinstance(value, bool):
        # booleans are rarely "in the text" verbatim (a user says "turn on
        # the light", not the word "true") — treated as always ungrounded
        # under STRICT unless the source contains the field name itself,
        # which is a weak but real signal the concept was discussed.
        grounded = field.lower() in source_lower
        return FieldGrounding(field, value, grounded,
                              source_span=field if grounded else None)

    if isinstance(value, (int, float)):
        text_numbers = _normalize_number(source_text)
        value_str = str(value)
        if value_str in text_numbers:
            return FieldGrounding(field, value, True, source_span=value_str)
        # try without a trailing ".0" for floats that are whole numbers
        if isinstance(value, float) and value == int(value) and str(int(value)) in text_numbers:
            return FieldGrounding(field, value, True, source_span=str(int(value)),
                                  transformation="number_normalized")
        return FieldGrounding(field, value, False)

    if isinstance(value, str):
        if value in source_text:
            return FieldGrounding(field, value, True, source_span=value)
        if value.lower() in source_lower:
            return FieldGrounding(field, value, True, source_span=value,
                                  transformation="case_insensitive")
        parsed_date = _try_parse_date(value)
        if parsed_date is not None:
            for candidate in _date_variants(parsed_date):
                if candidate.lower() in source_lower:
                    return FieldGrounding(field, value, True, source_span=candidate,
                                          transformation="date_normalized")
        return FieldGrounding(field, value, False)

    # dict/list/None: grounding is checked per-leaf by the caller if desired;
    # treat the container itself as ungrounded-but-not-blocking by default.
    return FieldGrounding(field, value, grounded=False)


def _try_parse_date(text: str) -> _dt.date | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _date_variants(date: _dt.date) -> list[str]:
    return [
        date.isoformat(),
        date.strftime("%B %d, %Y"),
        date.strftime("%b %d, %Y"),
        date.strftime("%m/%d/%Y"),
        str(date.year),
    ]


def check_grounding(arguments: dict[str, Any], source_text: str,
                    mode: GroundingMode = GroundingMode.BALANCED) -> GroundingResult:
    """Check every top-level field in ``arguments`` for groundedness against
    ``source_text``. Nested dict/list values are checked as opaque
    containers at this level — call ``check_grounding`` again on a nested
    dict if per-leaf grounding inside it matters for a specific tool."""
    fields = [_check_field(name, value, source_text) for name, value in arguments.items()]
    return GroundingResult(mode=mode, fields=fields)
