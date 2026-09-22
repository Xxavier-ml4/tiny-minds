"""Distillation interfaces, per the engineering brief section 21 — "one of
the most important parts of the project," and also one that structurally
cannot start until Phase 3/4 produce a student model and Phase 4 produces a
trained base to distill into (brief section 44: distillation is Phase 5,
after pretraining exists). ``Teacher``, ``Filter``, and ``Scorer`` are the
real interfaces; nothing implements them yet because there is no teacher
endpoint configured and no student model to train in this delivery.

The one substantive design decision worth recording now, before any
implementation: per brief section 21 ("do not copy massive private
chain-of-thought into training data... prefer concise, useful reasoning
traces"), ``DistillationExample.reasoning_trace`` below is typed and
documented as a *short* field for exactly that reason — the dataclass
shape itself is a small guard against the failure mode the brief calls
out, even before there's a generator to populate it.
"""
from __future__ import annotations

import abc
import dataclasses


@dataclasses.dataclass
class DistillationExample:
    prompt: str
    target: dict  # same shape as tinymind.data.validate's `target` field
    reasoning_trace: str | None = None
    """Short by design — brief section 21: prefer '17 chickens - 5 = 12'
    over a lengthy hidden chain-of-thought transcript. Not enforced with a
    hard length limit here (that belongs in tinymind.distillation.filters,
    once it exists) but the field is documented as short so a future
    generator isn't reaching for a different convention."""
    source: str = "teacher"
    verified: bool = False


class Teacher(abc.ABC):
    """A large model used to generate distillation demonstrations. No
    concrete implementation ships in this delivery — it needs a configured
    endpoint (local or remote) and API credentials that are deployment-
    specific, not something this delivery can hard-code. Needle's own
    approach (needle-analysis.md section 11: an OpenRouter-compatible chat
    endpoint) is a reasonable pattern to follow when this is implemented."""

    @abc.abstractmethod
    def generate(self, prompt: str, tools: list[dict] | None = None) -> DistillationExample:
        raise NotImplementedError(
            "Teacher needs a configured model endpoint; none is configured in this "
            "delivery — see STATUS.md")


class Filter(abc.ABC):
    """Accept or reject a ``DistillationExample`` before it enters a
    training set (brief section 21: preference filtering, verifier
    filtering)."""

    @abc.abstractmethod
    def accepts(self, example: DistillationExample) -> bool: ...


class Scorer(abc.ABC):
    """Score a ``DistillationExample``'s quality, for curriculum ordering
    or filtering thresholds."""

    @abc.abstractmethod
    def score(self, example: DistillationExample) -> float: ...


class VerifierFilter(Filter):
    """A real, working ``Filter``: reject any example whose target is a
    tool_call/structured output that doesn't validate against its own
    declared schema, or (for a math answer) whose answer doesn't match
    ``tinymind.tools.builtins.calculator``'s recomputation. Distillation
    example generation itself needs a Teacher (not implemented); *checking*
    an example against a schema or recomputing arithmetic needs neither —
    this filter is real for exactly the same reason
    ``tinymind.runtime.verification.verifier`` is real."""

    def __init__(self) -> None:
        from tinymind.runtime.verification.verifier import MathVerifier
        self._math_verifier = MathVerifier()

    def accepts(self, example: DistillationExample) -> bool:
        from tinymind.tools.validation import validate as validate_schema

        target_type = example.target.get("type")
        if target_type == "tool_call":
            arguments = example.target.get("arguments", {})
            schema = example.target.get("_schema")  # caller-supplied, not part of the stored target
            if schema is None:
                return True  # nothing to check against
            return validate_schema(arguments, schema).valid
        if target_type == "answer":
            # only meaningfully checkable when the prompt is itself a bare
            # arithmetic expression; anything else passes through unfiltered
            outcome = self._math_verifier.verify(example.prompt, example.target.get("content"), {})
            return outcome.passed if _looks_arithmetic(example.prompt) else True
        return True


def _looks_arithmetic(text: str) -> bool:
    import re
    return bool(re.fullmatch(r"[\d\s+\-*/().]+", text.strip()))
