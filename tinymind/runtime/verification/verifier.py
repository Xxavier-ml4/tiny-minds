"""``Verifier``: a second pass that checks a response before it's trusted,
per the engineering brief section 39.

Four verifiers are real and implemented, because each is a deterministic
check over something already in this codebase: is this valid JSON against
this schema (reuses ``tinymind.runtime.constraints.json_schema``), does
this arithmetic expression actually equal the claimed result (reuses
``tinymind.tools.builtins.calculator``), does this tool call match a
registered tool's schema (reuses ``tinymind.tools.validation``), and is
this argument grounded in its source text (reuses
``tinymind.runtime.grounding``). A neural self-consistency or
teacher-model verifier needs a trained model and is future work (see
``STATUS.md``); the ``Verifier`` interface those will implement is defined
here so adding one later doesn't change how ``VerifierChain`` or its
callers work.
"""
from __future__ import annotations

import abc
import dataclasses
from typing import Any

from tinymind.runtime.constraints.json_schema import validate_structured_output
from tinymind.runtime.grounding import GroundingMode, check_grounding
from tinymind.tools.builtins import CalculatorError, calculator
from tinymind.tools.registry import ToolRegistry
from tinymind.tools.validation import validate


@dataclasses.dataclass
class VerificationOutcome:
    verifier: str
    passed: bool
    detail: str = ""


class Verifier(abc.ABC):
    name: str = "verifier"

    @abc.abstractmethod
    def verify(self, request: Any, response: Any, context: dict[str, Any]) -> VerificationOutcome: ...


class JsonSchemaVerifier(Verifier):
    """Checks ``response`` (a parsed value) against ``context["schema"]``."""
    name = "json_schema"

    def verify(self, request: Any, response: Any, context: dict[str, Any]) -> VerificationOutcome:
        schema = context.get("schema")
        if schema is None:
            return VerificationOutcome(self.name, passed=False, detail="context['schema'] is required")
        result = validate_structured_output(response, schema)
        detail = "valid" if result.valid else "; ".join(result.errors)
        return VerificationOutcome(self.name, passed=result.valid, detail=detail)


class MathVerifier(Verifier):
    """Re-derives an arithmetic answer with the deterministic calculator and
    checks it matches. ``request`` is the expression string; ``response``
    is the claimed numeric result."""
    name = "math"

    def verify(self, request: Any, response: Any, context: dict[str, Any]) -> VerificationOutcome:
        try:
            actual = calculator(str(request))["result"]
        except CalculatorError as exc:
            return VerificationOutcome(self.name, passed=False, detail=f"could not evaluate: {exc}")
        passed = _numbers_equal(actual, response)
        detail = f"expected {actual}, got {response}" if not passed else "matches"
        return VerificationOutcome(self.name, passed=passed, detail=detail)


def _numbers_equal(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return a == b


class ToolSchemaVerifier(Verifier):
    """Checks that ``response`` (a ``{"name": ..., "arguments": {...}}``
    dict) names a real tool in ``context["registry"]`` and that its
    arguments validate against that tool's schema."""
    name = "tool_schema"

    def verify(self, request: Any, response: Any, context: dict[str, Any]) -> VerificationOutcome:
        registry: ToolRegistry | None = context.get("registry")
        if registry is None:
            return VerificationOutcome(self.name, passed=False, detail="context['registry'] is required")
        name = response.get("name") if isinstance(response, dict) else None
        if name is None or name not in registry:
            return VerificationOutcome(self.name, passed=False, detail=f"unknown tool: {name!r}")
        entry = registry.get(name)
        result = validate(response.get("arguments", {}), entry.schema.parameters)
        detail = "valid" if result.valid else "; ".join(str(e) for e in result.errors)
        return VerificationOutcome(self.name, passed=result.valid, detail=detail)


class GroundingVerifier(Verifier):
    """Checks that ``response`` (a tool-call arguments dict) is grounded in
    ``context["source_text"]``."""
    name = "grounding"

    def verify(self, request: Any, response: Any, context: dict[str, Any]) -> VerificationOutcome:
        source_text = context.get("source_text", str(request))
        mode = context.get("grounding_mode", GroundingMode.BALANCED)
        result = check_grounding(response, source_text, mode=mode)
        detail = ("all fields grounded" if result.all_grounded
                 else f"ungrounded: {result.ungrounded_fields}")
        return VerificationOutcome(self.name, passed=not result.blocks_execution(), detail=detail)


class VerifierChain:
    """Run a sequence of verifiers and combine their outcomes."""

    def __init__(self, verifiers: list[Verifier]) -> None:
        self._verifiers = verifiers

    def run(self, request: Any, response: Any, context: dict[str, Any]) -> list[VerificationOutcome]:
        return [v.verify(request, response, context) for v in self._verifiers]

    def all_passed(self, request: Any, response: Any, context: dict[str, Any]) -> bool:
        return all(o.passed for o in self.run(request, response, context))
