"""Multi-step tool plans: one tool call's result feeding another's argument.

Split deliberately into two pieces with very different maturity:

- ``PlanExecutor`` runs an already-built ``Plan`` — resolving ``Ref``
  placeholders against earlier steps' results and calling
  ``ToolExecutor.execute`` in order. This has no dependency on a trained
  model and is real, tested code (the "search_for_contact then
  send_instant_message with the returned contact_id" pattern the
  engineering brief's own Needle analysis called out, needle-analysis.md
  behavior notes under doc/apis.md).
- ``Planner`` is the interface for *building* a ``Plan`` from a natural-
  language goal, which genuinely needs a reasoning model or a hand-written
  rule set specific to a domain. ``SingleStepPlanner`` is the one concrete
  implementation shipped here — it wraps one already-decided tool call into
  a one-step plan — because it is honestly implementable without a model;
  a model-driven planner is future work tracked in ``STATUS.md``.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any

from tinymind.tools.executor import ToolExecutor
from tinymind.tools.results import ToolCall, ToolResult


@dataclasses.dataclass(frozen=True)
class Ref:
    """A placeholder inside a later step's arguments pointing at an earlier
    step's result. ``path`` is a dotted lookup into that result when it's a
    dict (``""`` means "the whole result")."""
    step_id: str
    path: str = ""


@dataclasses.dataclass
class PlanStep:
    step_id: str
    tool_name: str
    arguments: dict[str, Any]  # values may contain Ref(...) placeholders, anywhere in the tree


@dataclasses.dataclass
class Plan:
    steps: list[PlanStep]

    def validate_refs(self) -> None:
        seen: set[str] = set()
        for step in self.steps:
            for ref in _find_refs(step.arguments):
                if ref.step_id not in seen:
                    raise PlanError(
                        f"step {step.step_id!r} references {ref.step_id!r}, which "
                        "has not run yet (or does not exist) — Refs may only point "
                        "at earlier steps")
            seen.add(step.step_id)


class PlanError(ValueError):
    pass


def _find_refs(value: Any) -> list[Ref]:
    if isinstance(value, Ref):
        return [value]
    if isinstance(value, dict):
        return [ref for v in value.values() for ref in _find_refs(v)]
    if isinstance(value, list):
        return [ref for item in value for ref in _find_refs(item)]
    return []


def _resolve(value: Any, results: dict[str, Any]) -> Any:
    if isinstance(value, Ref):
        if value.step_id not in results:
            raise PlanError(f"cannot resolve Ref to unknown or not-yet-run step {value.step_id!r}")
        target = results[value.step_id]
        if not value.path:
            return target
        for part in value.path.split("."):
            if not isinstance(target, dict) or part not in target:
                raise PlanError(f"path {value.path!r} does not exist in the result of "
                                f"step {value.step_id!r}")
            target = target[part]
        return target
    if isinstance(value, dict):
        return {k: _resolve(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(item, results) for item in value]
    return value


@dataclasses.dataclass
class PlanResult:
    step_results: dict[str, ToolResult]
    ordered_results: list[ToolResult]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.ordered_results)


class PlanExecutor:
    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor

    def run(self, plan: Plan, *, confirmed: bool = False, stop_on_error: bool = True) -> PlanResult:
        plan.validate_refs()
        step_results: dict[str, ToolResult] = {}
        raw_values: dict[str, Any] = {}
        ordered: list[ToolResult] = []
        for step in plan.steps:
            try:
                resolved_args = _resolve(step.arguments, raw_values)
            except PlanError as exc:
                result = ToolResult(call=ToolCall(name=step.tool_name, arguments={},
                                                  call_id=step.step_id),
                                    ok=False, error=str(exc), error_code="unresolved_ref")
            else:
                call = ToolCall(name=step.tool_name, arguments=resolved_args, call_id=step.step_id)
                result = self._executor.execute(call, confirmed=confirmed)
            step_results[step.step_id] = result
            raw_values[step.step_id] = result.value if result.ok else None
            ordered.append(result)
            if not result.ok and stop_on_error:
                break
        return PlanResult(step_results=step_results, ordered_results=ordered)


class Planner:
    """Interface for building a ``Plan`` from a natural-language goal.

    A real implementation needs either a trained reasoning model or a
    domain-specific rule set; neither exists in this delivery (see
    ``STATUS.md``). Subclass and implement ``plan()`` when one does.
    """

    def plan(self, goal: str, available_tools: list[str]) -> Plan:
        raise NotImplementedError(
            "Planner.plan() needs a reasoning model or a domain-specific rule set; "
            "see tinymind/tools/planner.py module docstring and STATUS.md. Use "
            "SingleStepPlanner for the common case of one already-decided call, or "
            "build a Plan by hand.")


class SingleStepPlanner(Planner):
    """Wrap one already-decided tool call into a one-step ``Plan``.

    This is the realistic case when something upstream (a router, a rule,
    a person) has already decided the tool and arguments — it needs no
    model, and most TinyMind tool use in this delivery goes through exactly
    this path (see ``tinymind.runtime.session``).
    """

    def plan_call(self, tool_name: str, arguments: dict[str, Any], step_id: str = "step_0") -> Plan:
        return Plan(steps=[PlanStep(step_id=step_id, tool_name=tool_name, arguments=arguments)])
