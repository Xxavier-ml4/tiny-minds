"""An example ``AcceptanceSuite`` over an independently-designed tool
domain: adjustable-desk / desk-lamp / focus-timer controls. Chosen
specifically because it overlaps with none of Needle's own six
``environments/`` domains (smart home, media player, productivity,
wearable, kitchen appliance, data capture) — this is a fresh example, not a
renamed copy of one of those (see needle-analysis.md section 21 and
docs/architecture/tinymind-design.md section 13).

Two things here have different maturity, on purpose:

1. The tools (``adjust_desk_height``, ``toggle_desk_lamp``,
   ``set_focus_timer``) and their JSON-Schema range constraints are real,
   registered ``tinymind.tools`` tools, validated through the real
   ``ToolExecutor`` — an INVALID-category case (e.g. a desk height outside
   the tools' declared range) is rejected by the *actual* validation logic
   in ``tinymind.tools.validation``, not a separately hand-coded range
   check duplicated here.
2. ``_demo_predict`` — the thing standing in for "a model deciding which
   tool to call" — is a small, explicitly rule-based (regex/keyword)
   function. It exists only to give the suite something to run against
   today, so the *suite mechanism itself* (category pass/fail, critical-
   failure gating) is demonstrably real and working. It is not a claim
   about what TinyMind's eventual trained model would score on this suite,
   and it should never be cited as a benchmark result for TinyMind the
   product — only for TinyMind's evaluation *framework*.

Run directly (``python -m benchmarks.tools.desk_suite``) to see the report.
"""
from __future__ import annotations

import re
from typing import Annotated

from tinymind.evaluation.suite import AcceptanceSuite, Case, Category, PredictedCall, SuiteResult
from tinymind.tools.executor import ToolExecutor
from tinymind.tools.permissions import Capability, ToolPermissions
from tinymind.tools.registry import ToolRegistry
from tinymind.tools.results import ToolCall
from tinymind.tools.schema import Constraint


def adjust_desk_height(height_cm: Annotated[int, Constraint(
        minimum=60, maximum=130, description="Target desk height in centimeters, 60-130.")]) -> dict:
    """Raise or lower the standing desk."""
    return {"height_cm": height_cm}


def toggle_desk_lamp(on: Annotated[bool, Constraint(description="Whether the lamp should be on.")],
                     brightness: Annotated[int, Constraint(
                         minimum=0, maximum=100, description="Brightness percentage, 0-100.")] = 100) -> dict:
    """Turn the desk lamp on or off, optionally setting its brightness."""
    return {"on": on, "brightness": brightness}


def set_focus_timer(minutes: Annotated[int, Constraint(
        minimum=1, maximum=180, description="Timer duration in minutes, 1-180.")]) -> dict:
    """Start a focus/pomodoro-style countdown timer."""
    return {"minutes": minutes}


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    permissions = ToolPermissions(capabilities=Capability.LOCAL_WRITE)
    registry.register(adjust_desk_height, permissions=permissions)
    registry.register(toggle_desk_lamp, permissions=permissions)
    registry.register(set_focus_timer, permissions=permissions)
    return registry


_NEGATION_RE = re.compile(r"\b(don't|do not|won't|please don't|please do not)\b")



def _extract_clause(clause: str, executor: ToolExecutor) -> PredictedCall | None:
    lower = clause.lower()
    negated = bool(_NEGATION_RE.search(lower))

    if "lamp" in lower:
        if negated:
            return None
        brightness_match = re.search(r"(\d+)\s*percent", lower)
        on = "off" not in lower
        arguments: dict = {"on": on}
        if brightness_match:
            arguments["brightness"] = int(brightness_match.group(1))
        result = executor.execute(ToolCall(name="toggle_desk_lamp", arguments=arguments), confirmed=True)
        return PredictedCall("toggle_desk_lamp", arguments) if result.ok else None

    if "desk" in lower or ("height" in lower and "cm" in lower):
        if negated:
            return None
        height_match = re.search(r"(-?\d+)\s*(?:cm|centimeters?)", lower)
        if not height_match:
            return None
        arguments = {"height_cm": int(height_match.group(1))}
        result = executor.execute(ToolCall(name="adjust_desk_height", arguments=arguments), confirmed=True)
        return PredictedCall("adjust_desk_height", arguments) if result.ok else None

    if "timer" in lower or "focus" in lower:
        if negated:
            return None
        minutes_match = re.search(r"(-?\d+)\s*minute", lower)
        if not minutes_match:
            return None
        arguments = {"minutes": int(minutes_match.group(1))}
        result = executor.execute(ToolCall(name="set_focus_timer", arguments=arguments), confirmed=True)
        return PredictedCall("set_focus_timer", arguments) if result.ok else None

    return None


def make_demo_predictor(executor: ToolExecutor):
    def predict(text: str) -> list[PredictedCall]:
        clauses = re.split(r"\band\b", text)
        calls = [_extract_clause(clause, executor) for clause in clauses]
        return [c for c in calls if c is not None]
    return predict


_CASES = [
    # POSITIVE
    Case("pos_height", "Raise my desk to 110 centimeters", Category.POSITIVE,
        expect_call=True, expected_tool="adjust_desk_height", expected_arguments={"height_cm": 110}),
    Case("pos_lamp", "Turn on the desk lamp at 80 percent", Category.POSITIVE,
        expect_call=True, expected_tool="toggle_desk_lamp", expected_arguments={"on": True, "brightness": 80}),
    Case("pos_timer", "Start a 25 minute focus timer", Category.POSITIVE,
        expect_call=True, expected_tool="set_focus_timer", expected_arguments={"minutes": 25}),
    # MISSING
    Case("missing_height", "Adjust my desk height", Category.MISSING, expect_call=False),
    Case("missing_timer", "Set a focus timer", Category.MISSING, expect_call=False),
    Case("missing_lamp_brightness_ok", "Turn on the desk lamp", Category.MISSING,
        # brightness has a default, so this one IS a complete, callable
        # request — included specifically to check the suite doesn't
        # penalize a truly-optional field being absent.
        expect_call=True, expected_tool="toggle_desk_lamp", expected_arguments={"on": True}),
    # IRRELEVANT
    Case("irrelevant_weather", "What's the weather like today?", Category.IRRELEVANT, expect_call=False),
    Case("irrelevant_joke", "Tell me a joke", Category.IRRELEVANT, expect_call=False),
    # NEGATION
    Case("negation_lamp", "Don't turn on the desk lamp", Category.NEGATION, expect_call=False),
    Case("negation_height", "Please don't change my desk height right now", Category.NEGATION, expect_call=False),
    # INVALID
    Case("invalid_height", "Raise my desk to 500 centimeters", Category.INVALID, expect_call=False),
    Case("invalid_timer", "Set a focus timer for -10 minutes", Category.INVALID, expect_call=False),
    Case("invalid_brightness", "Turn on the desk lamp at 300 percent", Category.INVALID, expect_call=False),
    # PARALLEL
    Case("parallel_height_timer", "Raise my desk to 110 centimeters and start a 30 minute focus timer",
        Category.PARALLEL, expect_call=True, expected_call_count=2),
    Case("parallel_lamp_height", "Turn on the desk lamp at 60 percent and raise the desk to 100 centimeters",
        Category.PARALLEL, expect_call=True, expected_call_count=2),
]


def build_suite() -> tuple[AcceptanceSuite, ToolExecutor]:
    registry = _build_registry()
    executor = ToolExecutor(registry)
    return AcceptanceSuite("desk", _CASES), executor


def run() -> SuiteResult:
    suite, executor = build_suite()
    predict = make_demo_predictor(executor)
    return suite.run(predict)


if __name__ == "__main__":
    result = run()
    print(result.summary())
    if result.critical_failures:
        print("\nCritical failures:")
        for failure in result.critical_failures:
            print(f"  {failure.case.case_id}: {failure.detail}")
