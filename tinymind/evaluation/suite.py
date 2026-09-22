"""A frozen, categorized acceptance-suite runner for tool-use evaluation.

This generalizes the strongest pattern the Needle analysis found
(needle-analysis.md sections 21 and 22: hand-curated ``environments/``,
each a 32-case suite across six categories, with certain categories marked
``critical`` so one critical regression fails the suite regardless of the
aggregate pass rate) into a reusable library feature, per that analysis's
own recommendation (needle-analysis.md section 24) — rather than a
convention repeated by hand for every new tool domain, as it is in the
source it's inspired by.

This module is the reusable framework: ``Category``, ``Case``,
``AcceptanceSuite``. A concrete example suite over an independently
designed tool domain (adjustable-desk/focus-timer controls — a different
domain from anything in the Needle source, chosen specifically so nothing
here is a renamed copy of Needle's own smart-home/media-player/etc.
examples) lives in ``benchmarks/tools/desk_suite.py`` and is real, runnable
code today against the rule-based demo predictor defined there — it is not
a claim that TinyMind's *eventual trained model* would score anything in
particular; see that file's module docstring.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable


class Category(enum.Enum):
    POSITIVE = "positive"      # a clear, satisfiable request: should produce the right call
    MISSING = "missing"        # required info is absent: should NOT call with a guessed value
    IRRELEVANT = "irrelevant"  # no tool in this catalogue applies: should not force a call
    NEGATION = "negation"      # a negated request: should not call the naively-matched tool
    INVALID = "invalid"        # the request implies an out-of-range/invalid argument: should not call with it
    PARALLEL = "parallel"      # the request needs more than one call


_CRITICAL_CATEGORIES = frozenset({Category.MISSING, Category.NEGATION, Category.INVALID})


@dataclasses.dataclass
class Case:
    case_id: str
    input_text: str
    category: Category
    expect_call: bool
    """Whether a tool call is expected at all. False for most MISSING/
    IRRELEVANT/NEGATION/INVALID cases; True for POSITIVE and PARALLEL."""
    expected_tool: str | None = None
    expected_arguments: dict[str, Any] | None = None
    expected_call_count: int = 1
    """>1 only meaningful for PARALLEL cases."""

    @property
    def critical(self) -> bool:
        return self.category in _CRITICAL_CATEGORIES


@dataclasses.dataclass
class PredictedCall:
    tool: str
    arguments: dict[str, Any]


@dataclasses.dataclass
class CaseResult:
    case: Case
    passed: bool
    detail: str
    predicted: list[PredictedCall]


@dataclasses.dataclass
class SuiteResult:
    suite_name: str
    results: list[CaseResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def critical_failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed and r.case.critical]

    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 1.0

    def by_category(self) -> dict[str, tuple[int, int]]:
        """category -> (passed, total)"""
        out: dict[str, tuple[int, int]] = {}
        for result in self.results:
            key = result.case.category.value
            passed, total = out.get(key, (0, 0))
            out[key] = (passed + (1 if result.passed else 0), total + 1)
        return out

    def summary(self) -> str:
        lines = [f"{self.suite_name}: {sum(1 for r in self.results if r.passed)}/{len(self.results)} passed "
                f"({self.pass_rate():.0%}), {len(self.critical_failures)} critical failure(s)"]
        for category, (passed, total) in sorted(self.by_category().items()):
            lines.append(f"  {category:12s} {passed}/{total}")
        return "\n".join(lines)


class AcceptanceSuite:
    def __init__(self, name: str, cases: list[Case]) -> None:
        self.name = name
        self.cases = cases

    def run(self, predict: Callable[[str], list[PredictedCall]]) -> SuiteResult:
        results = [self._check(case, predict(case.input_text)) for case in self.cases]
        return SuiteResult(suite_name=self.name, results=results)

    @staticmethod
    def _check(case: Case, predicted: list[PredictedCall]) -> CaseResult:
        if not case.expect_call:
            if not predicted:
                return CaseResult(case, True, "correctly made no call", predicted)
            return CaseResult(case, False, f"expected no call, got {[p.tool for p in predicted]}", predicted)

        if not predicted:
            return CaseResult(case, False, "expected a call, got none", predicted)

        if len(predicted) != case.expected_call_count:
            return CaseResult(case, False,
                              f"expected {case.expected_call_count} call(s), got {len(predicted)}", predicted)

        if case.expected_tool is not None:
            actual_tools = [p.tool for p in predicted]
            if case.expected_tool not in actual_tools:
                return CaseResult(case, False,
                                  f"expected tool {case.expected_tool!r}, got {actual_tools}", predicted)

        if case.expected_arguments is not None:
            matched = next((p for p in predicted if p.tool == case.expected_tool), predicted[0])
            for key, value in case.expected_arguments.items():
                if matched.arguments.get(key) != value:
                    return CaseResult(case, False,
                                      f"argument {key!r}: expected {value!r}, got {matched.arguments.get(key)!r}",
                                      predicted)

        return CaseResult(case, True, "matched expectations", predicted)
