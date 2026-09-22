"""Stage-promotion gate (brief section 51): a stage is promoted because it
met measurable criteria, not because training finished.

``criteria`` is JSON::

    {"expect_stage": "stage1",
     "require_stage_complete": true,
     "checks": [{"metric": "summary.final_validation.val_loss", "max": 1.5},
                {"metric": "eval.capabilities.tool_arithmetic.accuracy", "min": 0.30}]}

A metric path starts with ``summary.`` (the run's ``training_summary.json``)
or ``eval.`` (a ``tinymind eval`` result) and then follows dictionary keys.
A missing metric FAILS its check (an unmeasured thing is not a passed thing).
The thresholds live in ``configs/stages/*.gate.json`` and are *policy*, chosen
and recorded in ``docs/training/three-stage-plan.md``; this module only applies them.
Checkpoint integrity and identity are checked separately by
``tinymind verify-checkpoint`` (the workflow runs both).
"""
from __future__ import annotations

from typing import Any


def _lookup(root: dict[str, Any] | None, path: list[str]) -> Any:
    cur: Any = root
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def evaluate_gate(criteria: dict[str, Any], summary: dict[str, Any] | None = None,
                  eval_results: dict[str, Any] | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, value: Any = None, requirement: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "value": value, "requirement": requirement})

    if "expect_stage" in criteria:
        got = (summary or {}).get("stage")
        add("stage", got == criteria["expect_stage"], got, f"== {criteria['expect_stage']!r}")
    if criteria.get("require_stage_complete"):
        got = (summary or {}).get("stage_complete")
        add("stage_complete", got is True, got, "is true")
    if criteria.get("require_no_divergence", True):
        got = (summary or {}).get("stop_reason")
        add("no_divergence", got is not None and got != "diverged", got, "!= diverged")
    roots = {"summary": summary, "eval": eval_results}
    for spec in criteria.get("checks", []):
        head, *path = spec["metric"].split(".")
        value = _lookup(roots.get(head), path)
        req = []
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok and "min" in spec:
            req.append(f">= {spec['min']}")
            ok = value >= spec["min"]
        if ok and "max" in spec:
            req.append(f"<= {spec['max']}")
            ok = value <= spec["max"]
        if "min" not in spec and "max" not in spec:
            req.append("(no bound given)")
            ok = False
        add(spec["metric"], ok, value, " and ".join(req) if req else "")
    return {"passed": all(c["ok"] for c in checks) and bool(checks), "checks": checks}
