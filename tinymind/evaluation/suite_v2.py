"""v2 capability evaluation report (brief section 10).

The existing ``tinymind.evaluation.tiny_suite`` already reports each capability
*category* on its own — ``capabilities[category] = {n, accuracy, ...}`` grouped
by each record's ``category`` — plus a separate ``tool_behavior`` block, and it
deliberately computes **no** single composite "quality score". v2 keeps that
suite unchanged and adds, on top of it, the explicit eleven-category view the
brief asks for:

    language  grammar  reasoning  math  knowledge  comprehension
    instruction  dialogue  tools  safety  integration

Each of the eleven gets its own metrics; several map to more than one raw
record category (e.g. ``grammar`` rolls up grammar/spelling/punctuation), and
the roll-up is example-count weighted so it is a true accuracy, not an average
of averages. **Every** individual raw metric is preserved alongside — nothing
is collapsed or dropped — so the full evaluation JSON still carries all of it
(brief section 10: "never collapse everything into one quality score").

``build_capability_report`` is a pure function of a tiny-suite result dict, so
it is testable without a model; ``run_v2_suite`` runs the suite and attaches
the report.
"""
from __future__ import annotations

from typing import Any

# Which raw record categories roll up into each of the eleven v2 categories.
# A raw category may appear in exactly one v2 category.
V2_CATEGORY_MAP: dict[str, tuple[str, ...]] = {
    "language": ("language", "paraphrase"),
    "grammar": ("grammar", "spelling", "punctuation"),
    "reasoning": ("reasoning",),
    "math": ("math",),
    "knowledge": ("knowledge",),
    "comprehension": ("comprehension", "extraction"),
    "instruction": ("instruction",),
    "dialogue": ("dialogue",),
    "tools": ("tools",),
    "safety": ("safety",),
    "integration": ("integration",),
}


def _weighted_accuracy(entries: list[dict[str, Any]]) -> tuple[int, float | None]:
    """Example-count-weighted accuracy over per-category entries that each have
    ``n`` and ``accuracy`` (accuracy may be None if n == 0)."""
    total_n = sum(e["n"] for e in entries)
    if total_n == 0:
        return 0, None
    correct = sum(e["n"] * e["accuracy"] for e in entries if e.get("accuracy") is not None)
    return total_n, correct / total_n


def build_capability_report(eval_result: dict[str, Any]) -> dict[str, Any]:
    """Return the eleven-category v2 report for a tiny-suite ``eval_result``.

    Shape::

        {"categories": {<v2 category>: {"n", "accuracy", "raw": {<raw cat>: {...}}}},
         "tool_behavior": {..., passthrough},
         "present": [...], "missing": [...],
         "note": "individual metrics preserved; no composite score"}

    ``present``/``missing`` name which of the eleven categories the eval set
    actually covered (a stage's test set covers its own categories plus any it
    replays), so a gate or a reader can tell measured from unmeasured.
    """
    caps = eval_result.get("capabilities", {})
    tool_behavior = eval_result.get("tool_behavior", {})
    categories: dict[str, Any] = {}
    present: list[str] = []
    missing: list[str] = []
    for v2_cat, raw_cats in V2_CATEGORY_MAP.items():
        entries = [caps[c] for c in raw_cats if c in caps]
        n, acc = _weighted_accuracy(entries)
        raw = {c: caps[c] for c in raw_cats if c in caps}
        # tools also reports its behavioural rates from tool_behavior.
        block: dict[str, Any] = {"n": n, "accuracy": acc, "raw": raw}
        if v2_cat == "tools" and tool_behavior:
            block["correct_tool_rate"] = tool_behavior.get("correct_tool_rate")
            block["wrong_tool_rate"] = tool_behavior.get("wrong_tool_rate")
            block["false_positive_call_rate"] = tool_behavior.get("false_positive_call_rate")
        categories[v2_cat] = block
        (present if n > 0 else missing).append(v2_cat)
    return {
        "categories": categories,
        "tool_behavior": tool_behavior,
        "present": present,
        "missing": missing,
        "note": "v2 eleven-category report; every individual raw metric is preserved and no composite score is computed",
    }


def run_v2_suite(pkg: Any, eval_records: Any, val_records: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Run the capability suite and attach the v2 eleven-category report.

    Delegates generation and scoring to ``tiny_suite.run_suite`` (unchanged), so
    every existing metric is produced exactly as before; this only adds
    ``v2_capability_report`` under the result. The suite JSON therefore stays a
    superset of the v1 one.
    """
    from tinymind.evaluation.tiny_suite import run_suite, strip_rows

    result = run_suite(pkg, eval_records, val_records, **kwargs)
    result = strip_rows(result)
    result["v2_capability_report"] = build_capability_report(result)
    return result
