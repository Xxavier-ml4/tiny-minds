"""Scoring of a model's *generated text* against an eval record's ``meta``.

Every scorer is a pure function of (generated text, meta) and returns a small
dict, so a result can be re-derived from a stored generation. The model's output
is untrusted text: nothing here executes it. ``safe_eval`` reads a calculator
expression with ``ast`` and evaluates only integer ``+ - *`` and parentheses,
which is what lets "arithmetic accuracy" mean *the model wrote an expression that
evaluates to the right value* — the runtime, not the model, does the sums.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any

TOOL_SCHEMAS: dict[str, dict[str, type]] = {
    "calculator": {"expr": str}, "get_weather": {"city": str}, "set_timer": {"minutes": int},
    "lookup": {"query": str}, "remember": {"key": str, "value": str},
}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
REFUSAL_STARTS = ("i can't", "i cannot", "i can’t", "sorry, i can't", "sorry, i can’t", "sorry, i cannot")


def safe_eval(expr: str) -> int | None:
    """Integer value of ``expr`` if it is a plain +,-,* expression over integers (<= 60 chars), else ``None``."""
    if not isinstance(expr, str) or len(expr) > 60:
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        raise ValueError("unsupported")

    try:
        return walk(tree)
    except (ValueError, RecursionError, OverflowError):
        return None


def looks_like_call(text: str) -> bool:
    """Did the model *attempt* a tool call (whether or not it is well formed)?"""
    t = text.strip()
    return t.startswith("{") and '"name"' in t or t.startswith('[{"name"')


def parse_tool_call(text: str) -> dict[str, Any]:
    """``{"status": "ok"|"malformed"|"none", "name":..., "arguments":..., "why":...}``. ``ok`` means valid JSON
    of the exact shape ``{"name": <known tool>, "arguments": <object matching that tool's schema>}``."""
    t = text.strip()
    if not looks_like_call(t):
        return {"status": "none"}
    try:
        obj = json.loads(t)
    except ValueError:
        return {"status": "malformed", "why": "not valid JSON"}
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"} or not isinstance(obj["arguments"], dict):
        return {"status": "malformed", "why": "wrong shape"}
    name, args = obj["name"], obj["arguments"]
    schema = TOOL_SCHEMAS.get(name)
    if schema is None:
        return {"status": "malformed", "why": f"unknown tool {name!r}", "name": name, "arguments": args}
    if set(args) != set(schema) or any(type(args[k]) is not typ for k, typ in schema.items()):
        return {"status": "malformed", "why": "arguments do not match the tool's schema", "name": name, "arguments": args}
    return {"status": "ok", "name": name, "arguments": args}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def score(text: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Score one generation. Always returns ``{"ok": bool, ...detail}``."""
    kind = meta["score"]
    out = _norm(text)
    if kind == "exact":
        return {"ok": out == _norm(str(meta["expected"]))}
    if kind in ("contains", "contains_ci"):
        hay = out.lower() if kind == "contains_ci" else out
        need = [str(x).lower() if kind == "contains_ci" else str(x) for x in meta["expected"]]
        return {"ok": all(x in hay for x in need) and not looks_like_call(out)}
    if kind == "json":
        try:
            return {"ok": json.loads(text.strip()) == meta["expected"]}
        except ValueError:
            return {"ok": False, "why": "invalid JSON"}
    if kind == "clarify":
        return {"ok": out.endswith("?") and not looks_like_call(out)}
    if kind == "refusal":
        return {"ok": out.lower().startswith(REFUSAL_STARTS) and not looks_like_call(out)}
    if kind == "no_tool":
        return {"ok": not looks_like_call(out) and all(x in out.lower() for x in meta.get("expected", [])), "called_tool": looks_like_call(out)}
    if kind in ("tool_call", "tool_name"):
        call = parse_tool_call(text)
        want = meta["tool"]
        name_ok = call["status"] == "ok" and call["name"] == want
        detail = {"status": call["status"], "called": call.get("name"), "name_ok": name_ok}
        if kind == "tool_name":
            return {"ok": name_ok, **detail}
        args_ok = False
        if name_ok:
            if want == "calculator":
                args_ok = safe_eval(call["arguments"]["expr"]) == meta["expr_value"]
            else:
                args_ok = call["arguments"] == meta["args"]
        return {"ok": name_ok and args_ok, "args_ok": args_ok, **detail}
    raise ValueError(f"unknown scorer {kind!r}")


def repetition(text: str, n: int = 6) -> float:
    """Share of character n-grams that repeat an earlier one (0 = none; ~1 = a loop)."""
    t = text.strip()
    if len(t) < n + 6:
        return 0.0
    grams = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)
