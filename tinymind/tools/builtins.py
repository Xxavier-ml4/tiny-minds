"""Deterministic built-in tools, per engineering brief section 8 ("hybrid
reasoning" — arithmetic, date math, and unit conversion should be computed,
not generated). ``register_builtins()`` adds all of them to a
``ToolRegistry`` so a TinyMind application gets guaranteed-correct answers
for these cases by default, the same way the brief's example
(``calculator(847, "*", 39)`` -> deterministic executor -> ``33033``) wants.

The calculator specifically does **not** use ``eval()`` — see
``_safe_eval`` — per the hard security rule in
docs/architecture/tinymind-design.md section 12 and the engineering brief
section 30: nothing in TinyMind's tool layer evaluates a string as code, a
built-in tool included.
"""
from __future__ import annotations

import ast
import datetime as _dt
import operator
from typing import Annotated, Literal

from tinymind.tools.schema import Constraint
from tinymind.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# calculator: arithmetic via a whitelisted AST walk, never eval()
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPRESSION_LENGTH = 200
_MAX_POWER_EXPONENT = 12


class CalculatorError(ValueError):
    pass


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalculatorError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER_EXPONENT:
            raise CalculatorError(f"exponent {right} exceeds the allowed maximum "
                                  f"({_MAX_POWER_EXPONENT}) to bound computation")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: Annotated[str, Constraint(
        description="An arithmetic expression, e.g. '847 * 39' or '(12 + 3) / 5'.",
        max_length=_MAX_EXPRESSION_LENGTH)]) -> dict:
    """Evaluate an arithmetic expression exactly, without asking a model to
    compute it. Supports + - * / // % ** and parentheses; nothing else.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"expression exceeds {_MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse expression: {exc}") from exc
    result = _safe_eval(tree)
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# unit conversion: a small deterministic table, no model involvement
# ---------------------------------------------------------------------------

# Each entry converts to a common base unit for its dimension.
_LENGTH_TO_METERS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                     "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}
_MASS_TO_GRAMS = {"mg": 0.001, "g": 1.0, "kg": 1000.0, "oz": 28.349523125,
                  "lb": 453.59237}
_VOLUME_TO_LITERS = {"ml": 0.001, "l": 1.0, "tsp": 0.00492892, "tbsp": 0.0147868,
                     "cup": 0.24, "floz": 0.0295735, "pt": 0.473176,
                     "qt": 0.946353, "gal": 3.78541}

Unit = Literal[
    "mm", "cm", "m", "km", "in", "ft", "yd", "mi",
    "mg", "g", "kg", "oz", "lb",
    "ml", "l", "tsp", "tbsp", "cup", "floz", "pt", "qt", "gal",
    "c", "f", "k",
]


class UnitConversionError(ValueError):
    pass


def _dimension_table(unit: str) -> tuple[dict, str]:
    if unit in _LENGTH_TO_METERS:
        return _LENGTH_TO_METERS, "length"
    if unit in _MASS_TO_GRAMS:
        return _MASS_TO_GRAMS, "mass"
    if unit in _VOLUME_TO_LITERS:
        return _VOLUME_TO_LITERS, "volume"
    raise UnitConversionError(f"unknown unit: {unit!r}")


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == "c":
        kelvin = value + 273.15
    elif from_unit == "f":
        kelvin = (value - 32) * 5 / 9 + 273.15
    else:
        kelvin = value
    if to_unit == "c":
        return kelvin - 273.15
    if to_unit == "f":
        return (kelvin - 273.15) * 9 / 5 + 32
    return kelvin


def unit_convert(value: float, from_unit: Unit, to_unit: Unit) -> dict:
    """Convert a numeric value between units of the same dimension exactly.

    Args:
        value: The numeric value to convert.
        from_unit: The unit the value is currently in.
        to_unit: The unit to convert to.
    """
    temp_units = {"c", "f", "k"}
    if from_unit in temp_units or to_unit in temp_units:
        if from_unit not in temp_units or to_unit not in temp_units:
            raise UnitConversionError(
                f"cannot convert between temperature unit {from_unit!r}/{to_unit!r} "
                "and a non-temperature unit")
        result = _convert_temperature(value, from_unit, to_unit)
    else:
        table_from, dim_from = _dimension_table(from_unit)
        table_to, dim_to = _dimension_table(to_unit)
        if dim_from != dim_to:
            raise UnitConversionError(
                f"cannot convert {from_unit!r} ({dim_from}) to {to_unit!r} ({dim_to})")
        result = value * table_from[from_unit] / table_to[to_unit]
    return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": result}


# ---------------------------------------------------------------------------
# date arithmetic: stdlib datetime, no model involvement
# ---------------------------------------------------------------------------

class DateArithmeticError(ValueError):
    pass


def _parse_date(value: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise DateArithmeticError(f"expected an ISO date (YYYY-MM-DD), got {value!r}") from exc


def add_days(date: str, days: int) -> dict:
    """Add (or subtract, with a negative value) a number of days to an ISO date.

    Args:
        date: The starting date, as YYYY-MM-DD.
        days: Number of days to add; negative subtracts.
    """
    result = _parse_date(date) + _dt.timedelta(days=days)
    return {"date": date, "days": days, "result": result.isoformat()}


def days_between(start_date: str, end_date: str) -> dict:
    """Compute the number of days between two ISO dates.

    Args:
        start_date: The earlier date, as YYYY-MM-DD.
        end_date: The later date, as YYYY-MM-DD.
    """
    delta = _parse_date(end_date) - _parse_date(start_date)
    return {"start_date": start_date, "end_date": end_date, "days": delta.days}


def register_builtins(registry: ToolRegistry) -> ToolRegistry:
    """Register the calculator, unit_convert, add_days, and days_between
    tools on ``registry``, all tagged read-only (see
    ``tinymind.tools.permissions``: pure functions of their arguments, no
    filesystem/network/state access). Returns ``registry`` for chaining."""
    from tinymind.tools.permissions import Capability, ToolPermissions

    read_only = ToolPermissions(capabilities=Capability.READ_ONLY)
    for fn in (calculator, unit_convert, add_days, days_between):
        registry.register(fn, permissions=read_only)
    return registry
