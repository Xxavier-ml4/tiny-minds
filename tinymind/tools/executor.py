"""``ToolExecutor``: the only code path in TinyMind that calls a tool function.

Security posture (engineering brief sections 4, 30, 31, and
docs/architecture/tinymind-design.md section 12): this module never calls
``eval``, ``exec``, or a subprocess with ``shell=True``, and it never calls
anything except a function object already present in a ``ToolRegistry`` —
there is no path from model output, a tool's arguments, or the confirmation
flag to arbitrary code execution. A ``DESTRUCTIVE`` or ``SENSITIVE`` tool
additionally requires ``confirmed=True`` from the *caller* of ``execute()``;
the model itself never supplies that flag.
"""
from __future__ import annotations

from tinymind.tools.registry import ToolRegistry
from tinymind.tools.results import ToolCall, ToolResult, timed_result
from tinymind.tools.validation import validate


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, call: ToolCall, *, confirmed: bool = False) -> ToolResult:
        if call.name not in self._registry:
            return ToolResult(call=call, ok=False, error=f"unknown tool: {call.name}",
                              error_code="unknown_tool")

        entry = self._registry.get(call.name)

        validation = validate(call.arguments, entry.schema.parameters)
        if not validation.valid:
            detail = "; ".join(str(e) for e in validation.errors)
            return ToolResult(call=call, ok=False, error=f"invalid arguments: {detail}",
                              error_code="validation_failed")

        if entry.permissions.needs_confirmation and not confirmed:
            return ToolResult(
                call=call, ok=False,
                error=(f"tool {call.name!r} requires confirmation "
                      f"(capabilities: {entry.permissions.describe()})"),
                error_code="confirmation_required")

        return timed_result(call, lambda: entry.fn(**call.arguments))

    def execute_many(self, calls: list[ToolCall], *, confirmed: bool = False) -> list[ToolResult]:
        return [self.execute(call, confirmed=confirmed) for call in calls]
