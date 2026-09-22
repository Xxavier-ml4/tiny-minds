"""``ToolCall`` / ``ToolResult``: the shapes that flow through the tool loop.

Kept intentionally small — this is data, not behavior. ``executor.py``
produces ``ToolResult`` objects; ``tinymind.runtime.session`` decides what
to do with them.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any


@dataclasses.dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclasses.dataclass
class ToolResult:
    call: ToolCall
    ok: bool
    value: Any = None
    error: str | None = None
    error_code: str | None = None
    """One of: "unknown_tool", "validation_failed", "confirmation_required",
    "execution_error", or None when ok is True."""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        out = {"name": self.call.name, "ok": self.ok}
        if self.ok:
            out["value"] = self.value
        else:
            out["error"] = self.error
            out["error_code"] = self.error_code
        return out


def timed_result(call: ToolCall, fn) -> ToolResult:
    """Run ``fn()`` (a zero-arg closure) and wrap it as a ``ToolResult``,
    measuring latency and catching exceptions rather than letting a broken
    tool crash the caller — see the brief section 54, "never hide errors":
    the exception's message is preserved in ``error``, not swallowed."""
    start = time.monotonic()
    try:
        value = fn()
        return ToolResult(call=call, ok=True, value=value,
                          latency_ms=(time.monotonic() - start) * 1000.0)
    except Exception as exc:  # noqa: BLE001 - intentionally broad: any tool may raise
        return ToolResult(call=call, ok=False, error=str(exc), error_code="execution_error",
                          latency_ms=(time.monotonic() - start) * 1000.0)
