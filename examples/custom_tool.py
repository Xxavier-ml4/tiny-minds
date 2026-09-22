"""Registering a custom tool with argument constraints and permissions, and
running it through the full real pipeline (validate -> execute -> ground ->
score confidence). Run with: python examples/custom_tool.py

This is the part of TinyMind that works completely independently of
whether a trained model exists — see tinymind/runtime/session.py's
module docstring.
"""
from typing import Annotated

from tinymind.runtime.engine import Engine
from tinymind.model.backends.echo import EchoBackend
from tinymind.tools import Capability, Constraint, ToolPermissions, ToolRegistry


def set_thermostat(room: str, temperature: Annotated[int, Constraint(
        minimum=10, maximum=30, description="Target temperature in Celsius, 10-30.")]) -> dict:
    """Set a room's thermostat target temperature.

    Args:
        room: Which room's thermostat to set.
        temperature: Target temperature in Celsius.
    """
    return {"room": room, "temperature": temperature}


registry = ToolRegistry()
registry.register(set_thermostat, permissions=ToolPermissions(capabilities=Capability.LOCAL_WRITE))

engine = Engine(EchoBackend(), registry)
session = engine.create_session("demo")

print("=== a request whose arguments are grounded in the text ===")
result = session.execute_tool_call(
    "set_thermostat", {"room": "bedroom", "temperature": 21},
    source_text="set the bedroom to 21 degrees", tool_ranked_scores=[9.0, 0.1])
print(f"ok={result.ok} value={result.tool_result.value} confidence={result.confidence.confidence:.2f}")

print("\n=== an out-of-range argument is rejected before it ever runs ===")
result = session.execute_tool_call("set_thermostat", {"room": "bedroom", "temperature": 45}, confirmed=True)
print(f"ok={result.ok} error={result.tool_result.error}")

print("\n=== an invented (ungrounded) argument is caught even though it's in-range ===")
result = session.execute_tool_call(
    "set_thermostat", {"room": "bedroom", "temperature": 25},
    source_text="make the bedroom a reasonable temperature", tool_ranked_scores=[9.0, 0.1])
print(f"ok={result.ok} reason={result.reason}")
