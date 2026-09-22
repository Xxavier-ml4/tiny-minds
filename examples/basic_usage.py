"""Basic TinyMind usage, matching the API shapes in the engineering brief
section 43. Run with: python examples/basic_usage.py

Uses the default EchoBackend (see tinymind/model/backends/echo.py) since
there is no trained TinyMind model in this delivery — every output below
is an honest echo/failure, not a claim about model quality. This example
exists to show the *shape* of the API working end to end, which it does,
today, for real.
"""
from tinymind import Model
from tinymind.tools.builtins import calculator, unit_convert

print("=== plain generation ===")
model = Model("models/tinymind-150m-q4.tm")
print(model.generate("Explain photosynthesis in one sentence."))

print("\n=== with tools ===")
model = Model("models/tinymind-150m-q4.tm", tools=[calculator, unit_convert])
result = model.run("What's the capital of France?")  # no tool needed -> CHAT mode
print(f"mode={result.mode.value} ok={result.ok} text={result.text!r}")

print("\n=== direct tool execution (the part that works fully without a trained model) ===")
result = model._session.execute_tool_call(
    "calculator", {"expression": "847 * 39"}, tool_ranked_scores=[9.5, 0.2])
print(f"ok={result.ok} value={result.tool_result.value} "
     f"confidence={result.confidence.confidence:.2f}")

print("\n=== streaming ===")
for chunk in model.stream("stream this back please"):
    print(chunk, end="")
print()
