import dataclasses
import unittest
from typing import Annotated, Literal

from tinymind.tools import (
    Capability, Constraint, LexicalRetriever, Plan, PlanExecutor, PlanStep, Ref,
    SingleStepPlanner, ToolCall, ToolExecutor, ToolPermissions, ToolRegistry,
    ToolRegistryError, add_days, calculator, days_between, register_builtins,
    schema_of, tool, unit_convert, validate,
)


class TestSchemaFromFunction(unittest.TestCase):
    def test_basic_types_and_docstring(self):
        def set_brightness(room: str, brightness: int) -> dict:
            """Set the brightness of a room light.

            Args:
                room: Which room.
                brightness: 0-100.
            """
            return {}

        schema = schema_of(set_brightness)
        self.assertEqual(schema.name, "set_brightness")
        self.assertEqual(schema.description, "Set the brightness of a room light.")
        self.assertEqual(schema.parameters["properties"]["room"]["type"], "string")
        self.assertEqual(schema.parameters["properties"]["brightness"]["type"], "integer")
        self.assertEqual(schema.parameters["properties"]["room"]["description"], "Which room.")
        self.assertEqual(set(schema.parameters["required"]), {"room", "brightness"})

    def test_default_value_makes_field_optional(self):
        def f(a: int, b: int = 5) -> int:
            return a + b
        schema = schema_of(f)
        self.assertEqual(schema.parameters["required"], ["a"])

    def test_constraint_annotation_applies_bounds(self):
        def f(x: Annotated[int, Constraint(minimum=0, maximum=10)]) -> int:
            return x
        schema = schema_of(f)
        self.assertEqual(schema.parameters["properties"]["x"]["minimum"], 0)
        self.assertEqual(schema.parameters["properties"]["x"]["maximum"], 10)

    def test_literal_becomes_enum(self):
        def f(mode: Literal["heat", "cool", "auto"]) -> str:
            return mode
        schema = schema_of(f)
        self.assertEqual(schema.parameters["properties"]["mode"]["enum"], ["heat", "cool", "auto"])

    def test_tool_decorator_caches_schema(self):
        @tool
        def f(x: int) -> int:
            return x
        self.assertTrue(hasattr(f, "_tinymind_schema"))
        self.assertIs(schema_of(f), f._tinymind_schema)


class TestSchemaFromDataclass(unittest.TestCase):
    def test_dataclass_schema(self):
        @dataclasses.dataclass
        class Receipt:
            total: float
            merchant: str
            note: str = ""

        schema = schema_of(Receipt)
        self.assertEqual(schema.name, "Receipt")
        self.assertEqual(set(schema.parameters["required"]), {"total", "merchant"})
        self.assertEqual(schema.parameters["properties"]["total"]["type"], "number")


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {
                "room": {"type": "string"},
                "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["room", "brightness"],
        }

    def test_valid_passes(self):
        result = validate({"room": "bedroom", "brightness": 30}, self.schema)
        self.assertTrue(result.valid)

    def test_missing_required_fails(self):
        result = validate({"room": "bedroom"}, self.schema)
        self.assertFalse(result.valid)
        self.assertIn("brightness", str(result.errors[0]))

    def test_out_of_range_fails(self):
        result = validate({"room": "bedroom", "brightness": 150}, self.schema)
        self.assertFalse(result.valid)

    def test_nested_array(self):
        schema = {"type": "array", "items": {"type": "integer", "minimum": 0}}
        self.assertTrue(validate([1, 2, 3], schema).valid)
        self.assertFalse(validate([1, -2, 3], schema).valid)

    def test_enum(self):
        schema = {"type": "string", "enum": ["a", "b"]}
        self.assertTrue(validate("a", schema).valid)
        self.assertFalse(validate("c", schema).valid)


class TestRegistry(unittest.TestCase):
    def test_register_and_get(self):
        registry = ToolRegistry()
        register_builtins(registry)
        self.assertIn("calculator", registry)
        self.assertEqual(len(registry.list()), 4)

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        register_builtins(registry)
        with self.assertRaises(ToolRegistryError):
            register_builtins(registry)

    def test_unregister(self):
        registry = ToolRegistry()
        register_builtins(registry)
        registry.unregister("calculator")
        self.assertNotIn("calculator", registry)


class TestBuiltins(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculator("847 * 39")["result"], 33033)
        self.assertEqual(calculator("(2 + 3) * 4")["result"], 20)

    def test_calculator_rejects_non_arithmetic(self):
        from tinymind.tools.builtins import CalculatorError
        with self.assertRaises(CalculatorError):
            calculator("__import__('os')")

    def test_calculator_bounds_exponent(self):
        from tinymind.tools.builtins import CalculatorError
        with self.assertRaises(CalculatorError):
            calculator("2 ** 999999")

    def test_unit_convert_length(self):
        result = unit_convert(100, "km", "mi")
        self.assertAlmostEqual(result["result"], 62.137, places=2)

    def test_unit_convert_temperature(self):
        result = unit_convert(0, "c", "f")
        self.assertAlmostEqual(result["result"], 32.0, places=5)

    def test_unit_convert_rejects_mismatched_dimension(self):
        from tinymind.tools.builtins import UnitConversionError
        with self.assertRaises(UnitConversionError):
            unit_convert(1, "km", "kg")

    def test_add_days(self):
        self.assertEqual(add_days("2026-01-01", 31)["result"], "2026-02-01")

    def test_days_between(self):
        self.assertEqual(days_between("2026-01-01", "2026-01-31")["days"], 30)


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        register_builtins(self.registry)
        self.executor = ToolExecutor(self.registry)

    def test_execute_success(self):
        result = self.executor.execute(ToolCall(name="calculator", arguments={"expression": "2+2"}))
        self.assertTrue(result.ok)
        self.assertEqual(result.value["result"], 4)

    def test_execute_unknown_tool(self):
        result = self.executor.execute(ToolCall(name="nope", arguments={}))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "unknown_tool")

    def test_execute_invalid_arguments(self):
        result = self.executor.execute(ToolCall(name="add_days", arguments={"date": "2026-01-01"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "validation_failed")

    def test_destructive_tool_requires_confirmation(self):
        def delete_thing(path: str) -> dict:
            """Delete something.

            Args:
                path: what to delete.
            """
            return {"deleted": path}
        self.registry.register(delete_thing, permissions=ToolPermissions(capabilities=Capability.DESTRUCTIVE))
        unconfirmed = self.executor.execute(ToolCall(name="delete_thing", arguments={"path": "/x"}))
        self.assertFalse(unconfirmed.ok)
        self.assertEqual(unconfirmed.error_code, "confirmation_required")
        confirmed = self.executor.execute(ToolCall(name="delete_thing", arguments={"path": "/x"}), confirmed=True)
        self.assertTrue(confirmed.ok)

    def test_executor_never_uses_eval_on_bad_tool_output(self):
        # A tool that raises is caught, not propagated as a crash.
        def broken() -> dict:
            """A tool that always raises."""
            raise RuntimeError("boom")
        self.registry.register(broken)
        result = self.executor.execute(ToolCall(name="broken", arguments={}))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "execution_error")
        self.assertIn("boom", result.error)


class TestRetrieval(unittest.TestCase):
    def test_ranks_relevant_tool_first(self):
        registry = ToolRegistry()
        register_builtins(registry)
        retriever = LexicalRetriever()
        top = retriever.retrieve("convert miles to kilometers", registry.list(), k=1)
        self.assertEqual(top[0].name, "unit_convert")

    def test_zero_overlap_scores_zero(self):
        registry = ToolRegistry()
        register_builtins(registry)
        scored = LexicalRetriever().retrieve_scored("xyzzy plugh", registry.list(), k=10)
        self.assertTrue(all(score == 0.0 for _tool, score in scored))

    def test_always_ranks_even_below_k(self):
        # Regression test: retrieve() must rank the whole catalogue even
        # when len(tools) <= k, not just return registration order.
        registry = ToolRegistry()
        register_builtins(registry)
        top = LexicalRetriever().retrieve("convert 10 miles to km", registry.list(), k=10)
        self.assertEqual(top[0].name, "unit_convert")


class TestPlanner(unittest.TestCase):
    def test_multi_step_plan_with_ref(self):
        registry = ToolRegistry()

        def lookup_contact(name: str) -> dict:
            """Find a contact.

            Args:
                name: contact name.
            """
            return {"contact_id": "c_42"}

        def send_message(contact_id: str, text: str) -> dict:
            """Send a message.

            Args:
                contact_id: recipient id.
                text: message body.
            """
            return {"sent_to": contact_id, "text": text}

        registry.register(lookup_contact)
        registry.register(send_message)
        executor = ToolExecutor(registry)

        plan = Plan(steps=[
            PlanStep(step_id="lookup", tool_name="lookup_contact", arguments={"name": "Maya"}),
            PlanStep(step_id="send", tool_name="send_message",
                    arguments={"contact_id": Ref("lookup", "contact_id"), "text": "hi"}),
        ])
        result = PlanExecutor(executor).run(plan)
        self.assertTrue(result.ok)
        self.assertEqual(result.step_results["send"].value["sent_to"], "c_42")

    def test_plan_stops_on_error_by_default(self):
        registry = ToolRegistry()
        register_builtins(registry)
        executor = ToolExecutor(registry)
        plan = Plan(steps=[
            PlanStep(step_id="a", tool_name="calculator", arguments={"expression": "not math"}),
            PlanStep(step_id="b", tool_name="calculator", arguments={"expression": "1+1"}),
        ])
        result = PlanExecutor(executor).run(plan)
        self.assertFalse(result.ok)
        self.assertEqual(len(result.ordered_results), 1)  # step b never ran

    def test_single_step_planner(self):
        plan = SingleStepPlanner().plan_call("calculator", {"expression": "1+1"})
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].tool_name, "calculator")


if __name__ == "__main__":
    unittest.main()
