import unittest

from tinymind.confidence import ConfidenceEstimator, ConfidencePolicy
from tinymind.model.backends.echo import EchoBackend
from tinymind.routing.router import ComputeLevel, ResponseMode, Router
from tinymind.runtime.engine import Engine, EngineError
from tinymind.runtime.grounding import GroundingMode, check_grounding
from tinymind.runtime.verification.verifier import GroundingVerifier, MathVerifier
from tinymind.tools.registry import ToolRegistry
from tinymind.tools.builtins import register_builtins
from tinymind.tools.validation import validate


class TestGrounding(unittest.TestCase):
    def test_grounded_fields(self):
        result = check_grounding({"room": "bedroom", "brightness": 30},
                                 "dim the bedroom lights to 30 percent")
        self.assertTrue(result.all_grounded)

    def test_invented_value_not_grounded(self):
        result = check_grounding({"room": "bedroom", "brightness": 50},
                                 "set a reasonable brightness in the bedroom",
                                 mode=GroundingMode.STRICT)
        self.assertIn("brightness", result.ungrounded_fields)
        self.assertTrue(result.blocks_execution())

    def test_date_normalization(self):
        result = check_grounding({"date": "2026-09-11"}, "remind me on September 11, 2026")
        self.assertTrue(result.fields[0].grounded)
        self.assertEqual(result.fields[0].transformation, "date_normalized")

    def test_permissive_mode_never_blocks(self):
        result = check_grounding({"x": 999}, "nothing relevant here", mode=GroundingMode.PERMISSIVE)
        self.assertFalse(result.blocks_execution())


class TestVerifiers(unittest.TestCase):
    def test_math_verifier_passes(self):
        outcome = MathVerifier().verify("17 * 8", 136, {})
        self.assertTrue(outcome.passed)

    def test_math_verifier_fails(self):
        outcome = MathVerifier().verify("17 * 8", 999, {})
        self.assertFalse(outcome.passed)

    def test_grounding_verifier(self):
        outcome = GroundingVerifier().verify(
            "dim bedroom to 30", {"room": "bedroom", "brightness": 30},
            {"source_text": "dim bedroom to 30"})
        self.assertTrue(outcome.passed)


class TestConfidence(unittest.TestCase):
    def test_min_of_components(self):
        est = ConfidenceEstimator()
        validation = validate({"a": 1}, {"type": "object", "properties": {"a": {"type": "integer"}}})
        result = est.estimate(tool_ranked_scores=[10.0, 1.0], validation=validation)
        self.assertAlmostEqual(result.confidence, min(result.components.tool_selection,
                                                       result.components.arguments))
        self.assertNotIn("model", result.valid_for)

    def test_no_signal_gives_zero_confidence(self):
        est = ConfidenceEstimator()
        result = est.estimate()
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.valid_for, [])

    def test_policy_thresholds(self):
        policy = ConfidencePolicy(execute_threshold=0.9, verify_threshold=0.7)
        self.assertEqual(policy.decide(0.95), "execute")
        self.assertEqual(policy.decide(0.8), "verify")
        self.assertEqual(policy.decide(0.5), "clarify")


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        register_builtins(self.registry)
        self.router = Router(self.registry)

    def test_greeting_is_chat(self):
        decision = self.router.route("hello")
        self.assertEqual(decision.mode, ResponseMode.CHAT)
        self.assertEqual(decision.compute_level, ComputeLevel.FAST)

    def test_general_knowledge_is_chat_not_forced_tool_call(self):
        decision = self.router.route("What is the capital of France?")
        self.assertEqual(decision.mode, ResponseMode.CHAT)

    def test_arithmetic_routes_to_calculator(self):
        decision = self.router.route("What is 2 + 2?")
        self.assertEqual(decision.mode, ResponseMode.TOOL_CALL)
        self.assertEqual(decision.candidate_tools[0], "calculator")

    def test_unit_conversion_routes_to_unit_convert(self):
        decision = self.router.route("Convert 10 miles to km")
        self.assertEqual(decision.candidate_tools[0], "unit_convert")

    def test_date_with_hyphens_not_misread_as_subtraction(self):
        decision = self.router.route("Add 5 days to 2026-01-01")
        self.assertEqual(decision.candidate_tools[0], "add_days")

    def test_multi_step_language_routes_to_plan(self):
        decision = self.router.route("Plan a 5-day trip and then book the hotel")
        self.assertEqual(decision.mode, ResponseMode.PLAN)

    def test_output_schema_forces_structured_output(self):
        decision = self.router.route("give me the data", output_schema={"type": "object"})
        self.assertEqual(decision.mode, ResponseMode.STRUCTURED_OUTPUT)

    def test_empty_input_asks_clarification(self):
        decision = self.router.route("   ")
        self.assertEqual(decision.mode, ResponseMode.ASK_CLARIFICATION)


class TestSessionAndEngine(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        register_builtins(self.registry)
        self.backend = EchoBackend()
        self.backend.load("dummy")
        self.engine = Engine(self.backend, self.registry)

    def test_chat_mode_roundtrip(self):
        session = self.engine.create_session("s1")
        result = session.run("What is the capital of France?")
        self.assertEqual(result.mode, ResponseMode.CHAT)
        self.assertTrue(result.ok)
        self.assertIn("capital of France", result.text)

    def test_tool_call_without_real_model_reports_honestly(self):
        session = self.engine.create_session("s1")
        result = session.run("Calculate 847 times 39")
        self.assertEqual(result.mode, ResponseMode.TOOL_CALL)
        self.assertFalse(result.ok)
        self.assertIn("EchoBackend", result.reason)

    def test_execute_tool_call_direct_pipeline_works(self):
        session = self.engine.create_session("s1")
        result = session.execute_tool_call("calculator", {"expression": "2+2"},
                                           tool_ranked_scores=[10.0, 0.1])
        self.assertTrue(result.ok)
        self.assertEqual(result.tool_result.value["result"], 4)
        self.assertGreaterEqual(result.confidence.confidence, 0.9)

    def test_execute_tool_call_blocks_ungrounded_argument(self):
        session = self.engine.create_session("s1")
        result = session.execute_tool_call(
            "add_days", {"date": "2026-01-01", "days": 999},
            source_text="add some days to Jan 1 2026", tool_ranked_scores=[9.0, 0.5])
        self.assertFalse(result.ok)
        self.assertIn("days", result.grounding.ungrounded_fields)

    def test_engine_rejects_duplicate_session_id(self):
        self.engine.create_session("dup")
        with self.assertRaises(EngineError):
            self.engine.create_session("dup")

    def test_engine_sessions_are_independent(self):
        s1 = self.engine.create_session("s1")
        s2 = self.engine.create_session("s2")
        s1.chat("hello from s1")
        self.assertEqual(len(s1.history.all()), 2)  # user + assistant
        self.assertEqual(len(s2.history.all()), 0)


if __name__ == "__main__":
    unittest.main()
