import unittest

from tinymind.evaluation import AcceptanceSuite, Case, Category, PredictedCall


class TestAcceptanceSuiteFramework(unittest.TestCase):
    def test_positive_case_pass(self):
        suite = AcceptanceSuite("t", [
            Case("c1", "turn it on", Category.POSITIVE, expect_call=True,
                expected_tool="toggle", expected_arguments={"on": True}),
        ])
        result = suite.run(lambda text: [PredictedCall("toggle", {"on": True})])
        self.assertTrue(result.passed)

    def test_positive_case_fail_wrong_tool(self):
        suite = AcceptanceSuite("t", [
            Case("c1", "turn it on", Category.POSITIVE, expect_call=True, expected_tool="toggle"),
        ])
        result = suite.run(lambda text: [PredictedCall("wrong_tool", {})])
        self.assertFalse(result.passed)

    def test_missing_case_expects_no_call(self):
        suite = AcceptanceSuite("t", [
            Case("c1", "do something vague", Category.MISSING, expect_call=False),
        ])
        passing = suite.run(lambda text: [])
        self.assertTrue(passing.passed)
        failing = suite.run(lambda text: [PredictedCall("guessed", {})])
        self.assertFalse(failing.passed)

    def test_critical_categories_flagged(self):
        cases = [
            Case("c1", "x", Category.MISSING, expect_call=False),
            Case("c2", "y", Category.POSITIVE, expect_call=True, expected_tool="t"),
        ]
        self.assertTrue(cases[0].critical)
        self.assertFalse(cases[1].critical)

    def test_critical_failures_surfaced(self):
        suite = AcceptanceSuite("t", [
            Case("c1", "x", Category.NEGATION, expect_call=False),
        ])
        result = suite.run(lambda text: [PredictedCall("shouldnt_be_called", {})])
        self.assertEqual(len(result.critical_failures), 1)

    def test_parallel_case_requires_call_count(self):
        suite = AcceptanceSuite("t", [
            Case("c1", "do a and b", Category.PARALLEL, expect_call=True, expected_call_count=2),
        ])
        result = suite.run(lambda text: [PredictedCall("a", {})])  # only one call, expected two
        self.assertFalse(result.passed)

    def test_by_category_breakdown(self):
        suite = AcceptanceSuite("t", [
            Case("c1", "x", Category.POSITIVE, expect_call=True, expected_tool="t"),
            Case("c2", "y", Category.IRRELEVANT, expect_call=False),
        ])
        result = suite.run(lambda text: [] if "y" in text else [PredictedCall("t", {})])
        breakdown = result.by_category()
        self.assertEqual(breakdown["positive"], (1, 1))
        self.assertEqual(breakdown["irrelevant"], (1, 1))


class TestDeskSuiteExample(unittest.TestCase):
    """The example suite (benchmarks/tools/desk_suite.py) demonstrates the
    framework end to end against a rule-based demo predictor — see that
    module's docstring on why the predictor itself is not a claim about a
    trained model's quality."""

    def test_desk_suite_passes_against_its_own_demo_predictor(self):
        from benchmarks.tools.desk_suite import run
        result = run()
        self.assertTrue(result.passed, result.summary())
        self.assertEqual(len(result.critical_failures), 0)

    def test_desk_suite_covers_every_category(self):
        from benchmarks.tools.desk_suite import _CASES
        categories = {case.category for case in _CASES}
        self.assertEqual(categories, set(Category))


if __name__ == "__main__":
    unittest.main()
