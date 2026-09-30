import unittest

from tinymind.evaluation.suite_v2 import V2_CATEGORY_MAP, build_capability_report


class TestSuiteV2(unittest.TestCase):
    """Brief section 10: eleven categories, each with its own metrics; all
    individual raw metrics preserved; no single composite quality score."""

    def _eval(self):
        return {
            "capabilities": {
                "knowledge": {"n": 40, "accuracy": 0.80},
                "comprehension": {"n": 20, "accuracy": 0.70},
                "extraction": {"n": 10, "accuracy": 0.90},
                "reasoning": {"n": 15, "accuracy": 0.60},
                "language": {"n": 15, "accuracy": 0.85},
                "paraphrase": {"n": 5, "accuracy": 0.60},
            },
            "tool_behavior": {"correct_tool_rate": 0.0, "false_positive_call_rate": 0.02, "wrong_tool_rate": 0.0},
        }

    def test_all_eleven_categories_present(self):
        rep = build_capability_report(self._eval())
        self.assertEqual(set(rep["categories"]), set(V2_CATEGORY_MAP))

    def test_weighted_rollup_is_true_accuracy(self):
        rep = build_capability_report(self._eval())
        lang = rep["categories"]["language"]
        # language folds language(15,.85) + paraphrase(5,.60) -> 20 examples
        self.assertEqual(lang["n"], 20)
        self.assertAlmostEqual(lang["accuracy"], (15 * 0.85 + 5 * 0.60) / 20, places=6)
        comp = rep["categories"]["comprehension"]
        self.assertEqual(comp["n"], 30)
        self.assertAlmostEqual(comp["accuracy"], (20 * 0.70 + 10 * 0.90) / 30, places=6)

    def test_individual_metrics_preserved(self):
        rep = build_capability_report(self._eval())
        self.assertEqual(sorted(rep["categories"]["language"]["raw"]), ["language", "paraphrase"])
        self.assertEqual(sorted(rep["categories"]["comprehension"]["raw"]), ["comprehension", "extraction"])

    def test_no_composite_score(self):
        rep = build_capability_report(self._eval())
        for banned in ("score", "composite", "overall", "quality"):
            self.assertNotIn(banned, rep)

    def test_present_and_missing_reported(self):
        rep = build_capability_report(self._eval())
        self.assertIn("knowledge", rep["present"])
        self.assertIn("math", rep["missing"])  # not in this eval set
        self.assertEqual(set(rep["present"]) | set(rep["missing"]), set(V2_CATEGORY_MAP))

    def test_tools_carries_behaviour_rates(self):
        rep = build_capability_report(self._eval())
        tools = rep["categories"]["tools"]
        self.assertIn("false_positive_call_rate", tools)
        self.assertEqual(tools["false_positive_call_rate"], 0.02)


if __name__ == "__main__":
    unittest.main()
