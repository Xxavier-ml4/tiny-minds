import json
import unittest
from pathlib import Path

from tinymind.training.gate import evaluate_gate

GATES = Path(__file__).resolve().parents[2] / "configs" / "stages"


class TestStageGatesV2(unittest.TestCase):
    """Brief section 9: stages 1-6 have machine-checkable promotion gates that
    reference real metrics; a bad metric fails the gate. Stage 7 has none."""

    def test_gate_files_exist_for_stage1_through_6(self):
        for i in range(1, 7):
            self.assertTrue((GATES / f"stage{i}.gate.json").is_file(), f"missing stage{i}.gate.json")
        self.assertFalse((GATES / "stage7.gate.json").is_file(), "stage7 must have no gate")

    def test_gates_parse_and_reference_known_metric_heads(self):
        for i in range(1, 7):
            gate = json.loads((GATES / f"stage{i}.gate.json").read_text())
            self.assertIn("checks", gate)
            self.assertEqual(gate["expect_stage"], f"stage{i}")
            for check in gate["checks"]:
                head = check["metric"].split(".")[0]
                self.assertIn(head, ("summary", "eval"))
                self.assertTrue(("min" in check) or ("max" in check))

    def _summary(self, stage, val_loss, complete=True, diverged=False):
        return {"stage": stage, "stage_complete": complete,
                "stop_reason": "diverged" if diverged else "complete",
                "final_validation": {"val_loss": val_loss}}

    def test_stage3_gate_passes_and_fails_correctly(self):
        gate = json.loads((GATES / "stage3.gate.json").read_text())

        def ev(k, c, r, l):
            return {"capabilities": {"knowledge": {"n": 40, "accuracy": k},
                                     "comprehension": {"n": 20, "accuracy": c},
                                     "reasoning": {"n": 15, "accuracy": r},
                                     "language": {"n": 15, "accuracy": l}}}

        self.assertTrue(evaluate_gate(gate, self._summary("stage3", 0.9), ev(0.6, 0.5, 0.4, 0.7))["passed"])
        # regression below the reasoning floor
        self.assertFalse(evaluate_gate(gate, self._summary("stage3", 0.9), ev(0.6, 0.5, 0.02, 0.7))["passed"])
        # validation loss too high
        self.assertFalse(evaluate_gate(gate, self._summary("stage3", 5.0), ev(0.6, 0.5, 0.4, 0.7))["passed"])
        # stage not complete
        self.assertFalse(evaluate_gate(gate, self._summary("stage3", 0.9, complete=False), ev(0.6, 0.5, 0.4, 0.7))["passed"])
        # a required metric is missing entirely
        missing = {"capabilities": {"comprehension": {"n": 20, "accuracy": 0.5},
                                    "reasoning": {"n": 15, "accuracy": 0.4},
                                    "language": {"n": 15, "accuracy": 0.7}}}
        self.assertFalse(evaluate_gate(gate, self._summary("stage3", 0.9), missing)["passed"])

    def test_stage5_tool_behaviour_thresholds(self):
        gate = json.loads((GATES / "stage5.gate.json").read_text())

        def ev(correct, false_pos):
            return {"capabilities": {"reasoning": {"n": 10, "accuracy": 0.4},
                                     "knowledge": {"n": 10, "accuracy": 0.4}},
                    "tool_behavior": {"correct_tool_rate": correct, "false_positive_call_rate": false_pos}}

        self.assertTrue(evaluate_gate(gate, self._summary("stage5", 0.9), ev(0.5, 0.1))["passed"])
        self.assertFalse(evaluate_gate(gate, self._summary("stage5", 0.9), ev(0.5, 0.9))["passed"])  # too many false calls
        self.assertFalse(evaluate_gate(gate, self._summary("stage5", 0.9), ev(0.05, 0.1))["passed"])  # too little routing


if __name__ == "__main__":
    unittest.main()
