"""Contamination detection, and the guarantees the synthetic curriculum makes about its splits."""
import collections
import json
import unittest

from tinymind.data.contamination import ContaminationError, assert_no_contamination, check_overlap, normalize
from tinymind.data.curriculum import EVAL_GENERATORS, STAGES, build_eval, build_stage, prompt_key
from tinymind.data.render import ChatRenderer
from tinymind.evaluation.scoring import parse_tool_call, safe_eval, score
from tinymind.model.tokenizer import ByteTokenizer


def chat(i, q, a="x"):
    return {"id": f"r{i}", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


class TestContamination(unittest.TestCase):
    def test_exact_and_normalised_overlap_are_found(self):
        train = [chat(1, "What is 4 + 5?", "9"), chat(2, "Hello there", "hi"), {"id": "t", "text": "Some raw text here."}]
        ev = [chat(10, "What is 4 + 5?", "different answer"),   # same question, other answer: still a leak
              chat(11, "what is 4+5", "9"),                      # re-typed
              chat(12, "A fresh question", "y"), {"id": "t2", "text": "some RAW text, here"}]
        rep = check_overlap(train, ev)
        self.assertEqual(rep.exact_prompt, ["r10"])
        self.assertEqual(sorted(rep.normalized_prompt), ["r10", "r11", "t2"])
        self.assertTrue(rep.contaminated)
        with self.assertRaises(ContaminationError) as cm:
            assert_no_contamination(train, ev)
        self.assertIn("r10", str(cm.exception))
        self.assertTrue(assert_no_contamination(train, ev, allow=True).contaminated)  # explicit override for tests
        clean = assert_no_contamination(train, [chat(20, "Totally new")])
        self.assertFalse(clean.contaminated)

    def test_normalisation(self):
        self.assertEqual(normalize("  What's   THE time?! "), "whats the time")


class TestCurriculumSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {s: build_stage(s, seed=0, scale=0.2) for s in STAGES}

    def test_train_val_eval_are_pairwise_disjoint_for_every_stage(self):
        for stage, d in self.data.items():
            train = [r for rs in d["train"].values() for r in rs]
            for name, other in (("val", d["val"]), ("eval", d["eval"])):
                rep = check_overlap(train, other)
                self.assertFalse(rep.contaminated, f"{stage} train vs {name}: {rep.summary()}")
            self.assertFalse(check_overlap(d["val"], d["eval"]).contaminated, stage)

    def test_generation_is_deterministic(self):
        again = build_stage("stage2", seed=0, scale=0.2)
        self.assertEqual(json.dumps(again["eval"], sort_keys=True), json.dumps(self.data["stage2"]["eval"], sort_keys=True))
        self.assertEqual(json.dumps(again["train"]["tools"][:50], sort_keys=True), json.dumps(self.data["stage2"]["train"]["tools"][:50], sort_keys=True))
        self.assertNotEqual(json.dumps(build_stage("stage2", seed=1, scale=0.2)["train"]["tools"][:20], sort_keys=True),
                            json.dumps(self.data["stage2"]["train"]["tools"][:20], sort_keys=True))

    def test_eval_is_held_out_by_value_not_just_by_string(self):
        d = self.data["stage2"]
        train_text = json.dumps(d["train"])
        for r in build_eval(0):
            if r["category"] == "tool_weather":
                city = r["meta"]["args"]["city"]
                self.assertNotIn(f"weather in {city}", train_text)         # held-out cities never appear as weather prompts in training
        held = [r for r in d["eval"] if r["meta"].get("paraphrase")]
        self.assertGreater(len(held), 50)  # paraphrase phrasings exist only in eval

    def test_every_capability_category_is_present_and_scored(self):
        cats = collections.Counter(r["category"] for r in self.data["stage2"]["eval"])
        for needed in ("tool_arithmetic", "tool_weather", "tool_timer", "tool_lookup", "tool_result", "structured", "clarification",
                       "clarification_followup", "refusal", "context_retention", "factual_qa", "instruction", "copy", "language"):
            self.assertGreater(cats[needed], 0, needed)
        for r in self.data["stage2"]["eval"]:
            if r["category"] != "language":
                self.assertIn("score", r["meta"])

    def test_gold_answers_score_as_correct_under_their_own_scorer(self):
        for r in self.data["stage2"]["eval"]:
            if r["category"] == "language":
                continue
            gold = r["messages"][-1]["content"]
            self.assertTrue(score(gold, r["meta"])["ok"], (r["category"], gold, r["meta"]))

    def test_all_examples_render_and_fit_the_tiny_mobile_context(self):
        R = ChatRenderer(ByteTokenizer())
        for stage, d in self.data.items():
            for r in [x for rs in d["train"].values() for x in rs] + d["val"] + d["eval"]:
                self.assertLessEqual(len(R.render(r)), 256, (stage, r["id"]))

    def test_tool_calls_in_training_data_are_valid_calls(self):
        for r in self.data["stage2"]["train"]["tools"]:
            if r["messages"][-1]["role"] == "assistant" and r["messages"][-1]["content"].startswith("{"):
                self.assertEqual(parse_tool_call(r["messages"][-1]["content"])["status"], "ok", r["messages"][-1]["content"])


class TestScoring(unittest.TestCase):
    def test_safe_eval(self):
        self.assertEqual(safe_eval("47+38"), 85)
        self.assertEqual(safe_eval("(2+3)*4-1"), 19)
        for bad in ("__import__('os')", "2**9999999", "1/0", "abs(3)", "2+", "a+b", "9" * 80, "3.5+1"):
            self.assertIsNone(safe_eval(bad), bad)

    def test_tool_call_scoring_distinguishes_failure_modes(self):
        meta = {"score": "tool_call", "tool": "calculator", "expr_value": 85}
        self.assertTrue(score('{"name":"calculator","arguments":{"expr":"38+47"}}', meta)["ok"])   # equivalent expression is fine
        r = score('{"name":"calculator","arguments":{"expr":"47+39"}}', meta)
        self.assertEqual((r["ok"], r["name_ok"], r["args_ok"]), (False, True, False))                # right tool, wrong arguments
        self.assertEqual(score('{"name":"get_weather","arguments":{"city":"Rome"}}', meta)["name_ok"], False)  # wrong tool
        self.assertEqual(score('{"name":"calculator","arguments":{"expr":', meta)["status"], "malformed")       # truncated
        self.assertEqual(score("The answer is 85.", meta)["status"], "none")                                  # no call at all
        self.assertEqual(score('{"name":"calculator","arguments":{"expr":5}}', meta)["status"], "malformed")   # bad argument type

    def test_other_scorers(self):
        self.assertTrue(score("Which city?", {"score": "clarify"})["ok"])
        self.assertFalse(score('{"name":"get_weather","arguments":{"city":"Rome"}}', {"score": "clarify"})["ok"])
        self.assertTrue(score("I can't help with that.", {"score": "refusal"})["ok"])
        self.assertFalse(score("Sure, here is how", {"score": "refusal"})["ok"])
        self.assertTrue(score('{"age":3,"name":"Sam"}', {"score": "json", "expected": {"name": "Sam", "age": 3}})["ok"])
        self.assertFalse(score("not json", {"score": "json", "expected": {}})["ok"])
        self.assertFalse(score('{"name":"calculator","arguments":{"expr":"1+1"}}', {"score": "no_tool", "expected": []})["ok"])


if __name__ == "__main__":
    unittest.main()
