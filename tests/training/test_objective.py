"""Stage objective (tinymind.training.objective) as pure functions: the
measurements, the all-must-pass verdict with a minimum budget, regression
detection (same stage, across stages, against the promotion reference), report
rendering with verbatim raw generations, and the repository's objective configs."""
import json
import unittest
from pathlib import Path

from tinymind.training.objective import (DEFAULT_OBJECTIVE_DIR, StageObjective, StageObjectiveError, compare_reports,
                                         generation_metrics, grammar_metrics, render_history, render_markdown)

REPO = Path(__file__).resolve().parents[2]


def row(text, finish="max_new_tokens", n=20):
    return {"prompt": "p", "generation": text, "tokens_generated": n, "finish_reason": finish}


def report(stage="stage1", step=100, **metrics):
    base = {"loss": {"val_loss": 2.0}, "generation": {"mean_repetition": 0.1, "mean_distinct_2": 0.8},
            "grammar": {"word_like_fraction": 0.9, "known_word_fraction": 0.9}}
    for key, value in metrics.items():
        section, name = key.split("__")
        base.setdefault(section, {})[name] = value
    return {"stage": stage, "step": step, "metrics": base}


REGRESSION = {"max_val_loss_increase": 0.2, "checks": [{"metric": "generation.mean_repetition", "max_increase": 0.1},
                                                       {"metric": "grammar.word_like_fraction", "max_decrease": 0.1}]}


class TestMeasurements(unittest.TestCase):
    def test_generation_metrics_separate_language_from_loops_and_noise(self):
        good = generation_metrics([row(" hills. The birds sang in the tall trees, and a farmer walked home."),
                                   row(" quietly. She smiled at the dog.", "eos")])
        loop = generation_metrics([row(" the the the the the the the the the the the the the the")])
        self.assertEqual(good["non_empty_rate"], 1.0)
        self.assertLess(good["mean_repetition"], 0.1)
        self.assertGreater(loop["mean_repetition"], 0.5)
        self.assertEqual(loop["looping_rate"], 1.0)
        self.assertLess(loop["mean_distinct_2"], good["mean_distinct_2"])
        self.assertEqual(good["eos_rate"], 0.5)
        empty = generation_metrics([row("   ")])
        self.assertEqual(empty["non_empty_rate"], 0.0)
        self.assertIsNone(empty["mean_repetition"])  # an empty output is not "non-repetitive"

    def test_grammar_metrics(self):
        g = grammar_metrics([row(" hills. The birds sang, and a farmer walked home.")], lexicon={"hills", "the", "birds"})
        self.assertEqual(g["sentence_start_capitalized"], 1.0)
        self.assertEqual(g["space_after_punctuation"], 1.0)
        self.assertEqual(g["word_like_fraction"], 1.0)
        self.assertAlmostEqual(g["known_word_fraction"], 3 / 9)
        bad = grammar_metrics([row("ly. sl t,x .q oollaye")], lexicon={"the"})
        self.assertEqual(bad["sentence_start_capitalized"], 0.0)
        self.assertLess(bad["space_after_punctuation"], 1.0)
        self.assertEqual(bad["known_word_fraction"], 0.0)  # invented words are not known words
        self.assertIsNone(grammar_metrics([row("no punctuation at all here")])["space_after_punctuation"])
        self.assertIsNone(grammar_metrics([row("x")])["known_word_fraction"])  # no lexicon: not measured
        self.assertIsNone(grammar_metrics([row("3.5 and 1,000")])["space_after_punctuation"])  # numbers are exempt


class TestVerdict(unittest.TestCase):
    def objective(self, **extra):
        cfg = {"stage": "s1", "generation": {"prompts": ["a"]},
               "measurements": [{"name": "loss", "metric": "loss.val_loss", "max": 3.0},
                                {"name": "loops", "metric": "generation.mean_repetition", "max": 0.3}]}
        cfg.update(extra)
        return StageObjective(cfg)

    def decide(self, obj, loss, rep, tokens=100, min_tokens=100, **kw):
        # evaluate_checkpoint's decision logic without a model: generations are injected
        import tinymind.training.objective as O
        saved = O.generate_continuations
        O.generate_continuations = lambda *a, **k: [row(" the the the the the the the the the the the the")] if rep else \
            [row(" A clear and varied sentence about ships, harbours and grain.")]
        try:
            return obj.evaluate_checkpoint(model=None, tokenizer=None, held_out={"val_loss": loss}, step=1, tokens=tokens,
                                           min_tokens=min_tokens, **kw)
        finally:
            O.generate_continuations = saved

    def test_every_measurement_must_pass_no_loss_only_promotion(self):
        obj = self.objective()
        self.assertTrue(self.decide(obj, loss=1.0, rep=False)["objective_met"])
        r = self.decide(obj, loss=0.01, rep=True)  # an excellent loss cannot buy promotion past a failing check
        self.assertFalse(r["objective_met"])
        self.assertTrue(r["verdict"]["reasons"])
        self.assertFalse(self.decide(obj, loss=9.0, rep=False)["objective_met"])

    def test_token_budget_is_a_minimum_not_a_criterion(self):
        obj = self.objective()
        r = self.decide(obj, loss=1.0, rep=False, tokens=99, min_tokens=100)
        self.assertFalse(r["objective_met"])
        self.assertFalse(r["verdict"]["budget_reached"])
        self.assertTrue(r["verdict"]["measurements_pass"])

    def test_baseline_never_promotes_and_regression_blocks(self):
        obj = self.objective(regression={"max_val_loss_increase": 0.1})
        self.assertFalse(self.decide(obj, loss=1.0, rep=False, baseline=True)["objective_met"])
        prev = report("s1", 10, loss__val_loss=0.5)
        r = self.decide(obj, loss=1.0, rep=False, previous=prev)  # passes every floor but got worse
        self.assertTrue(r["verdict"]["measurements_pass"])
        self.assertTrue(r["verdict"]["regressed"])
        self.assertFalse(r["objective_met"])

    def test_config_validation(self):
        for cfg in ({"stage": "s", "measurements": []},
                    {"stage": "s", "measurements": [{"metric": "made_up.x", "max": 1}]},
                    {"stage": "s", "measurements": [{"metric": "loss.val_loss"}]},
                    {"stage": "s", "measurements": [{"metric": "generation.mean_repetition", "max": 1}]},
                    {"stage": "s", "measurements": [{"metric": "loss.val_loss", "max": "1"}]},
                    {"stage": "s", "measurements": [{"metric": "loss.val_loss", "max": 1}], "retain": ["s"]}):
            with self.assertRaises(StageObjectiveError, msg=str(cfg)):
                StageObjective(cfg)

    def test_data_requirements_are_decidable_before_training(self):
        obj = StageObjective({"stage": "s", "measurements": [{"metric": "data.natural_train_bytes", "min": 100},
                                                              {"metric": "loss.val_loss", "max": 1}]})
        self.assertEqual(len(obj.data_failures({"natural_train_bytes": 5})), 1)
        self.assertEqual(obj.data_failures({"natural_train_bytes": 500}), [])
        self.assertEqual(len(obj.data_failures({})), 1)  # an unmeasured requirement is not met


class TestRegressionDetection(unittest.TestCase):
    def test_same_stage_degradation_is_detected(self):
        prev = report(step=100)
        worse = report(step=200, loss__val_loss=2.5, generation__mean_repetition=0.3)
        res = compare_reports(worse, prev, REGRESSION)
        self.assertTrue(res["regressed"])
        failed = {c["metric"] for c in res["checks"] if not c["ok"]}
        self.assertEqual(failed, {"loss.val_loss", "generation.mean_repetition"})
        better = report(step=200, loss__val_loss=1.5, generation__mean_repetition=0.05)
        self.assertFalse(compare_reports(better, prev, REGRESSION)["regressed"])
        small = report(step=200, loss__val_loss=2.15, grammar__word_like_fraction=0.85)  # within tolerance
        self.assertFalse(compare_reports(small, prev, REGRESSION)["regressed"])
        self.assertFalse(compare_reports(worse, None, REGRESSION)["regressed"])  # nothing to regress from

    def test_later_stage_cannot_hide_a_lost_capability(self):
        stage1_final = report("stage1", 3600)  # stage1's own metrics when it was promoted
        retained_ok = {"stage1": {"metrics": {"generation": {"mean_repetition": 0.12, "mean_distinct_2": 0.8},
                                              "grammar": {"word_like_fraction": 0.88}}, "regression": REGRESSION}}
        retained_lost = {"stage1": {"metrics": {"generation": {"mean_repetition": 0.45, "mean_distinct_2": 0.3},
                                                "grammar": {"word_like_fraction": 0.5}}, "regression": REGRESSION}}
        cur_ok = {**report("stage2", 200, loss__val_loss=9.0), "retained": retained_ok}
        cur_lost = {**report("stage2", 200), "retained": retained_lost}
        # stage2's own loss is on other data: not compared with stage1's, so 9.0 is no "regression" here
        self.assertFalse(compare_reports(cur_ok, stage1_final, {"max_val_loss_increase": 0.1})["regressed"])
        res = compare_reports(cur_lost, stage1_final, {})
        self.assertTrue(res["regressed"])
        self.assertTrue(all(c["scope"].startswith("retained stage1") for c in res["checks"]))
        # slow drift: fine against the previous checkpoint, but not against stage1 at promotion
        prev = {**report("stage2", 100), "retained": {"stage1": {"metrics": retained_lost["stage1"]["metrics"]}}}
        drift = compare_reports(cur_lost, prev, {}, reference={"stage1": stage1_final["metrics"]})
        self.assertTrue(drift["regressed"])
        self.assertTrue(any("at promotion" in c["scope"] and not c["ok"] for c in drift["checks"]))

    def test_natural_text_compared_across_stages_on_the_same_split(self):
        reg = {"checks": [{"metric": "loss.text_val_bpb", "max_increase": 0.1}]}
        a = report("stage1", 10, loss__text_val_bpb=1.2, data__natural_val_sha256="abc")
        b = report("stage2", 10, loss__text_val_bpb=1.6, data__natural_val_sha256="abc")
        c = report("stage2", 10, loss__text_val_bpb=1.6, data__natural_val_sha256="other")
        self.assertTrue(compare_reports(b, a, reg)["regressed"])
        self.assertFalse(compare_reports(c, a, reg)["regressed"])  # different held-out split: not comparable


class TestRendering(unittest.TestCase):
    def full_report(self, generations):
        return {"stage": "stage1", "step": 1200, "tokens": 19_660_800, "real_tokens": 19_000_000, "min_tokens": 60_000_000,
                "baseline": False, "objective_met": False,
                "metrics": {"loss": {"val_loss": 1.9, "val_ppl": 6.7, "text_val_bpb": 1.8},
                            "generation": generation_metrics(generations), "grammar": grammar_metrics(generations),
                            "data": {"natural_train_bytes": 25_000_000}},
                "generations": generations, "retained": {},
                "regression": {"regressed": False, "checks": [], "notes": ["no previous checkpoint to compare against"]},
                "verdict": {"objective_met": False, "budget_reached": False, "measurements_pass": True,
                            "retained_pass": True, "regressed": False, "reasons": ["minimum training budget not reached"],
                            "checks": [{"name": "loss", "metric": "loss.val_loss", "value": 1.9,
                                        "requirement": "<= 3.0", "ok": True}]}}

    def test_markdown_shows_exact_prompts_and_raw_generations_verbatim(self):
        gens = [{"prompt": "The sun rose over the", "generation": " hills.\nThen ```code``` and `ticks`\x07!",
                 "tokens_generated": 30, "finish_reason": "max_new_tokens"},
                {"prompt": "Once upon a time, there was a", "generation": "", "tokens_generated": 1, "finish_reason": "eos"}]
        md = render_markdown(self.full_report(gens))
        self.assertIn("The sun rose over the", md)
        self.assertIn("Once upon a time, there was a", md)
        self.assertIn(" hills.\nThen ```code``` and `ticks`\\x07!", md)  # verbatim; control char made visible
        self.assertIn("````text", md)  # fence longer than the generation's own backtick runs
        self.assertIn("_(empty generation)_", md)
        for fragment in ("validation loss", "bits per byte", "repetition", "distinct-2", "word-like tokens",
                         "sentence starts capitalised", "minimum training budget not reached", "Regression checks",
                         "19,660,800"):
            self.assertIn(fragment, md)

    def test_markdown_template_does_not_editorialise(self):
        from tinymind.ci.trained_model_report import _BANNED_WORDS
        md = render_markdown(self.full_report([row("x")])).lower()
        for word in _BANNED_WORDS:
            self.assertNotIn(f" {word} ", md)

    def test_history_table(self):
        text = render_history([{"step": 200, "tokens": 10, "val_loss": 2.0, "objective_met": False, "regressed": True},
                               {"step": 400, "tokens": 20, "val_loss": 1.5, "objective_met": True, "regressed": False}])
        self.assertIn("| 400 | 20 | 1.5000 |", text)
        self.assertIn("MET", text)


class TestRepositoryObjectives(unittest.TestCase):
    """The shipped configs load, stage 1 is a real language gate, later stages retain it, and no objective can be met
    while the same stage's promotion gate would fail on the same validation-loss number."""

    def test_all_stages_load(self):
        for i in range(1, 8):
            obj = StageObjective.from_stage(f"stage{i}")
            self.assertIsNotNone(obj, f"stage{i}")
            if i > 1:
                self.assertEqual([r.stage for r in obj.retained], ["stage1"])

    def test_stage1_is_a_multi_signal_language_gate(self):
        obj = StageObjective.from_stage("stage1")
        sections = {m["metric"].split(".")[0] for m in obj.measurements}
        self.assertEqual(sections, {"loss", "generation", "grammar", "data"})
        self.assertGreaterEqual(len(obj.prompts), 6)
        metrics = {m["metric"] for m in obj.measurements}
        self.assertTrue({"loss.text_val_bpb", "grammar.known_word_fraction", "grammar.sentence_start_capitalized",
                         "generation.mean_repetition", "data.natural_train_bytes"} <= metrics)
        self.assertTrue(obj.regression.get("checks"))

    def test_objective_loss_ceiling_never_looser_than_the_promotion_gate(self):
        for gate_file in sorted(DEFAULT_OBJECTIVE_DIR.glob("stage*.gate.json")):
            stage = gate_file.name.split(".")[0]
            gate = json.loads(gate_file.read_text())
            gate_max = next(c["max"] for c in gate["checks"] if c["metric"] == "summary.final_validation.val_loss")
            obj = StageObjective.from_stage(stage)
            obj_max = next(m["max"] for m in obj.measurements if m["metric"] == "loss.val_loss")
            self.assertLessEqual(obj_max, gate_max, stage)


if __name__ == "__main__":
    unittest.main()
