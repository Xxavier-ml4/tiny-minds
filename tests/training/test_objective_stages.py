"""Objective-driven stages in the training engine: a stage completes only when
its objective is met; a failing objective at the budget leaves it INCOMPLETE; the
same stage is continued with a larger budget (never resumed past its horizon);
every evaluation checkpoint writes a report with the raw generations; regression
against the previous checkpoint blocks completion; later stages re-check earlier
capabilities; and the same mechanism works unchanged across model sizes."""
import json
import unittest

from tinymind.training import checkpoint as ck
from tinymind.training.checkpoint import ResumeMismatchError
from tinymind.training.config import TrainingConfigError
from tinymind.training.data import TokenizedDataset
from tinymind.training.objective import StageObjective

from ._helpers import RENDERER, make_engine, tmpdir

PROMPTS = ["The cat", "repeat 3"]


def objective(stage="stage0", max_loss=1e9, *, extra=None, **config):
    measurements = [{"name": "loss", "metric": "loss.val_loss", "max": max_loss},
                    {"name": "any_output", "metric": "generation.n", "min": 0}]
    cfg = {"stage": stage, "generation": {"max_new_tokens": 6, "prompts": PROMPTS},
           "measurements": measurements + (extra or [])}
    cfg.update(config)
    return StageObjective(cfg)


HARD = objective(max_loss=-1.0)  # can never be met
EASY = objective()               # always met once the minimum budget is reached


def reports(out):
    return sorted(p.name for p in (out / "objective_reports").glob("step-*.json"))


class TestObjectiveCompletion(unittest.TestCase):
    def test_stage_is_incomplete_when_the_objective_fails_at_the_budget(self):
        out = tmpdir()
        s = make_engine(out, train_over=dict(max_steps=6, eval_interval=3), objective=HARD).train()
        self.assertEqual((s["stop_reason"], s["stage_complete"], s["final_step"]), ("gate_failed", False, 6))
        self.assertFalse(s["objective"]["objective_met"])
        latest = ck.find_latest_valid(out / "checkpoints")[0]
        self.assertFalse(json.loads((latest / "manifest.json").read_text())["stage_complete"])

    def test_stage_completes_when_the_objective_is_met(self):
        s = make_engine(tmpdir(), train_over=dict(max_steps=6, eval_interval=3), objective=EASY).train()
        self.assertEqual((s["stop_reason"], s["stage_complete"]), ("gate_passed", True))

    def test_budget_is_a_minimum_training_chunk_not_the_criterion(self):
        # default minimum = the stage budget: even an always-true objective waits for it...
        s = make_engine(tmpdir(), train_over=dict(max_steps=12, eval_interval=3), objective=EASY).train()
        self.assertEqual(s["final_step"], 12)
        # ...and once a (smaller) minimum is reached, the stage ends as soon as the objective is met
        s = make_engine(tmpdir(), train_over=dict(max_steps=12, eval_interval=3), objective=EASY,
                        objective_min_tokens=4 * 96 * 6).train()
        self.assertEqual((s["stop_reason"], s["final_step"], s["stage_complete"]), ("gate_passed", 6, True))

    def test_without_an_objective_the_legacy_budget_completion_is_unchanged(self):
        s = make_engine(tmpdir(), train_over=dict(max_steps=6, eval_interval=3)).train()
        self.assertEqual((s["stop_reason"], s["stage_complete"]), ("complete", True))
        self.assertFalse(s["objective"]["configured"])

    def test_a_time_stop_keeps_its_reason_and_the_stage_incomplete(self):
        s = make_engine(tmpdir(), train_over=dict(max_steps=12, eval_interval=3), objective=EASY,
                        stop_after_steps=4).train()
        self.assertEqual((s["stop_reason"], s["stage_complete"]), ("step_limit", False))

    def test_data_that_can_never_satisfy_the_objective_is_refused_up_front(self):
        needs_text = objective(extra=[{"metric": "data.natural_train_bytes", "min": 1000}])
        with self.assertRaises(TrainingConfigError):
            make_engine(tmpdir(), objective=needs_text, objective_data={"natural_train_bytes": 10})
        make_engine(tmpdir(), objective=needs_text, objective_data={"natural_train_bytes": 5000})


class TestReports(unittest.TestCase):
    def test_every_evaluation_checkpoint_reports_prompts_and_raw_generations(self):
        out = tmpdir()
        make_engine(out, train_over=dict(max_steps=6, eval_interval=2), objective=HARD).train()
        self.assertEqual(reports(out), ["step-00000000.json", "step-00000002.json", "step-00000004.json",
                                        "step-00000006.json"])
        for name in reports(out):
            rep = json.loads((out / "objective_reports" / name).read_text())
            md = (out / "objective_reports" / name.replace(".json", ".md")).read_text()
            self.assertEqual([g["prompt"] for g in rep["generations"]], PROMPTS)
            for g in rep["generations"]:
                self.assertIn(g["prompt"], md)
                if g["generation"]:
                    self.assertIn(g["generation"].replace("\x00", "\\x00"), md)  # the raw text itself, not a score
        history = (out / "objective_reports" / "history.jsonl").read_text().splitlines()
        self.assertEqual(len(history), 4)
        self.assertTrue(json.loads(history[0])["baseline"])

    def test_bits_per_byte_on_natural_held_out_text(self):
        texts = [{"id": f"t{i}", "text": "The cat sat on the mat."} for i in range(3)]
        tv = TokenizedDataset.from_records(texts, RENDERER, 96, name="val_text")
        e = make_engine(tmpdir(), train_over=dict(max_steps=4, eval_interval=2), objective=EASY, objective_validation=tv)
        ev = e.evaluate_text()
        self.assertEqual(ev["text_val_bytes"], 3 * len("The cat sat on the mat."))
        self.assertAlmostEqual(ev["text_val_bpb"], ev["text_val_loss"] * ev["text_val_tokens"] /
                               (ev["text_val_bytes"] * 0.6931471805599453), places=6)


class TestSameStageContinuation(unittest.TestCase):
    def setUp(self):
        self.first = tmpdir()
        make_engine(self.first, train_over=dict(max_steps=6, eval_interval=3), objective=HARD).train()
        self.ckpt = self.first / "checkpoints"

    def test_exhausted_budget_cannot_be_resumed_only_continued(self):
        with self.assertRaises(ResumeMismatchError) as cm:
            make_engine(tmpdir(), train_over=dict(max_steps=6, eval_interval=3), objective=HARD, resume=self.ckpt)
        self.assertIn("--continue-stage", str(cm.exception))

    def test_continuation_extends_the_budget_and_keeps_identity(self):
        out = tmpdir()
        watched = objective(max_loss=-1.0, regression={"max_val_loss_increase": 100.0})
        e = make_engine(out, train_over=dict(max_steps=12, eval_interval=3), objective=watched, continue_from=self.ckpt,
                        carry_optimizer=True)
        self.assertEqual((e.step, e.total_steps), (6, 12))
        s = e.train()
        self.assertEqual((s["final_step"], s["stop_reason"], s["stage_complete"]), (12, "gate_failed", False))
        self.assertTrue(s["parent"]["continuation"])
        self.assertEqual(s["parent"]["previous_total_steps"], 6)
        before = json.loads((ck.find_latest_valid(self.ckpt)[0] / "manifest.json").read_text())
        after = json.loads((ck.find_latest_valid(out / "checkpoints")[0] / "manifest.json").read_text())
        for key in ("model_config_hash", "tokenizer_hash", "dataset_hash"):
            self.assertEqual(before[key], after[key], key)
        # the first report of the continued run is compared with the last one of the previous run (another job)
        first = json.loads((out / "objective_reports" / "step-00000009.json").read_text())
        self.assertEqual([c["scope"] for c in first["regression"]["checks"]], ["vs previous checkpoint (step 6)"])
        self.assertEqual(reports(out), ["step-00000009.json", "step-00000012.json"])  # no baseline: not a fresh stage

    def test_resume_from_before_the_first_evaluation_decides_like_the_uninterrupted_run(self):
        # Audit finding: a checkpoint taken before the first evaluation stored the untrained step-0 baseline as the
        # regression reference, so a resumed run compared its first checkpoint with a random model.
        watched = objective(regression={"max_val_loss_increase": 0.0,
                                        "checks": [{"metric": "generation.mean_distinct_1", "max_decrease": 0.0}]})
        over = dict(max_steps=8, eval_interval=4, checkpoint_interval=2, keep_checkpoints=10)
        cont = tmpdir()
        make_engine(cont, train_over=over, objective=watched).train()
        stored = json.loads((cont / "checkpoints" / "step-00000002" / "state.json").read_text())
        self.assertIsNone(stored["metrics"]["objective_report"])  # no reference yet: the baseline is not one
        res = tmpdir()
        make_engine(res, train_over=over, objective=watched, resume=cont / "checkpoints" / "step-00000002").train()
        for step in ("00000004", "00000008"):
            a = json.loads((cont / "objective_reports" / f"step-{step}.json").read_text())
            b = json.loads((res / "objective_reports" / f"step-{step}.json").read_text())
            self.assertEqual(a["regression"], b["regression"], step)
            self.assertEqual((a["objective_met"], a["verdict"]["reasons"]), (b["objective_met"], b["verdict"]["reasons"]))
            self.assertEqual(a["generations"], b["generations"])
        self.assertEqual(json.loads((cont / "objective_reports" / "step-00000004.json").read_text())["regression"]["checks"], [])

    def test_continuation_can_finally_pass(self):
        s = make_engine(tmpdir(), train_over=dict(max_steps=12, eval_interval=3), objective=EASY,
                        continue_from=self.ckpt, objective_min_tokens=0).train()
        self.assertEqual((s["stop_reason"], s["stage_complete"]), ("gate_passed", True))

    def test_continuation_refusals(self):
        with self.assertRaises(ResumeMismatchError):  # not a larger budget
            make_engine(tmpdir(), train_over=dict(max_steps=6, eval_interval=3), objective=HARD, continue_from=self.ckpt)
        with self.assertRaises(ResumeMismatchError):  # another stage: that is --init-from
            make_engine(tmpdir(), train_over=dict(stage="stage1", max_steps=12, eval_interval=3),
                        objective=objective("stage1"), continue_from=self.ckpt)
        with self.assertRaises(ResumeMismatchError):  # a different trajectory (learning rate) is not a continuation
            make_engine(tmpdir(), train_over=dict(max_steps=12, eval_interval=3, learning_rate=1e-2), objective=HARD,
                        continue_from=self.ckpt)
        with self.assertRaises(TrainingConfigError):
            make_engine(tmpdir(), objective=HARD, continue_from=self.ckpt, resume=self.ckpt)

    def test_a_complete_stage_is_reopened_only_on_request(self):
        done = tmpdir()
        make_engine(done, train_over=dict(max_steps=6, eval_interval=3), objective=EASY).train()
        with self.assertRaises(ResumeMismatchError):
            make_engine(tmpdir(), train_over=dict(max_steps=12, eval_interval=3), objective=EASY,
                        continue_from=done / "checkpoints")
        s = make_engine(tmpdir(), train_over=dict(max_steps=12, eval_interval=3), objective=EASY,
                        continue_from=done / "checkpoints", reopen=True).train()
        # a reopened stage trains its whole extension before it may complete again
        self.assertEqual((s["final_step"], s["stage_complete"], s["parent"]["reopened"]), (12, True, True))


class TestRegressionBlocksCompletion(unittest.TestCase):
    def test_passing_floors_are_not_enough_when_the_model_got_worse(self):
        # loss must IMPROVE by at least 1.0 between checkpoints: impossible here, so every comparison regresses
        strict = objective(regression={"max_val_loss_increase": -1.0})
        s = make_engine(tmpdir(), train_over=dict(max_steps=6, eval_interval=3), objective=strict).train()
        self.assertEqual((s["stop_reason"], s["stage_complete"]), ("gate_failed", False))
        self.assertTrue(s["objective"]["verdict"]["measurements_pass"])
        self.assertTrue(s["objective"]["verdict"]["regressed"])

    def test_a_fresh_stage_is_not_compared_with_its_random_initialisation(self):
        strict = objective(regression={"max_val_loss_increase": -1.0})
        out = tmpdir()
        make_engine(out, train_over=dict(max_steps=6, eval_interval=3), objective=strict).train()
        first = json.loads((out / "objective_reports" / "step-00000003.json").read_text())
        self.assertEqual(first["regression"]["checks"], [])  # nothing before the first trained checkpoint


class TestNextStageNeedsACompleteParent(unittest.TestCase):
    def test_init_from_is_rejected_until_the_parent_objective_is_met(self):
        failed = tmpdir()
        make_engine(failed, train_over=dict(max_steps=6, eval_interval=3), objective=HARD).train()
        with self.assertRaises(ResumeMismatchError) as cm:
            make_engine(tmpdir(), train_over=dict(stage="stage1", max_steps=6), init_from=failed / "checkpoints")
        self.assertIn("not complete", str(cm.exception))
        # explicit engineering override
        make_engine(tmpdir(), train_over=dict(stage="stage1", max_steps=6), init_from=failed / "checkpoints",
                    allow_incomplete_parent=True)
        # after the SAME stage is continued until its objective passes, the next stage is accepted
        passed = tmpdir()
        make_engine(passed, train_over=dict(max_steps=12, eval_interval=3), objective=EASY,
                    continue_from=failed / "checkpoints", objective_min_tokens=0).train()
        e = make_engine(tmpdir(), train_over=dict(stage="stage1", max_steps=6), init_from=passed / "checkpoints")
        self.assertEqual(e.parent["stage"], "stage0")


class TestAcrossStages(unittest.TestCase):
    def test_later_stage_retains_earlier_capabilities(self):
        a_cfg = {"stage": "stageA", "generation": {"max_new_tokens": 6, "prompts": ["hi", "the cat"]},
                 "measurements": [{"metric": "loss.val_loss", "max": 1e9}, {"metric": "generation.n", "min": 0}],
                 "regression": {"checks": [{"metric": "generation.mean_distinct_1", "max_decrease": 0.5}]}}
        out_a = tmpdir()
        make_engine(out_a, train_over=dict(stage="stageA", max_steps=4, eval_interval=2),
                    objective=StageObjective(a_cfg)).train()
        b = StageObjective({"stage": "stageB", "retain": [a_cfg], "generation": {"max_new_tokens": 6, "prompts": ["yes"]},
                            "measurements": [{"metric": "loss.val_loss", "max": 1e9}]})
        out_b = tmpdir()
        e = make_engine(out_b, train_over=dict(stage="stageB", max_steps=6, eval_interval=2), objective=b,
                        init_from=out_a / "checkpoints")
        self.assertIn("stageA", e._capability_reference)  # stageA's metrics at promotion
        s = e.train()
        rep = json.loads((out_b / "objective_reports" / "step-00000002.json").read_text())
        self.assertEqual([g["prompt"] for g in rep["retained"]["stageA"]["generations"]], ["hi", "the cat"])
        self.assertTrue(any(c["scope"].startswith("retained stageA") for c in rep["regression"]["checks"]))
        state = json.loads((ck.find_latest_valid(out_b / "checkpoints")[0] / "state.json").read_text())
        self.assertIn("stageA", state["metrics"]["capability_reference"])  # carried into stage C via this checkpoint
        self.assertEqual(s["parent"]["stage"], "stageA")

    def test_failing_retained_capability_blocks_the_later_stage(self):
        a_cfg = {"stage": "stageA", "generation": {"max_new_tokens": 6, "prompts": ["hi"]},
                 "measurements": [{"metric": "generation.mean_distinct_1", "min": 2.0}]}  # can never hold
        b = StageObjective({"stage": "stageB", "retain": [a_cfg], "measurements": [{"metric": "loss.val_loss", "max": 1e9}]})
        s = make_engine(tmpdir(), train_over=dict(stage="stageB", max_steps=6, eval_interval=3), objective=b).train()
        self.assertEqual((s["stop_reason"], s["stage_complete"]), ("gate_failed", False))
        self.assertFalse(s["objective"]["verdict"]["retained_pass"])


class TestSizeIndependence(unittest.TestCase):
    def test_the_same_mechanism_at_two_model_sizes(self):
        for over in (dict(hidden_size=32, num_layers=2), dict(hidden_size=64, num_layers=3, intermediate_size=128)):
            with self.subTest(**over):
                first = tmpdir()
                s = make_engine(first, model_over=over, train_over=dict(max_steps=6, eval_interval=3),
                                objective=HARD).train()
                self.assertEqual((s["stop_reason"], s["stage_complete"]), ("gate_failed", False))
                s = make_engine(tmpdir(), model_over=over, train_over=dict(max_steps=12, eval_interval=3),
                                objective=EASY, continue_from=first / "checkpoints", objective_min_tokens=0).train()
                self.assertEqual((s["stop_reason"], s["stage_complete"]), ("gate_passed", True))


if __name__ == "__main__":
    unittest.main()
