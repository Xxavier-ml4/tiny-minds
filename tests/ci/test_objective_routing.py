"""Objective-driven routing between GitHub jobs (tinymind.ci.stage_io): a bundle
whose stage used its whole budget without meeting its objective is CONTINUED
(larger budget), never resumed and never promoted; the next stage is rejected
until the parent's objective is met and its promotion gate passes, and accepted
after; a complete stage whose own promotion gate fails is reopened. Bundles are
real: produced by the training engine and bundle_run, then verified."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from tinymind.ci.stage_io import BundleError, bundle_run, check_incoming, main, resolve_mode, summary_markdown, verify_bundle
from tinymind.training.objective import StageObjective
from tests.training._helpers import make_engine, model_config


def _objective(stage, max_loss):
    return StageObjective({"stage": stage, "generation": {"max_new_tokens": 6, "prompts": ["The cat", "repeat 1"]},
                           "measurements": [{"metric": "loss.val_loss", "max": max_loss},
                                            {"metric": "generation.n", "min": 0}]})


class TestObjectiveRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tm-route-"))
        cls.profile = cls.tmp / "profile.yaml"
        cls.profile.write_text(yaml.safe_dump({"model": model_config().to_dict()}))
        for name, verdict in (("gates_fail", {"max": -1.0}), ("gates_pass", {"max": 1e9})):
            d = cls.tmp / name
            d.mkdir()
            d.joinpath("stage1.gate.json").write_text(json.dumps({
                "expect_stage": "stage1", "require_stage_complete": True,
                "checks": [{"metric": "summary.final_validation.val_loss", **verdict}]}))
        cls.failed = cls.job("failed", _objective("stage1", -1.0), max_steps=6)       # budget used, objective not met
        cls.stopped = cls.job("stopped", _objective("stage1", 1e9), max_steps=12, stop_after_steps=3)  # out of time
        cls.passed = cls.job("passed", _objective("stage1", 1e9), max_steps=6)        # objective met

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def job(cls, name, objective, *, max_steps, stop_after_steps=None):
        out = cls.tmp / name / "out"
        make_engine(out, train_over=dict(stage="stage1", max_steps=max_steps, eval_interval=3), objective=objective,
                    stop_after_steps=stop_after_steps).train()
        bundle = cls.tmp / name / "bundle"
        bundle_run(out, bundle, profile="ci", stage="stage1", run_id=f"r{name}")
        return bundle

    def incoming(self, bundle):
        dst = Path(tempfile.mkdtemp(dir=self.tmp)) / "incoming"
        shutil.copytree(bundle, dst)
        return dst

    def check(self, bundle, stage, **kw):
        return check_incoming(self.incoming(bundle), stage=stage, profile_config=self.profile, **kw)

    def test_bundle_records_why_the_stage_stopped(self):
        m, _ = verify_bundle(self.failed)
        self.assertEqual((m["stop_reason"], m["stage_complete"], m["budget_exhausted"]), ("gate_failed", False, True))
        self.assertEqual(m["objective"], {"configured": True, "objective_met": False, "report_step": 6})
        self.assertEqual(m["budget_tokens"], m["total_steps"] * m["effective_batch_tokens"])
        m, _ = verify_bundle(self.stopped)
        self.assertEqual((m["stop_reason"], m["stage_complete"], m["budget_exhausted"]), ("step_limit", False, False))
        m, _ = verify_bundle(self.passed)
        self.assertEqual((m["stop_reason"], m["stage_complete"]), ("gate_passed", True))

    def test_bundle_carries_every_objective_report(self):
        reports = sorted(p.name for p in (self.failed / "objective_reports").glob("step-*.md"))
        self.assertEqual(reports, ["step-00000000.md", "step-00000003.md", "step-00000006.md"])
        self.assertIn("Raw generation 1", (self.failed / "objective_reports" / "latest.md").read_text())
        m, _ = verify_bundle(self.failed)
        self.assertIn("objective_reports/latest.json", m["files"])  # hashed like every other bundle file

    def test_objective_not_met_at_budget_continues_the_same_stage(self):
        d = self.check(self.failed, "stage1")
        self.assertEqual((d["mode"], d["reopen"]), ("continue", False))
        self.assertGreater(d["parent_budget_tokens"], 0)
        with self.assertRaises(BundleError) as cm:  # an exact resume cannot extend a used-up budget
            self.check(self.failed, "stage1", requested_mode="resume")
        self.assertIn("continue", str(cm.exception))

    def test_next_stage_is_rejected_while_the_parent_objective_is_not_met(self):
        for bundle in (self.failed, self.stopped):
            with self.assertRaises(BundleError) as cm:
                self.check(bundle, "stage2", skip_gate=True)
            self.assertIn("objective must be met", str(cm.exception))
            with self.assertRaises(BundleError):
                self.check(bundle, "stage2", requested_mode="init-from", skip_gate=True)

    def test_time_interrupted_stage_resumes(self):
        self.assertEqual(self.check(self.stopped, "stage1")["mode"], "resume")

    def test_next_stage_accepted_after_the_objective_and_gate_pass(self):
        d = self.check(self.passed, "stage2", gate_dir=self.tmp / "gates_pass")
        self.assertEqual((d["mode"], d["gate"]["passed"]), ("init-from", True))
        with self.assertRaises(BundleError):  # the same complete stage has nothing left to do
            self.check(self.passed, "stage1", gate_dir=self.tmp / "gates_pass")

    def test_complete_stage_whose_promotion_gate_fails_is_reopened(self):
        with self.assertRaises(BundleError) as cm:
            self.check(self.passed, "stage2", gate_dir=self.tmp / "gates_fail")
        self.assertIn("promotion gate FAILED", str(cm.exception))
        self.assertIn("reopen", str(cm.exception))
        d = self.check(self.passed, "stage1", gate_dir=self.tmp / "gates_fail")
        self.assertEqual((d["mode"], d["reopen"], d["gate"]["passed"]), ("continue", True, False))

    def test_resolve_mode_table(self):
        base = {"stage": "stage1", "global_step": 6}
        self.assertEqual(resolve_mode({**base, "stage_complete": False, "budget_exhausted": True}, stage="stage1"), "continue")
        self.assertEqual(resolve_mode({**base, "stage_complete": False, "budget_exhausted": False}, stage="stage1"), "resume")
        self.assertEqual(resolve_mode({**base, "stage_complete": False}, stage="stage1"), "resume")  # pre-objective bundle
        self.assertEqual(resolve_mode({**base, "stage_complete": True}, stage="stage2"), "init-from")
        self.assertEqual(resolve_mode({**base, "stage_complete": True}, stage="stage1", own_gate_failed=True), "continue")
        for m, stage, requested in (({**base, "stage_complete": True}, "stage1", "auto"),
                                    ({**base, "stage_complete": True}, "stage1", "continue"),
                                    ({**base, "stage_complete": False, "budget_exhausted": True}, "stage2", "continue"),
                                    ({**base, "stage_complete": False, "budget_exhausted": True}, "stage2", "auto")):
            with self.assertRaises(BundleError, msg=(m, stage, requested)):
                resolve_mode(m, stage=stage, requested=requested)

    def test_job_summary_names_the_next_action(self):
        for bundle, expected in ((self.failed, "mode continue"), (self.stopped, "mode resume"),
                                 (self.passed, "mode init-from")):
            m, _ = verify_bundle(bundle)
            self.assertIn(expected, summary_markdown(m, None))

    def test_routing_decision_and_rejection_are_written_to_the_job_summary(self):
        import os
        summary = self.tmp / "step_summary.md"
        for bundle, stage, expect in ((self.failed, "stage1", "continue (same stage, larger token budget)"),
                                      (self.failed, "stage2", "REJECTED for stage2")):
            summary.write_text("")
            os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    main(["check-incoming", "--dir", str(self.incoming(bundle)), "--stage", stage,
                          "--profile-config", str(self.profile)])
            finally:
                os.environ.pop("GITHUB_STEP_SUMMARY", None)
            text = summary.read_text()
            self.assertIn(expect, text)
        self.assertIn("objective must be met", text)  # the rejection says why

    def test_cli_outputs_what_the_workflow_reads(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["check-incoming", "--dir", str(self.incoming(self.failed)), "--stage", "stage1",
                         "--profile-config", str(self.profile)])
        self.assertEqual(code, 0)
        values = dict(line.split("=", 1) for line in out.getvalue().splitlines() if "=" in line)
        self.assertEqual((values["mode"], values["reopen"], values["parent_stop_reason"]), ("continue", "false", "gate_failed"))
        self.assertGreater(int(values["parent_budget_tokens"]), 0)


if __name__ == "__main__":
    unittest.main()
