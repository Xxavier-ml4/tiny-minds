"""Verifying and reporting on a completed training artifact — never training.
Covers: hard-fail-on-tamper (no partial report), clean-process loading,
deterministic smoke prompts, the exact metric list the brief asks for,
native inference reported separately (and honestly, when unsupported), and
that the rendered report contains no quality judgement language.
"""
import contextlib
import io
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from tinymind.ci import stage_io, trained_model_report as tmr
from tinymind.data.curriculum import build_eval
from tinymind.evaluation.tiny_suite import run_suite, strip_rows
from tinymind.export import load_package
from tinymind.export.package import PackageError
from tinymind.native_bridge import HAS_CXX
from tinymind.training.checkpoint import CheckpointError, sha256_file
from tinymind.training.exporter import package_exporter

from tests.training._helpers import RENDERER, make_engine, tmpdir

REPO = Path(__file__).resolve().parents[2]


def cli_main(argv):
    from tinymind.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class Fixture(unittest.TestCase):
    """One trained bundle (a handful of real steps, tiny model) + real held-out data, shared by every test in
    this module — training is the expensive part, so it happens once."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tmpdir()
        out = cls.tmp / "run"
        engine = make_engine(out, train_over=dict(max_steps=15, checkpoint_interval=0, log_interval=0),
                             exporter=package_exporter)
        engine.train()
        cls.bundle = cls.tmp / "bundle"
        stage_io.bundle_run(out, cls.bundle, profile="ci-fixture", stage="stage0")
        cls.data_dir = cls.tmp / "data"
        cls.data_dir.mkdir()
        eval_records = build_eval(seed=0, scale=0.02)
        (cls.data_dir / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_records) + "\n")
        (cls.data_dir / "val.jsonl").write_text("\n".join(json.dumps(r) for r in eval_records[:5]) + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def fresh_copy(self):
        dst = tmpdir() / "bundle"
        shutil.copytree(self.bundle, dst)
        return dst


class TestVerifyArtifact(Fixture):
    def test_good_bundle_verifies(self):
        result = tmr.verify_artifact(self.bundle)
        self.assertIn("bundle_manifest", result)
        self.assertIn("package_manifest", result)
        self.assertTrue(Path(result["export_dir"]).is_dir())

    def test_corrupt_checkpoint_is_refused(self):
        d = self.fresh_copy()
        f = next((d / "checkpoints").glob("step-*/model.npz"))
        f.write_bytes(f.read_bytes()[:-30])
        with self.assertRaises(tmr.TrainedModelReportError):
            tmr.verify_artifact(d)

    def test_corrupt_package_is_refused(self):
        d = self.fresh_copy()
        (d / "export" / "model.tm").write_bytes(b"garbage")
        with self.assertRaises(tmr.TrainedModelReportError) as cm:
            tmr.verify_artifact(d)
        self.assertTrue(cm.exception.errors)

    def test_package_swapped_for_a_different_models_package_is_refused(self):
        # a bundle whose checkpoint and exported package do not describe the same model: caught here by the
        # bundle's own per-file integrity check (the swapped files no longer match bundle_manifest.json's
        # recorded hashes) — the separate model_config_hash/tokenizer_hash cross-check in verify_artifact is
        # defense in depth for a bundle_manifest that was itself regenerated to match tampered content, which
        # this simpler swap does not exercise.
        other_out = tmpdir() / "other"
        other = make_engine(other_out, model_over=dict(hidden_size=48), train_over=dict(max_steps=5, checkpoint_interval=0, log_interval=0),
                            exporter=package_exporter)
        other.train()
        d = self.fresh_copy()
        shutil.rmtree(d / "export")
        shutil.copytree(other_out / "export", d / "export")
        with self.assertRaises(tmr.TrainedModelReportError) as cm:
            tmr.verify_artifact(d)
        self.assertTrue(cm.exception.errors)

    def test_cross_check_catches_a_manifest_rewritten_to_match_mismatched_content(self):
        # the harder case: bundle_manifest.json's file hashes are updated to match a swapped-in export (as an
        # adversarial or buggy re-bundler might do), so the per-file integrity check alone would pass — only
        # the model_config_hash/tokenizer_hash cross-check in verify_artifact catches this.
        other_out = tmpdir() / "other2"
        other = make_engine(other_out, model_over=dict(hidden_size=48), train_over=dict(max_steps=5, checkpoint_interval=0, log_interval=0),
                            exporter=package_exporter)
        other.train()
        d = self.fresh_copy()
        shutil.rmtree(d / "export")
        shutil.copytree(other_out / "export", d / "export")
        manifest = json.loads((d / "bundle_manifest.json").read_text())
        for f in (d / "export").iterdir():
            manifest["files"][f"export/{f.name}"] = {"sha256": sha256_file(f), "size": f.stat().st_size}
        (d / "bundle_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(tmr.TrainedModelReportError) as cm:
            tmr.verify_artifact(d)
        self.assertIn("model_config_hash", "; ".join(cm.exception.errors))

    def test_missing_bundle_manifest(self):
        d = tmpdir()
        with self.assertRaises(tmr.TrainedModelReportError):
            tmr.verify_artifact(d)


class TestCleanProcessLoad(Fixture):
    def test_loads_with_training_code_unreachable(self):
        info = tmr.load_in_clean_process(self.bundle / "export", REPO)
        self.assertEqual(info, {"loaded": True, "parameters": info["parameters"], "training_importable": False})
        self.assertGreater(info["parameters"], 0)

    def test_a_broken_package_fails_loudly_not_silently(self):
        d = self.fresh_copy()
        (d / "export" / "package.json").write_text("{not json")
        with self.assertRaises(tmr.TrainedModelReportError):
            tmr.load_in_clean_process(d / "export", REPO)


class TestSmokePrompts(Fixture):
    def test_fixed_prompt_set_and_shape(self):
        pkg = load_package(self.bundle / "export")
        rows = tmr.run_smoke_prompts(pkg, max_new_tokens=8)
        self.assertEqual([r["label"] for r in rows], [p[0] for p in tmr.DEFAULT_SMOKE_PROMPTS])
        for row in rows:
            for key in ("prompt", "tools", "completion", "tokens_generated", "finish_reason"):
                self.assertIn(key, row)
            self.assertIn(row["finish_reason"], ("stop", "length"))

    def test_deterministic_within_and_across_package_loads(self):
        pkg = load_package(self.bundle / "export")
        self.assertTrue(tmr.smoke_prompts_are_deterministic(pkg, max_new_tokens=8))
        a = tmr.run_smoke_prompts(pkg, max_new_tokens=8)
        b = tmr.run_smoke_prompts(load_package(self.bundle / "export"), max_new_tokens=8)  # a SEPARATE load
        self.assertEqual([r["completion"] for r in a], [r["completion"] for r in b])


class TestTmArtifact(Fixture):
    def test_direct_forward_pass(self):
        result = tmr.test_tm_artifact(self.bundle / "export")
        self.assertTrue(result["forward_pass_ran"])
        self.assertTrue(result["output_all_finite"])
        self.assertGreater(result["size_bytes"], 0)
        self.assertGreater(result["parameters"], 0)

    def test_is_deterministic(self):
        a = tmr.test_tm_artifact(self.bundle / "export")
        b = tmr.test_tm_artifact(self.bundle / "export")
        self.assertEqual(a["output_abs_max"], b["output_abs_max"])


class TestNativeInference(Fixture):
    def test_reports_supported_matches_environment(self):
        pkg = load_package(self.bundle / "export")
        result = tmr.test_native_inference(self.bundle / "export", pkg)
        self.assertEqual(result["supported"], HAS_CXX)
        if HAS_CXX:
            self.assertIsInstance(result["max_abs_logit_diff"], float)
            self.assertIsInstance(result["greedy_token_mismatches"], int)
            self.assertGreaterEqual(result["greedy_token_mismatches"], 0)
            self.assertLess(result["max_abs_logit_diff"], 1e-2)
        else:
            self.assertIn("reason", result)

    def test_reports_unsupported_honestly_when_no_compiler_is_available(self):
        pkg = load_package(self.bundle / "export")
        with patch.object(tmr, "HAS_CXX", False):
            result = tmr.test_native_inference(self.bundle / "export", pkg)
        self.assertEqual(result["supported"], False)
        self.assertIn("compiler", result["reason"])


class TestEvalExtraction(Fixture):
    EXPECTED_FIELDS = ("validation_loss", "held_out_loss", "perplexity", "copy_accuracy", "instruction_accuracy",
                       "factual_qa_accuracy", "structured_output_accuracy", "clarification_accuracy",
                       "refusal_accuracy", "correct_tool_rate", "argument_accuracy", "wrong_tool_rate",
                       "malformed_call_rate", "false_positive_call_rate")

    def test_every_required_metric_is_present(self):
        self.assertEqual(set(tmr.KEY_METRIC_PATHS), set(self.EXPECTED_FIELDS))
        self.assertEqual(len(self.EXPECTED_FIELDS), 14)

    def test_key_metrics_match_run_suite_directly(self):
        pkg = load_package(self.bundle / "export")
        eval_records = [json.loads(l) for l in (self.data_dir / "eval.jsonl").read_text().splitlines()]
        val_records = [json.loads(l) for l in (self.data_dir / "val.jsonl").read_text().splitlines()]
        direct = strip_rows(run_suite(pkg, eval_records, val_records, max_new_tokens=8, limit_per_category=2))
        via_report = tmr.run_eval(pkg, eval_records, val_records, limit_per_category=2, max_new_tokens=8)
        self.assertEqual(via_report["key_metrics"]["validation_loss"], direct["loss"]["validation"]["loss"])
        self.assertEqual(via_report["key_metrics"]["correct_tool_rate"], direct["tool_behavior"]["correct_tool_rate"])
        for name in self.EXPECTED_FIELDS:
            self.assertIn(name, via_report["key_metrics"])

    def test_missing_metric_path_is_none_not_a_crash(self):
        self.assertIsNone(tmr._dig({"loss": {}}, ("loss", "eval", "loss")))
        self.assertIsNone(tmr._dig({}, ("a", "b")))


class TestBuildReportEndToEnd(Fixture):
    def test_full_report_structure(self):
        report = tmr.build_report(self.bundle, repo_root=REPO, data_dir=self.data_dir, eval_limit_per_category=2, max_new_tokens=8)
        for key in ("kind", "format_version", "artifact_name", "stage", "global_step", "stage_complete",
                    "parameter_count", "verification", "clean_process_load", "smoke_prompts", "tm_artifact",
                    "native_inference", "evaluation"):
            self.assertIn(key, report)
        self.assertEqual(report["kind"], tmr.REPORT_KIND)
        self.assertTrue(report["verification"]["ok"])
        self.assertEqual(len(report["smoke_prompts"]["prompts"]), len(tmr.DEFAULT_SMOKE_PROMPTS))
        self.assertIsNotNone(report["evaluation"])
        json.dumps(report)  # must be JSON-serialisable

    def test_no_data_dir_skips_evaluation_without_failing(self):
        report = tmr.build_report(self.bundle, repo_root=REPO, data_dir=None, max_new_tokens=8)
        self.assertIsNone(report["evaluation"])
        self.assertTrue(report["verification"]["ok"])

    def test_tampered_bundle_raises_and_produces_no_report(self):
        d = self.fresh_copy()
        (d / "checkpoints" / "latest.json").write_text('{"checkpoint": "does-not-exist", "manifest_sha256": "x", "global_step": 0}')
        with self.assertRaises((tmr.TrainedModelReportError, CheckpointError)):
            tmr.build_report(d, repo_root=REPO, data_dir=self.data_dir, max_new_tokens=8)

    def test_report_is_deterministic_end_to_end(self):
        a = tmr.build_report(self.bundle, repo_root=REPO, data_dir=self.data_dir, eval_limit_per_category=2, max_new_tokens=8)
        b = tmr.build_report(self.bundle, repo_root=REPO, data_dir=self.data_dir, eval_limit_per_category=2, max_new_tokens=8)
        self.assertEqual([r["completion"] for r in a["smoke_prompts"]["prompts"]],
                         [r["completion"] for r in b["smoke_prompts"]["prompts"]])
        self.assertEqual(a["evaluation"]["key_metrics"], b["evaluation"]["key_metrics"])
        self.assertEqual(a["tm_artifact"]["output_abs_max"], b["tm_artifact"]["output_abs_max"])


class TestMarkdownReportsMeasurementsOnly(Fixture):
    def test_no_quality_verdict_language(self):
        report = tmr.build_report(self.bundle, repo_root=REPO, data_dir=self.data_dir, eval_limit_per_category=2, max_new_tokens=8)
        text = tmr.render_markdown(report)
        # everything except the closing disclaimer, which legitimately says "it does NOT judge... good" —
        # that negation is the one place those words are allowed to appear
        body = text.rsplit("---", 1)[0].lower()
        for word in tmr._BANNED_WORDS:
            self.assertNotIn(word, body, word)

    def test_contains_the_required_sections(self):
        report = tmr.build_report(self.bundle, repo_root=REPO, data_dir=self.data_dir, eval_limit_per_category=2, max_new_tokens=8)
        text = tmr.render_markdown(report)
        for heading in ("## Verification", "## `.tm` artifact", "## Native inference", "## Smoke prompts",
                       "## Held-out evaluation"):
            self.assertIn(heading, text)
        for name in TestEvalExtraction.EXPECTED_FIELDS:
            self.assertIn(name.replace("_", " "), text)

    def test_no_data_dir_says_so_plainly(self):
        report = tmr.build_report(self.bundle, repo_root=REPO, data_dir=None, max_new_tokens=8)
        text = tmr.render_markdown(report)
        self.assertIn("Not run: no evaluation data supplied.", text)


class TestWriteReports(Fixture):
    def test_writes_valid_json_and_nonempty_markdown(self):
        report = tmr.build_report(self.bundle, repo_root=REPO, data_dir=self.data_dir, eval_limit_per_category=2, max_new_tokens=8)
        out = tmpdir()
        tmr.write_reports(report, out / "r.json", out / "r.md")
        reloaded = json.loads((out / "r.json").read_text())
        self.assertEqual(reloaded["artifact_name"], report["artifact_name"])
        self.assertGreater(len((out / "r.md").read_text()), 100)


class TestCLI(Fixture):
    def test_success_exit_zero_and_writes_both_files(self):
        out = tmpdir()
        code = tmr.main(["--bundle", str(self.bundle), "--data", str(self.data_dir), "--eval-limit-per-category", "2",
                         "--max-new-tokens", "8", "--repo-root", str(REPO),
                         "--out-json", str(out / "r.json"), "--out-md", str(out / "r.md")])
        self.assertEqual(code, 0)
        self.assertTrue((out / "r.json").is_file())
        self.assertTrue((out / "r.md").is_file())

    def test_failure_exit_one_and_writes_nothing(self):
        d = self.fresh_copy()
        (d / "export" / "model.tm").write_bytes(b"garbage")
        out = tmpdir()
        code = tmr.main(["--bundle", str(d), "--repo-root", str(REPO),
                         "--out-json", str(out / "r.json"), "--out-md", str(out / "r.md")])
        self.assertEqual(code, 1)
        self.assertFalse((out / "r.json").exists())
        self.assertFalse((out / "r.md").exists())

    def test_via_tinymind_ci_module_entrypoint_subprocess(self):
        import subprocess
        import sys
        out = tmpdir()
        proc = subprocess.run([sys.executable, "-m", "tinymind.ci.trained_model_report", "--bundle", str(self.bundle),
                               "--eval-limit-per-category", "2", "--max-new-tokens", "8",
                               "--out-json", str(out / "r.json"), "--out-md", str(out / "r.md")],
                              cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((out / "r.json").is_file())


if __name__ == "__main__":
    unittest.main()
