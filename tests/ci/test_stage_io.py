"""Phase 3B, phases J-K: the artifact hand-off between workflow jobs, simulated offline. Each "job" runs in a fresh
directory and sees only what a downloaded artifact would give it."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tinymind.ci import stage_io
from tinymind.ci.stage_io import BundleError, artifact_name, bundle_run, check_incoming, resolve_mode, runtime_seconds, verify_bundle
from tinymind.cli import main
from tinymind.model import ModelConfig
from tinymind.training import checkpoint as ck

PROFILE_YAML = """profile: ci_tiny
model:
  hidden_size: 32
  num_layers: 2
  num_heads: 4
  num_kv_heads: 2
  intermediate_size: 64
  max_seq_len: 256
  vocab_size: 260
tokenizer:
  type: byte
training:
  batch_size: 4
  max_seq_len: 256
  max_steps: 12
  warmup_steps: 2
  learning_rate: 0.003
  min_learning_rate: 0.0003
  eval_interval: 0
  checkpoint_interval: 100
  log_interval: 0
"""


def cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestStageHandoff(unittest.TestCase):
    """stage1 (interrupted, resumed) -> stage2 -> stage3, each job in its own directory; the chain is built once and the
    incoming-artifact checks then run against copies of its bundles."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tm-ci-"))
        cls.profile = cls.tmp / "ci_tiny.yaml"
        cls.profile.write_text(PROFILE_YAML)
        cls.gates = cls.tmp / "gates"
        cls.gates.mkdir()
        cls.data = {}
        for stage in ("stage1", "stage2", "stage3"):
            d = cls.tmp / f"data_{stage}"
            code, _, err = cli(["data", "build-curriculum", "--stage", stage, "--out", str(d), "--scale", "0.01"])
            assert code == 0, err
            cls.data[stage] = d
        cls.gates.joinpath("stage1.gate.json").write_text(json.dumps({"expect_stage": "stage1", "require_stage_complete": True,
                                                                     "checks": [{"metric": "summary.final_validation.val_loss", "max": 50.0}]}))
        # --- job 1: stage1 from scratch, stops after 5 of 12 steps (as if it ran out of time)
        cls.b1 = cls.train_job("job1", "stage1", stop_after=5)
        # --- job 2: resumes stage1 in a clean directory; the run must finish the stage
        cls.b2 = cls.train_job("job2", "stage1", incoming=cls.b1)
        # --- job 3: stage2 from the completed stage1 (promotion) — gated
        cls.b3 = cls.train_job("job3", "stage2", incoming=cls.b2, gate_dir=cls.gates)
        # --- job 4: stage3 from stage2
        cls.b4 = cls.train_job("job4", "stage3", incoming=cls.b3, skip_gate=True)
        cls.artifacts = {"b1": cls.b1, "b2": cls.b2, "b3": cls.b3, "b4": cls.b4}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def train_job(cls, name, stage, *, incoming=None, mode="auto", stop_after=None, gate_dir=None, skip_gate=False, expect_sha=None):
        """One workflow job: verify incoming -> train -> bundle. Returns the bundle path (the 'artifact')."""
        work = cls.tmp / name
        args = ["train", "--config", str(cls.profile), "--dataset", str(cls.data[stage]), "--output", str(work / "out"),
                "--stage", stage, "--seed", "0", "--overflow", "drop"]
        if incoming is not None:
            local = work / "incoming"
            shutil.copytree(incoming, local)  # "download"
            decision = check_incoming(local, stage=stage, profile_config=cls.profile, requested_mode=mode,
                                      gate_dir=gate_dir, skip_gate=skip_gate, expect_manifest_sha256=expect_sha)
            args += ["--resume" if decision["mode"] == "resume" else "--init-from", decision["checkpoint"]]
        if stop_after:
            args += ["--stop-after-steps", str(stop_after)]
        code, out, err = cli(args)
        assert code == 0, err
        bundle = work / "bundle"
        bundle_run(work / "out", bundle, profile="ci_tiny", stage=stage, run_id=f"run-{name}")
        return bundle

    def test_full_three_stage_chain_with_an_interrupted_stage(self):
        m1, _ = verify_bundle(self.b1)
        self.assertEqual((m1["stage"], m1["global_step"], m1["stage_complete"]), ("stage1", 5, False))
        self.assertEqual(m1["artifact_name"].rsplit("-", 1)[0], "tinymind-ci_tiny-stage1-step000005")
        m2, _ = verify_bundle(self.b2)
        self.assertEqual((m2["global_step"], m2["stage_complete"]), (12, True))
        self.assertEqual(json.loads((self.b2 / "training_summary.json").read_text())["resumed_from"].split("/")[-1], "step-00000005")
        s3 = json.loads((self.b3 / "training_summary.json").read_text())
        self.assertEqual((s3["stage"], s3["parent"]["stage"], s3["parent"]["global_step"], s3["cumulative_steps"]), ("stage2", "stage1", 12, 24))
        self.assertEqual(s3["parent"]["manifest_sha256"], verify_bundle(self.b2)[0]["checkpoint_manifest_sha256"])  # lineage points at the exact parent
        s4 = json.loads((self.b4 / "training_summary.json").read_text())
        self.assertEqual((s4["stage"], s4["parent"]["stage"], s4["cumulative_steps"]), ("stage3", "stage2", 36))
        names = {verify_bundle(b)[0]["artifact_name"] for b in (self.b1, self.b2, self.b3, self.b4)}
        self.assertEqual(len(names), 4)  # deterministic, and no two stages/steps collide
        from tinymind.export import load_package
        self.assertIsNotNone(load_package(self.b4 / "export").model)  # each bundle carries a loadable inference package

    def fresh_copy(self, key):
        dst = Path(tempfile.mkdtemp(dir=self.tmp)) / "incoming"
        shutil.copytree(self.artifacts[key], dst)
        return dst

    def check(self, key, **kw):
        return check_incoming(self.fresh_copy(key), profile_config=self.profile, **kw)

    def test_good_artifacts_are_accepted_with_the_right_mode(self):
        self.assertEqual(self.check("b1", stage="stage1")["mode"], "resume")
        self.assertEqual(self.check("b2", stage="stage2", skip_gate=True)["mode"], "init-from")

    def test_corrupted_or_partial_download_is_refused(self):
        d = self.fresh_copy("b1")
        target = next((d / "checkpoints").glob("step-*/model.npz"))
        target.write_bytes(target.read_bytes()[:-100])
        with self.assertRaises(BundleError) as cm:
            check_incoming(d, stage="stage1", profile_config=self.profile)
        self.assertIn("does not match", str(cm.exception))
        d = self.fresh_copy("b1")
        next((d / "export").glob("*.tm")).unlink()
        with self.assertRaises(BundleError):
            check_incoming(d, stage="stage1", profile_config=self.profile)

    def test_a_bundle_whose_manifest_disagrees_with_its_checkpoint_is_refused(self):
        d = self.fresh_copy("b1")
        m = json.loads((d / "bundle_manifest.json").read_text())
        m["dataset_hash"] = "0" * 64
        (d / "bundle_manifest.json").write_text(json.dumps(m))
        with self.assertRaises(BundleError) as cm:
            check_incoming(d, stage="stage1", profile_config=self.profile)
        self.assertIn("dataset_hash", str(cm.exception))

    def test_different_architecture_is_refused_before_any_training(self):
        other = self.tmp / "other.yaml"
        other.write_text(PROFILE_YAML.replace("num_layers: 2", "num_layers: 3"))
        with self.assertRaises(BundleError) as cm:
            check_incoming(self.fresh_copy("b1"), stage="stage1", profile_config=other)
        self.assertIn("model architecture differs", str(cm.exception))

    def test_pinned_manifest_hash(self):
        good = verify_bundle(self.b1)[0]["checkpoint_manifest_sha256"]
        self.assertEqual(self.check("b1", stage="stage1", expect_manifest_sha256=good)["mode"], "resume")
        with self.assertRaises(BundleError):
            self.check("b1", stage="stage1", expect_manifest_sha256="f" * 64)

    def test_mode_rules(self):
        m_incomplete = verify_bundle(self.fresh_copy("b1"))[0]
        m_complete = verify_bundle(self.fresh_copy("b2"))[0]
        with self.assertRaises(BundleError):  # starting stage2 from an unfinished stage1
            resolve_mode(m_incomplete, stage="stage2")
        with self.assertRaises(BundleError):  # nothing left to resume
            resolve_mode(m_complete, stage="stage1")
        with self.assertRaises(BundleError):  # resuming into a different stage
            resolve_mode(m_incomplete, stage="stage2", requested="resume")
        with self.assertRaises(BundleError):
            resolve_mode(m_incomplete, stage="stage2", requested="init-from")
        self.assertEqual(resolve_mode(m_complete, stage="stage3", requested="init-from"), "init-from")

    def test_promotion_gate_blocks_and_can_be_skipped_but_never_silently_missing(self):
        gates = self.tmp / "strict"
        gates.mkdir(exist_ok=True)
        (gates / "stage1.gate.json").write_text(json.dumps({"checks": [{"metric": "summary.final_validation.val_loss", "max": 0.0001}]}))
        with self.assertRaises(BundleError) as cm:
            self.check("b2", stage="stage2", gate_dir=gates)
        self.assertIn("promotion gate FAILED", str(cm.exception))
        self.assertEqual(self.check("b2", stage="stage2", gate_dir=gates, skip_gate=True)["mode"], "init-from")
        empty = self.tmp / "nogates"
        empty.mkdir(exist_ok=True)
        decision = self.check("b2", stage="stage2", gate_dir=empty)
        self.assertIn("not gated", decision["gate"]["note"])  # reported, not hidden


class TestReadTrainingContext(unittest.TestCase):
    """Phase 3B+: recovering (stage, seed) from a bundle so a downstream job can regenerate its eval data
    without the operator having to retype them (and without them ever drifting out of sync)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tm-ctx-"))
        cls.profile = cls.tmp / "ci_tiny.yaml"
        cls.profile.write_text(PROFILE_YAML)
        cls.data = cls.tmp / "data"
        code, _, err = cli(["data", "build-curriculum", "--stage", "stage0", "--out", str(cls.data), "--scale", "0.01"])
        assert code == 0, err
        cls.out = cls.tmp / "run"
        code, _, err = cli(["train", "--config", str(cls.profile), "--dataset", str(cls.data), "--output", str(cls.out),
                            "--stage", "stage0", "--seed", "7", "--overflow", "drop"])
        assert code == 0, err
        cls.bundle = cls.tmp / "bundle"
        bundle_run(cls.out, cls.bundle, profile="ci_tiny", stage="stage0")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_matches_what_the_run_was_actually_configured_with(self):
        ctx = stage_io.read_training_context(self.bundle)
        self.assertEqual(ctx, {"stage": "stage0", "seed": 7})

    def test_refuses_a_corrupt_bundle_the_same_as_everything_else(self):
        broken = self.tmp / "broken"
        shutil.copytree(self.bundle, broken)
        (broken / "bundle_manifest.json").write_text("{not json")
        with self.assertRaises(BundleError):
            stage_io.read_training_context(broken)

    def test_cli_subcommand_writes_github_output(self):
        gh_out = self.tmp / "gh_output.txt"
        import os
        old = os.environ.get("GITHUB_OUTPUT")
        os.environ["GITHUB_OUTPUT"] = str(gh_out)
        try:
            code = stage_io.main(["training-context", "--dir", str(self.bundle)])
        finally:
            if old is None:
                os.environ.pop("GITHUB_OUTPUT", None)
            else:
                os.environ["GITHUB_OUTPUT"] = old
        self.assertEqual(code, 0)
        self.assertEqual(gh_out.read_text(), "stage=stage0\nseed=7\n")


class TestSmallPieces(unittest.TestCase):
    def test_runtime_budget(self):
        self.assertEqual(runtime_seconds(5.0), 5 * 3600 - 15 * 60)
        for bad in (0.0, 0.01, 5.6, 6.0, 24.0):
            with self.assertRaises(BundleError):
                runtime_seconds(bad)

    def test_artifact_names_are_deterministic_and_informative(self):
        self.assertEqual(artifact_name("tiny_mobile", "stage1", 500, "abcdef123456"), "tinymind-tiny_mobile-stage1-step000500-abcdef1")
        self.assertNotEqual(artifact_name("tiny_mobile", "stage1", 500, "abcdef1"), artifact_name("tiny_mobile", "stage1", 501, "abcdef1"))
        self.assertNotEqual(artifact_name("tiny_mobile", "stage1", 500, "abcdef1"), artifact_name("tiny_mobile", "stage2", 500, "abcdef1"))

    def test_missing_or_foreign_bundle(self):
        d = Path(tempfile.mkdtemp())
        with self.assertRaises(BundleError):
            verify_bundle(d)
        (d / "bundle_manifest.json").write_text("{not json")
        with self.assertRaises(BundleError):
            verify_bundle(d)
        (d / "bundle_manifest.json").write_text(json.dumps({"kind": "something-else"}))
        with self.assertRaises(BundleError):
            verify_bundle(d)

    def test_cli_entry_points_report_errors_with_exit_code_1(self):
        d = Path(tempfile.mkdtemp())
        self.assertEqual(stage_io.main(["check-incoming", "--dir", str(d), "--stage", "stage1", "--profile-config", "configs/tiny_debug.yaml"]), 1)
        self.assertEqual(stage_io.main(["runtime", "--max-hours", "9"]), 1)


if __name__ == "__main__":
    unittest.main()
