"""Phase 3B CLI: train / checkpoint-info / verify-checkpoint / export / eval / data / stage-gate / budget."""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tinymind.cli import main
from tinymind.export import load_package

REPO = Path(__file__).resolve().parents[1]


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def cli_subprocess(argv, **kw):
    return subprocess.run([sys.executable, "-m", "tinymind.cli", *argv], cwd=REPO, capture_output=True, text=True, timeout=280, **kw)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tm-cli-"))
        cls.data = cls.tmp / "stage0"
        code, _, err = run(["data", "build-curriculum", "--stage", "stage0", "--out", str(cls.data), "--scale", "0.05"])
        assert code == 0, err

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def train(self, out, *extra, config="tiny_debug", steps="30"):
        return run(["train", "--config", config, "--dataset", str(self.data), "--output", str(out), "--max-steps", steps,
                    "--checkpoint-interval", "10", "--eval-interval", "10", "--log-interval", "0", "--seed", "1", *extra])


class TestData(Base):
    def test_curriculum_files_and_manifest(self):
        names = sorted(p.name for p in self.data.iterdir())
        self.assertEqual(names, ["curriculum.json", "eval.jsonl", "train_sanity.jsonl", "val.jsonl"])
        m = json.loads((self.data / "curriculum.json").read_text())
        self.assertEqual((m["stage"], m["mixture"]), ("stage0", {"sanity": 1.0}))

    def test_contamination_command_detects_a_leak(self):
        train, ev = self.data / "train_sanity.jsonl", self.data / "eval.jsonl"
        code, out, _ = run(["data", "check-contamination", "--train", str(train), "--eval", str(ev)])
        self.assertEqual(code, 0)
        leaked = self.tmp / "leaked_train.jsonl"
        leaked.write_text(train.read_text() + ev.read_text().splitlines()[0] + "\n")
        code, out, _ = run(["data", "check-contamination", "--train", str(leaked), "--eval", str(ev)])
        self.assertEqual(code, 1)
        self.assertTrue(json.loads(out)["contaminated"])

    def test_render_shows_the_loss_mask(self):
        code, out, _ = run(["data", "render", str(self.data / "train_sanity.jsonl"), "-n", "1"])
        self.assertEqual(code, 0)
        self.assertIn("<BOS>user:", out)
        self.assertIn("assistant:\\n\n[", out)  # everything before the response is untrained, the response is bracketed
        self.assertIn("<EOS>]", out)


class TestTrainCommand(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = cls.tmp / "run"
        cls.code, cls.stdout, cls.stderr = cls.train(cls, cls.out)

    def test_trains_checkpoints_evaluates_and_exports(self):
        self.assertEqual(self.code, 0, self.stderr)
        result = json.loads(self.stdout)
        self.assertTrue(result["stage_complete"])
        summary = json.loads((self.out / "training_summary.json").read_text())
        self.assertEqual((summary["final_step"], summary["stop_reason"]), (30, "complete"))
        self.assertLess(summary["final_validation"]["val_loss"], summary["initial_validation"]["val_loss"])
        self.assertTrue((self.out / "export" / "model.tm").exists())
        self.assertTrue((self.out / "checkpoints" / "latest.json").exists())
        self.assertIn("contamination check", self.stderr)

    def test_checkpoint_info_and_verify(self):
        root = str(self.out / "checkpoints")
        code, out, err = run(["checkpoint-info", root])
        info = json.loads(out)
        self.assertEqual((code, info["valid"], info["progress"]["global_step"]), (0, True, 30))
        code, out, err = run(["verify-checkpoint", root, "--expect-stage", "stage0", "--expect-step", "30", "--expect-complete", "true"])
        ok = json.loads(out)
        self.assertEqual((code, ok["ok"]), (0, True))
        code, _, err = run(["verify-checkpoint", root, "--expect-stage", "stage9"])
        self.assertEqual(code, 1)
        self.assertIn("expected 'stage9'", err)
        code, _, err = run(["verify-checkpoint", root, "--expect-dataset-hash", "0" * 64])
        self.assertEqual(code, 1)
        code, _, err = run(["verify-checkpoint", root, "--expect-manifest-sha256", ok["manifest_sha256"]])
        self.assertEqual(code, 0)

    def test_corrupt_checkpoint_fails_clearly(self):
        bad = self.tmp / "bad"
        shutil.copytree(self.out / "checkpoints", bad)
        for d in bad.glob("step-*"):
            f = d / "model.npz"
            f.write_bytes(f.read_bytes()[:-50])
        code, _, err = run(["verify-checkpoint", str(bad)])
        self.assertEqual(code, 1)
        self.assertIn("no valid checkpoint", err)
        code, _, err = run(["checkpoint-info", str(bad)])
        self.assertEqual(code, 1)

    def test_export_and_verify_package_and_eval(self):
        pkg = self.tmp / "exported"
        code, out, err = run(["export", "--checkpoint", str(self.out / "checkpoints"), "--output", str(pkg)])
        self.assertEqual(code, 0, err)
        self.assertEqual(run(["verify-package", str(pkg)])[0], 0)
        a, b = load_package(pkg), load_package(self.out / "export")
        ids = np.array([[1, 5, 6, 7]])
        np.testing.assert_array_equal(a.model(ids).logits.data, b.model(ids).logits.data)
        res = self.tmp / "eval.json"
        code, out, err = run(["eval", "--package", str(pkg), "--eval", str(self.data / "eval.jsonl"), "--val", str(self.data / "val.jsonl"),
                              "--out", str(res), "--limit-per-category", "3", "--max-new-tokens", "6"])
        self.assertEqual(code, 0, err)
        r = json.loads(res.read_text())
        for key in ("loss", "capabilities", "tool_behavior", "generation", "timing"):
            self.assertIn(key, r)
        self.assertNotIn("_rows", r)
        self.assertIn("copy", r["capabilities"])
        code, out, _ = run(["eval-compare", str(res), str(res), "--json"])
        self.assertTrue(all(row["delta"] == 0 for row in json.loads(out)))

    def test_stage_gate(self):
        crit = self.tmp / "gate.json"
        summary = str(self.out / "training_summary.json")
        crit.write_text(json.dumps({"expect_stage": "stage0", "require_stage_complete": True,
                                    "checks": [{"metric": "summary.final_validation.val_loss", "max": 100.0}]}))
        self.assertEqual(run(["stage-gate", "--criteria", str(crit), "--summary", summary])[0], 0)
        crit.write_text(json.dumps({"checks": [{"metric": "summary.final_validation.val_loss", "max": 0.0001},
                                               {"metric": "eval.capabilities.tool_arithmetic.accuracy", "min": 0.1}]}))
        code, out, _ = run(["stage-gate", "--criteria", str(crit), "--summary", summary])
        verdict = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual([c["ok"] for c in verdict["checks"][-2:]], [False, False])  # a missing metric fails, it is not skipped

    def test_budget(self):
        code, out, _ = run(["budget", "--config", "tiny_mobile"])
        b = json.loads(out)
        self.assertEqual((code, b["parameters"], b["kv_cache"]["256"]["bytes"]), (0, 1214592, 786432))


class TestTrainErrors(Base):
    def test_failures_are_clear_and_nonzero(self):
        out = self.tmp / "e1"
        code, _, err = run(["train", "--config", "tiny_debug", "--dataset", str(self.data / "train_sanity.jsonl"), "--output", str(out),
                            "--max-steps", "3", "--warmup-steps", "1"])
        self.assertEqual(code, 0, err)  # validation-fraction split is the default, so a bare file still has validation
        code, _, err = run(["train", "--config", "tiny_debug", "--dataset", str(self.data), "--output", str(self.tmp / "e0"), "--max-steps", "3"])
        self.assertEqual(code, 1)
        self.assertIn("warmup_steps (10)", err)  # the profile's warmup does not fit a 3-step run: an error, not a silent change
        code, _, err = run(["train", "--config", "tiny_debug", "--dataset", str(self.data / "train_sanity.jsonl"), "--output", str(self.tmp / "e2"),
                            "--max-steps", "3", "--warmup-steps", "1", "--validation-fraction", "0"])
        self.assertEqual(code, 1)
        self.assertIn("no validation data", err)
        code, _, err = run(["train", "--config", "tiny_debug", "--dataset", str(self.data), "--output", str(self.tmp / "e3"), "--max-steps", "3",
                            "--warmup-steps", "1", "--resume", str(self.tmp / "nowhere")])
        self.assertEqual(code, 1)
        code, _, err = run(["train", "--config", "tiny_debug", "--dataset", str(self.data), "--output", str(self.tmp / "e4"), "--max-steps", "3",
                            "--warmup-steps", "1", "--resume", str(out / "checkpoints"), "--init-from", str(out / "checkpoints")])
        self.assertEqual(code, 1)
        self.assertIn("different operations", err)

    def test_phase3a_style_empty_target_data_is_rejected_not_silently_trained(self):
        f = self.tmp / "empty.jsonl"
        f.write_text("\n".join(json.dumps({"id": f"e{i}", "messages": [{"role": "user", "content": "text"}],
                                           "target": {"type": "answer", "content": ""}}) for i in range(8)) + "\n")
        code, _, err = run(["train", "--config", "tiny_debug", "--dataset", str(f), "--output", str(self.tmp / "e5"), "--max-steps", "3", "--warmup-steps", "1"])
        self.assertEqual(code, 1)
        self.assertIn("empty", err)

    def test_contaminated_eval_blocks_training(self):
        train = self.tmp / "leak_train.jsonl"
        ev = self.data / "eval.jsonl"
        train.write_text((self.data / "train_sanity.jsonl").read_text() + ev.read_text().splitlines()[0] + "\n")
        common = ["train", "--config", "tiny_debug", "--dataset", str(train), "--eval-dataset", str(ev), "--max-steps", "3", "--warmup-steps", "1"]
        code, _, err = run([*common, "--output", str(self.tmp / "e6")])
        self.assertEqual(code, 1)
        self.assertIn("overlaps the training data", err)  # caught even though the leaked line may land in the validation split
        code, _, err = run([*common, "--output", str(self.tmp / "e7"), "--allow-contamination"])
        self.assertEqual(code, 0, err)

    def test_vocab_mismatch(self):
        cfg = self.tmp / "bad.yaml"
        cfg.write_text("model:\n  hidden_size: 16\n  num_layers: 1\n  num_heads: 4\n  num_kv_heads: 2\n  intermediate_size: 32\n  max_seq_len: 64\n  vocab_size: 999\n")
        code, _, err = run(["train", "--config", str(cfg), "--dataset", str(self.data), "--output", str(self.tmp / "e8"), "--max-steps", "2"])
        self.assertEqual(code, 1)
        self.assertIn("vocab_size", err)


class TestResumeAcrossProcesses(Base):
    def test_time_budget_stop_then_resume_in_a_new_process_equals_one_uninterrupted_run(self):
        common = ["train", "--config", "tiny_debug", "--dataset", str(self.data), "--max-steps", "60", "--checkpoint-interval", "1000",
                  "--eval-interval", "0", "--log-interval", "0", "--seed", "4"]
        whole = self.tmp / "whole"
        p = cli_subprocess([*common, "--output", str(whole)])
        self.assertEqual(p.returncode, 0, p.stderr)
        split = self.tmp / "split"
        p1 = cli_subprocess([*common, "--output", str(split), "--max-runtime", "3", "--safety-margin", "2.4"])
        self.assertEqual(p1.returncode, 0, p1.stderr)
        s1 = json.loads((split / "training_summary.json").read_text())
        self.assertEqual(s1["stop_reason"], "time_budget")
        self.assertTrue(0 < s1["final_step"] < 60, s1["final_step"])
        p2 = cli_subprocess([*common, "--output", str(split), "--resume", str(split / "checkpoints")])
        self.assertEqual(p2.returncode, 0, p2.stderr)
        s2 = json.loads((split / "training_summary.json").read_text())
        self.assertEqual((s2["final_step"], s2["stage_complete"], s2["initial_step"]), (60, True, s1["final_step"]))
        a, b = load_package(whole / "export"), load_package(split / "export")
        for (n, x), (_, y) in zip(a.model.named_parameters(), b.model.named_parameters()):
            self.assertTrue(np.array_equal(x.data, y.data), n)

    def test_changed_hyperparameter_on_resume_is_rejected(self):
        out = self.tmp / "r1"
        common = ["train", "--config", "tiny_debug", "--dataset", str(self.data), "--max-steps", "40", "--log-interval", "0", "--eval-interval", "0", "--seed", "4"]
        p1 = cli_subprocess([*common, "--output", str(out), "--max-runtime", "3", "--safety-margin", "2.4"])
        self.assertEqual(p1.returncode, 0, p1.stderr)
        p2 = cli_subprocess([*common, "--output", str(out), "--resume", str(out / "checkpoints"), "--learning-rate", "0.02"])
        self.assertEqual(p2.returncode, 1)
        self.assertIn("training configuration differs", p2.stderr)
        self.assertIn("learning_rate", p2.stderr)


if __name__ == "__main__":
    unittest.main()
