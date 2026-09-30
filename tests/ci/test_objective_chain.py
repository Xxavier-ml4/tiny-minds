"""Objective-driven stage chain through the REAL command lines, job by job, the way
.github/workflows/train-50m.yml runs it: each job trains in its own directory
and hands state on only as a stage bundle, which the next job verifies with
`python -m tinymind.ci.stage_io check-incoming` (GitHub outputs included).

stage1 objective not met at its budget (gate_failed) -> next stage REJECTED ->
same stage continued with a larger budget (mode continue) -> objective met ->
parent promotion gate fails: next stage REJECTED, same stage REOPENED ->
promotion gate passes: stage2 init-from, stage1 capabilities re-checked."""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tinymind.ci.stage_io import main as stage_io
from tinymind.cli import main as tinymind

REPO = Path(__file__).resolve().parents[2]
PROFILE = "tiny_mobile"
COMMON = ["--config", PROFILE, "--warmup-steps", "2", "--eval-interval", "3", "--checkpoint-interval", "3",
          "--seed", "0", "--max-steps", "0", "--batch-size", "2", "--max-seq-len", "256"]
TARGET = 2 * 256 * 6  # six optimizer steps: the stage's minimum training chunk


def run(fn, *argv, outputs=None):
    env = {"GITHUB_OUTPUT": str(outputs)} if outputs else {}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = fn([str(a) for a in argv])
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return code, out.getvalue(), err.getvalue()


class TestObjectiveDrivenChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = Path(tempfile.mkdtemp(prefix="tm-chain-"))
        manifest = json.loads((REPO / "datasets/v2/manifest.json").read_text())
        sample = dict(manifest["datasets"][0], shards=[str(REPO / "datasets/v2/samples/stage1_language_sample.txt")],
                      chunk_chars=200, group_chunks=2)
        (cls.w / "manifest.json").write_text(json.dumps({"version": 2, "datasets": [sample]}))
        assert run(tinymind, "data", "prepare-corpus", "--manifest", cls.w / "manifest.json", "--out", cls.w / "corpus",
                   "--validation", "0.1", "--test", "0.1")[0] == 0
        for stage in ("stage1", "stage2"):
            code, _, err = run(tinymind, "data", "build-curriculum-v2", "--stage", stage, "--out", cls.w / stage,
                               "--scale", "0.1", "--corpus", cls.w / "corpus")
            assert code == 0, err
        # thresholds a seconds-long tiny run can reach (the shipped objectives are not modified)
        for name, max_loss in (("hard", -1.0), ("easy", 100.0)):
            d = cls.w / name
            d.mkdir()
            s1 = {"stage": "stage1", "generation": {"max_new_tokens": 12, "prompts": ["The sun rose over the"]},
                  "measurements": [{"metric": "loss.val_loss", "max": max_loss},
                                   {"metric": "loss.text_val_bpb", "max": 100.0},
                                   {"metric": "generation.non_empty_rate", "min": 0.0},
                                   {"metric": "data.natural_train_bytes", "min": 1000}],
                  "regression": {"checks": [{"metric": "generation.mean_repetition", "max_increase": 1.0}]}}
            (d / "stage1.objective.json").write_text(json.dumps(s1))
            (d / "stage2.objective.json").write_text(json.dumps({
                "stage": "stage2", "retain": ["stage1"], "generation": {"max_new_tokens": 12, "prompts": ["The"]},
                "measurements": [{"metric": "loss.val_loss", "max": 100.0}]}))
        for name, mx in (("gates_fail", -1.0), ("gates_pass", 1e9)):
            (cls.w / name).mkdir()
            (cls.w / name / "stage1.gate.json").write_text(json.dumps({
                "expect_stage": "stage1", "require_stage_complete": True,
                "checks": [{"metric": "summary.final_validation.val_loss", "max": mx}]}))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.w, ignore_errors=True)

    def job(self, name, stage, objectives, target, *extra):
        out = self.w / name / "out"
        code, stdout, err = run(tinymind, "train", "--dataset", self.w / stage, "--output", out, "--stage", stage,
                                "--target-tokens", target, "--objective-dir", self.w / objectives, *COMMON, *extra)
        self.assertEqual(code, 0, err[-2000:])
        bundle = self.w / name / "bundle"
        self.assertEqual(run(stage_io, "bundle", "--run", out, "--dest", bundle, "--profile", PROFILE, "--stage", stage)[0], 0)
        return json.loads(stdout), bundle

    def incoming(self, bundle, name, stage, *extra):
        local = self.w / name / "incoming"
        shutil.copytree(bundle, local)  # "download"
        outputs = self.w / name / "github_output.txt"
        outputs.write_text("")
        code, _, err = run(stage_io, "check-incoming", "--dir", local, "--stage", stage, "--profile-config",
                           REPO / "configs" / f"{PROFILE}.yaml", *extra, outputs=outputs)
        values = dict(line.split("=", 1) for line in outputs.read_text().splitlines() if "=" in line)
        return code, values, err

    def test_chain(self):
        # job 1: stage1's objective is not met at its budget -> incomplete, never promoted
        r1, b1 = self.job("j1", "stage1", "hard", TARGET)
        self.assertEqual((r1["stop_reason"], r1["stage_complete"]), ("gate_failed", False))
        code, _, err = self.incoming(b1, "j1x", "stage2", "--gate-dir", self.w / "gates_pass")
        self.assertNotEqual(code, 0)
        self.assertIn("objective must be met", err)

        # job 2: the same stage again -> continue with a larger budget (the workflow's arithmetic)
        code, v2, err = self.incoming(b1, "j2", "stage1", "--gate-dir", self.w / "gates_pass")
        self.assertEqual((code, v2["mode"], v2["reopen"]), (0, "continue", "false"), err)
        r2, b2 = self.job("j2", "stage1", "easy", int(v2["parent_budget_tokens"]) + TARGET // 2,
                          "--continue-stage", v2["checkpoint"])
        self.assertEqual((r2["stop_reason"], r2["stage_complete"]), ("gate_passed", True))
        self.assertGreater(r2["total_steps"], r1["total_steps"])

        # job 3: stage1's promotion gate fails -> stage2 rejected; stage1 re-run is REOPENED and continued
        code, _, err = self.incoming(b2, "j3x", "stage2", "--gate-dir", self.w / "gates_fail")
        self.assertNotEqual(code, 0)
        self.assertIn("promotion gate FAILED", err)
        code, v3, err = self.incoming(b2, "j3", "stage1", "--gate-dir", self.w / "gates_fail")
        self.assertEqual((code, v3["mode"], v3["reopen"]), (0, "continue", "true"), err)
        r3, _ = self.job("j3", "stage1", "easy", int(v3["parent_budget_tokens"]) + TARGET // 2,
                         "--continue-stage", v3["checkpoint"], "--reopen-stage")
        self.assertTrue(r3["stage_complete"])
        self.assertEqual(r3["final_step"], r3["total_steps"])  # a reopened stage trains its whole extension

        # job 4: promotion gate passes -> stage2 starts from stage1 and re-checks stage1's capabilities
        code, v4, err = self.incoming(b2, "j4", "stage2", "--gate-dir", self.w / "gates_pass")
        self.assertEqual((code, v4["mode"], v4["gate"]), (0, "init-from", "passed"), err)
        r4, b4 = self.job("j4", "stage2", "easy", TARGET, "--init-from", v4["checkpoint"])
        report = json.loads((b4 / "objective_reports" / "latest.json").read_text())
        self.assertIn("stage1", report["retained"])
        self.assertTrue(any(c["scope"].startswith("retained stage1") for c in report["regression"]["checks"]))
        summary = json.loads((b4 / "training_summary.json").read_text())
        self.assertEqual(summary["parent"]["stage"], "stage1")
        self.assertTrue((b4 / "data_provenance" / "corpus_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
