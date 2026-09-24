"""The runner-hardware benchmark script: a short, bounded real-training run,
not a training run in its own right. Checks structure/consistency of what it
reports; does not assert on absolute timing (that varies by machine)."""
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

import runner_benchmark  # noqa: E402

from tests.training._helpers import tmpdir


class TestRunBenchmarkDirectly(unittest.TestCase):
    """Calls run_benchmark() in-process (fast) against tiny_debug."""

    @classmethod
    def setUpClass(cls):
        cls.work = tmpdir()
        cls.result = runner_benchmark.run_benchmark(profile="tiny_debug", steps=6, batch=4, seq=None, seed=0,
                                                     data_scale=0.02, work_dir=cls.work / "run")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_required_top_level_fields(self):
        for key in ("profile", "hardware", "model", "training", "memory", "checkpoint", "estimated_tokens_per_hour"):
            self.assertIn(key, self.result)

    def test_hardware_fields(self):
        for key in ("cpu", "cpu_count", "allowed_cores", "ram_mb", "blas", "threads_env"):
            self.assertIn(key, self.result["hardware"])

    def test_model_fields_match_the_real_profile(self):
        from tinymind.model.config import ModelConfig, count_parameters
        cfg = ModelConfig.from_yaml(REPO / "configs" / "tiny_debug.yaml")
        self.assertEqual(self.result["model"]["parameters"], count_parameters(cfg))
        self.assertEqual(self.result["model"]["hidden_size"], cfg.hidden_size)

    def test_training_fields_and_consistency(self):
        t = self.result["training"]
        for key in ("batch_size", "sequence_length", "steps_requested", "steps_completed", "wall_seconds",
                   "train_seconds", "steps_per_sec", "tokens_per_sec", "tokens_processed"):
            self.assertIn(key, t)
        self.assertEqual(t["batch_size"], 4)
        self.assertEqual(t["steps_requested"], 6)
        self.assertEqual(t["steps_completed"], 6)  # a bounded benchmark always completes its requested steps
        self.assertGreater(t["tokens_per_sec"], 0)
        self.assertGreater(t["wall_seconds"], 0)

    def test_estimated_tokens_per_hour_is_exactly_tokens_per_sec_times_3600(self):
        self.assertAlmostEqual(self.result["estimated_tokens_per_hour"],
                               round(self.result["training"]["tokens_per_sec"] * 3600), delta=1)

    def test_checkpoint_sizes_are_real_files_not_estimates(self):
        ck = self.result["checkpoint"]
        d = Path(ck["directory"])
        self.assertTrue(d.is_dir())
        self.assertEqual(ck["model_weights_bytes"], (d / "model.npz").stat().st_size)
        self.assertEqual(ck["optimizer_state_bytes"], (d / "optimizer.npz").stat().st_size)
        self.assertEqual(ck["checkpoint_total_bytes"], sum(f.stat().st_size for f in d.iterdir() if f.is_file()))
        # AdamW keeps first and second moments, each the same size as the weights themselves
        self.assertAlmostEqual(ck["optimizer_state_bytes"] / ck["model_weights_bytes"], 2.0, delta=0.05)

    def test_memory_is_recorded(self):
        self.assertGreater(self.result["memory"]["peak_rss_mb"], 0)

    def test_result_is_json_serialisable(self):
        json.dumps(self.result)

    def test_rejects_a_sequence_length_longer_than_the_profile_supports(self):
        with self.assertRaises(ValueError):
            runner_benchmark.run_benchmark(profile="tiny_debug", steps=2, batch=2, seq=99999, seed=0,
                                           data_scale=0.02, work_dir=tmpdir() / "run")


class TestCLI(unittest.TestCase):
    def test_writes_the_json_file_and_prints_it(self):
        work = tmpdir()
        out = work / "bench.json"
        proc = subprocess.run([sys.executable, str(REPO / "benchmarks" / "runner_benchmark.py"), "--profile", "tiny_debug",
                              "--steps", "4", "--batch", "2", "--data-scale", "0.02", "--out", str(out),
                              "--work-dir", str(work / "run")],
                             cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(out.is_file())
        data = json.loads(out.read_text())
        self.assertEqual(data["training"]["steps_completed"], 4)
        printed = json.loads(proc.stdout)
        self.assertEqual(printed["training"]["steps_completed"], 4)


if __name__ == "__main__":
    unittest.main()
