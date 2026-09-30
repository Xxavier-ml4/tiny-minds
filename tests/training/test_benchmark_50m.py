import unittest

from tinymind.model.config import load_preset
from tinymind.training.benchmark_50m import measure_shape, run


class TestBenchmark50M(unittest.TestCase):
    """Brief section 7: the 50M memory/throughput preflight builds the real model
    and optimizer and produces real per-shape measurements. Small shapes are used
    so the test is quick; the measured numbers are genuine, not mocked."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_preset("50m")

    def test_single_shape_measurement(self):
        m = measure_shape(self.config, batch_size=1, gradient_accumulation=1, seq_len=64, seed=0, warmup=False)
        self.assertEqual(m.seq_len, 64)
        self.assertEqual(m.effective_batch_tokens, 64)
        self.assertGreater(m.forward_seconds, 0.0)
        self.assertGreater(m.backward_seconds, 0.0)
        self.assertGreater(m.optimizer_seconds, 0.0)
        self.assertGreater(m.tokens_per_sec, 0.0)
        self.assertGreater(m.peak_rss_mb, 0.0)

    def test_run_reports_exact_param_count_and_passes(self):
        result = run(self.config, shapes=((1, 2),), seq_len=64, seed=0, warmup=False)
        self.assertEqual(result["parameter_count"], 50_370_624)
        self.assertTrue(result["passed"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["shapes"]), 1)
        self.assertGreater(result["model_init_seconds"], 0.0)

    def test_seq_len_cannot_exceed_max(self):
        with self.assertRaises(ValueError):
            measure_shape(self.config, 1, 1, seq_len=self.config.max_seq_len + 1)


if __name__ == "__main__":
    unittest.main()
