"""Model-size accounting (Phase 3B, phase F): the analytic parameter count
must equal what the instantiated model really has, for every attention
layout, and the shipped mobile profiles must respect the size budget."""
import itertools
import re
import unittest
from pathlib import Path

from tinymind.model import ModelConfig, TinyMindTransformer
from tinymind.model.config import count_parameters, parameter_shapes

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def cfg(h, L, H, kv, I, V, tie, T=32):
    return ModelConfig(hidden_size=h, num_layers=L, num_heads=H, num_kv_heads=kv, intermediate_size=I, max_seq_len=T,
                       vocab_size=V, tie_embeddings=tie)


class TestCountMatchesTheModel(unittest.TestCase):
    def test_grid_of_layouts(self):
        shapes = [(16, 1, 4, 4, 32, 20), (32, 2, 4, 2, 64, 260), (48, 3, 6, 1, 96, 100), (24, 2, 6, 3, 72, 50), (64, 2, 8, 2, 128, 300)]
        for (h, L, H, kv, I, V), tie in itertools.product(shapes, (True, False)):
            c = cfg(h, L, H, kv, I, V, tie)
            model = TinyMindTransformer(c, seed=0)
            self.assertEqual(count_parameters(c), model.count_parameters(), (h, L, H, kv, I, V, tie))
            self.assertEqual(c.count_parameters(), count_parameters(c))

    def test_shapes_names_and_order_match_named_parameters(self):
        for tie in (True, False):
            c = cfg(32, 2, 4, 2, 64, 40, tie)
            model = TinyMindTransformer(c, seed=0)
            actual = [(n, tuple(p.data.shape)) for n, p in model.named_parameters()]
            self.assertEqual(actual, list(parameter_shapes(c).items()))

    def test_phase3a_estimate_bugs_are_gone(self):
        c = cfg(128, 6, 4, 2, 384, 260, False, T=256)
        self.assertEqual(c.approx_param_count, count_parameters(c))       # untied head is now counted
        self.assertGreater(count_parameters(c), count_parameters(cfg(128, 6, 4, 2, 384, 260, True, T=256)))
        self.assertEqual(count_parameters(c) - count_parameters(cfg(128, 6, 4, 2, 384, 260, True, T=256)), 260 * 128)

    def test_the_rejected_32k_vocabulary_example(self):
        c = cfg(192, 6, 6, 2, 512, 32000, True, T=256)
        self.assertEqual(c.vocab_size * c.hidden_size, 6_144_000)
        self.assertGreater(count_parameters(c), 5_000_000)  # exceeds the mobile budget on its own


class TestProfiles(unittest.TestCase):
    def load(self, name):
        return ModelConfig.from_yaml(CONFIGS / f"{name}.yaml")

    def test_budget(self):
        debug, mobile, plus = (count_parameters(self.load(n)) for n in ("tiny_debug", "tiny_mobile", "tiny_mobile_plus"))
        self.assertTrue(debug < mobile < plus < 5_000_000)
        self.assertGreaterEqual(mobile, 1_000_000)
        self.assertLess(mobile, 3_000_000)  # preferred first mobile candidate: 1M-3M
        self.assertLess(debug, 500_000)

    def test_ranges_requested_for_each_profile(self):
        d, m, p = (self.load(n) for n in ("tiny_debug", "tiny_mobile", "tiny_mobile_plus"))
        self.assertTrue(64 <= d.hidden_size <= 96 and 2 <= d.num_layers <= 4 and 128 <= d.max_seq_len <= 256)
        self.assertTrue(96 <= m.hidden_size <= 160 and 4 <= m.num_layers <= 8 and m.max_seq_len == 256)
        self.assertTrue(160 <= p.hidden_size <= 224 and 6 <= p.num_layers <= 10 and 256 <= p.max_seq_len <= 512)
        for c in (d, m, p):
            self.assertIn(c.num_kv_heads, (1, 2))
            self.assertEqual(c.hidden_size % c.num_heads, 0)
            self.assertTrue(2 * c.hidden_size <= c.intermediate_size <= 4 * c.hidden_size)
            self.assertEqual(c.vocab_size, 260)  # byte tokenizer: vocabulary is part of the budget

    def test_header_comment_states_the_exact_count(self):
        for name in ("tiny_debug", "tiny_mobile", "tiny_mobile_plus"):
            first = (CONFIGS / f"{name}.yaml").read_text().splitlines()[0]
            stated = int(re.search(r"([\d,]+) parameters", first).group(1).replace(",", ""))
            self.assertEqual(stated, count_parameters(self.load(name)), name)

    def test_profiles_build_and_train_configs_load(self):
        from tinymind.training.config import TrainingConfig
        for name in ("tiny_debug", "tiny_mobile", "tiny_mobile_plus"):
            c = self.load(name)
            c.require_supported()
            t = TrainingConfig.from_yaml(CONFIGS / f"{name}.yaml")
            self.assertLessEqual(t.max_seq_len, c.max_seq_len)
        TinyMindTransformer(self.load("tiny_debug"), seed=0)


if __name__ == "__main__":
    unittest.main()
