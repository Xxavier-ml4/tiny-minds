import unittest

from tinymind.model import ModelConfig
from tinymind.model.config import count_parameters, load_preset


class Test50MProfile(unittest.TestCase):
    """Brief section 1: the 50m profile is EXACTLY 50,370,624 parameters and has
    the geometry the profile fixes."""

    EXPECT = 50_370_624

    def setUp(self):
        self.config = load_preset("50m")

    def test_exact_parameter_count(self):
        self.assertEqual(count_parameters(self.config), self.EXPECT)
        self.assertEqual(self.config.count_parameters() if hasattr(self.config, "count_parameters")
                         else count_parameters(self.config), self.EXPECT)

    def test_geometry(self):
        c = self.config
        self.assertEqual(c.hidden_size, 576)
        self.assertEqual(c.num_layers, 12)
        self.assertEqual(c.num_heads, 6)
        self.assertEqual(c.num_kv_heads, 2)
        self.assertEqual(c.intermediate_size, 1472)
        self.assertEqual(c.vocab_size, 16000)
        self.assertEqual(c.max_seq_len, 1024)
        self.assertTrue(c.tie_embeddings)

    def test_max_seq_len_does_not_change_count(self):
        # RoPE has no learned positional table, so context length is free.
        base = dict(vocab_size=16000, hidden_size=576, num_layers=12, num_heads=6,
                    num_kv_heads=2, intermediate_size=1472, tie_embeddings=True)
        a = count_parameters(ModelConfig(max_seq_len=1024, **base))
        b = count_parameters(ModelConfig(max_seq_len=4096, **base))
        self.assertEqual(a, b)
        self.assertEqual(a, self.EXPECT)

    def test_breakdown_adds_up(self):
        c = self.config
        embed = c.vocab_size * c.hidden_size
        self.assertEqual(embed, 9_216_000)
        total = count_parameters(c)
        # tied embeddings: the 9.216M embedding is counted once; the rest is blocks + final norm.
        self.assertEqual(total, self.EXPECT)
        self.assertGreater(total - embed, 40_000_000)


if __name__ == "__main__":
    unittest.main()
