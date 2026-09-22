import unittest

from tinymind.model.config import ModelConfig, ModelConfigError


class TestModelConfigPhase3AFields(unittest.TestCase):
    def test_new_fields_have_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.norm_epsilon, 1e-6)
        self.assertEqual(config.dropout, 0.0)
        self.assertEqual(config.dtype, "float32")

    def test_invalid_dtype_rejected(self):
        with self.assertRaises(ModelConfigError):
            ModelConfig(dtype="int4")

    def test_invalid_dropout_rejected(self):
        with self.assertRaises(ModelConfigError):
            ModelConfig(dropout=1.5)
        with self.assertRaises(ModelConfigError):
            ModelConfig(dropout=-0.1)

    def test_invalid_norm_epsilon_rejected(self):
        with self.assertRaises(ModelConfigError):
            ModelConfig(norm_epsilon=0.0)

    def test_still_backward_compatible_with_phase1_fields(self):
        # Every Phase 1 field must still construct and round-trip.
        config = ModelConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
                             intermediate_size=128, max_seq_len=64, vocab_size=100)
        data = config.to_dict()
        restored = ModelConfig.from_dict(data)
        self.assertEqual(config, restored)


if __name__ == "__main__":
    unittest.main()
