import unittest

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.generation import ModelGenerationConfig, generate, generate_with_cache_ids
from tinymind.model.model import TinyMindTransformer
from tinymind.model.tokenizer import ByteTokenizer


def _tiny_model(seed=0, vocab_size=20, max_seq_len=32):
    config = ModelConfig(hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2,
                         intermediate_size=32, max_seq_len=max_seq_len, vocab_size=vocab_size)
    return TinyMindTransformer(config, seed=seed)


class TestGreedyGeneration(unittest.TestCase):
    def test_deterministic_by_default(self):
        model = _tiny_model()
        prompt = np.array([[1, 2, 3]])
        config = ModelGenerationConfig(max_new_tokens=5, do_sample=False)
        out1 = generate_with_cache_ids(model, prompt, config)
        out2 = generate_with_cache_ids(model, prompt, config)
        self.assertTrue(np.array_equal(out1, out2))

    def test_default_do_sample_is_false(self):
        self.assertFalse(ModelGenerationConfig().do_sample)

    def test_generates_requested_number_of_tokens_absent_eos(self):
        model = _tiny_model()
        prompt = np.array([[1, 2]])
        config = ModelGenerationConfig(max_new_tokens=6, do_sample=False, eos_token_id=None)
        out = generate_with_cache_ids(model, prompt, config)
        self.assertEqual(out.shape[1], len(prompt[0]) + 6)

    def test_stops_at_eos(self):
        # Force eos to be whatever the model greedily picks first, then
        # confirm generation stops there rather than continuing.
        model = _tiny_model()
        prompt = np.array([[1, 2]])
        probe_config = ModelGenerationConfig(max_new_tokens=1, do_sample=False, eos_token_id=None)
        probe = generate_with_cache_ids(model, prompt, probe_config)
        first_generated_token = int(probe[0, -1])

        config = ModelGenerationConfig(max_new_tokens=10, do_sample=False, eos_token_id=first_generated_token)
        out = generate_with_cache_ids(model, prompt, config)
        self.assertEqual(out.shape[1], len(prompt[0]) + 1)  # stopped right after hitting eos

    def test_max_new_tokens_clamped_to_context_window_not_an_error(self):
        model = _tiny_model(max_seq_len=10)
        prompt = np.array([[1, 2, 3]])
        config = ModelGenerationConfig(max_new_tokens=1000, do_sample=False, eos_token_id=None)
        out = generate_with_cache_ids(model, prompt, config)  # must not raise
        self.assertLessEqual(out.shape[1], model.config.max_seq_len)

    def test_prompt_longer_than_max_seq_len_raises_clearly(self):
        model = _tiny_model(max_seq_len=4)
        prompt = np.array([[1, 2, 3, 4, 5, 6]])  # longer than max_seq_len
        with self.assertRaises(ValueError):
            generate_with_cache_ids(model, prompt, ModelGenerationConfig(max_new_tokens=1))


class TestSamplingGeneration(unittest.TestCase):
    def test_seed_reproducibility(self):
        model = _tiny_model()
        prompt = np.array([[1, 2, 3]])
        config = ModelGenerationConfig(max_new_tokens=8, do_sample=True, temperature=1.0, seed=42)
        out1 = generate_with_cache_ids(model, prompt, config)
        out2 = generate_with_cache_ids(model, prompt, config)
        self.assertTrue(np.array_equal(out1, out2))

    def test_different_seeds_can_diverge(self):
        model = _tiny_model()
        prompt = np.array([[1, 2, 3]])
        out1 = generate_with_cache_ids(model, prompt,
            ModelGenerationConfig(max_new_tokens=8, do_sample=True, temperature=2.0, seed=1))
        out2 = generate_with_cache_ids(model, prompt,
            ModelGenerationConfig(max_new_tokens=8, do_sample=True, temperature=2.0, seed=2))
        self.assertFalse(np.array_equal(out1, out2))

    def test_temperature_top_k_top_p_do_not_crash(self):
        model = _tiny_model()
        prompt = np.array([[1, 2, 3]])
        for cfg in [
            ModelGenerationConfig(max_new_tokens=4, do_sample=True, temperature=0.7),
            ModelGenerationConfig(max_new_tokens=4, do_sample=True, temperature=1.0, top_k=5),
            ModelGenerationConfig(max_new_tokens=4, do_sample=True, temperature=1.0, top_p=0.9),
        ]:
            out = generate_with_cache_ids(model, prompt, cfg)
            self.assertEqual(out.shape[1], 3 + 4)


class TestTextGeneration(unittest.TestCase):
    def test_generate_text_round_trips_through_tokenizer(self):
        tok = ByteTokenizer()
        model = _tiny_model(vocab_size=tok.vocab_size, max_seq_len=32)
        text = generate(model, tok, "hi", ModelGenerationConfig(max_new_tokens=5, do_sample=False))
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
