import unittest

from tinymind.model.backends.echo import EchoBackend
from tinymind.model.config import ModelConfig, ModelConfigError, load_preset
from tinymind.model.tokenizer import ByteTokenizer
from tinymind.runtime.sampling import argmax, select_token, softmax, top_k_filter, top_p_filter


class TestModelConfig(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = ModelConfig()
        self.assertGreater(config.approx_param_count, 0)

    def test_hidden_size_must_divide_by_heads(self):
        with self.assertRaises(ModelConfigError):
            ModelConfig(hidden_size=100, num_heads=7)

    def test_kv_heads_must_divide_heads(self):
        with self.assertRaises(ModelConfigError):
            ModelConfig(hidden_size=512, num_heads=8, num_kv_heads=3)

    def test_all_named_presets_load_and_scale_up(self):
        names = ["50m", "100m", "150m", "300m", "500m", "1b"]
        counts = [load_preset(name).approx_param_count for name in names]
        self.assertEqual(counts, sorted(counts))  # each preset is bigger than the last

    def test_unknown_field_rejected(self):
        with self.assertRaises(ModelConfigError):
            ModelConfig.from_dict({"not_a_real_field": 1})


class TestByteTokenizer(unittest.TestCase):
    def test_round_trip(self):
        tok = ByteTokenizer()
        text = "hello, world! \u00e9\u00e8 unicode too"
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)

    def test_bos_eos(self):
        tok = ByteTokenizer()
        ids = tok.encode("hi", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.bos_token_id)
        self.assertEqual(ids[-1], tok.eos_token_id)

    def test_call_interface_truncates(self):
        tok = ByteTokenizer()
        out = tok(["hello world"], max_length=3)
        self.assertEqual(len(out["input_ids"][0]), 3)


class TestEchoBackend(unittest.TestCase):
    def test_deterministic_generation(self):
        backend = EchoBackend()
        backend.load("dummy")
        r1 = backend.generate("hello there")
        r2 = backend.generate("hello there")
        self.assertEqual(r1.text, r2.text)

    def test_is_not_a_real_model(self):
        backend = EchoBackend()
        self.assertFalse(backend.is_real_model)

    def test_embed_similar_text_closer_than_dissimilar(self):
        import math
        backend = EchoBackend()

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            return dot / ((math.sqrt(sum(x * x for x in a)) or 1) * (math.sqrt(sum(y * y for y in b)) or 1))

        v1 = backend.embed("convert miles to kilometers")
        v2 = backend.embed("convert kilometers to miles")
        v3 = backend.embed("what is the weather today")
        self.assertGreater(cosine(v1, v2), cosine(v1, v3))


class TestSampling(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        probs = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_argmax(self):
        self.assertEqual(argmax([1.0, 5.0, 2.0]), 1)

    def test_top_k_filter_zeroes_rest(self):
        filtered = top_k_filter([0.1, 0.4, 0.3, 0.2], k=2)
        self.assertEqual(sum(1 for p in filtered if p == 0.0), 2)
        self.assertAlmostEqual(sum(filtered), 1.0, places=6)

    def test_top_p_filter_keeps_minimal_nucleus(self):
        filtered = top_p_filter([0.5, 0.3, 0.15, 0.05], p=0.8)
        self.assertEqual(filtered[2], 0.0)
        self.assertEqual(filtered[3], 0.0)

    def test_greedy_is_deterministic(self):
        self.assertEqual(select_token([1.0, 5.0, 1.0], temperature=0.0), 1)


if __name__ == "__main__":
    unittest.main()
