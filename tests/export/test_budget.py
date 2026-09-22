"""The mobile budget formulas match the objects they describe."""
import unittest
from pathlib import Path

import numpy as np

from tinymind.export.budget import budget, kv_cache_bytes, weight_bytes_fp32, weight_bytes_int8
from tinymind.model import ModelConfig, TinyMindTransformer
from tinymind.model.model import KVCache
from tinymind.quantization.model_quantizer import quantize_model

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def small(kv=2, tie=True, T=64):
    return ModelConfig(hidden_size=32, num_layers=3, num_heads=4, num_kv_heads=kv, intermediate_size=64, max_seq_len=T,
                       vocab_size=50, tie_embeddings=tie)


class TestBudget(unittest.TestCase):
    def test_fp32_bytes_equal_the_real_weights(self):
        for tie in (True, False):
            c = small(tie=tie)
            m = TinyMindTransformer(c, seed=0)
            self.assertEqual(weight_bytes_fp32(c), sum(p.data.nbytes for p in m.parameters()))

    def test_kv_cache_bytes_equal_the_allocated_arrays_at_full_length(self):
        for kv in (1, 2, 4):
            c = small(kv=kv, T=64)
            cache = KVCache(c, batch_size=1)
            self.assertEqual(kv_cache_bytes(c, 64), cache._keys.nbytes + cache._values.nbytes)
            self.assertEqual(kv_cache_bytes(c, 32) * 2, kv_cache_bytes(c, 64))

    def test_int8_bytes_equal_what_the_quantizer_produces(self):
        c = small()
        m = TinyMindTransformer(c, seed=0)
        for embeddings in (True, False):
            q = quantize_model(m, "int8", quantize_embeddings=embeddings)
            actual = sum(len(t.quantized_bytes) + 4 * len(t.metadata["scales"]) for t in q.quantized.values()) \
                + sum(a.nbytes for a in q.unquantized.values())
            self.assertEqual(weight_bytes_int8(c, embeddings), actual, embeddings)
        self.assertNotIn("embed_tokens", quantize_model(m, "int8", quantize_embeddings=False).quantized)
        self.assertIn("embed_tokens", quantize_model(m, "int8").quantized)  # Phase 3A default unchanged

    def test_int8_is_roughly_a_quarter_and_quantized_model_still_runs(self):
        c = ModelConfig.from_yaml(CONFIGS / "tiny_mobile.yaml")
        self.assertLess(weight_bytes_int8(c, False), 0.30 * weight_bytes_fp32(c))
        m = TinyMindTransformer(small(), seed=0)
        from tinymind.quantization.model_quantizer import dequantize_to_model
        deq = dequantize_to_model(quantize_model(m, "int8", quantize_embeddings=False))
        ids = np.array([[1, 5, 9, 12]])
        a, b = m(ids).logits.data, deq(ids).logits.data
        self.assertLess(float(np.abs(a - b).max()), 0.1 * float(np.abs(a).max()) + 1e-3)

    def test_budget_report_for_the_profiles(self):
        for name, kv_256 in (("tiny_mobile", 786432), ("tiny_debug", None)):
            c = ModelConfig.from_yaml(CONFIGS / f"{name}.yaml")
            b = budget(c)
            self.assertEqual(b["kv_cache"]["256"]["bytes"], kv_cache_bytes(c, 256))
            if kv_256:
                self.assertEqual(b["kv_cache"]["256"]["bytes"], kv_256)  # 2 x 6 layers x 2 kv heads x 32 x 256 x 4 B
            self.assertEqual(b["kv_cache"]["128"]["bytes"] * 2, b["kv_cache"]["256"]["bytes"])
            self.assertGreater(b["runtime_ram_estimate"]["256"]["fp32_MiB"], b["fp32_weight_MiB"])


if __name__ == "__main__":
    unittest.main()
