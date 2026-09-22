import unittest

import numpy as np

from tinymind.model.attention import CausalSelfAttention, _repeat_kv
from tinymind.model.config import ModelConfig
from tinymind.model.model import KVCache
from tinymind.model.tensor import Tensor


def _config(**overrides):
    defaults = dict(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=4,
                    intermediate_size=32, max_seq_len=16, vocab_size=10)
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestCausalSelfAttentionShapes(unittest.TestCase):
    def test_output_shape_matches_input(self):
        attn = CausalSelfAttention(_config())
        x = Tensor(np.random.randn(2, 5, 16))
        out = attn(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_gqa_shape(self):
        attn = CausalSelfAttention(_config(num_heads=4, num_kv_heads=2))
        x = Tensor(np.random.randn(2, 5, 16))
        out = attn(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_mqa_shape(self):
        attn = CausalSelfAttention(_config(num_heads=4, num_kv_heads=1))
        x = Tensor(np.random.randn(2, 5, 16))
        out = attn(x)
        self.assertEqual(out.shape, (2, 5, 16))


class TestRepeatKV(unittest.TestCase):
    def test_no_op_when_n_rep_is_one(self):
        x = Tensor(np.random.randn(1, 4, 3, 2))
        out = _repeat_kv(x, 1)
        self.assertIs(out, x)

    def test_repeats_each_head_contiguously(self):
        x = Tensor(np.arange(2 * 3 * 4).reshape(1, 2, 3, 4).astype(np.float32))
        out = _repeat_kv(x, 2)
        self.assertEqual(out.shape, (1, 4, 3, 4))
        self.assertTrue(np.array_equal(out.data[0, 0], x.data[0, 0]))
        self.assertTrue(np.array_equal(out.data[0, 1], x.data[0, 0]))  # kv head 0 repeated
        self.assertTrue(np.array_equal(out.data[0, 2], x.data[0, 1]))  # kv head 1 repeated

    def test_gradient_sums_across_repeats(self):
        x = Tensor(np.random.randn(1, 2, 3, 4), requires_grad=True)
        out = _repeat_kv(x, 3)
        out.sum().backward()
        # Each repeated copy contributes 1.0 per element to the sum, so the
        # gradient w.r.t. the original should be exactly n_rep everywhere.
        self.assertTrue(np.allclose(x.grad, 3.0))


class TestCausalMasking(unittest.TestCase):
    def test_causal_mask_blocks_future_positions(self):
        # A hand-checkable property: changing a FUTURE token must not
        # change an EARLIER position's output (that would mean the earlier
        # position attended to the future — a broken causal mask).
        attn = CausalSelfAttention(_config())
        x1 = Tensor(np.random.randn(1, 5, 16))
        x2_data = x1.data.copy()
        x2_data[0, 4, :] = np.random.randn(16)  # change only the LAST position
        x2 = Tensor(x2_data)

        out1 = attn(x1).data
        out2 = attn(x2).data
        # positions 0..3 must be unaffected by changing position 4
        self.assertTrue(np.allclose(out1[0, :4], out2[0, :4], atol=1e-5))
        # position 4 itself should differ (it legitimately sees its own new input)
        self.assertFalse(np.allclose(out1[0, 4], out2[0, 4], atol=1e-5))


class TestAttentionWithKVCache(unittest.TestCase):
    def test_cached_and_uncached_prefill_agree(self):
        config = _config(num_layers=2)
        attn = CausalSelfAttention(config)
        x = Tensor(np.random.randn(1, 4, 16))

        out_no_cache = attn(x, layer_idx=0)

        cache = KVCache(config, batch_size=1)
        out_with_cache = attn(x, kv_cache=cache, layer_idx=0, position_ids=np.arange(4)[None, :])

        self.assertTrue(np.allclose(out_no_cache.data, out_with_cache.data, atol=1e-4))


class TestAttentionGradientFlow(unittest.TestCase):
    def test_gradients_exist_and_finite_for_all_projections(self):
        attn = CausalSelfAttention(_config())
        x = Tensor(np.random.randn(2, 4, 16), requires_grad=True)
        out = attn(x)
        out.sum().backward()
        for name, param in attn.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} has no gradient")
            self.assertTrue(np.all(np.isfinite(param.grad)), f"{name} has non-finite gradient")
        self.assertIsNotNone(x.grad)
        self.assertTrue(np.all(np.isfinite(x.grad)))


if __name__ == "__main__":
    unittest.main()
