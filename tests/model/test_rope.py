import unittest

import numpy as np

from tinymind.model.positional import apply_rotary_pos_emb, precompute_rope_cache
from tinymind.model.tensor import Tensor


class TestRopeCache(unittest.TestCase):
    def test_cache_shape(self):
        cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=16, theta=10000.0)
        self.assertEqual(cos.shape, (16, 8))
        self.assertEqual(sin.shape, (16, 8))

    def test_odd_head_dim_rejected(self):
        with self.assertRaises(ValueError):
            precompute_rope_cache(head_dim=7, max_seq_len=16)

    def test_position_zero_is_identity_rotation(self):
        cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=16)
        self.assertTrue(np.allclose(cos[0], 1.0))
        self.assertTrue(np.allclose(sin[0], 0.0))

    def test_deterministic(self):
        cos1, sin1 = precompute_rope_cache(head_dim=8, max_seq_len=16, theta=10000.0)
        cos2, sin2 = precompute_rope_cache(head_dim=8, max_seq_len=16, theta=10000.0)
        self.assertTrue(np.array_equal(cos1, cos2))
        self.assertTrue(np.array_equal(sin1, sin2))

    def test_different_theta_gives_different_cache(self):
        cos1, _ = precompute_rope_cache(head_dim=8, max_seq_len=16, theta=10000.0)
        cos2, _ = precompute_rope_cache(head_dim=8, max_seq_len=16, theta=500.0)
        self.assertFalse(np.allclose(cos1, cos2))


class TestApplyRope(unittest.TestCase):
    def test_output_shape_preserved(self):
        cos, sin = precompute_rope_cache(head_dim=4, max_seq_len=10)
        x = Tensor(np.random.randn(2, 3, 5, 4))  # [B, H, T, D]
        out = apply_rotary_pos_emb(x, cos, sin)
        self.assertEqual(out.shape, (2, 3, 5, 4))

    def test_preserves_vector_norm(self):
        # RoPE is a rotation: it must not change each position's vector norm.
        cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=10)
        x = Tensor(np.random.randn(1, 1, 5, 8))
        out = apply_rotary_pos_emb(x, cos, sin)
        norms_before = np.linalg.norm(x.data, axis=-1)
        norms_after = np.linalg.norm(out.data, axis=-1)
        self.assertTrue(np.allclose(norms_before, norms_after, atol=1e-4))

    def test_position_sensitivity_different_positions_rotate_differently(self):
        cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=10)
        same_vector_at_two_positions = Tensor(np.tile(np.random.randn(1, 1, 1, 8), (1, 1, 2, 1)))
        out = apply_rotary_pos_emb(same_vector_at_two_positions, cos, sin)
        self.assertFalse(np.allclose(out.data[0, 0, 0], out.data[0, 0, 1]))

    def test_position_ids_override_default_positions(self):
        cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=20)
        x = Tensor(np.random.randn(1, 1, 1, 8))
        out_default = apply_rotary_pos_emb(x, cos, sin)  # position 0
        out_pos5 = apply_rotary_pos_emb(x, cos, sin, position_ids=np.array([[5]]))
        self.assertFalse(np.allclose(out_default.data, out_pos5.data))

    def test_gradient_flows_through_rope(self):
        cos, sin = precompute_rope_cache(head_dim=4, max_seq_len=10)
        x = Tensor(np.random.randn(1, 1, 3, 4), requires_grad=True)
        out = apply_rotary_pos_emb(x, cos, sin)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(np.all(np.isfinite(x.grad)))

    def test_numerical_stability_at_long_sequence_length(self):
        cos, sin = precompute_rope_cache(head_dim=16, max_seq_len=2048, theta=10000.0)
        self.assertTrue(np.all(np.isfinite(cos)))
        self.assertTrue(np.all(np.isfinite(sin)))
        x = Tensor(np.random.randn(1, 2, 2048, 16))
        out = apply_rotary_pos_emb(x, cos, sin)
        self.assertTrue(np.all(np.isfinite(out.data)))


if __name__ == "__main__":
    unittest.main()
