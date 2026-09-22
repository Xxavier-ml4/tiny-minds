import unittest

import numpy as np

from tinymind.model.norm import RMSNorm
from tinymind.model.tensor import Tensor
from tests.model.test_tensor import assert_gradients_close, numerical_gradient


class TestRMSNorm(unittest.TestCase):
    def test_output_shape_matches_input(self):
        norm = RMSNorm(hidden_size=8)
        x = Tensor(np.random.randn(2, 5, 8))
        out = norm(x)
        self.assertEqual(out.shape, (2, 5, 8))

    def test_unit_weight_normalizes_rms_to_one(self):
        norm = RMSNorm(hidden_size=8, eps=0.0)
        x = Tensor(np.random.randn(3, 8) * 10)  # large scale
        out = norm(x)
        rms = np.sqrt((out.data ** 2).mean(axis=-1))
        self.assertTrue(np.allclose(rms, 1.0, atol=1e-4))

    def test_configurable_epsilon_prevents_division_by_zero(self):
        norm = RMSNorm(hidden_size=4, eps=1e-6)
        x = Tensor(np.zeros((1, 4)))  # all-zero input: rms would be 0 without eps
        out = norm(x)
        self.assertTrue(np.all(np.isfinite(out.data)))

    def test_gradient_matches_numerical(self):
        hidden_size = 6
        x_data = np.random.randn(2, hidden_size).astype(np.float64)
        norm = RMSNorm(hidden_size=hidden_size)
        weight_data = norm.weight.data.astype(np.float64)

        def fn(x_):
            n = RMSNorm(hidden_size=hidden_size)
            n.weight.data = weight_data.astype(np.float32)
            return float(n(Tensor(x_.astype(np.float32))).sum().data)

        numeric = numerical_gradient(fn, [x_data])
        x = Tensor(x_data, requires_grad=True)
        norm(x).sum().backward()
        assert_gradients_close(self, x.grad, numeric[0], "rmsnorm")

    def test_weight_gradient_exists_and_finite(self):
        norm = RMSNorm(hidden_size=6)
        x = Tensor(np.random.randn(2, 6), requires_grad=True)
        norm(x).sum().backward()
        self.assertIsNotNone(norm.weight.grad)
        self.assertTrue(np.all(np.isfinite(norm.weight.grad)))


if __name__ == "__main__":
    unittest.main()
