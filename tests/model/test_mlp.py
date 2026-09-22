import unittest

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.mlp import SwiGLUMLP
from tinymind.model.tensor import Tensor


def _config(**overrides):
    defaults = dict(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=4,
                    intermediate_size=32, max_seq_len=16, vocab_size=10)
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestSwiGLUMLP(unittest.TestCase):
    def test_output_shape_matches_input(self):
        mlp = SwiGLUMLP(_config())
        x = Tensor(np.random.randn(2, 5, 16))
        out = mlp(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_deterministic_given_same_weights(self):
        mlp = SwiGLUMLP(_config())
        x = Tensor(np.random.randn(2, 5, 16))
        out1 = mlp(x).data.copy()
        out2 = mlp(x).data.copy()
        self.assertTrue(np.array_equal(out1, out2))

    def test_configurable_intermediate_size(self):
        mlp = SwiGLUMLP(_config(intermediate_size=64))
        self.assertEqual(mlp.gate_proj.weight.shape, (64, 16))
        self.assertEqual(mlp.down_proj.weight.shape, (16, 64))

    def test_gradients_exist_and_finite(self):
        mlp = SwiGLUMLP(_config())
        x = Tensor(np.random.randn(2, 4, 16), requires_grad=True)
        mlp(x).sum().backward()
        for name, param in mlp.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} missing gradient")
            self.assertTrue(np.all(np.isfinite(param.grad)), f"{name} non-finite gradient")
        self.assertTrue(np.all(np.isfinite(x.grad)))

    def test_numerical_gradient_check(self):
        from tests.model.test_tensor import assert_gradients_close, numerical_gradient
        mlp = SwiGLUMLP(_config(hidden_size=4, intermediate_size=6))
        x_data = np.random.randn(2, 4).astype(np.float64)

        gate_w = mlp.gate_proj.weight.data.astype(np.float64)
        up_w = mlp.up_proj.weight.data.astype(np.float64)
        down_w = mlp.down_proj.weight.data.astype(np.float64)

        def fn(x_):
            mlp.gate_proj.weight.data = gate_w.astype(np.float32)
            mlp.up_proj.weight.data = up_w.astype(np.float32)
            mlp.down_proj.weight.data = down_w.astype(np.float32)
            return float(mlp(Tensor(x_.astype(np.float32))).sum().data)

        numeric = numerical_gradient(fn, [x_data])
        x = Tensor(x_data, requires_grad=True)
        mlp(x).sum().backward()
        assert_gradients_close(self, x.grad, numeric[0], "swiglu_mlp")


if __name__ == "__main__":
    unittest.main()
