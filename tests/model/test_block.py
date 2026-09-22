import unittest

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.layers import TransformerBlock
from tinymind.model.tensor import Tensor


def _config(**overrides):
    defaults = dict(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2,
                    intermediate_size=32, max_seq_len=16, vocab_size=10)
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestTransformerBlock(unittest.TestCase):
    def test_output_shape_matches_input(self):
        block = TransformerBlock(_config())
        x = Tensor(np.random.randn(2, 5, 16))
        out = block(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_residual_connections_present(self):
        # A block of all-zero-initialized weights should be the identity
        # (attention/mlp outputs are exactly zero, so x + 0 + 0 == x) — a
        # direct, hand-checkable way to confirm both residual adds are
        # actually wired in, not just "shape matches by coincidence".
        block = TransformerBlock(_config())
        for _name, param in block.named_parameters():
            param.data[...] = 0.0
        x = Tensor(np.random.randn(1, 3, 16))
        out = block(x)
        self.assertTrue(np.allclose(out.data, x.data, atol=1e-5))

    def test_gradients_reach_every_sublayer(self):
        block = TransformerBlock(_config())
        x = Tensor(np.random.randn(2, 4, 16), requires_grad=True)
        block(x).sum().backward()
        for name, param in block.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} missing gradient")
            self.assertTrue(np.all(np.isfinite(param.grad)), f"{name} non-finite gradient")

    def test_no_tool_or_runtime_logic_imported(self):
        # Brief section 8/18: the transformer block must stay a pure
        # neural component. A cheap, real guard: the module must not
        # import anything from tinymind.tools or tinymind.runtime.
        import tinymind.model.layers as layers_module
        source = layers_module.__file__
        with open(source) as f:
            content = f.read()
        self.assertNotIn("tinymind.tools", content)
        self.assertNotIn("tinymind.runtime", content)


if __name__ == "__main__":
    unittest.main()
