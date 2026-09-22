import unittest

import numpy as np

from tinymind.model.loss import causal_lm_loss
from tinymind.model.tensor import Tensor


class TestCausalLMLoss(unittest.TestCase):
    def test_shift_by_one(self):
        # Craft logits that perfectly predict input_ids[1:] at every
        # position (huge logit on the correct next token) — loss should be
        # very small, confirming the shift direction is correct (predict
        # t+1 from t, not t from t+1).
        vocab_size = 5
        input_ids = np.array([[0, 1, 2, 3]])
        logits_data = np.full((1, 4, vocab_size), -10.0, dtype=np.float32)
        for t in range(3):  # positions 0,1,2 predict input_ids[1],[2],[3]
            logits_data[0, t, input_ids[0, t + 1]] = 10.0
        logits = Tensor(logits_data)
        loss = causal_lm_loss(logits, input_ids)
        self.assertLess(loss.item(), 0.01)

    def test_wrong_shift_direction_gives_high_loss(self):
        vocab_size = 5
        input_ids = np.array([[0, 1, 2, 3]])
        logits_data = np.full((1, 4, vocab_size), -10.0, dtype=np.float32)
        for t in range(3):  # predicting input_ids[t] instead of input_ids[t+1] -- wrong shift
            logits_data[0, t, input_ids[0, t]] = 10.0
        logits = Tensor(logits_data)
        loss = causal_lm_loss(logits, input_ids)
        self.assertGreater(loss.item(), 5.0)

    def test_padding_mask_excludes_padded_targets(self):
        input_ids = np.array([[1, 2, 3, 0]])  # last token is padding
        attention_mask = np.array([[1, 1, 1, 0]])
        logits = Tensor(np.random.randn(1, 4, 10))
        loss_masked = causal_lm_loss(logits, input_ids, attention_mask)
        loss_unmasked = causal_lm_loss(logits, input_ids, attention_mask=None)
        # They should generally differ since one excludes a target the other doesn't.
        self.assertNotAlmostEqual(loss_masked.item(), loss_unmasked.item(), places=3)

    def test_gradient_flows_to_logits(self):
        logits = Tensor(np.random.randn(2, 5, 8), requires_grad=True)
        input_ids = np.random.randint(0, 8, size=(2, 5))
        loss = causal_lm_loss(logits, input_ids)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(np.all(np.isfinite(logits.grad)))

    def test_loss_is_scalar(self):
        logits = Tensor(np.random.randn(3, 6, 4))
        input_ids = np.random.randint(0, 4, size=(3, 6))
        loss = causal_lm_loss(logits, input_ids)
        self.assertEqual(loss.data.size, 1)


if __name__ == "__main__":
    unittest.main()
