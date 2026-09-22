import unittest

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.model import TinyMindTransformer


def _tiny_config(**overrides):
    defaults = dict(hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2,
                    intermediate_size=32, max_seq_len=16, vocab_size=20)
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestModelForward(unittest.TestCase):
    def test_logits_shape(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        out = model(np.array([[1, 2, 3, 4]]))
        self.assertEqual(out.logits.shape, (1, 4, 20))

    def test_no_softmax_applied_logits_not_bounded_0_1(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        out = model(np.array([[1, 2, 3]]))
        # Real logits should routinely fall outside [0, 1] — if they never
        # did, that would suggest a softmax snuck into forward().
        self.assertTrue(np.any(out.logits.data < 0) or np.any(out.logits.data > 1))

    def test_loss_is_none_without_labels(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        out = model(np.array([[1, 2, 3]]))
        self.assertIsNone(out.loss)

    def test_loss_computed_with_labels(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        out = model(np.array([[1, 2, 3]]), labels=np.array([[1, 2, 3]]))
        self.assertIsNotNone(out.loss)
        self.assertTrue(np.isfinite(out.loss.item()))

    def test_rejects_out_of_vocab_token(self):
        model = TinyMindTransformer(_tiny_config(vocab_size=10), seed=0)
        with self.assertRaises(ValueError):
            model(np.array([[1, 2, 999]]))

    def test_tied_embeddings_share_weight_object(self):
        model = TinyMindTransformer(_tiny_config(tie_embeddings=True), seed=0)
        self.assertIsNone(model.lm_head)  # forward() reuses embed_tokens directly

    def test_untied_embeddings_have_separate_head(self):
        model = TinyMindTransformer(_tiny_config(tie_embeddings=False), seed=0)
        self.assertIsNotNone(model.lm_head)
        self.assertFalse(np.array_equal(model.lm_head.weight.data, model.embed_tokens.data))

    def test_deterministic_given_seed(self):
        model1 = TinyMindTransformer(_tiny_config(), seed=123)
        model2 = TinyMindTransformer(_tiny_config(), seed=123)
        out1 = model1(np.array([[1, 2, 3]])).logits.data
        out2 = model2(np.array([[1, 2, 3]])).logits.data
        self.assertTrue(np.array_equal(out1, out2))

    def test_different_seeds_give_different_weights(self):
        model1 = TinyMindTransformer(_tiny_config(), seed=1)
        model2 = TinyMindTransformer(_tiny_config(), seed=2)
        self.assertFalse(np.array_equal(model1.embed_tokens.data, model2.embed_tokens.data))


class TestParameterCounting(unittest.TestCase):
    def test_count_parameters_positive(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        self.assertGreater(model.count_parameters(), 0)

    def test_count_parameters_stable_across_calls(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        self.assertEqual(model.count_parameters(), model.count_parameters())

    def test_num_parameters_matches_manual_sum(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        manual = sum(p.data.size for p in model.parameters())
        self.assertEqual(model.num_parameters(), manual)

    def test_larger_config_has_more_parameters(self):
        small = TinyMindTransformer(_tiny_config(hidden_size=16), seed=0)
        large = TinyMindTransformer(_tiny_config(hidden_size=64, num_heads=8, num_kv_heads=4), seed=0)
        self.assertGreater(large.count_parameters(), small.count_parameters())

    def test_untied_embeddings_have_more_parameters(self):
        tied = TinyMindTransformer(_tiny_config(tie_embeddings=True), seed=0)
        untied = TinyMindTransformer(_tiny_config(tie_embeddings=False), seed=0)
        self.assertGreater(untied.count_parameters(), tied.count_parameters())


class TestGradientFlowThroughFullModel(unittest.TestCase):
    def test_every_parameter_gets_a_finite_gradient(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        out = model(np.array([[1, 2, 3, 4]]), labels=np.array([[1, 2, 3, 4]]))
        out.loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} has no gradient")
            self.assertTrue(np.all(np.isfinite(param.grad)), f"{name} has a non-finite gradient")

    def test_parameters_actually_change_after_optimizer_step(self):
        from tinymind.model.optim import AdamW
        model = TinyMindTransformer(_tiny_config(), seed=0)
        before = {name: param.data.copy() for name, param in model.named_parameters()}

        opt = AdamW(model.parameters(), learning_rate=1e-2)
        out = model(np.array([[1, 2, 3, 4]]), labels=np.array([[1, 2, 3, 4]]))
        out.loss.backward()
        opt.step()

        for name, param in model.named_parameters():
            self.assertFalse(np.array_equal(before[name], param.data), f"{name} did not change after opt.step()")


class TestTokenizerCompatibility(unittest.TestCase):
    def test_compatible_tokenizer_passes(self):
        from tinymind.model.tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        model = TinyMindTransformer(_tiny_config(vocab_size=tok.vocab_size), seed=0)
        model.check_tokenizer_compatibility(tok)  # must not raise

    def test_incompatible_tokenizer_raises(self):
        from tinymind.model.tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        model = TinyMindTransformer(_tiny_config(vocab_size=10), seed=0)  # smaller than tokenizer's 260
        with self.assertRaises(ValueError):
            model.check_tokenizer_compatibility(tok)


if __name__ == "__main__":
    unittest.main()
