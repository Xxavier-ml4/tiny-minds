"""Training-time dropout: the autodiff op, the fused attention kernel (the path the trainer actually runs), and the
whole model.

The properties that matter, each pinned below:

* dropout exists only when a generator is passed, so evaluation/generation can never get it by accident;
* inverted scaling preserves the expected activation (checked on the op AND inside the fused kernel);
* gradients are exact (finite differences with the mask held fixed by re-seeding), in the op and the kernel;
* the fused kernel and the composed reference path draw the SAME mask from the same generator state, so they
  agree on outputs and on every parameter gradient — not merely "both look plausible";
* dropout=0 is bit-identical to a model that never heard of dropout.
"""
import unittest

import numpy as np

from tinymind.model import ModelConfig, TinyMindTransformer, fused
from tinymind.model.generation import ModelGenerationConfig, generate_with_cache_ids
from tinymind.model.model import KVCache
from tinymind.model.tensor import Tensor, dropout, keep_mask, no_grad

from tests.model.test_tensor import assert_gradients_close, numerical_gradient

BASE = dict(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64, max_seq_len=16,
            vocab_size=30)
IDS = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9, 1]])
SEGMENTS = np.array([[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2, 2, 2]])


def rng(seed=0):
    return np.random.default_rng(seed)


class TestDropoutOp(unittest.TestCase):
    def test_identity_without_a_generator_or_at_rate_zero(self):
        x = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
        self.assertIs(dropout(x, 0.5, None), x)
        self.assertIs(dropout(x, 0.0, rng()), x)

    def test_invalid_rates_are_rejected_even_without_a_generator(self):
        x = Tensor(np.ones(4, dtype=np.float32))
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                dropout(x, bad, None)
            with self.assertRaises(ValueError):
                keep_mask((4,), bad, rng())

    def test_survivors_are_scaled_and_the_rest_are_zero(self):
        out = dropout(Tensor(np.ones((200, 50), dtype=np.float32)), 0.25, rng(1)).data
        self.assertEqual(set(np.unique(out).tolist()), {0.0, np.float32(1.0 / 0.75)})

    def test_keep_fraction_and_expectation(self):
        x = Tensor(np.full((400, 100), 2.0, dtype=np.float32))
        for rate in (0.05, 0.3, 0.6):
            out = dropout(x, rate, rng(2)).data
            self.assertAlmostEqual(float((out != 0).mean()), 1 - rate, delta=0.01)
            self.assertAlmostEqual(float(out.mean()), 2.0, delta=0.05)  # inverted scaling keeps E[out] = x

    def test_mask_is_a_pure_function_of_the_generator(self):
        x = Tensor(np.ones((50, 50), dtype=np.float32))
        a, b, c = (dropout(x, 0.4, rng(s)).data for s in (7, 7, 8))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_gradient_is_the_scaled_mask(self):
        x = Tensor(np.ones((6, 7), dtype=np.float32), requires_grad=True)
        out = dropout(x, 0.5, rng(3))
        out.sum().backward()
        self.assertTrue(np.array_equal(x.grad, out.data))  # d/dx of sum(x * m) is m, and out = 1 * m here

    def test_finite_difference_gradient(self):
        x = np.random.default_rng(4).standard_normal((3, 5))
        w = np.random.default_rng(5).standard_normal((3, 5))

        def fn(x_):
            return float((dropout(Tensor(x_.astype(np.float32)), 0.4, rng(9)) * Tensor(w.astype(np.float32))).sum().data)
        numeric = numerical_gradient(fn, [x])[0]
        tx = Tensor(x, requires_grad=True)
        (dropout(tx, 0.4, rng(9)) * Tensor(w.astype(np.float32))).sum().backward()
        assert_gradients_close(self, tx.grad, numeric, "dropout")

    def test_records_no_graph_under_no_grad(self):
        x = Tensor(np.ones(8, dtype=np.float32), requires_grad=True)
        with no_grad():
            out = dropout(x, 0.5, rng())
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._prev, ())


class TestFusedAttentionDropout(unittest.TestCase):
    B, H, HKV, T, D = 1, 4, 2, 5, 4

    def tensors(self, seed=0):
        r = np.random.default_rng(seed)
        q = r.standard_normal((self.B, self.H, self.T, self.D))
        k = r.standard_normal((self.B, self.HKV, self.T, self.D))
        v = r.standard_normal((self.B, self.HKV, self.T, self.D))
        return q, k, v

    @staticmethod
    def attend(q, k, v, **kw):
        tq, tk, tv = (Tensor(a.astype(np.float32), requires_grad=True) for a in (q, k, v))
        out = fused.attention(tq, tk, tv, scale=0.5, causal=True, **kw)
        return out, tq, tk, tv

    def test_rate_zero_or_no_generator_is_bit_identical_to_the_original_kernel(self):
        q, k, v = self.tensors()
        base = self.attend(q, k, v)[0].data
        self.assertTrue(np.array_equal(base, self.attend(q, k, v, dropout_rate=0.0, dropout_rng=rng())[0].data))
        self.assertTrue(np.array_equal(base, self.attend(q, k, v, dropout_rate=0.5, dropout_rng=None)[0].data))

    def test_gradients_match_finite_differences_with_a_fixed_mask(self):
        q, k, v = self.tensors(1)
        w = np.random.default_rng(2).standard_normal((self.B, self.H, self.T, self.D)).astype(np.float32)

        def fn(q_, k_, v_):
            out = self.attend(q_, k_, v_, dropout_rate=0.3, dropout_rng=rng(11))[0]
            return float((out * Tensor(w)).sum().data)
        numeric = numerical_gradient(fn, [q.copy(), k.copy(), v.copy()])

        out, tq, tk, tv = self.attend(q, k, v, dropout_rate=0.3, dropout_rng=rng(11))
        (out * Tensor(w)).sum().backward()
        for name, analytic, num in zip("qkv", (tq.grad, tk.grad, tv.grad), numeric):
            assert_gradients_close(self, analytic, num, f"fused attention d{name} under dropout")

    def test_inverted_scaling_preserves_the_expected_output(self):
        q, k, v = self.tensors(3)
        exact = self.attend(q, k, v)[0].data
        mean = np.mean([self.attend(q, k, v, dropout_rate=0.2, dropout_rng=rng(s))[0].data for s in range(600)], axis=0)
        np.testing.assert_allclose(mean, exact, atol=0.12)  # seeds are fixed, so this is deterministic, not flaky

    def test_dropout_actually_changes_the_output(self):
        q, k, v = self.tensors(4)
        self.assertFalse(np.array_equal(self.attend(q, k, v)[0].data,
                                        self.attend(q, k, v, dropout_rate=0.5, dropout_rng=rng(0))[0].data))


class TestModelDropout(unittest.TestCase):
    def models(self, rate=0.3):
        return (TinyMindTransformer(ModelConfig(**BASE), seed=0),
                TinyMindTransformer(ModelConfig(**BASE, dropout=rate), seed=0))

    def test_dropout_does_not_touch_initialisation(self):
        plain, drop = self.models()
        for (n1, p1), (n2, p2) in zip(plain.named_parameters(), drop.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(np.array_equal(p1.data, p2.data), n1)

    def test_evaluation_path_is_dropout_free(self):
        plain, drop = self.models()
        self.assertTrue(np.array_equal(plain(IDS).logits.data, drop(IDS).logits.data))

    def test_generation_is_unaffected_by_the_dropout_setting(self):
        plain, drop = self.models()
        cfg = ModelGenerationConfig(max_new_tokens=6, do_sample=False)
        prompt = np.array([[1, 2, 3]])
        self.assertTrue(np.array_equal(generate_with_cache_ids(plain, prompt, cfg),
                                       generate_with_cache_ids(drop, prompt, cfg)))

    def test_a_generator_with_rate_zero_changes_nothing(self):
        plain, _ = self.models()
        self.assertTrue(np.array_equal(plain(IDS).logits.data, plain(IDS, dropout_rng=rng(1)).logits.data))

    def test_training_forward_is_random_but_reproducible_per_seed(self):
        _, drop = self.models()
        eval_logits = drop(IDS).logits.data
        a = drop(IDS, dropout_rng=rng(5)).logits.data
        b = drop(IDS, dropout_rng=rng(5)).logits.data
        c = drop(IDS, dropout_rng=rng(6)).logits.data
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(a, eval_logits))

    def test_dropout_with_a_kv_cache_is_refused(self):
        _, drop = self.models()
        with self.assertRaises(ValueError):
            drop(IDS[:1], use_cache=True, past_key_values=KVCache(drop.config), dropout_rng=rng())

    def test_fused_and_reference_paths_agree_exactly_under_dropout(self):
        """Same generator state -> same masks in both paths -> same logits and the same gradient for every
        parameter, including with packed segments (the real training layout)."""
        _, drop = self.models(0.3)

        def run(fused_on):
            drop.zero_grad()
            with fused.use_fused(fused_on):
                out = drop(IDS, labels=IDS, segment_ids=SEGMENTS, dropout_rng=rng(5))
                out.loss.backward()
            return out.logits.data.copy(), float(out.loss.item()), {n: p.grad.copy() for n, p in drop.named_parameters()}

        lf, lossf, gf = run(True)
        lr, lossr, gr = run(False)
        np.testing.assert_allclose(lf, lr, atol=1e-4)
        self.assertAlmostEqual(lossf, lossr, places=4)
        for name in gf:
            np.testing.assert_allclose(gf[name], gr[name], atol=1e-4, err_msg=name)

    def test_every_parameter_receives_a_finite_gradient_with_dropout(self):
        _, drop = self.models(0.2)
        drop(IDS, labels=IDS, dropout_rng=rng(1)).loss.backward()
        for name, p in drop.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(np.isfinite(p.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
