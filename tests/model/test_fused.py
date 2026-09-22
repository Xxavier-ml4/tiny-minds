"""The fused kernels must compute the same function as the composed reference
ops — forward values and every gradient — and the model built on them must
agree with the reference model. The composed path is itself checked against
finite differences in ``test_tensor.py``; this file adds finite differences for
the fused attention directly, so the guarantee does not rest on one chain of
comparisons.

Documented tolerance: float32 forward values agree to rtol 1e-4 / atol 1e-5,
gradients to rtol 1e-3 / atol 1e-5 (summation order differs between a batched
GEMM plus un-broadcast and one flattened 2-D GEMM).
"""
import unittest

import numpy as np

from tinymind.model import fused
from tinymind.model.attention import CausalSelfAttention, block_causal_bias, segment_positions
from tinymind.model.config import ModelConfig
from tinymind.model.model import KVCache, TinyMindTransformer
from tinymind.model.norm import RMSNorm
from tinymind.model.positional import apply_rotary_pos_emb, precompute_rope_cache
from tinymind.model.mlp import SwiGLUMLP
from tinymind.model.linear import Linear
from tinymind.model.tensor import Tensor, no_grad

FWD = dict(rtol=1e-4, atol=1e-5)
BWD = dict(rtol=1e-3, atol=1e-5)


def _run(module_fn, flag, inputs):
    """Run ``module_fn(*tensors)`` under the given fused flag; return output and input grads."""
    tensors = [Tensor(a.copy(), requires_grad=True) for a in inputs]
    with fused.use_fused(flag):
        out = module_fn(*tensors)
        weights = np.random.default_rng(99).normal(size=out.shape).astype(np.float32)
        (out * Tensor(weights)).sum().backward()
    return out.data.copy(), [t.grad.copy() for t in tensors]


class TestFusedFlag(unittest.TestCase):
    def test_flag_round_trips_and_restores_on_exception(self):
        before = fused.enabled()
        try:
            with fused.use_fused(not before):
                self.assertEqual(fused.enabled(), not before)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertEqual(fused.enabled(), before)


class TestModuleEquivalence(unittest.TestCase):
    def _compare(self, module, inputs):
        out_f, grads_f = _run(module, True, inputs)
        params_f = {n: p.grad.copy() for n, p in module_params(module)}
        zero(module)
        out_r, grads_r = _run(module, False, inputs)
        params_r = {n: p.grad.copy() for n, p in module_params(module)}
        np.testing.assert_allclose(out_f, out_r, **FWD)
        for gf, gr in zip(grads_f, grads_r):
            np.testing.assert_allclose(gf, gr, **BWD)
        for name in params_f:
            np.testing.assert_allclose(params_f[name], params_r[name], **BWD, err_msg=name)

    def test_linear_3d_and_2d(self):
        rng = np.random.default_rng(0)
        layer = Linear(7, 5, rng=rng)
        self._compare(layer, [rng.normal(size=(2, 6, 7)).astype(np.float32)])
        zero(layer)
        self._compare(layer, [rng.normal(size=(9, 7)).astype(np.float32)])

    def test_rmsnorm(self):
        rng = np.random.default_rng(1)
        norm = RMSNorm(12, eps=1e-6)
        norm.weight.data[...] = rng.normal(1.0, 0.3, size=12).astype(np.float32)
        self._compare(norm, [rng.normal(size=(3, 5, 12)).astype(np.float32)])

    def test_swiglu_mlp(self):
        rng = np.random.default_rng(2)
        cfg = ModelConfig(hidden_size=16, num_layers=1, num_heads=2, num_kv_heads=2,
                          intermediate_size=24, max_seq_len=8, vocab_size=10)
        mlp = SwiGLUMLP(cfg, rng=rng)
        self._compare(mlp, [rng.normal(size=(2, 4, 16)).astype(np.float32)])

    def test_swiglu_handles_large_negative_gate_without_nan(self):
        gate = Tensor(np.array([[-200.0, -50.0, 0.0, 50.0]], dtype=np.float32), requires_grad=True)
        up = Tensor(np.ones((1, 4), dtype=np.float32), requires_grad=True)
        with np.errstate(all="raise"):
            out = fused.swiglu(gate, up)
            out.sum().backward()
        self.assertTrue(np.isfinite(out.data).all() and np.isfinite(gate.grad).all())

    def test_rope_unbatched_and_batched_positions(self):
        rng = np.random.default_rng(3)
        cos, sin = precompute_rope_cache(8, 32, 10000.0)
        x = rng.normal(size=(2, 3, 5, 8)).astype(np.float32)
        for position_ids in (None, np.array([[0, 1, 2, 3, 4], [7, 8, 9, 10, 11]]), np.array([3, 4, 5, 6, 7])):
            fn = lambda t, p=position_ids: apply_rotary_pos_emb(t, cos, sin, position_ids=p)  # noqa: E731
            out_f, g_f = _run(fn, True, [x])
            out_r, g_r = _run(fn, False, [x])
            np.testing.assert_allclose(out_f, out_r, **FWD)
            np.testing.assert_allclose(g_f[0], g_r[0], **BWD)

    def test_attention_module_mha_gqa_mqa_prefill(self):
        rng = np.random.default_rng(4)
        for heads, kv in ((4, 4), (4, 2), (4, 1), (6, 2)):
            cfg = ModelConfig(hidden_size=heads * 4, num_layers=1, num_heads=heads, num_kv_heads=kv,
                              intermediate_size=16, max_seq_len=16, vocab_size=10)
            attn = CausalSelfAttention(cfg, rng=np.random.default_rng(5))
            x = rng.normal(size=(2, 7, heads * 4)).astype(np.float32)
            with self.subTest(heads=heads, kv=kv):
                self._compare(attn, [x])
                zero(attn)

    def test_attention_with_block_diagonal_bias(self):
        rng = np.random.default_rng(6)
        cfg = ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2,
                          intermediate_size=16, max_seq_len=16, vocab_size=10)
        attn = CausalSelfAttention(cfg, rng=np.random.default_rng(7))
        seg = np.array([[1, 1, 1, 2, 2, 3, 3, 3], [1, 1, 2, 2, 2, 2, 0, 0]])
        bias = block_causal_bias(seg)
        pos = segment_positions(seg)
        x = rng.normal(size=(2, 8, 16)).astype(np.float32)
        fn = lambda t: attn(t, position_ids=pos, attention_bias=bias)  # noqa: E731
        out_f, g_f = _run(fn, True, [x])
        out_r, g_r = _run(fn, False, [x])
        np.testing.assert_allclose(out_f, out_r, **FWD)
        np.testing.assert_allclose(g_f[0], g_r[0], **BWD)


def module_params(module):
    return list(module.named_parameters())


def zero(module):
    module.zero_grad()


class TestAttentionFiniteDifferences(unittest.TestCase):
    """Independent of the reference path: central differences on the fused op."""

    def test_gradients_wrt_q_k_v(self):
        rng = np.random.default_rng(8)
        b, h, hkv, t, tk, d = 1, 4, 2, 3, 5, 4
        arrays = [rng.normal(size=(b, h, t, d)), rng.normal(size=(b, hkv, tk, d)), rng.normal(size=(b, hkv, tk, d))]
        weights = rng.normal(size=(b, h, t, d))
        bias = np.where(rng.random((b, 1, t, tk)) < 0.25, -1e9, 0.0).astype(np.float32)
        bias[..., 0] = 0.0  # keep every row with at least one visible key

        def scalar(q_, k_, v_):
            out = fused.attention(Tensor(q_.astype(np.float32)), Tensor(k_.astype(np.float32)),
                                  Tensor(v_.astype(np.float32)), scale=0.5, causal=False, bias=bias)
            return float((out.data * weights).sum())

        tq, tk_, tv = (Tensor(a.astype(np.float32), requires_grad=True) for a in arrays)
        (fused.attention(tq, tk_, tv, scale=0.5, causal=False, bias=bias) * Tensor(weights.astype(np.float32))).sum().backward()
        for idx, (analytic, name) in enumerate(((tq.grad, "q"), (tk_.grad, "k"), (tv.grad, "v"))):
            numeric = np.zeros_like(arrays[idx])
            it = np.nditer(arrays[idx], flags=["multi_index"])
            for _ in it:
                pos = it.multi_index
                orig = arrays[idx][pos]
                arrays[idx][pos] = orig + 1e-3
                plus = scalar(*arrays)
                arrays[idx][pos] = orig - 1e-3
                minus = scalar(*arrays)
                arrays[idx][pos] = orig
                numeric[pos] = (plus - minus) / 2e-3
            np.testing.assert_allclose(analytic, numeric, rtol=5e-2, atol=5e-3, err_msg=name)


class TestFullModelEquivalence(unittest.TestCase):
    def _model(self, **kw):
        base = dict(hidden_size=24, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=48,
                    max_seq_len=16, vocab_size=30)
        base.update(kw)
        return TinyMindTransformer(ModelConfig(**base), seed=11)

    def _grads(self, model, flag, ids, labels, **kw):
        model.zero_grad()
        with fused.use_fused(flag):
            out = model(ids, labels=labels, **kw)
            out.loss.backward()
        return out.logits.data.copy(), out.loss.item(), {n: p.grad.copy() for n, p in model.named_parameters()}

    def test_logits_loss_and_every_parameter_gradient_match(self):
        ids = np.random.default_rng(0).integers(4, 30, size=(3, 10))
        for kw in ({}, {"tie_embeddings": False}, {"num_kv_heads": 1}, {"num_kv_heads": 4}):
            model = self._model(**kw)
            with self.subTest(**kw):
                lf, loss_f, gf = self._grads(model, True, ids, ids)
                lr, loss_r, gr = self._grads(model, False, ids, ids)
                np.testing.assert_allclose(lf, lr, **FWD)
                self.assertAlmostEqual(loss_f, loss_r, places=5)
                for name in gf:
                    np.testing.assert_allclose(gf[name], gr[name], **BWD, err_msg=name)

    def test_packed_rows_match_reference(self):
        ids = np.random.default_rng(1).integers(4, 30, size=(2, 12))
        seg = np.array([[1] * 5 + [2] * 4 + [3] * 3, [1] * 7 + [2] * 3 + [0] * 2])
        labels = np.where(seg == 0, -100, ids)
        model = self._model()
        lf, _, gf = self._grads(model, True, ids, labels, segment_ids=seg)
        lr, _, gr = self._grads(model, False, ids, labels, segment_ids=seg)
        np.testing.assert_allclose(lf, lr, **FWD)
        for name in gf:
            np.testing.assert_allclose(gf[name], gr[name], **BWD, err_msg=name)

    def test_kv_cache_decode_matches_full_forward_with_fused_ops(self):
        model = self._model()
        ids = np.random.default_rng(2).integers(4, 30, size=(1, 9))
        with no_grad():
            full = model(ids).logits.data
            cache = KVCache(model.config, batch_size=1)
            pre = model(ids[:, :5], use_cache=True, past_key_values=cache, position_ids=np.arange(5)[None]).logits.data
            steps = [pre[0, -1]]
            for i in range(5, 9):
                out = model(ids[:, i:i + 1], use_cache=True, past_key_values=cache, position_ids=np.array([[cache.length]]))
                steps.append(out.logits.data[0, -1])
        np.testing.assert_allclose(np.stack(steps), full[0, 4:9], rtol=1e-3, atol=1e-4)

    def test_deterministic_within_process(self):
        ids = np.random.default_rng(3).integers(4, 30, size=(2, 8))
        model = self._model()
        a = self._grads(model, True, ids, ids)
        b = self._grads(model, True, ids, ids)
        self.assertTrue(np.array_equal(a[0], b[0]))
        for name in a[2]:
            self.assertTrue(np.array_equal(a[2][name], b[2][name]), name)


class TestGraphLifecycle(unittest.TestCase):
    def test_no_grad_records_nothing(self):
        model = TinyMindTransformer(ModelConfig(hidden_size=16, num_layers=1, num_heads=2, num_kv_heads=1,
                                                intermediate_size=32, max_seq_len=8, vocab_size=20), seed=0)
        ids = np.array([[1, 2, 3, 4]])
        with no_grad():
            out = model(ids, labels=ids)
        self.assertFalse(out.logits.requires_grad)
        self.assertEqual(out.logits._prev, ())
        self.assertFalse(out.loss.requires_grad)
        with fused.use_fused(False), no_grad():  # reference ops honour it too
            ref = model(ids)
        self.assertEqual(ref.logits._prev, ())
        np.testing.assert_allclose(out.logits.data, ref.logits.data, **FWD)

    def test_no_grad_restores_state_after_exception(self):
        from tinymind.model import tensor as tmod
        try:
            with no_grad():
                raise ValueError
        except ValueError:
            pass
        self.assertTrue(tmod.grad_enabled())

    def test_release_graph_gives_identical_parameter_gradients_and_frees_interior(self):
        model = TinyMindTransformer(ModelConfig(hidden_size=16, num_layers=2, num_heads=2, num_kv_heads=1,
                                                intermediate_size=32, max_seq_len=8, vocab_size=20), seed=0)
        ids = np.array([[1, 2, 3, 4, 5]])
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        kept = {n: p.grad.copy() for n, p in model.named_parameters()}
        model.zero_grad()
        out = model(ids, labels=ids)
        out.loss.backward(retain_graph=False)
        for n, p in model.named_parameters():
            self.assertTrue(np.array_equal(p.grad, kept[n]), n)
        self.assertEqual(out.loss._prev, ())            # root released
        self.assertIsNone(out.logits.grad)               # interior gradient freed

    def test_backward_handles_graphs_deeper_than_the_recursion_limit(self):
        import sys
        x = Tensor(np.ones(1, dtype=np.float32), requires_grad=True)
        y = x
        for _ in range(sys.getrecursionlimit() + 500):
            y = y + x
        y.backward()
        self.assertEqual(float(x.grad[0]), sys.getrecursionlimit() + 501)


class TestUnsupportedConfigsAreRejected(unittest.TestCase):
    """Phase 3A audit finding: these built fine and were silently ignored."""

    BASE = dict(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=4, intermediate_size=64,
                max_seq_len=16, vocab_size=30)

    def test_each_silent_knob_now_fails_loudly(self):
        from tinymind.model.config import ModelConfigError
        for override in (dict(norm_type="layernorm"), dict(mlp_type="gelu_mlp"), dict(sliding_window=2),
                         dict(dropout=0.5), dict(dtype="float16"), dict(attention_type="mqa"),
                         dict(attention_type="mha", num_kv_heads=2)):
            with self.subTest(**override):
                with self.assertRaises(ModelConfigError):
                    TinyMindTransformer(ModelConfig(**{**self.BASE, **override}), seed=0)

    def test_supported_configs_still_build(self):
        for attention_type, kv in (("mha", 4), ("gqa", 2), ("gqa", 4), ("mqa", 1)):
            TinyMindTransformer(ModelConfig(**{**self.BASE, "attention_type": attention_type, "num_kv_heads": kv}), seed=0)


if __name__ == "__main__":
    unittest.main()
