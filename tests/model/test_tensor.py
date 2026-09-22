"""Numerical (finite-difference) gradient checks for
``tinymind.model.tensor``. This is the correctness spec for the from-scratch
autodiff engine that the rest of ``tinymind/model/`` is built on — see that
module's docstring for why it exists. Every op is checked against a
central-difference numerical gradient on small random inputs, the standard,
framework-agnostic way to verify a backward pass is mathematically correct
independent of the forward implementation.
"""
import unittest
from pathlib import Path

import numpy as np

from tinymind.model.tensor import Tensor, concat, cross_entropy

_EPS = 1e-3
_RTOL = 5e-2  # finite differences are inherently imprecise; this is generous on purpose
_ATOL = 5e-3


def numerical_gradient(fn, inputs: list[np.ndarray], eps: float = _EPS) -> list[np.ndarray]:
    """``fn(*arrays) -> scalar``. Returns one gradient array per input, via
    central differences: (f(x+eps) - f(x-eps)) / (2*eps) per element."""
    grads = []
    for i, arr in enumerate(inputs):
        grad = np.zeros_like(arr, dtype=np.float64)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = arr[idx]
            arr[idx] = original + eps
            plus = fn(*inputs)
            arr[idx] = original - eps
            minus = fn(*inputs)
            arr[idx] = original
            grad[idx] = (plus - minus) / (2 * eps)
        grads.append(grad)
    return grads


def assert_gradients_close(test_case: unittest.TestCase, analytic: np.ndarray, numeric: np.ndarray,
                           msg: str = "") -> None:
    ok = np.allclose(analytic, numeric, rtol=_RTOL, atol=_ATOL)
    if not ok:
        diff = np.abs(analytic - numeric)
        test_case.fail(f"{msg}: gradient mismatch, max abs diff {diff.max():.6f}\n"
                       f"analytic=\n{analytic}\nnumeric=\n{numeric}")


class TestBasicArithmeticGradients(unittest.TestCase):
    def test_add(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.random.randn(3, 4).astype(np.float64)

        def fn(a_, b_):
            return float((Tensor(a_.astype(np.float32)) + Tensor(b_.astype(np.float32))).sum().data)
        numeric = numerical_gradient(fn, [a, b])

        ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
        (ta + tb).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "add/a")
        assert_gradients_close(self, tb.grad, numeric[1], "add/b")

    def test_add_broadcasting(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.random.randn(1, 4).astype(np.float64)

        def fn(a_, b_):
            return float((Tensor(a_.astype(np.float32)) + Tensor(b_.astype(np.float32))).sum().data)
        numeric = numerical_gradient(fn, [a, b])

        ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
        (ta + tb).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "broadcast-add/a")
        assert_gradients_close(self, tb.grad, numeric[1], "broadcast-add/b")

    def test_mul(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.random.randn(3, 4).astype(np.float64)

        def fn(a_, b_):
            return float((Tensor(a_.astype(np.float32)) * Tensor(b_.astype(np.float32))).sum().data)
        numeric = numerical_gradient(fn, [a, b])

        ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
        (ta * tb).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "mul/a")
        assert_gradients_close(self, tb.grad, numeric[1], "mul/b")

    def test_div(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = (np.random.randn(3, 4).astype(np.float64) + 3.0)  # keep away from 0

        def fn(a_, b_):
            return float((Tensor(a_.astype(np.float32)) / Tensor(b_.astype(np.float32))).sum().data)
        numeric = numerical_gradient(fn, [a, b])

        ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
        (ta / tb).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "div/a")
        assert_gradients_close(self, tb.grad, numeric[1], "div/b")

    def test_pow_and_sqrt(self):
        a = np.random.rand(3, 4).astype(np.float64) + 0.5  # keep positive for sqrt

        def fn_pow(a_):
            return float((Tensor(a_.astype(np.float32)) ** 2).sum().data)
        numeric = numerical_gradient(fn_pow, [a])
        ta = Tensor(a, requires_grad=True)
        (ta ** 2).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "pow2")

        def fn_sqrt(a_):
            return float(Tensor(a_.astype(np.float32)).sqrt().sum().data)
        numeric = numerical_gradient(fn_sqrt, [a])
        ta2 = Tensor(a, requires_grad=True)
        ta2.sqrt().sum().backward()
        assert_gradients_close(self, ta2.grad, numeric[0], "sqrt")

    def test_rsqrt(self):
        a = np.random.rand(3, 4).astype(np.float64) + 0.5

        def fn(a_):
            return float(Tensor(a_.astype(np.float32)).rsqrt().sum().data)
        numeric = numerical_gradient(fn, [a])
        ta = Tensor(a, requires_grad=True)
        ta.rsqrt().sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "rsqrt")

    def test_exp_log(self):
        a = np.random.rand(3, 4).astype(np.float64) + 0.5

        def fn_exp(a_):
            return float(Tensor(a_.astype(np.float32)).exp().sum().data)
        numeric = numerical_gradient(fn_exp, [a])
        ta = Tensor(a, requires_grad=True)
        ta.exp().sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "exp")

        def fn_log(a_):
            return float(Tensor(a_.astype(np.float32)).log().sum().data)
        numeric = numerical_gradient(fn_log, [a])
        ta2 = Tensor(a, requires_grad=True)
        ta2.log().sum().backward()
        assert_gradients_close(self, ta2.grad, numeric[0], "log")

    def test_sigmoid_silu(self):
        a = np.random.randn(3, 4).astype(np.float64)

        def fn_sig(a_):
            return float(Tensor(a_.astype(np.float32)).sigmoid().sum().data)
        numeric = numerical_gradient(fn_sig, [a])
        ta = Tensor(a, requires_grad=True)
        ta.sigmoid().sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "sigmoid")

        def fn_silu(a_):
            return float(Tensor(a_.astype(np.float32)).silu().sum().data)
        numeric = numerical_gradient(fn_silu, [a])
        ta2 = Tensor(a, requires_grad=True)
        ta2.silu().sum().backward()
        assert_gradients_close(self, ta2.grad, numeric[0], "silu")


class TestShapeAndReductionGradients(unittest.TestCase):
    def test_reshape(self):
        a = np.random.randn(2, 6).astype(np.float64)

        def fn(a_):
            return float(Tensor(a_.astype(np.float32)).reshape(3, 4).sum().data)
        numeric = numerical_gradient(fn, [a])
        ta = Tensor(a, requires_grad=True)
        ta.reshape(3, 4).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "reshape")

    def test_transpose(self):
        a = np.random.randn(2, 3, 4).astype(np.float64)

        def fn(a_):
            return float((Tensor(a_.astype(np.float32)).transpose(0, 2, 1) * 2.0).sum().data)
        numeric = numerical_gradient(fn, [a])
        ta = Tensor(a, requires_grad=True)
        (ta.transpose(0, 2, 1) * 2.0).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "transpose")

    def test_sum_axis(self):
        a = np.random.randn(3, 4).astype(np.float64)

        def fn(a_):
            return float(Tensor(a_.astype(np.float32)).sum(axis=1).sum().data)
        numeric = numerical_gradient(fn, [a])
        ta = Tensor(a, requires_grad=True)
        ta.sum(axis=1).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "sum-axis")

    def test_mean(self):
        a = np.random.randn(3, 4).astype(np.float64)

        def fn(a_):
            return float(Tensor(a_.astype(np.float32)).mean(axis=-1, keepdims=True).sum().data)
        numeric = numerical_gradient(fn, [a])
        ta = Tensor(a, requires_grad=True)
        ta.mean(axis=-1, keepdims=True).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "mean")


class TestMatmulGradient(unittest.TestCase):
    def test_2d_matmul(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.random.randn(4, 5).astype(np.float64)

        def fn(a_, b_):
            return float((Tensor(a_.astype(np.float32)) @ Tensor(b_.astype(np.float32))).sum().data)
        numeric = numerical_gradient(fn, [a, b])

        ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
        (ta @ tb).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "matmul/a")
        assert_gradients_close(self, tb.grad, numeric[1], "matmul/b")

    def test_batched_matmul(self):
        a = np.random.randn(2, 3, 4).astype(np.float64)
        b = np.random.randn(2, 4, 5).astype(np.float64)

        def fn(a_, b_):
            return float((Tensor(a_.astype(np.float32)) @ Tensor(b_.astype(np.float32))).sum().data)
        numeric = numerical_gradient(fn, [a, b])

        ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
        (ta @ tb).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "batched-matmul/a")
        assert_gradients_close(self, tb.grad, numeric[1], "batched-matmul/b")


class TestSoftmaxAndCrossEntropy(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        a = Tensor(np.random.randn(2, 5))
        probs = a.softmax(axis=-1)
        self.assertTrue(np.allclose(probs.data.sum(axis=-1), 1.0, atol=1e-5))

    def test_softmax_gradient(self):
        a = np.random.randn(2, 5).astype(np.float64)
        weights = np.random.randn(2, 5).astype(np.float64)  # to make a non-trivial scalar loss

        def fn(a_):
            probs = Tensor(a_.astype(np.float32)).softmax(axis=-1)
            return float((probs * Tensor(weights.astype(np.float32))).sum().data)
        numeric = numerical_gradient(fn, [a])

        ta = Tensor(a, requires_grad=True)
        (ta.softmax(axis=-1) * Tensor(weights)).sum().backward()
        assert_gradients_close(self, ta.grad, numeric[0], "softmax")

    def test_cross_entropy_matches_numerical_gradient(self):
        logits = np.random.randn(4, 6).astype(np.float64)
        targets = np.array([1, 3, 0, 5])

        def fn(logits_):
            return float(cross_entropy(Tensor(logits_.astype(np.float32)), targets).data)
        numeric = numerical_gradient(fn, [logits])

        t_logits = Tensor(logits, requires_grad=True)
        cross_entropy(t_logits, targets).backward()
        assert_gradients_close(self, t_logits.grad, numeric[0], "cross_entropy")

    def test_cross_entropy_ignore_index(self):
        logits = Tensor(np.random.randn(3, 6), requires_grad=True)
        targets = np.array([1, -100, 2])  # middle position ignored
        loss = cross_entropy(logits, targets, ignore_index=-100)
        loss.backward()
        self.assertTrue(np.all(logits.grad[1] == 0.0))  # ignored row gets zero gradient

    def test_cross_entropy_decreases_when_logit_moves_toward_target(self):
        logits = Tensor(np.array([[0.0, 0.0, 0.0]]), requires_grad=True)
        targets = np.array([1])
        loss_before = cross_entropy(logits, targets).item()
        logits2 = Tensor(np.array([[0.0, 2.0, 0.0]]))
        loss_after = cross_entropy(logits2, targets).item()
        self.assertLess(loss_after, loss_before)


class TestEmbeddingLookup(unittest.TestCase):
    def test_forward_shape(self):
        table = Tensor(np.random.randn(10, 4))
        indices = np.array([[1, 2, 3], [4, 5, 6]])
        out = table.embedding_lookup(indices)
        self.assertEqual(out.shape, (2, 3, 4))

    def test_gradient_accumulates_for_repeated_indices(self):
        table = Tensor(np.random.randn(5, 3), requires_grad=True)
        indices = np.array([0, 0, 1])  # index 0 used twice
        out = table.embedding_lookup(indices)
        out.sum().backward()
        self.assertTrue(np.allclose(table.grad[0], np.array([2.0, 2.0, 2.0])))
        self.assertTrue(np.allclose(table.grad[1], np.array([1.0, 1.0, 1.0])))
        self.assertTrue(np.allclose(table.grad[2], np.array([0.0, 0.0, 0.0])))

    def test_matches_numerical_gradient(self):
        table_data = np.random.randn(5, 3).astype(np.float64)
        indices = np.array([0, 2, 0, 4])

        def fn(table_):
            return float(Tensor(table_.astype(np.float32)).embedding_lookup(indices).sum().data)
        numeric = numerical_gradient(fn, [table_data])

        table = Tensor(table_data, requires_grad=True)
        table.embedding_lookup(indices).sum().backward()
        assert_gradients_close(self, table.grad, numeric[0], "embedding_lookup")


class TestConcat(unittest.TestCase):
    def test_forward_and_backward(self):
        a = Tensor(np.random.randn(2, 3), requires_grad=True)
        b = Tensor(np.random.randn(2, 5), requires_grad=True)
        out = concat([a, b], axis=1)
        self.assertEqual(out.shape, (2, 8))
        out.sum().backward()
        self.assertEqual(a.grad.shape, (2, 3))
        self.assertEqual(b.grad.shape, (2, 5))


class TestGradientAccumulationAcrossFanOut(unittest.TestCase):
    def test_tensor_used_twice_accumulates(self):
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        y = (x * x) + x  # dy/dx = 2x + 1
        y.sum().backward()
        expected = 2 * x.data + 1
        self.assertTrue(np.allclose(x.grad, expected))


class TestDeterminism(unittest.TestCase):
    """Regression coverage for a real bug found via
    tests/training/test_training.py::test_reproducible_given_seed: ``_prev``
    used to be a ``set`` of ``Tensor`` objects, and since ``Tensor`` has no
    custom ``__hash__``, that set ordered by the default identity
    (memory-address-based) hash — which differs between *separate process
    runs* even for identical code and inputs, since absolute memory
    addresses aren't guaranteed to repeat across process invocations. That
    let backward()'s topological sort visit a multi-parent node's children
    in a different order run to run, and because float addition isn't
    exactly associative, accumulated gradients occasionally differed at the
    ~1e-7 level between two otherwise-identical runs. ``_prev`` is a tuple
    now — see ``tinymind/model/tensor.py``'s constructor comment.

    A same-process repeated call would not reliably have caught the
    original bug (freshly-constructed objects within one process often
    reuse the same freed addresses, masking exactly this class of issue) —
    so this test spawns two independent subprocesses, which is what
    actually exposed the bug in the first place.
    """

    def test_backward_is_identical_across_separate_process_invocations(self):
        import subprocess
        import sys

        script = (
            "import numpy as np\n"
            "from tinymind.model.tensor import Tensor\n"
            "x = Tensor(np.random.RandomState(0).randn(20, 20), requires_grad=True)\n"
            "a = x * 2.0\n"
            "b = x * 3.0\n"
            "c = x.exp()\n"
            "d = x.sum(axis=0, keepdims=True)\n"
            "out = (a * b) + c.sum() + d.sum()\n"
            "out.sum().backward()\n"
            "print(repr(x.grad.tobytes()))\n"
        )
        results = []
        for _ in range(3):
            proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                  cwd=str(Path(__file__).resolve().parents[2]), check=True)
            results.append(proc.stdout.strip())
        self.assertTrue(all(r == results[0] for r in results),
                        "gradient bytes differ across separate process invocations for identical "
                        "inputs — this is the exact bug tuple-ifying Tensor._prev fixed")


if __name__ == "__main__":
    unittest.main()
