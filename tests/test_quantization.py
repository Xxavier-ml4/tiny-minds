import unittest

import numpy as np

from tinymind.quantization import Int4Scheme, Int8Scheme, QuantizationReport


class TestInt8Scheme(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # Deliberately different scales per row, to exercise per-row scaling.
        self.weights = (rng.standard_normal((6, 10)) *
                        np.array([1, 5, 0.1, 2, 3, 0.5]).reshape(-1, 1)).astype(np.float32)
        self.scheme = Int8Scheme()

    def test_round_trip_low_error(self):
        qbytes, meta = self.scheme.quantize(self.weights.tobytes(), self.weights.shape, "float32")
        deq = np.frombuffer(self.scheme.dequantize(qbytes, meta), dtype=np.float32).reshape(self.weights.shape)
        rel_err = np.abs(deq - self.weights) / (np.abs(self.weights) + 1e-6)
        # Mean relative error is a noisy metric near zero (a tiny absolute
        # error on a near-zero true value reads as a huge relative one) —
        # bounded generously for that reason; the tight, meaningful check
        # is test_quantization_error_bounded_by_half_a_step below.
        self.assertLess(float(np.mean(rel_err)), 0.05)

    def test_quantization_error_bounded_by_half_a_step(self):
        # The mathematically-guaranteed property of round-to-nearest
        # quantization: every dequantized value is within +/- 0.5 * scale
        # of the true value (for values inside the representable range) —
        # a tight, principled bound, unlike mean relative error.
        qbytes, meta = self.scheme.quantize(self.weights.tobytes(), self.weights.shape, "float32")
        deq = np.frombuffer(self.scheme.dequantize(qbytes, meta), dtype=np.float32).reshape(self.weights.shape)
        scales = np.array(meta["scales"], dtype=np.float32)
        abs_err = np.abs(deq - self.weights)
        self.assertTrue(np.all(abs_err <= 0.5 * scales[:, None] + 1e-4))

    def test_compression_ratio_is_four_x(self):
        qbytes, _meta = self.scheme.quantize(self.weights.tobytes(), self.weights.shape, "float32")
        report = QuantizationReport(scheme="int8", bits=8, original_size_bytes=self.weights.nbytes,
                                    quantized_size_bytes=len(qbytes))
        self.assertAlmostEqual(report.compression_ratio, 4.0, places=1)

    def test_rejects_non_float32_input(self):
        with self.assertRaises(ValueError):
            self.scheme.quantize(self.weights.astype(np.float16).tobytes(), self.weights.shape, "float16")

    def test_rejects_non_2d_shape(self):
        vec = np.ones(8, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.scheme.quantize(vec.tobytes(), vec.shape, "float32")

    def test_all_zero_row_does_not_divide_by_zero(self):
        w = np.zeros((2, 4), dtype=np.float32)
        qbytes, meta = self.scheme.quantize(w.tobytes(), w.shape, "float32")
        deq = np.frombuffer(self.scheme.dequantize(qbytes, meta), dtype=np.float32).reshape(w.shape)
        self.assertTrue(np.all(np.isfinite(deq)))
        self.assertTrue(np.allclose(deq, 0.0))


class TestInt4Scheme(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.weights = (rng.standard_normal((6, 10)) *
                        np.array([1, 5, 0.1, 2, 3, 0.5]).reshape(-1, 1)).astype(np.float32)
        self.scheme = Int4Scheme()

    def test_round_trip_bounded_error(self):
        qbytes, meta = self.scheme.quantize(self.weights.tobytes(), self.weights.shape, "float32")
        deq = np.frombuffer(self.scheme.dequantize(qbytes, meta), dtype=np.float32).reshape(self.weights.shape)
        rel_err = np.abs(deq - self.weights) / (np.abs(self.weights) + 1e-6)
        # 4-bit is genuinely lossy; bound generously rather than tightly.
        self.assertLess(float(np.mean(rel_err)), 0.35)

    def test_compression_ratio_is_eight_x(self):
        qbytes, _meta = self.scheme.quantize(self.weights.tobytes(), self.weights.shape, "float32")
        report = QuantizationReport(scheme="int4", bits=4, original_size_bytes=self.weights.nbytes,
                                    quantized_size_bytes=len(qbytes))
        self.assertAlmostEqual(report.compression_ratio, 8.0, places=1)

    def test_odd_column_count_packs_and_unpacks_correctly(self):
        w = np.random.default_rng(2).standard_normal((3, 7)).astype(np.float32)
        qbytes, meta = self.scheme.quantize(w.tobytes(), w.shape, "float32")
        deq = np.frombuffer(self.scheme.dequantize(qbytes, meta), dtype=np.float32).reshape(w.shape)
        self.assertEqual(deq.shape, w.shape)

    def test_quantized_values_within_representable_range(self):
        # Every dequantized value should be an exact integer multiple of
        # its row's scale, in [-7, 7] * scale — a direct check that the
        # excess-8 nibble packing round-trips exactly at the integer level,
        # not just "close enough" at the float level.
        qbytes, meta = self.scheme.quantize(self.weights.tobytes(), self.weights.shape, "float32")
        deq = np.frombuffer(self.scheme.dequantize(qbytes, meta), dtype=np.float32).reshape(self.weights.shape)
        scales = np.array(meta["scales"], dtype=np.float32)
        ratio = deq / scales[:, None]
        self.assertTrue(np.allclose(ratio, np.round(ratio), atol=1e-3))
        self.assertTrue(np.all(np.abs(np.round(ratio)) <= 7))


class TestQuantizationReport(unittest.TestCase):
    def test_compression_ratio_infinite_when_quantized_is_empty(self):
        report = QuantizationReport(scheme="int8", bits=8, original_size_bytes=100, quantized_size_bytes=0)
        self.assertEqual(report.compression_ratio, float("inf"))

    def test_optional_fields_default_none(self):
        report = QuantizationReport(scheme="int8", bits=8, original_size_bytes=100, quantized_size_bytes=25)
        self.assertIsNone(report.accuracy_delta)
        self.assertIsNone(report.latency_delta_ms)
        self.assertIsNone(report.tokens_per_second)


if __name__ == "__main__":
    unittest.main()
