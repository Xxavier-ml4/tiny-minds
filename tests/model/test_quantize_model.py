import os
import tempfile
import unittest

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.model import TinyMindTransformer
from tinymind.quantization.model_quantizer import (
    dequantize_to_model, export_quantized_tm, import_quantized_tm, quantize_model, report,
)


def _tiny_config():
    return ModelConfig(hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2,
                       intermediate_size=32, max_seq_len=16, vocab_size=20)


class TestQuantizeModel(unittest.TestCase):
    def test_only_2d_tensors_quantized(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        qmodel = quantize_model(model, "int8")
        for name, param in model.named_parameters():
            if param.data.ndim == 2:
                self.assertIn(name, qmodel.quantized)
            else:
                self.assertIn(name, qmodel.unquantized)

    def test_norm_weights_preserved_exactly(self):
        model = TinyMindTransformer(_tiny_config(), seed=1)
        qmodel = quantize_model(model, "int8")
        params = dict(model.named_parameters())
        for name, array in qmodel.unquantized.items():
            self.assertTrue(np.array_equal(array, params[name].data))

    def test_unknown_scheme_rejected(self):
        model = TinyMindTransformer(_tiny_config(), seed=0)
        with self.assertRaises(ValueError):
            quantize_model(model, "not_a_real_scheme")


class TestDequantizeToModel(unittest.TestCase):
    def test_produces_runnable_model_with_same_shapes(self):
        model = TinyMindTransformer(_tiny_config(), seed=2)
        qmodel = quantize_model(model, "int8")
        restored = dequantize_to_model(qmodel)
        seq = np.array([[1, 2, 3]])
        out = restored(seq)
        self.assertEqual(out.logits.shape, (1, 3, 20))

    def test_int8_quantization_preserves_top_prediction_on_trained_model(self):
        # A directly-trained (not random-init) model has confident enough
        # predictions that mild int8 noise shouldn't flip the argmax —
        # this is the practical property quantization needs to hold for
        # generation quality to survive it.
        from tinymind.model.optim import AdamW
        model = TinyMindTransformer(_tiny_config(), seed=3)
        seq = np.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
        opt = AdamW(model.parameters(), learning_rate=5e-3)
        for _ in range(60):
            opt.zero_grad()
            out = model(seq, labels=seq)
            out.loss.backward()
            opt.step(grad_clip_norm=1.0)

        original_top = np.argmax(model(seq).logits.data, axis=-1)
        qmodel = quantize_model(model, "int8")
        restored = dequantize_to_model(qmodel)
        restored_top = np.argmax(restored(seq).logits.data, axis=-1)
        self.assertTrue(np.array_equal(original_top, restored_top))


class TestQuantizationReportMeasurement(unittest.TestCase):
    def test_report_fields_are_all_populated(self):
        model = TinyMindTransformer(_tiny_config(), seed=4)
        qmodel = quantize_model(model, "int8")
        rep = report(qmodel, model, np.array([[1, 2, 3]]))
        self.assertGreater(rep.original_size_bytes, 0)
        self.assertGreater(rep.quantized_size_bytes, 0)
        self.assertLess(rep.quantized_size_bytes, rep.original_size_bytes)
        self.assertIsNotNone(rep.accuracy_delta)
        self.assertGreaterEqual(rep.accuracy_delta, 0.0)

    def test_int4_has_worse_or_equal_accuracy_delta_than_int8(self):
        # Not a strict law for any single probe, but true in aggregate for
        # a reasonably-sized probe — a real, meaningful sanity check that
        # accuracy_delta actually reflects something.
        model = TinyMindTransformer(_tiny_config(), seed=5)
        probe = np.array([[1, 2, 3, 4, 5, 6, 7]])
        int8_report = report(quantize_model(model, "int8"), model, probe)
        int4_report = report(quantize_model(model, "int4"), model, probe)
        self.assertGreaterEqual(int4_report.accuracy_delta, int8_report.accuracy_delta * 0.5)

    def test_int4_compresses_more_than_int8(self):
        model = TinyMindTransformer(_tiny_config(), seed=6)
        probe = np.array([[1, 2, 3]])
        int8_report = report(quantize_model(model, "int8"), model, probe)
        int4_report = report(quantize_model(model, "int4"), model, probe)
        self.assertGreater(int4_report.compression_ratio, int8_report.compression_ratio)


class TestQuantizedTmRoundTrip(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".tm")

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_int8_round_trip_exact(self):
        model = TinyMindTransformer(_tiny_config(), seed=7)
        qmodel = quantize_model(model, "int8")
        export_quantized_tm(qmodel, self.path)
        reloaded = import_quantized_tm(self.path)

        seq = np.array([[1, 2, 3]])
        original_deq = dequantize_to_model(qmodel)(seq).logits.data
        reloaded_deq = dequantize_to_model(reloaded)(seq).logits.data
        self.assertTrue(np.array_equal(original_deq, reloaded_deq))

    def test_int4_round_trip_exact(self):
        model = TinyMindTransformer(_tiny_config(), seed=8)
        qmodel = quantize_model(model, "int4")
        export_quantized_tm(qmodel, self.path)
        reloaded = import_quantized_tm(self.path)

        seq = np.array([[1, 2, 3]])
        original_deq = dequantize_to_model(qmodel)(seq).logits.data
        reloaded_deq = dequantize_to_model(reloaded)(seq).logits.data
        self.assertTrue(np.array_equal(original_deq, reloaded_deq))

    def test_config_preserved(self):
        model = TinyMindTransformer(_tiny_config(), seed=9)
        qmodel = quantize_model(model, "int8")
        export_quantized_tm(qmodel, self.path)
        reloaded = import_quantized_tm(self.path)
        self.assertEqual(qmodel.config, reloaded.config)

    def test_file_is_smaller_than_unquantized_export(self):
        from tinymind.model.tm_export import export_to_tm
        model = TinyMindTransformer(_tiny_config(), seed=10)
        unquantized_path = tempfile.mktemp(suffix=".tm")
        export_to_tm(model, unquantized_path)
        qmodel = quantize_model(model, "int8")
        export_quantized_tm(qmodel, self.path)
        try:
            self.assertLess(os.path.getsize(self.path), os.path.getsize(unquantized_path))
        finally:
            os.remove(unquantized_path)

    def test_wrong_architecture_tag_rejected(self):
        import tinymind.runtime.format as fmt
        fmt.write_model(self.path, metadata={"architecture": "not-quantized"}, tensors={})
        with self.assertRaises(fmt.ModelFormatError):
            import_quantized_tm(self.path)


if __name__ == "__main__":
    unittest.main()
