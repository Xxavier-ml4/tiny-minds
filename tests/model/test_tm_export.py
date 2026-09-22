import os
import tempfile
import unittest

import numpy as np

import tinymind.runtime.format as fmt
from tinymind.model.config import ModelConfig
from tinymind.model.model import TinyMindTransformer
from tinymind.model.tm_export import TmExportError, export_to_tm, import_from_tm


def _tiny_config():
    return ModelConfig(hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2,
                       intermediate_size=32, max_seq_len=16, vocab_size=20)


class TestTmExportRoundTrip(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".tm")

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_logits_match_exactly_after_round_trip(self):
        model = TinyMindTransformer(_tiny_config(), seed=1)
        seq = np.array([[1, 2, 3]])
        before = model(seq).logits.data.copy()

        export_to_tm(model, self.path)
        loaded = import_from_tm(self.path)
        after = loaded(seq).logits.data

        self.assertTrue(np.array_equal(before, after))

    def test_config_preserved(self):
        model = TinyMindTransformer(_tiny_config(), seed=2)
        export_to_tm(model, self.path)
        loaded = import_from_tm(self.path)
        self.assertEqual(model.config, loaded.config)

    def test_checksum_protected(self):
        model = TinyMindTransformer(_tiny_config(), seed=3)
        export_to_tm(model, self.path)
        with open(self.path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            last_byte = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([last_byte[0] ^ 0xFF]))
        with self.assertRaises(fmt.CorruptModelFileError):
            import_from_tm(self.path)


class TestTmExportMalformedRejection(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".tm")

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_wrong_architecture_tag_rejected(self):
        fmt.write_model(self.path, metadata={"architecture": "not-tinymind"}, tensors={})
        with self.assertRaises(TmExportError):
            import_from_tm(self.path)

    def test_missing_model_config_rejected(self):
        fmt.write_model(self.path, metadata={"architecture": "tinymind-transformer-v1"}, tensors={})
        with self.assertRaises(TmExportError):
            import_from_tm(self.path)

    def test_missing_tensor_rejected(self):
        model = TinyMindTransformer(_tiny_config(), seed=4)
        export_to_tm(model, self.path)
        model_file = fmt.read_model(self.path)
        tensors = {name: (entry.dtype, entry.shape, model_file.read_tensor(name))
                  for name, entry in model_file.tensors.items()}
        dropped_name = next(iter(tensors))
        del tensors[dropped_name]
        fmt.write_model(self.path, metadata=model_file.metadata, tensors=tensors)
        with self.assertRaises(TmExportError):
            import_from_tm(self.path)

    def test_shape_mismatch_rejected(self):
        model = TinyMindTransformer(_tiny_config(), seed=5)
        export_to_tm(model, self.path)
        model_file = fmt.read_model(self.path)
        tensors = {name: (entry.dtype, entry.shape, model_file.read_tensor(name))
                  for name, entry in model_file.tensors.items()}
        name = "embed_tokens"
        dtype, shape, raw = tensors[name]
        tensors[name] = (dtype, (shape[1], shape[0]), raw)  # transposed shape, same byte count
        fmt.write_model(self.path, metadata=model_file.metadata, tensors=tensors)
        with self.assertRaises(TmExportError):
            import_from_tm(self.path)


if __name__ == "__main__":
    unittest.main()
