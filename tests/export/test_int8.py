"""INT8 package variant: real quantization, honest scope."""
import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from tinymind.export import PackageError, export_package, load_package, verify_package
from tinymind.export.budget import weight_bytes_int8
from tinymind.export.int8 import INT8_FILE, add_int8, load_int8_package
from tinymind.model import TinyMindTransformer
from tinymind.model.tokenizer import ByteTokenizer
from tinymind.quantization.model_quantizer import dequantize_to_model, import_quantized_tm, quantize_model

from tests.training._helpers import model_config, tmpdir


def build():
    m = TinyMindTransformer(model_config(), seed=9)
    d = tmpdir() / "pkg"
    export_package(m, ByteTokenizer(), d)
    return m, d


class TestInt8Package(unittest.TestCase):
    def test_round_trip_size_and_quantization_error(self):
        m, d = build()
        ids = np.array([[1, 12, 13, 14, 15, 16]])
        ref = m(ids).logits.data
        sizes = {}
        for embed in (False, True):
            entry = add_int8(d, quantize_embeddings=embed)
            sizes[embed] = entry["size"]
            q = load_int8_package(d)
            err = float(np.abs(q.model(ids).logits.data - ref).max())
            self.assertLess(err, 0.15 * float(np.abs(ref).max()) + 0.05)     # bounded, not zero: it IS lossy
            self.assertGreater(err, 0.0)
            self.assertTrue(verify_package(d).ok)                              # the fp32 files still verify
        self.assertLess(sizes[True], sizes[False])                             # quantizing the embedding saves more
        self.assertLess(sizes[False], 0.6 * (d / "model.tm").stat().st_size)
        manifest = json.loads((d / "package.json").read_text())
        self.assertEqual(manifest["weights_int8"]["file"], INT8_FILE)
        self.assertIn("not readable by the native runtime", manifest["weights_int8"]["runtime"])

    def test_int8_matches_the_quantizer_exactly_and_is_deterministic(self):
        m, d = build()
        add_int8(d, quantize_embeddings=False)
        direct = dequantize_to_model(quantize_model(m, "int8", quantize_embeddings=False))
        via_pkg = load_int8_package(d).model
        for (n, a), (_, b) in zip(direct.named_parameters(), via_pkg.named_parameters()):
            self.assertTrue(np.array_equal(a.data, b.data), n)
        self.assertNotIn("embed_tokens", import_quantized_tm(d / INT8_FILE).quantized)

    def test_int8_is_a_size_claim_only(self):
        m, d = build()
        add_int8(d)
        q = load_int8_package(d)
        self.assertEqual(sum(p.data.nbytes for p in q.model.parameters()), sum(p.data.nbytes for p in m.parameters()))  # dequantized in RAM

    def test_tampered_or_missing_int8_is_refused(self):
        m, d = build()
        with self.assertRaises(PackageError):
            load_int8_package(d)  # no int8 yet
        add_int8(d)
        f = d / INT8_FILE
        f.write_bytes(f.read_bytes()[:-20])
        with self.assertRaises(PackageError):
            load_int8_package(d)

    def test_size_matches_the_budget_formula(self):
        m, d = build()
        for embed in (False, True):
            add_int8(d, quantize_embeddings=embed)
            payload = weight_bytes_int8(m.config, embed)
            self.assertGreaterEqual((d / INT8_FILE).stat().st_size, payload)            # payload plus container overhead
            self.assertLess((d / INT8_FILE).stat().st_size, payload * 1.35 + 4096)      # ... which is small


if __name__ == "__main__":
    unittest.main()
