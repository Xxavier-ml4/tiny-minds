"""Phase 3B, phase L: the inference package is self-contained, verified, and
loadable without any training code or state."""
import hashlib
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np

from tinymind.export import PackageError, export_package, load_package, verify_package
from tinymind.model import TinyMindTransformer
from tinymind.model.tokenizer import ByteTokenizer

from tests.training._helpers import make_engine, model_config, tmpdir

REPO = Path(__file__).resolve().parents[2]
TOK = ByteTokenizer()


def build():
    model = TinyMindTransformer(model_config(), seed=5)
    out = tmpdir() / "pkg"
    manifest = export_package(model, TOK, out, provenance={"stage": "stage0", "global_step": 7})
    return model, out, manifest


class TestPackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.dir, cls.manifest = build()

    def test_contains_only_inference_files(self):
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["model.tm", "package.json", "tokenizer.json"])
        for key in ("optimizer_state", "training_data", "training_framework", "network"):
            self.assertFalse(self.manifest["requires"][key])

    def test_manifest_content_and_checksums(self):
        m = json.loads((self.dir / "package.json").read_text())
        self.assertEqual((m["kind"], m["architecture"], m["parameter_count"]), ("tinymind-inference-package", "tinymind-transformer-v1", self.model.count_parameters()))
        self.assertEqual(m["model_config"], self.model.config.to_dict())
        self.assertEqual(m["tokenizer"]["type"], "byte")
        self.assertEqual(m["renderer"]["template"], "tinymind-chat-v1")
        self.assertEqual(m["provenance"]["global_step"], 7)
        for name, info in m["files"].items():
            self.assertEqual(info["sha256"], hashlib.sha256((self.dir / name).read_bytes()).hexdigest())
        self.assertTrue(verify_package(self.dir).ok)

    def test_load_reproduces_the_model_exactly(self):
        pkg = load_package(self.dir)
        ids = np.array([[1, 10, 20, 30, 40, 2]])
        np.testing.assert_array_equal(pkg.model(ids).logits.data, self.model(ids).logits.data)
        self.assertEqual(pkg.tokenizer.spec(), TOK.spec())

    def test_generation_uses_the_training_template(self):
        pkg = load_package(self.dir)
        ids = pkg.prompt_ids("What is 2+2?")
        self.assertEqual(TOK.decode(ids[1:]), "user:\nWhat is 2+2?\nassistant:\n")
        new, reason = pkg.generate_ids("hi", max_new_tokens=5)
        self.assertLessEqual(len(new), 5)
        self.assertIn(reason, ("stop", "length"))
        self.assertIsInstance(pkg.generate("hi", max_new_tokens=3), str)
        a = pkg.generate("hi", max_new_tokens=8, temperature=0.9, seed=4)
        self.assertEqual(a, pkg.generate("hi", max_new_tokens=8, temperature=0.9, seed=4))

    def corrupt(self, name, mutate):
        _, d, _ = build()
        mutate(d / name)
        return d

    def test_any_tampering_is_detected(self):
        flip = lambda p: p.write_bytes(bytes(b ^ 1 if i == len(p.read_bytes()) // 2 else b for i, b in enumerate(p.read_bytes())))
        for name in ("model.tm", "tokenizer.json"):
            d = self.corrupt(name, flip)
            self.assertFalse(verify_package(d).ok, name)
            with self.assertRaises(PackageError):
                load_package(d)
        d = self.corrupt("package.json", lambda p: p.write_text(p.read_text().replace('"byte"', '"bpe"')))
        with self.assertRaises(PackageError):  # the manifest changed but the files did not: unusable, not silently fixed
            load_package(d)
        d = self.corrupt("model.tm", lambda p: p.unlink())
        self.assertIn("model.tm missing", verify_package(d).errors)

    def test_tokenizer_swap_is_detected_even_if_the_hash_is_recomputed(self):
        _, d, m = build()
        spec = json.loads((d / "tokenizer.json").read_text())
        spec["vocab_size"] = 300
        (d / "tokenizer.json").write_text(json.dumps(spec))
        m["files"]["tokenizer.json"] = {"sha256": hashlib.sha256((d / "tokenizer.json").read_bytes()).hexdigest(), "size": (d / "tokenizer.json").stat().st_size}
        (d / "package.json").write_text(json.dumps(m))
        self.assertFalse(verify_package(d).ok)  # the spec no longer matches the copy embedded in model.tm / manifest hash

    def test_loads_in_a_process_where_training_code_cannot_be_imported(self):
        expected = hashlib.sha256(self.model(np.array([[1, 5, 6, 7, 2]])).logits.data.tobytes()).hexdigest()
        code = textwrap.dedent(f"""
            import sys, hashlib, importlib.abc
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name == "tinymind.training" or name.startswith("tinymind.training."):
                        raise ImportError("training code is not available on the device")
            sys.meta_path.insert(0, Block())
            sys.path.insert(0, {str(REPO)!r})
            import numpy as np
            from tinymind.export import load_package
            pkg = load_package({str(self.dir)!r})
            print(hashlib.sha256(pkg.model(np.array([[1, 5, 6, 7, 2]])).logits.data.tobytes()).hexdigest())
            print(any(m.startswith("tinymind.training") for m in sys.modules))
            print(pkg.generate("ping", max_new_tokens=4) is not None)
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        digest, imported_training, generated = out.stdout.split()
        self.assertEqual(digest, expected)
        self.assertEqual((imported_training, generated), ("False", "True"))

    def test_failed_export_leaves_the_previous_package_intact(self):
        model, d, m = build()
        import tinymind.export.package as pkgmod
        real = pkgmod.export_to_tm
        pkgmod.export_to_tm = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            with self.assertRaises(OSError):
                export_package(model, TOK, d)
        finally:
            pkgmod.export_to_tm = real
        self.assertTrue(verify_package(d).ok)
        self.assertEqual([p.name for p in d.parent.iterdir()], ["pkg"])  # no temp litter

    def test_re_export_replaces_atomically(self):
        model, d, _ = build()
        model2 = TinyMindTransformer(model_config(), seed=6)
        export_package(model2, TOK, d)
        np.testing.assert_array_equal(load_package(d).model(np.array([[1, 2, 3]])).logits.data,
                                      model2(np.array([[1, 2, 3]])).logits.data)
        self.assertEqual([p.name for p in d.parent.iterdir()], ["pkg"])

    def test_vocab_mismatch_is_refused_at_export(self):
        bad = TinyMindTransformer(model_config(vocab_size=300), seed=1)
        with self.assertRaises(PackageError):
            export_package(bad, TOK, tmpdir() / "x")


class TestEngineExports(unittest.TestCase):
    def test_every_run_ends_with_a_verified_package(self):
        from tinymind.training.exporter import package_exporter
        out = tmpdir()
        e = make_engine(out, train_over=dict(max_steps=12, checkpoint_interval=6), exporter=package_exporter)
        summary = e.train()
        self.assertTrue(verify_package(out / "export").ok)
        pkg = load_package(out / "export")
        prov = pkg.manifest["provenance"]
        self.assertEqual((prov["stage"], prov["global_step"], prov["stage_complete"]), ("stage0", 12, True))
        self.assertEqual(prov["dataset_hash"], e.plan.dataset_hash())
        self.assertEqual(summary["exports"]["tm"], str(out / "export" / "model.tm"))
        ids = np.array([[1, 9, 8, 7]])
        np.testing.assert_array_equal(pkg.model(ids).logits.data, e.model(ids).logits.data)  # exports the trained weights


if __name__ == "__main__":
    unittest.main()
