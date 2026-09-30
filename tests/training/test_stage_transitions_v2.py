import tempfile
import unittest
from pathlib import Path

from tinymind.data import curriculum_v2 as C2
from tinymind.data.contamination import normalize
from tinymind.data.external import DatasetManifestError, prepare


class TestHeldOutIndependence(unittest.TestCase):
    """Brief section 11: the held-out test set is generated independently from
    held-out buckets and is disjoint from training even after normalisation."""

    def test_normalised_disjointness(self):
        data = C2.build_stage("stage2", seed=3, scale=0.12)
        train_norm = {normalize(C2.prompt_key(r)) for rs in data["train"].values() for r in rs}
        test_norm = {normalize(C2.prompt_key(r)) for r in data["test"]}
        self.assertEqual(train_norm & test_norm, set())

    def test_recompute_detects_a_tampered_answer(self):
        # A correct arithmetic record verifies; corrupting its stored answer must fail.
        import random
        ctx = C2.Ctx(random.Random(0), "train")
        for _ in range(50):
            r = C2.g_arith(ctx)
            if r["meta"]["sub"] == "arith":
                break
        self.assertTrue(C2.verify_record(r))
        bad = {**r, "meta": {**r["meta"], "expected": str(int(r["meta"]["expected"]) + 1)}}
        self.assertFalse(C2.verify_record(bad))


class TestExternalDataset(unittest.TestCase):
    """Brief section 13: external/synthetic datasets are prepared deterministically
    and a url entry is refused without an explicit network opt-in."""

    def test_synthetic_prepare_is_deterministic(self):
        manifest = {
            "version": 2,
            "datasets": [
                {"source": "lang", "type": "synthetic", "generator": "language",
                 "examples": 50, "seed": 0, "split": "train", "path": "x"},
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "manifest.json"
            import json
            m.write_text(json.dumps(manifest))
            a = prepare(str(m), str(Path(d) / "a"))
            b = prepare(str(m), str(Path(d) / "b"))
            self.assertEqual(a["entries"][0]["sha256"], b["entries"][0]["sha256"])
            self.assertEqual(a["entries"][0]["examples"], b["entries"][0]["examples"])

    def test_url_entry_refused_without_download(self):
        manifest = {"version": 2, "datasets": [
            {"source": "web", "type": "url", "path": "https://example.com/data.jsonl", "split": "train"}]}
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "manifest.json"
            import json
            m.write_text(json.dumps(manifest))
            with self.assertRaises(DatasetManifestError):
                prepare(str(m), str(Path(d) / "out"), allow_download=False)

    def test_repo_manifest_prepares(self):
        repo_manifest = Path(__file__).resolve().parents[2] / "datasets" / "v2" / "manifest.json"
        if not repo_manifest.is_file():
            self.skipTest("repo manifest not present")
        with tempfile.TemporaryDirectory() as d:
            # The default manifest must prepare HERMETICALLY (no allow_download). It now holds the committed
            # natural-text sample ('local', sha256-pinned) plus supplemental 'synthetic' entries; neither needs a
            # network. A real corpus is added through 'url' entries (datasets/v2/corpus.manifest.example.json).
            m = prepare(str(repo_manifest), d, allow_download=False)
            self.assertGreaterEqual(len(m["entries"]), 1)
            for e in m["entries"]:
                self.assertIn(e["type"], ("synthetic", "local"))  # default manifest is hermetic: no 'url' entry
            self.assertTrue(any(e["natural"] for e in m["entries"]), "expected a natural-text sample entry")


if __name__ == "__main__":
    unittest.main()
