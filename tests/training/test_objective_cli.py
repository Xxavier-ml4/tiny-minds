"""The objective-driven pipeline through the real command line: prepare-corpus ->
build-curriculum-v2 --corpus -> train (objective switched on automatically for
v2 curriculum data, natural held-out text and lexicon picked up, a
placeholder-sized corpus refused up front) -> objective-report (every checkpoint,
raw generations) -> continue-stage."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tinymind.cli import main

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "datasets" / "v2" / "samples" / "stage1_language_sample.txt"


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(map(str, argv)))
    return code, out.getvalue(), err.getvalue()


class TestObjectivePipelineCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp(prefix="tm-objcli-"))
        manifest = json.loads((REPO / "datasets" / "v2" / "manifest.json").read_text())
        sample = dict(manifest["datasets"][0], shards=[str(SAMPLE)], chunk_chars=200, group_chunks=2)
        (cls.d / "manifest.json").write_text(json.dumps({"version": 2, "datasets": [sample]}))
        code, out, err = cli("data", "prepare-corpus", "--manifest", cls.d / "manifest.json", "--out", cls.d / "corpus",
                             "--validation", "0.1", "--test", "0.1")
        assert code == 0, err
        cls.prepared = json.loads(out)
        code, out, err = cli("data", "build-curriculum-v2", "--stage", "stage1", "--out", cls.d / "data", "--scale", "0.1",
                             "--corpus", cls.d / "corpus")
        assert code == 0, err
        cls.built = json.loads(out)
        # thresholds for a two-minute test: the real stage1 objective, minus its 20 MB corpus requirement
        cls.objectives = cls.d / "objectives"
        cls.objectives.mkdir()
        obj = json.loads((REPO / "configs" / "stages_v2" / "stage1.objective.json").read_text())
        for m in obj["measurements"]:
            if m["metric"] == "data.natural_train_bytes":
                m["min"] = 1000
        (cls.objectives / "stage1.objective.json").write_text(json.dumps(obj))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def train(self, out, *extra, steps=12):
        return cli("train", "--config", "tiny_mobile", "--dataset", self.d / "data", "--output", self.d / out,
                   "--stage", "stage1", "--max-steps", steps, "--warmup-steps", 2, "--eval-interval", 6,
                   "--checkpoint-interval", 6, "--seed", 0, *extra)

    def test_prepare_and_build_report_provenance(self):
        self.assertEqual(self.prepared["natural_fraction"], 1.0)
        self.assertGreater(self.prepared["splits"]["val"]["records"], 0)
        self.assertEqual(len(self.prepared["corpus_sha256"]), 64)
        self.assertTrue(self.built["corpus"]["used"])
        self.assertIn("corpus", self.built["mixture"])

    def test_default_stage1_objective_refuses_a_placeholder_corpus(self):
        code, _, err = self.train("refused")
        self.assertEqual(code, 1)
        self.assertIn("data.natural_train_bytes", err)
        self.assertIn("never be met", err)
        self.assertFalse((self.d / "refused" / "checkpoints").exists())  # refused before any training

    def test_train_with_objective_then_continue_and_report(self):
        code, out, err = self.train("run", "--objective-dir", self.objectives)
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual((result["stop_reason"], result["stage_complete"], result["objective_met"]),
                         ("gate_failed", False, False))  # 12 tiny steps do not teach language
        self.assertIn("natural held-out text:", err)
        rep = json.loads((self.d / "run" / "objective_reports" / "latest.json").read_text())
        self.assertIn("text_val_bpb", rep["metrics"]["loss"])  # measured on val_text.jsonl
        self.assertIsNotNone(rep["metrics"]["grammar"]["known_word_fraction"])  # lexicon.txt was used
        # resume refuses a used-up budget; continue-stage extends it
        code, _, err = self.train("resumed", "--objective-dir", self.objectives, "--resume", self.d / "run" / "checkpoints")
        self.assertEqual(code, 1)
        self.assertIn("--continue-stage", err)
        code, out, err = self.train("continued", "--objective-dir", self.objectives, "--continue-stage",
                                    self.d / "run" / "checkpoints", steps=18)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["final_step"], 18)
        # the report command renders every checkpoint, the raw generations and the history
        md_file = self.d / "summary.md"
        code, out, err = cli("objective-report", "--run", self.d / "run", "--out", md_file)
        self.assertEqual(code, 0, err)
        text = md_file.read_text()
        self.assertIn("| step | budget tokens |", text)
        self.assertEqual(text.count("| baseline |") + text.count("| not met |"), 3)  # steps 0, 6, 12
        self.assertIn("The sun rose over the", text)
        self.assertIn("Raw generation 8", text)
        self.assertEqual(text.count("<details><summary>Checkpoint step-"), 2)  # steps 0 and 6, collapsible
        continued = json.loads((self.d / "continued" / "objective_reports" / "latest.json").read_text())
        self.assertEqual(continued["regression"]["checks"][0]["scope"], "vs previous checkpoint (step 12)")
        code, _, _ = cli("objective-report", "--run", self.d / "continued", "--fail-if-not-met")
        self.assertEqual(code, 1)

    def test_training_output_and_bundle_carry_the_data_provenance(self):
        from tinymind.ci.stage_io import bundle_run, verify_bundle
        code, _, err = self.train("prov", "--objective-dir", self.objectives, steps=6)
        self.assertEqual(code, 0, err)
        prov = self.d / "prov" / "data_provenance"
        corpus = json.loads((prov / "corpus_manifest.json").read_text())
        self.assertEqual(corpus["sources"][0]["license"], "CC0-1.0 (public domain dedication; written for this repository)")
        self.assertTrue(corpus["sources"][0]["shards"][0]["pinned"])
        self.assertEqual(json.loads((prov / "curriculum.json").read_text())["corpus"]["corpus_sha256"], corpus["corpus_sha256"])
        bundle = self.d / "prov_bundle"
        bundle_run(self.d / "prov", bundle, profile="tiny_mobile", stage="stage1", run_id="prov")
        m, _ = verify_bundle(bundle)
        self.assertIn("data_provenance/corpus_manifest.json", m["files"])  # hashed with the rest of the bundle

    def test_natural_text_dropped_for_length_does_not_count(self):
        # Audit finding: chunks longer than max_seq_len are dropped when rendered, but the objective's data fact
        # still counted them, so the Stage-1 natural-text requirement could pass on text that is never trained on.
        d = self.d / "long"
        d.mkdir()
        paragraphs = [p for p in SAMPLE.read_text().split("\n\n") if p.strip()]
        with (d / "docs.jsonl").open("w") as f:  # short documents fit the context, long ones do not
            for i in range(24):
                text = paragraphs[i % len(paragraphs)][:150] if i % 2 else " ".join(paragraphs[i % 10:i % 10 + 5])
                f.write(json.dumps({"text": f"Document {i}. {text}"}) + "\n")
        (d / "m.json").write_text(json.dumps({"version": 2, "datasets": [
            {"source": "mixed", "type": "local", "format": "jsonl", "shards": [str(d / "docs.jsonl")],
             "chunk_chars": 1000, "license": "CC0-1.0"}]}))
        self.assertEqual(cli("data", "prepare-corpus", "--manifest", d / "m.json", "--out", d / "corpus",
                             "--validation", "0.2", "--test", "0.2")[0], 0)
        code, out, err = cli("data", "build-curriculum-v2", "--stage", "stage1", "--out", d / "data", "--scale", "0.1",
                             "--corpus", d / "corpus")
        self.assertEqual(code, 0, err)
        listed = json.loads(out)["corpus"]["natural_train_bytes"]
        objectives = d / "objectives"
        objectives.mkdir()
        (objectives / "stage1.objective.json").write_text(json.dumps({
            "stage": "stage1", "generation": {"prompts": ["The"], "max_new_tokens": 4},
            "measurements": [{"metric": "loss.val_loss", "max": 1e9},
                             {"metric": "data.natural_train_bytes", "min": listed // 2}]}))
        code, _, err = cli("train", "--config", "tiny_mobile", "--dataset", d / "data", "--output", d / "run", "--stage",
                           "stage1", "--max-steps", 4, "--warmup-steps", 1, "--overflow", "drop",
                           "--objective-dir", objectives)
        self.assertEqual(code, 1)  # ~1000-character chunks cannot fit tiny_mobile's 256-token context
        self.assertIn("were dropped", err)
        self.assertIn("data.natural_train_bytes", err)

    def test_no_objective_is_an_explicit_engineering_choice(self):
        code, out, err = self.train("noobj", "--no-objective", steps=6)
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual((result["stop_reason"], result["stage_complete"], result["objective_met"]),
                         ("complete", True, None))

    def test_explicit_objective_dir_without_the_stage_config_is_an_error(self):
        empty = self.d / "empty_objectives"
        empty.mkdir(exist_ok=True)
        code, _, err = self.train("noconfig", "--objective-dir", empty, steps=6)
        self.assertEqual(code, 1)
        self.assertIn("stage1.objective.json", err)


if __name__ == "__main__":
    unittest.main()
