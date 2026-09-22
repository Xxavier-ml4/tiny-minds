"""Static checks of .github/workflows/train-stage.yml. It cannot be executed here (no runner, no network), so these
tests pin the properties that can be verified offline: that it parses, exposes the inputs the brief lists, references
only commands that exist, passes inputs through the environment, uses the artifact mechanism explicitly, and has no
embedded secrets. What a real run would additionally prove is listed in docs/training/implementation-report.md."""
import re
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "train-stage.yml"


def load():
    return yaml.safe_load(WORKFLOW.read_text())


class TestTrainStageWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text()
        cls.wf = load()
        cls.on = cls.wf.get("on", cls.wf.get(True))  # PyYAML parses the bare key `on` as boolean True
        cls.train = cls.wf["jobs"]["train"]
        cls.steps = cls.train["steps"]
        cls.runs = "\n".join(s["run"] for s in cls.steps if "run" in s)

    def test_dispatch_inputs_required_by_the_brief(self):
        inputs = self.on["workflow_dispatch"]["inputs"]
        for name in ("stage", "resume_artifact", "max_hours", "config", "dataset", "seed"):
            self.assertIn(name, inputs)
        self.assertEqual(inputs["stage"]["options"], ["stage0", "stage1", "stage2", "stage3"])
        self.assertEqual(list(self.on), ["workflow_dispatch"])  # no accidental triggers on push/PR

    def test_every_referenced_input_exists(self):
        declared = set(self.on["workflow_dispatch"]["inputs"])
        used = set(re.findall(r"inputs\.([a-z0-9_]+)", self.text))
        self.assertEqual(used - declared, set())
        self.assertEqual(declared - used, set(), "a declared input nothing reads is a dead knob")

    def test_inputs_are_never_interpolated_into_shell_scripts(self):
        self.assertNotIn("${{", self.runs)  # script-injection safe: inputs travel via env: and are quoted in the script

    def test_referenced_commands_exist(self):
        from tinymind.cli import build_parser
        parser = build_parser()
        sub = next(a for a in parser._actions if a.dest == "command").choices
        for match in re.finditer(r"python -m tinymind\.cli ([a-z-]+)(?: ([a-z-]+))?", self.text):
            command = match.group(1)
            self.assertIn(command, sub, command)
        import importlib
        from tinymind.ci import stage_io
        for cmd in set(re.findall(r"python -m tinymind\.ci\.stage_io ([a-z-]+)", self.text)):
            self.assertIn(cmd, {"runtime", "check-incoming", "bundle", "summary"})
        for path in re.findall(r"python (benchmarks/[a-z_]+\.py)", self.text):
            self.assertTrue((REPO / path).is_file(), path)
        importlib.import_module("tinymind.ci.publish_hf")
        self.assertTrue((REPO / "configs").is_dir())

    def test_stage_bundle_flags_are_accepted_by_the_real_argument_parsers(self):
        from tinymind.cli import build_parser
        p = build_parser()
        ns = p.parse_args(["train", "--config", "tiny_mobile", "--dataset", "data", "--output", "out", "--stage", "stage1", "--seed", "0",
                           "--max-runtime", "17100", "--safety-margin", "300", "--resume", "incoming/checkpoints"])
        self.assertEqual((ns.max_runtime, ns.safety_margin), (17100.0, 300.0))
        p.parse_args(["eval", "--package", "out/export", "--eval", "d/eval.jsonl", "--val", "d/val.jsonl", "--out", "o.json"])
        p.parse_args(["verify-checkpoint", "bundle/checkpoints"])
        p.parse_args(["verify-package", "bundle/export"])
        p.parse_args(["data", "build-curriculum", "--stage", "stage1", "--out", "data", "--seed", "0", "--scale", "1.0"])

    def test_artifact_transfer_is_explicit_and_verified(self):
        uses = [s.get("uses", "") for s in self.steps]
        self.assertTrue(any(u.startswith("actions/download-artifact@v4") for u in uses))
        self.assertTrue(any(u.startswith("actions/upload-artifact@v4") for u in uses))
        download = next(s for s in self.steps if s.get("uses", "").startswith("actions/download-artifact"))
        for key in ("name", "run-id", "github-token", "repository", "path"):
            self.assertIn(key, download["with"])  # cross-run download needs all of these
        names = [s["name"] for s in self.steps if "name" in s]
        self.assertLess(names.index("Download the previous artifact"), names.index("Verify the incoming artifact before trusting it"))
        self.assertLess(names.index("Verify the incoming artifact before trusting it"), names.index("Train"))
        for up in (s for s in self.steps if s.get("uses", "").startswith("actions/upload-artifact")):
            self.assertEqual(up["with"]["if-no-files-found"], "error")
            self.assertIn("steps.bundle.outputs.artifact_name", up["with"]["name"])  # deterministic, content-derived name
        self.assertLess(names.index("Verify the bundle we are about to upload"), names.index("Upload the stage bundle (checkpoints + inference package + summary)"))

    def test_the_trainer_owns_the_time_limit(self):
        self.assertIn("--max-runtime", self.runs)
        self.assertIn("--safety-margin", self.runs)
        self.assertLess(int(self.train["timeout-minutes"]), 360)  # only a backstop under GitHub's 6 h kill
        self.assertIn("runtime --max-hours", self.runs)

    def test_stage_promotion_paths(self):
        self.assertIn('--resume "$INCOMING_CKPT"', self.runs)
        self.assertIn('--init-from "$INCOMING_CKPT"', self.runs)
        self.assertIn("--gate-dir configs/stages", self.runs)

    def test_permissions_are_minimal(self):
        self.assertEqual(self.wf["permissions"], {"contents": "read", "actions": "read"})
        publish = self.wf["jobs"]["publish"]
        self.assertEqual(publish["permissions"], {"contents": "write"})
        self.assertTrue(publish["continue-on-error"])  # publishing can never fail the run

    def test_no_credentials_embedded(self):
        for pattern in (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_", r"AKIA[0-9A-Z]{12,}", r"hf_[A-Za-z0-9]{20,}", r"password\s*[:=]\s*\S+"):
            self.assertIsNone(re.search(pattern, self.text), pattern)
        secrets = set(re.findall(r"secrets\.([A-Z_]+)", self.text))
        self.assertEqual(secrets, {"HF_TOKEN"})  # the only secret, optional, and only in the publish job
        self.assertNotIn("secrets.", yaml.safe_dump(self.train))

    def test_publish_is_optional_and_guarded(self):
        publish = self.wf["jobs"]["publish"]
        self.assertIn("inputs.publish", publish["if"])
        hf = next(s for s in publish["steps"] if s.get("name", "").startswith("Hugging Face"))
        self.assertIn("env.HF_TOKEN", hf["if"])

    def test_no_deep_learning_framework_is_installed(self):
        for word in ("torch", "tensorflow", "jax"):
            self.assertNotRegex(self.runs, rf"pip install[^\n]*\b{word}\b")

    def test_gate_files_exist_for_every_promotable_stage(self):
        for stage in ("stage1", "stage2"):  # a stage that can be promoted from needs a gate file
            self.assertTrue((REPO / "configs" / "stages" / f"{stage}.gate.json").is_file(), stage)


if __name__ == "__main__":
    unittest.main()
