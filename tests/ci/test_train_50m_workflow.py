import json
import unittest
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

WF = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "train-50m.yml"


@unittest.skipIf(yaml is None, "PyYAML not available")
class TestTrain50MWorkflow(unittest.TestCase):
    """Brief section 12: a CI workflow runs the seven-stage 50M curriculum,
    mirroring the existing multi-stage pipeline (identity checks, gates,
    artifact hand-off) with token-budget training and v2 data preparation."""

    @classmethod
    def setUpClass(cls):
        cls.doc = yaml.safe_load(WF.read_text())
        # PyYAML parses the top-level `on:` key as boolean True (YAML 1.1).
        cls.on = cls.doc.get("on", cls.doc.get(True))
        cls.train_steps = cls.doc["jobs"]["train"]["steps"]
        cls.run_text = " ".join(s.get("run", "") for s in cls.train_steps)

    def test_seven_stage_options(self):
        opts = self.on["workflow_dispatch"]["inputs"]["stage"]["options"]
        self.assertEqual(opts, [f"stage{i}" for i in range(1, 8)])

    def test_defaults_to_50m(self):
        self.assertEqual(self.on["workflow_dispatch"]["inputs"]["config"]["default"], "50m")

    def test_prepares_v2_data(self):
        self.assertIn("build-curriculum-v2", self.run_text)
        self.assertIn("prepare-external", self.run_text)

    def test_runs_preflight(self):
        self.assertIn("benchmark train-step", self.run_text)

    def test_trains_with_token_budget(self):
        train = next(s for s in self.train_steps if s.get("name", "").startswith("Train"))
        self.assertIn("--target-tokens", train["run"])

    def test_applies_promotion_gates(self):
        self.assertIn("gate-dir configs/stages", self.run_text)

    def test_verifies_identity_and_bundles(self):
        self.assertIn("check-incoming", self.run_text)
        self.assertIn("verify-checkpoint", self.run_text)
        self.assertIn("verify-package", self.run_text)

    # ---- real corpus, objective-driven stages, visible reports -------------------------------------
    def step(self, prefix):
        return next(s for s in self.train_steps if s.get("name", "").startswith(prefix))

    def test_prepares_the_real_corpus_hermetically_by_default(self):
        data = self.step("Prepare data")["run"]
        self.assertIn("data prepare-corpus", data)
        self.assertIn("--corpus data/corpus", data)
        self.assertIn('if [ "$ALLOW_DOWNLOAD" = "true" ]; then dl+=(--allow-download); fi', data)
        self.assertFalse(self.on["workflow_dispatch"]["inputs"]["allow_download"]["default"])

    def test_objective_reports_go_to_the_job_summary_and_an_artifact(self):
        report = self.step("Stage objective reports")
        self.assertIn("objective-report --run out --github-summary", report["run"])
        self.assertEqual(report["if"], "${{ always() }}")
        upload = self.step("Upload the stage objective reports")
        self.assertTrue(upload["uses"].startswith("actions/upload-artifact@"))
        self.assertEqual(upload["with"]["path"], "out/objective_reports")

    def test_unmet_objective_continues_the_same_stage(self):
        self.assertIn("continue", self.on["workflow_dispatch"]["inputs"]["resume_mode"]["options"])
        train = self.step("Train")["run"]
        self.assertIn("--continue-stage", train)
        self.assertIn("--reopen-stage", train)
        self.assertIn("PARENT_BUDGET_TOKENS", train)  # a continuation trains beyond the budget already used

    def test_no_promotion_on_budget_alone(self):
        train = self.step("Train")["run"]
        self.assertIn('if [ "$OBJECTIVE_GATE" = "false" ]', train)  # --no-objective only when explicitly asked
        self.assertTrue(self.on["workflow_dispatch"]["inputs"]["objective_gate"]["default"])
        self.assertIn("stage_complete == 'true'", self.step("Held-out evaluation")["if"])
        self.assertIn("stage_complete == 'true'", self.doc["jobs"]["publish"]["if"])

    # ---- audit fixes ------------------------------------------------------------------------------
    def test_a_failing_command_in_a_pipeline_fails_the_step(self):
        # Without an explicit shell GitHub runs `bash -e` (no pipefail): `prepare-corpus ... | tee x` would pass.
        self.assertEqual(self.doc["defaults"]["run"]["shell"], "bash")
        self.assertIn("| tee", self.run_text)  # the pipelines this protects
        for step in self.train_steps:
            self.assertNotIn("shell", step, step.get("name"))  # no step opts out

    def test_corpus_split_and_tokenizer_do_not_depend_on_the_training_seed(self):
        data = self.step("Prepare data")["run"]
        corpus_cmd = data[data.index("data prepare-corpus"):data.index("| tee corpus_prepare.json")]
        self.assertNotIn("$SEED", corpus_cmd)  # every stage's job must derive the same corpus split and tokenizer
        tok = self.step("Ensure the BPE tokenizer")["run"]
        self.assertIn("data tokenizer-sample", tok)
        self.assertIn("--seed 0", tok)
        self.assertNotIn("head -n", tok)  # never "the first N records"

    def test_trainer_budget_excludes_time_already_spent(self):
        self.assertIn('JOB_START_EPOCH=$(date +%s)', self.step("Check the time budget")["run"])
        train = self.step("Train")["run"]
        self.assertIn('elapsed=$(( $(date +%s) - JOB_START_EPOCH ))', train)
        self.assertIn('--max-runtime "$runtime"', train)
        self.assertNotIn('--max-runtime "$RUNTIME_SECONDS"', train)

    def test_private_corpus_tokens_reach_data_preparation_only(self):
        self.assertEqual(set(self.step("Prepare data").get("env", {})), {"HF_TOKEN", "CORPUS_TOKEN"})
        for step in self.train_steps:
            if not step.get("name", "").startswith("Prepare data"):
                self.assertNotIn("secrets.", json.dumps(step), step.get("name"))

    def test_bundle_carries_the_tokenizer_sample_provenance(self):
        self.assertIn("tokenizer_sample_manifest.json", self.step("Bundle")["run"])

    def test_inputs_never_interpolated_into_shell_and_all_declared(self):
        import re
        self.assertNotIn("${{", self.run_text)
        text = WF.read_text()
        declared = set(self.on["workflow_dispatch"]["inputs"])
        self.assertEqual(set(re.findall(r"inputs\.([a-z0-9_]+)", text)), declared)


if __name__ == "__main__":
    unittest.main()
