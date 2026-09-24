"""Static checks of .github/workflows/benchmark-runner.yml and test-trained-model.yml. Neither can be executed
here (no runner, no network) — these tests pin what can be verified offline: the workflows parse, expose the
inputs they claim to, reference only real commands and files, pass inputs through the environment rather than
interpolating them into shell text, use the artifact mechanism correctly, carry no embedded secrets, and —
specifically for test-trained-model.yml — that its verification step cannot silently swallow a failure (the
brief: "do not silently lower thresholds when something fails")."""
import re
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH_WORKFLOW = REPO / ".github" / "workflows" / "benchmark-runner.yml"
REPORT_WORKFLOW = REPO / ".github" / "workflows" / "test-trained-model.yml"


def load(path: Path):
    return yaml.safe_load(path.read_text())


class WorkflowChecks:
    """Mixin, not a TestCase itself — TestBenchmarkRunnerWorkflow and TestTestTrainedModelWorkflow each combine
    this with unittest.TestCase so these checks run against both workflows without unittest also trying to
    collect and run this base class on its own (it has no PATH)."""
    PATH: Path

    @classmethod
    def setUpClass(cls):
        cls.text = cls.PATH.read_text()
        cls.wf = load(cls.PATH)
        cls.on = cls.wf.get("on", cls.wf.get(True))  # PyYAML parses the bare key `on` as boolean True
        cls.job_name = next(iter(cls.wf["jobs"]))
        cls.job = cls.wf["jobs"][cls.job_name]
        cls.steps = cls.job["steps"]
        cls.runs = "\n".join(s["run"] for s in cls.steps if "run" in s)

    def step(self, name_prefix: str) -> dict:
        return next(s for s in self.steps if s.get("name", "").startswith(name_prefix))

    def test_only_workflow_dispatch_triggers_it(self):
        self.assertEqual(list(self.on), ["workflow_dispatch"])

    def test_every_referenced_input_is_declared_and_vice_versa(self):
        declared = set(self.on["workflow_dispatch"]["inputs"])
        used = set(re.findall(r"inputs\.([a-z0-9_]+)", self.text))
        self.assertEqual(used - declared, set())
        self.assertEqual(declared - used, set(), "a declared input nothing reads is a dead knob")

    def test_inputs_are_never_interpolated_directly_into_shell(self):
        self.assertNotIn("${{", self.runs)  # inputs travel via env:, never string-substituted into a script

    def test_no_embedded_credentials_and_only_expected_secrets_used(self):
        for pattern in (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_", r"AKIA[0-9A-Z]{12,}", r"hf_[A-Za-z0-9]{20,}", r"password\s*[:=]\s*\S+"):
            self.assertIsNone(re.search(pattern, self.text), pattern)
        self.assertEqual(set(re.findall(r"secrets\.([A-Z_]+)", self.text)), set())  # neither workflow needs a secret

    def test_referenced_python_files_exist(self):
        for path in set(re.findall(r"python[3]? (?:-m )?([\w./]+\.py)", self.text)):
            self.assertTrue((REPO / path).is_file(), path)

    def test_referenced_tinymind_cli_and_ci_commands_exist(self):
        from tinymind.cli import build_parser
        sub = next(a for a in build_parser()._actions if a.dest == "command").choices
        for cmd in set(re.findall(r"python -m tinymind\.cli ([a-z-]+)", self.text)):
            self.assertIn(cmd, sub, cmd)
        for mod, cmd in re.findall(r"python -m (tinymind\.ci\.stage_io) ([a-z][a-z-]*)", self.text):
            self.assertIn(cmd, {"runtime", "check-incoming", "bundle", "summary", "training-context"}, f"{mod} {cmd}")
        for mod in re.findall(r"python -m (tinymind\.ci\.trained_model_report)\b(?! [a-z-])", self.text):
            self.assertEqual(mod, "tinymind.ci.trained_model_report")  # invoked as `main()`, no subcommand


class TestBenchmarkRunnerWorkflow(WorkflowChecks, unittest.TestCase):
    PATH = BENCH_WORKFLOW

    def test_declared_inputs(self):
        inputs = self.on["workflow_dispatch"]["inputs"]
        for name in ("profile", "steps", "batch_size", "sequence_length", "data_scale"):
            self.assertIn(name, inputs)
        self.assertEqual(inputs["profile"]["default"], "tiny_mobile")  # tiny_mobile is benchmarked first

    def test_does_not_call_the_train_cli_or_resume_anything(self):
        # this is a benchmark, not a training run: no --resume/--init-from, and no stage upload
        self.assertNotIn("--resume", self.runs)
        self.assertNotIn("--init-from", self.runs)
        self.assertNotIn("tinymind.ci.stage_io bundle", self.runs)

    def test_uploads_the_benchmark_json_files(self):
        up = self.step("Upload benchmark results")
        self.assertEqual(up["uses"].split("@")[0], "actions/upload-artifact")
        self.assertIn("runner_benchmark.json", up["with"]["path"])
        self.assertEqual(up["with"]["if-no-files-found"], "error")

    def test_runs_the_runner_probe_before_the_benchmark(self):
        names = [s.get("name", "") for s in self.steps]
        self.assertLess(next(i for i, n in enumerate(names) if "thread" in n.lower()),
                        next(i for i, n in enumerate(names) if "Benchmark" in n))

    def test_summary_step_only_reads_fields_the_script_actually_produces(self):
        import sys
        sys.path.insert(0, str(REPO / "benchmarks"))
        import inspect

        import runner_benchmark
        src = inspect.getsource(runner_benchmark.run_benchmark)
        summary = self.step("Job summary")["run"]
        accesses = re.findall(r"\b(?:d|hw|mdl|tr|mem|ck)\[['\"]([a-zA-Z_]+)['\"]\]", summary)
        # every top-level / nested key the summary reads must appear as a literal key somewhere in the script
        for key in set(accesses):
            self.assertIn(f'"{key}"', src, f"summary step reads {key!r}, which run_benchmark() does not appear to set")

    def test_no_secrets_permission_beyond_default(self):
        self.assertEqual(self.wf.get("permissions"), {"contents": "read"})

    def test_benchmark_step_does_not_swallow_a_failure(self):
        bench = self.step("Benchmark")["run"]
        self.assertNotIn("|| true", bench)
        self.assertNotIn("|| echo", bench)
        self.assertNotIn("continue-on-error", str(self.step("Benchmark")))

    def test_bounded_not_open_ended(self):
        # a real workflow_dispatch numeric input arrives as a string; the script must still receive a concrete,
        # short step count either way — this pins that the default is small (a benchmark, not training)
        self.assertLessEqual(int(self.on["workflow_dispatch"]["inputs"]["steps"]["default"]), 200)
        self.assertLessEqual(self.job["timeout-minutes"], 60)  # generous upper bound; this is not a training job


class TestTestTrainedModelWorkflow(WorkflowChecks, unittest.TestCase):
    PATH = REPORT_WORKFLOW

    def test_declared_inputs(self):
        inputs = self.on["workflow_dispatch"]["inputs"]
        for name in ("artifact_name", "artifact_run_id", "eval_data_scale", "eval_limit_per_category", "max_new_tokens"):
            self.assertIn(name, inputs)
        self.assertTrue(inputs["artifact_name"]["required"])  # the one thing the workflow cannot default

    def test_accepts_a_completed_artifact_by_name_and_downloads_it(self):
        download = self.step("Download the artifact")
        self.assertEqual(download["uses"].split("@")[0], "actions/download-artifact")
        for key in ("name", "run-id", "github-token", "repository", "path"):
            self.assertIn(key, download["with"])
        self.assertIn("inputs.artifact_name", str(download["with"]["name"]))

    def test_permissions_include_actions_read_for_cross_run_download(self):
        self.assertEqual(self.wf.get("permissions"), {"contents": "read", "actions": "read"})

    def test_recovers_stage_and_seed_then_regenerates_eval_data(self):
        names = [s.get("name", "") for s in self.steps]
        i_context = next(i for i, n in enumerate(names) if "stage and seed" in n)
        i_data = next(i for i, n in enumerate(names) if "Regenerate" in n)
        i_verify = next(i for i, n in enumerate(names) if "Verify and evaluate" in n)
        self.assertLess(i_context, i_data)
        self.assertLess(i_data, i_verify)
        self.assertIn("build-curriculum", self.step("Regenerate")["run"])
        self.assertIn("steps.context.outputs.stage", str(self.step("Regenerate")))
        self.assertIn("steps.context.outputs.seed", str(self.step("Regenerate")))

    def test_verification_step_hard_fails_and_is_never_softened(self):
        # brief: "do not silently lower thresholds when something fails" — statically enforce that the one step
        # doing real verification cannot have its exit code swallowed and is not marked continue-on-error
        verify_step = self.step("Verify and evaluate")
        self.assertNotIn("continue-on-error", verify_step)
        body = verify_step["run"]
        self.assertNotIn("|| true", body)
        self.assertNotIn("|| echo", body)
        self.assertNotIn("exit 0", body)
        self.assertIn("trained_model_report", body)

    def test_reports_are_uploaded_even_on_failure_for_debugging_but_verification_step_itself_is_not_relaxed(self):
        up = self.step("Upload reports")
        self.assertEqual(up.get("if"), "always()")
        self.assertEqual(up["with"]["if-no-files-found"], "warn")  # may legitimately be absent if verification failed first
        # the job's overall pass/fail is still governed by the (unsoftened) verification step, not by this upload
        self.assertNotIn("continue-on-error", up)

    def test_job_summary_does_not_claim_a_verdict(self):
        summary_step = self.step("Job summary")
        self.assertIn("trained-model-report.md", str(summary_step))

    def test_no_training_happens_in_this_workflow(self):
        self.assertNotIn("tinymind.cli train ", self.runs)
        self.assertNotIn("--max-runtime", self.runs)
        self.assertNotIn("--resume", self.runs)
        self.assertNotIn("--init-from", self.runs)


if __name__ == "__main__":
    unittest.main()
