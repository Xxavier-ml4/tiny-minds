import contextlib
import io
import unittest

from tinymind.cli import main


def _run(argv):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCLI(unittest.TestCase):
    def test_version(self):
        code, out, _err = _run(["version"])
        self.assertEqual(code, 0)
        self.assertIn("tinymind", out)

    def test_tools_list(self):
        code, out, _err = _run(["tools", "list"])
        self.assertEqual(code, 0)
        self.assertIn("calculator", out)

    def test_model_info(self):
        code, out, _err = _run(["model", "info", "150m"])
        self.assertEqual(code, 0)
        self.assertIn("hidden_size", out)

    def test_model_info_unknown_preset_fails_cleanly(self):
        code, _out, err = _run(["model", "info", "does-not-exist"])
        self.assertNotEqual(code, 0)
        self.assertIn("no preset", err)

    def test_run_chat_mode(self):
        code, out, _err = _run(["run", "models/x.tm", "hello there"])
        self.assertEqual(code, 0)
        self.assertIn("hello there", out)

    def test_run_tool_call_reports_honest_failure(self):
        code, _out, err = _run(["run", "models/x.tm", "Calculate 847 times 39"])
        self.assertEqual(code, 1)
        self.assertIn("EchoBackend", err)

    def test_not_implemented_command_exits_nonzero(self):
        # "train" itself is real now (Phase 3A) — use a command that's
        # still genuinely not implemented, per _NOT_YET_IMPLEMENTED.
        code, _out, err = _run(["finetune", "--config", "foo.yaml"])
        self.assertEqual(code, 2)
        self.assertIn("not implemented", err)

    def test_benchmark_tools(self):
        code, out, _err = _run(["benchmark", "tools"])
        self.assertEqual(code, 0)
        self.assertIn("desk", out)

    def test_benchmark_unpopulated_suite_fails_cleanly(self):
        code, _out, err = _run(["benchmark", "capability"])
        self.assertNotEqual(code, 0)
        self.assertIn("STATUS.md", err)

    def test_model_info_on_real_tm_file_shows_exact_count(self):
        import shutil
        import tempfile
        from tinymind.model import ModelConfig, TinyMindTransformer
        from tinymind.model.tm_export import export_to_tm

        config = ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2,
                             intermediate_size=32, max_seq_len=16, vocab_size=20)
        model = TinyMindTransformer(config, seed=0)
        path = tempfile.mktemp(suffix=".tm")
        export_to_tm(model, path)
        try:
            code, out, err = _run(["model", "info", path])
            self.assertEqual(code, 0)
            self.assertIn("hidden_size", out)
            self.assertIn("exact parameter count", err)
            self.assertIn(str(model.count_parameters()), err.replace(",", ""))
        finally:
            import os
            os.remove(path)

    def test_generate_command_produces_output(self):
        import tempfile
        from tinymind.model import ModelConfig, TinyMindTransformer, ByteTokenizer
        from tinymind.model.tm_export import export_to_tm

        tok = ByteTokenizer()
        config = ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2,
                             intermediate_size=32, max_seq_len=32, vocab_size=tok.vocab_size)
        model = TinyMindTransformer(config, seed=0)
        path = tempfile.mktemp(suffix=".tm")
        export_to_tm(model, path)
        try:
            code, out, err = _run(["generate", "--model", path, "--prompt", "hi", "--max-new-tokens", "5"])
            self.assertEqual(code, 0)
            self.assertIn("tokens", err)
        finally:
            import os
            os.remove(path)

    def test_generate_command_missing_model_fails_cleanly(self):
        code, _out, err = _run(["generate", "--model", "/nonexistent.tm", "--prompt", "hi"])
        self.assertNotEqual(code, 0)
        self.assertIn("error", err)

    def test_train_and_quantize_commands_end_to_end(self):
        import json
        import shutil
        import tempfile
        from pathlib import Path

        tmpdir = Path(tempfile.mkdtemp())
        try:
            with (tmpdir / "train.jsonl").open("w") as f:
                for i in range(4):
                    f.write(json.dumps({"id": f"ex_{i}", "messages": [{"role": "user", "content": "cli test"}],
                                        "target": {"type": "answer", "content": ""}}) + "\n")
            with (tmpdir / "config.yaml").open("w") as f:
                f.write("model:\n  hidden_size: 16\n  num_layers: 1\n  num_heads: 4\n  num_kv_heads: 2\n"
                       "  intermediate_size: 32\n  max_seq_len: 16\n  vocab_size: 260\n")

            code, out, err = _run(["train", "--config", str(tmpdir / "config.yaml"),
                                   "--dataset", str(tmpdir / "train.jsonl"),
                                   "--output", str(tmpdir / "checkpoint"),
                                   "--epochs", "3", "--batch-size", "2", "--legacy"])  # Phase 3A trainer, explicitly
            self.assertEqual(code, 0, err)
            self.assertIn("Phase 3A trainer", err)
            self.assertTrue((tmpdir / "checkpoint" / "weights.npz").exists())

            from tinymind.model.checkpoint import from_pretrained
            from tinymind.model.tm_export import export_to_tm
            model = from_pretrained(str(tmpdir / "checkpoint"))
            export_to_tm(model, str(tmpdir / "model.tm"))

            code, out, err = _run(["quantize", "--model", str(tmpdir / "model.tm"), "--scheme", "int8",
                                   "--output", str(tmpdir / "model-int8.tm")])
            self.assertEqual(code, 0, err)
            self.assertIn("compression_ratio", out)
            self.assertTrue((tmpdir / "model-int8.tm").exists())
        finally:
            shutil.rmtree(tmpdir)

    def test_train_vocab_mismatch_fails_cleanly(self):
        import tempfile
        from pathlib import Path
        tmpdir = Path(tempfile.mkdtemp())
        with (tmpdir / "config.yaml").open("w") as f:
            f.write("model:\n  vocab_size: 999\n")  # doesn't match ByteTokenizer's 260
        with (tmpdir / "train.jsonl").open("w") as f:
            f.write('{"id": "a", "messages": [{"role": "user", "content": "x"}], "target": {"type": "answer", "content": ""}}\n')
        code, _out, err = _run(["train", "--config", str(tmpdir / "config.yaml"),
                               "--dataset", str(tmpdir / "train.jsonl"), "--output", str(tmpdir / "out")])
        self.assertNotEqual(code, 0)
        self.assertIn("vocab_size", err)


if __name__ == "__main__":
    unittest.main()
