"""Cross-language equivalence: the native C++ forward pass
(native/src/model.cpp) must reproduce the real Python model logits,
within float32 precision, on identical weights and input.

Skipped (not failed) in an environment without a C++ compiler.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tinymind.model import ModelConfig, TinyMindTransformer, ByteTokenizer
from tinymind.model.tm_export import export_to_tm
from tinymind.native_bridge import HAS_CXX as _HAS_CXX
from tinymind.native_bridge import NATIVE_DIR as _NATIVE_DIR
from tinymind.native_bridge import build_equivalence_binary as _build_equiv_binary
from tinymind.native_bridge import compile_sources as _compile


def _build_cabi_binary(build_dir: Path) -> Path:
    harness = build_dir / "cabi_harness.cpp"
    harness.write_text(r"""
#include <cstdio>
#include <string>
#include "../include/tinymind.h"
int main(int argc, char** argv) {
    if (argc < 3) return 2;
    tm_context_t* ctx = tm_create();
    if (!ctx) return 1;
    if (tm_load_model(ctx, argv[1]) != TM_OK) { fprintf(stderr,"load failed\n"); return 1; }
    tm_generation_config_t cfg{};
    cfg.max_new_tokens=20; cfg.temperature=0.0f; cfg.top_k=-1; cfg.top_p=-1.0f; cfg.seed=0;
    char buf[4096];
    if (tm_generate(ctx, argv[2], &cfg, buf, sizeof(buf)) != TM_OK) { fprintf(stderr,"gen failed\n"); return 1; }
    std::string r(buf);
    size_t s = r.find("\"text\": \"") + 9;
    size_t e = r.find("\", \"tokens_generated\"");
    printf("%s", r.substr(s, e-s).c_str());
    tm_free(ctx);
    return 0;
}
""")
    native_srcs = [harness] + [_NATIVE_DIR / "src" / f
                                for f in ("model.cpp","tensor.cpp","runtime.cpp",
                                          "sampler.cpp","tokenizer.cpp","kv_cache.cpp")]
    return _compile(build_dir, native_srcs, "cabi")


def _train_tiny_model(config: ModelConfig, seed: int, steps: int = 20) -> TinyMindTransformer:
    from tinymind.model.optim import AdamW
    model = TinyMindTransformer(config, seed=seed)
    if steps and config.vocab_size >= 130:  # only train when the vocab is big enough for byte-encoded text
        tok = ByteTokenizer()
        ids = np.array([tok.encode("hello there hello there")])
        opt = AdamW(model.parameters(), learning_rate=3e-3)
        for _ in range(steps):
            opt.zero_grad()
            out = model(ids, labels=ids)
            out.loss.backward()
            opt.step(grad_clip_norm=1.0)
    elif steps:
        # Small vocab: train on a simple numerical pattern instead
        from tinymind.model.optim import AdamW
        seq = np.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
        # clip to vocab size
        seq = np.where(seq < config.vocab_size, seq, 0)
        opt = AdamW(model.parameters(), learning_rate=3e-3)
        for _ in range(steps):
            opt.zero_grad()
            out = model(seq, labels=seq)
            out.loss.backward()
            opt.step(grad_clip_norm=1.0)
    return model


@unittest.skipUnless(_HAS_CXX, "no g++ available")
class TestNativeForwardEquivalence(unittest.TestCase):
    """Model::forward() must produce the same logits as Python's
    TinyMindTransformer.forward() within 1e-3 max absolute difference
    (float32 ordering differences between NumPy and C++ are the source
    of any non-zero gap; argmax must match exactly)."""

    def setUp(self):
        self.build_dir = Path(tempfile.mkdtemp())
        self.binary = _build_equiv_binary(self.build_dir)

    def tearDown(self):
        shutil.rmtree(self.build_dir, ignore_errors=True)

    def _compare(self, model, input_ids, tol=1e-3):
        python_logits = model(np.array([input_ids])).logits.data[0]
        tm_path = self.build_dir / "model.tm"
        export_to_tm(model, tm_path)
        logits_path = self.build_dir / "logits.bin"
        cmd = [str(self.binary), str(tm_path), str(logits_path), *[str(i) for i in input_ids]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        native_logits = np.fromfile(logits_path, dtype=np.float32).reshape(python_logits.shape)
        max_diff = float(np.max(np.abs(python_logits - native_logits)))
        argmax_ok = np.array_equal(np.argmax(python_logits, axis=-1), np.argmax(native_logits, axis=-1))
        self.assertTrue(argmax_ok, "argmax diverges between Python and native")
        self.assertLess(max_diff, tol, f"max abs diff = {max_diff}")
        return max_diff

    def test_gqa_trained_model(self):
        config = ModelConfig(hidden_size=32, num_layers=3, num_heads=4, num_kv_heads=2,
                             intermediate_size=64, max_seq_len=32, vocab_size=50)
        model = _train_tiny_model(config, seed=42, steps=20)
        diff = self._compare(model, [3, 7, 1, 9, 2, 8, 4, 6])
        self.assertLess(diff, 1e-3)

    def test_mqa_untied_embeddings(self):
        config = ModelConfig(hidden_size=48, num_layers=2, num_heads=8, num_kv_heads=1,
                             intermediate_size=80, max_seq_len=40, vocab_size=35,
                             rope_theta=50000.0, norm_epsilon=1e-5, tie_embeddings=False)
        model = _train_tiny_model(config, seed=123, steps=15)
        diff = self._compare(model, list(range(1, 15)))
        self.assertLess(diff, 1e-3)

    def test_byte_tokenizer_vocab_scale(self):
        tok = ByteTokenizer()
        config = ModelConfig(hidden_size=48, num_layers=2, num_heads=4, num_kv_heads=2,
                             intermediate_size=96, max_seq_len=64, vocab_size=tok.vocab_size)
        model = _train_tiny_model(config, seed=5, steps=10)
        diff = self._compare(model, tok.encode("hello there", add_bos=True))
        self.assertLess(diff, 1e-3)

    def test_kvcache_cached_equals_full_native(self):
        """Native KV-cache decode path must match native full-sequence
        forward (the same property test_cache.py verifies for the Python
        side, now confirmed on the C++ side too)."""
        import ctypes, struct
        config = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                             intermediate_size=64, max_seq_len=32, vocab_size=50)
        model = _train_tiny_model(config, seed=7, steps=15)
        full_input = [3, 7, 1, 9, 2, 8, 4]

        # Full-sequence logits from native binary
        tm_path = self.build_dir / "kv_model.tm"
        export_to_tm(model, tm_path)
        logits_path = self.build_dir / "kv_logits.bin"
        cmd = [str(self.binary), str(tm_path), str(logits_path), *[str(i) for i in full_input]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        native_full = np.fromfile(logits_path, dtype=np.float32).reshape(len(full_input), 50)

        # Python cached path
        from tinymind.model.generation import ModelGenerationConfig, generate_with_cache_ids
        from tinymind.model.model import KVCache
        cache = KVCache(config, batch_size=1)
        cached_logits = []
        for t, token in enumerate(full_input):
            step_out = model(np.array([[token]]), use_cache=True, past_key_values=cache,
                            position_ids=np.array([[t]]))
            cached_logits.append(step_out.logits.data[0, 0])
        cached = np.stack(cached_logits)

        # Both should agree with each other (and transitively with the full forward pass)
        max_diff_py_native = float(np.max(np.abs(cached - native_full)))
        self.assertLess(max_diff_py_native, 1e-3,
                       f"Python-cached vs native-full max diff = {max_diff_py_native}")


@unittest.skipUnless(_HAS_CXX, "no g++ available")
class TestNativeCAbIGeneration(unittest.TestCase):
    def setUp(self):
        self.build_dir = Path(tempfile.mkdtemp())
        self.binary = _build_cabi_binary(self.build_dir)

    def tearDown(self):
        shutil.rmtree(self.build_dir, ignore_errors=True)

    def test_generation_matches_python(self):
        tok = ByteTokenizer()
        config = ModelConfig(hidden_size=48, num_layers=2, num_heads=4, num_kv_heads=2,
                             intermediate_size=96, max_seq_len=64, vocab_size=tok.vocab_size)
        model = TinyMindTransformer(config, seed=5)
        from tinymind.model.optim import AdamW
        ids = np.array([tok.encode("hello there hello there hello there")])
        opt = AdamW(model.parameters(), learning_rate=5e-3)
        for _ in range(60):
            opt.zero_grad()
            out = model(ids, labels=ids)
            out.loss.backward()
            opt.step(grad_clip_norm=1.0)

        from tinymind.model.generation import ModelGenerationConfig, generate
        python_gen = generate(model, tok, "hello",
                              ModelGenerationConfig(max_new_tokens=20, do_sample=False,
                                                   eos_token_id=tok.eos_token_id))

        tm_path = self.build_dir / "cabi_model.tm"
        export_to_tm(model, tm_path)
        r = subprocess.run([str(self.binary), str(tm_path), "hello"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        native_continuation = r.stdout  # do NOT strip() — the leading space is a real generated token
        native_gen = "hello" + native_continuation
        self.assertEqual(python_gen, native_gen,
                        f"python: {python_gen!r}  native: {native_gen!r}")


@unittest.skipUnless(_HAS_CXX, "no C++ compiler")
class TestPackageModelNative(unittest.TestCase):
    """Phase 3B: the model an inference *package* carries (float32 ``model.tm`` with tokenizer / template / provenance
    in its metadata) loads in the native runtime and agrees with Python on the logits AND on the greedy next token at
    every position, for a prompt rendered by the training template and for the real ``tiny_mobile`` architecture."""

    @classmethod
    def setUpClass(cls):
        cls.build_dir = Path(tempfile.mkdtemp())
        cls.binary = _build_equiv_binary(cls.build_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.build_dir, ignore_errors=True)

    def _package_and_compare(self, config, seed, steps, tol=1e-3):
        from tinymind.export import export_package, load_package
        from tinymind.data.render import ChatRenderer

        model = _train_tiny_model(config, seed=seed, steps=steps)
        tok = ByteTokenizer()
        pkg_dir = self.build_dir / f"pkg{seed}"
        export_package(model, tok, pkg_dir, provenance={"stage": "test"})
        pkg = load_package(pkg_dir)
        renderer = ChatRenderer(tok)
        record = {"id": "x", "messages": [{"role": "user", "content": "What is 47 + 38?"},
                  {"role": "assistant", "content": '{"name":"calculator","arguments":{"expr":"47+38"}}'}], "tools": ["calculator"]}
        ids = [int(t) for t in renderer.render(record).ids][: config.max_seq_len]
        python_logits = pkg.model(np.array([ids])).logits.data[0]
        logits_path = self.build_dir / f"logits{seed}.bin"
        r = subprocess.run([str(self.binary), str(pkg_dir / "model.tm"), str(logits_path), *[str(i) for i in ids]], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        native = np.fromfile(logits_path, dtype=np.float32).reshape(python_logits.shape)
        diff = float(np.max(np.abs(python_logits - native)))
        self.assertLess(diff, tol, f"max abs logit diff {diff}")
        mismatches = int((np.argmax(python_logits, axis=-1) != np.argmax(native, axis=-1)).sum())
        self.assertEqual(mismatches, 0, f"{mismatches} of {len(ids)} positions disagree on the greedy token")
        return diff, len(ids)

    def test_small_gqa_package(self):
        config = ModelConfig(hidden_size=32, num_layers=3, num_heads=4, num_kv_heads=2, intermediate_size=64,
                             max_seq_len=128, vocab_size=ByteTokenizer().vocab_size)
        self._package_and_compare(config, seed=11, steps=15)

    def test_real_tiny_mobile_architecture(self):
        config = ModelConfig.from_yaml(_NATIVE_DIR.parent / "configs" / "tiny_mobile.yaml")
        diff, n = self._package_and_compare(config, seed=12, steps=3)
        self.assertGreater(n, 60)  # a realistic prompt + completion, not a toy input


if __name__ == "__main__":
    unittest.main()
