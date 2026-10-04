"""Anti-loop decoding controls: ``no_repeat_ngram_size``, ``repetition_penalty`` validation, and that they are
reachable from the backend and the ``tinymind generate`` CLI while the DEFAULT stays plain deterministic greedy.
"""
import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from tinymind.cli import main
from tinymind.model import ByteTokenizer, ModelConfig, TinyMindTransformer
from tinymind.model.backends.transformer import TransformerBackend
from tinymind.model.generation import ModelGenerationConfig, banned_ngram_tokens, generate_with_cache_ids
from tinymind.model.tm_export import export_to_tm


def repeated_ngrams(seq, n):
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return len(grams) - len(set(grams))


class TestBannedNgramTokens(unittest.TestCase):
    def test_bans_the_token_that_would_complete_a_seen_ngram(self):
        self.assertEqual(banned_ngram_tokens([1, 2, 3, 1, 2], 3), {3})
        self.assertEqual(banned_ngram_tokens([5, 6, 5, 6, 5], 2), {6})

    def test_collects_every_continuation_seen_after_the_same_prefix(self):
        self.assertEqual(banned_ngram_tokens([1, 2, 7, 1, 2, 8, 1, 2], 3), {7, 8})

    def test_nothing_banned_when_the_prefix_is_new_or_the_text_is_short(self):
        self.assertEqual(banned_ngram_tokens([1, 2, 3, 4], 3), set())
        self.assertEqual(banned_ngram_tokens([1, 2], 3), set())
        self.assertEqual(banned_ngram_tokens([1, 2, 3, 1, 2], 0), set())

    def test_size_one_bans_every_token_already_used(self):
        self.assertEqual(banned_ngram_tokens([4, 5, 4], 1), {4, 5})

    def test_the_loop_from_the_bug_report(self):
        # "the town of the town of" -> after "the town of the town" the model must not be allowed to say "of" again
        the, town, of = 10, 11, 12
        self.assertEqual(banned_ngram_tokens([the, town, of, the, town], 3), {of})


class TestGenerationConfigValidation(unittest.TestCase):
    def test_defaults_leave_decoding_unchanged(self):
        cfg = ModelGenerationConfig()
        self.assertEqual((cfg.repetition_penalty, cfg.no_repeat_ngram_size), (1.0, 0))

    def test_invalid_values_are_rejected(self):
        for kw in (dict(repetition_penalty=0.0), dict(repetition_penalty=-1.0), dict(no_repeat_ngram_size=-1),
                   dict(max_new_tokens=-1)):
            with self.assertRaises(ValueError, msg=str(kw)):
                ModelGenerationConfig(**kw)


class TestNgramBanInGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyMindTransformer(ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                                                    intermediate_size=64, max_seq_len=96, vocab_size=40), seed=1)
        cls.prompt = np.array([[1, 2, 3]])

    def generate(self, **kw):
        cfg = ModelGenerationConfig(max_new_tokens=40, do_sample=False, **kw)
        return generate_with_cache_ids(self.model, self.prompt, cfg)[0].tolist()

    def test_default_greedy_decoding_is_unchanged_by_the_new_option(self):
        self.assertEqual(self.generate(), self.generate(no_repeat_ngram_size=0))

    def test_this_model_loops_without_the_ban_so_the_next_test_is_not_vacuous(self):
        self.assertGreater(repeated_ngrams(self.generate(), 3), 0)

    def test_no_ngram_of_the_banned_size_ever_repeats(self):
        for n in (2, 3, 4):
            self.assertEqual(repeated_ngrams(self.generate(no_repeat_ngram_size=n), n), 0, f"n={n}")

    def test_the_ban_also_holds_when_sampling(self):
        cfg = ModelGenerationConfig(max_new_tokens=40, do_sample=True, temperature=1.0, seed=3, no_repeat_ngram_size=3)
        out = generate_with_cache_ids(self.model, self.prompt, cfg)[0].tolist()
        self.assertEqual(repeated_ngrams(out, 3), 0)

    def test_the_ban_is_deterministic(self):
        self.assertEqual(self.generate(no_repeat_ngram_size=3), self.generate(no_repeat_ngram_size=3))


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestBackendAndCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cfg = ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=32,
                          max_seq_len=64, vocab_size=tok.vocab_size)
        cls.path = tempfile.mktemp(suffix=".tm")
        export_to_tm(TinyMindTransformer(cfg, seed=0), cls.path)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.path)

    def backend(self):
        b = TransformerBackend()
        b.load(self.path)
        return b

    def test_backend_defaults_are_unchanged_plain_greedy(self):
        b = self.backend()
        self.assertEqual(b.generate("hi", max_new_tokens=12).text,
                         b.generate("hi", max_new_tokens=12, repetition_penalty=1.0, no_repeat_ngram_size=0).text)

    def test_backend_accepts_the_anti_loop_controls(self):
        b = self.backend()
        banned = b.generate("hi", max_new_tokens=24, no_repeat_ngram_size=2)
        plain = b.generate("hi", max_new_tokens=24)
        self.assertIsInstance(banned.text, str)
        self.assertEqual(banned.text, b.generate("hi", max_new_tokens=24, no_repeat_ngram_size=2).text)
        self.assertNotEqual(banned.text, plain.text)  # this random model loops greedily; the ban must change it

    def test_backend_repetition_penalty_reaches_the_generator(self):
        b = self.backend()
        self.assertNotEqual(b.generate("hi", max_new_tokens=24).text,
                            b.generate("hi", max_new_tokens=24, repetition_penalty=3.0).text)

    def test_cli_accepts_the_new_flags(self):
        code, _out, err = _run(["generate", "--model", self.path, "--prompt", "hi", "--max-new-tokens", "8",
                                "--no-repeat-ngram-size", "3", "--repetition-penalty", "1.15"])
        self.assertEqual(code, 0, err)

    def test_cli_sampling_flags_work_together(self):
        code, _out, err = _run(["generate", "--model", self.path, "--prompt", "hi", "--max-new-tokens", "8",
                                "--temperature", "0.8", "--top-k", "10", "--top-p", "0.9", "--seed", "1"])
        self.assertEqual(code, 0, err)

    def test_cli_rejects_an_invalid_penalty_cleanly(self):
        code, _out, err = _run(["generate", "--model", self.path, "--prompt", "hi", "--repetition-penalty", "0"])
        self.assertEqual(code, 2)
        self.assertIn("error", err)


if __name__ == "__main__":
    unittest.main()
