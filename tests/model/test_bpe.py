"""The optional byte-level BPE tokenizer: correctness, identity, and that the whole pipeline accepts it."""
import json
import unittest

import numpy as np

from tinymind.data.render import ChatRenderer
from tinymind.model.bpe import BASE, BPETokenizer
from tinymind.model.tokenizer import ByteTokenizer, tokenizer_from_spec

CORPUS = ["What is 47 + 38?", "the cat sat on the mat", "the dog sat on the log", "Set a timer for 15 minutes.",
          '{"name":"calculator","arguments":{"expr":"47+38"}}', "user:\nhello\nassistant:\nHello! How can I help?"] * 5


class TestBPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer.train(CORPUS, 300)

    def test_round_trip_including_unicode_and_empty(self):
        for s in ["", "hello world", "  two  spaces ", "line\nbreak\t tab", "naïve café — ☃ 日本語 🙂", "47+38=85", "x" * 500]:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s, s)

    def test_zero_merges_is_exactly_the_byte_tokenizer(self):
        b, z = ByteTokenizer(), BPETokenizer([])
        for s in ["abc", "日本語", "line\n"]:
            self.assertEqual(z.encode(s), b.encode(s))
        self.assertEqual((z.vocab_size, z.bos_token_id, z.eos_token_id, z.pad_token_id), (260, 1, 2, 0))

    def test_it_actually_compresses_and_digits_stay_separate(self):
        s = "the cat sat on the mat"
        self.assertLess(len(self.tok.encode(s)), 0.6 * len(ByteTokenizer().encode(s)))
        self.assertEqual(len(self.tok.encode("47")), 2)  # a number is one token per digit whatever was learned
        self.assertEqual(self.tok.encode("hi", add_bos=True, add_eos=True)[0], 1)
        self.assertEqual(self.tok.encode("hi", add_eos=True)[-1], 2)

    def test_training_is_deterministic_and_order_independent(self):
        again = BPETokenizer.train(CORPUS, 300)
        shuffled = BPETokenizer.train(list(reversed(CORPUS)), 300)
        self.assertEqual(self.tok.spec(), again.spec())
        self.assertEqual(self.tok.spec_hash(), shuffled.spec_hash())
        self.assertNotEqual(self.tok.spec_hash(), BPETokenizer.train(CORPUS, 280).spec_hash())

    def test_merges_never_cross_word_boundaries(self):
        for i in range(BASE, self.tok.vocab_size):
            piece = self.tok._bytes[i].decode("utf-8", errors="replace")
            body = piece[1:] if piece.startswith(" ") else piece
            self.assertNotIn(" ", body, repr(piece))
            self.assertFalse(any(c.isdigit() for c in body) and len(body) > 1, repr(piece))

    def test_spec_round_trip_and_tampering(self):
        spec = json.loads(json.dumps(self.tok.spec()))
        rebuilt = tokenizer_from_spec(spec)
        self.assertEqual(rebuilt.encode("the cat sat"), self.tok.encode("the cat sat"))
        bad = dict(spec, vocab_size=spec["vocab_size"] + 1)
        with self.assertRaises(ValueError):
            tokenizer_from_spec(bad)
        with self.assertRaises(ValueError):
            tokenizer_from_spec(dict(spec, pretokenizer="something-else"))
        with self.assertRaises(ValueError):
            BPETokenizer([(10, 9999)])
        with self.assertRaises(ValueError):
            self.tok.decode([self.tok.vocab_size])

    def test_renderer_prompt_is_still_a_prefix_of_the_training_sequence(self):
        r = ChatRenderer(self.tok)
        rec = {"id": "a", "messages": [{"role": "user", "content": "What is 47 + 38?"},
                                       {"role": "assistant", "content": '{"name":"calculator","arguments":{"expr":"47+38"}}'}], "tools": ["calculator"]}
        ex = r.render(rec)
        prompt = r.render_prompt(rec["messages"][:1], tools=["calculator"])
        self.assertEqual(ex.ids[:len(prompt)].tolist(), prompt)
        self.assertEqual(r.decode_completion(ex.ids[len(prompt):].tolist()), rec["messages"][1]["content"])

    def test_engine_trains_resumes_and_exports_with_bpe(self):
        from tinymind.export import load_package
        from tinymind.training.data import DataSource, TokenizedDataset
        from tinymind.training.exporter import package_exporter
        from tinymind.training.checkpoint import ResumeMismatchError
        from tests.training._helpers import make_engine, records, tmpdir

        tok = self.tok
        r = ChatRenderer(tok)
        train = TokenizedDataset.from_records(records(40), r, 96, name="train")
        val = TokenizedDataset.from_records(records(8, 1000, "v"), r, 96, name="val")
        out = tmpdir()
        kw = dict(tokenizer=tok, model_over=dict(vocab_size=tok.vocab_size), train_data=train, val_data=val)
        e = make_engine(out, train_over=dict(max_steps=12, max_runtime_seconds=6.5, safety_margin_seconds=0), exporter=package_exporter,
                        clock=__import__("tests.training._helpers", fromlist=["FakeClock"]).FakeClock(), **kw)
        e.train()
        self.assertEqual(e.step, 6)
        resumed = make_engine(tmpdir(), train_over=dict(max_steps=12), resume=out / "checkpoints", exporter=package_exporter, **kw)
        resumed.train()
        pkg = load_package(resumed.output_dir / "export")
        self.assertEqual(pkg.tokenizer.spec_hash(), tok.spec_hash())
        self.assertIsInstance(pkg.generate("repeat 3", max_new_tokens=6), str)
        other = BPETokenizer.train(CORPUS, 290)  # a different vocabulary must be refused on resume
        r2 = ChatRenderer(other)
        with self.assertRaises(ResumeMismatchError):
            make_engine(tmpdir(), train_over=dict(max_steps=12), resume=out / "checkpoints", tokenizer=other,
                        model_over=dict(vocab_size=other.vocab_size),
                        train_data=TokenizedDataset.from_records(records(40), r2, 96, name="train"),
                        val_data=TokenizedDataset.from_records(records(8, 1000, "v"), r2, 96, name="val"))


if __name__ == "__main__":
    unittest.main()
