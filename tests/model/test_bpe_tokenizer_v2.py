import json
import tempfile
import unittest
from pathlib import Path

from tinymind.model.bpe_io import load_tokenizer, save_tokenizer, spec_hash, train_bpe


CORPUS = "\n".join([
    "the quick brown fox jumps over the lazy dog",
    "a banana is yellow and a lime is green",
    "she sells sea shells by the sea shore",
    "the farmer walks to the market before dawn",
    "reading books in a quiet library is calm",
] * 40)


class TestBpeTokenizerV2(unittest.TestCase):
    """Brief section 2: a first-class, deterministic BPE tokenizer that can be
    trained from a corpus, written to tokenizer.json, and reconstructed solely
    from that file."""

    def _corpus_file(self, tmp: Path) -> Path:
        p = tmp / "corpus.txt"
        p.write_text(CORPUS, encoding="utf-8")
        return p

    def test_train_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            corpus = self._corpus_file(tmp)
            a = train_bpe([str(corpus)], vocab_size=400)
            b = train_bpe([str(corpus)], vocab_size=400)
            self.assertEqual(spec_hash(a), spec_hash(b))
            self.assertEqual(a.spec()["merges"], b.spec()["merges"])

    def test_save_and_reload_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            corpus = self._corpus_file(tmp)
            tok = train_bpe([str(corpus)], vocab_size=400)
            out = tmp / "tok.json"
            digest = save_tokenizer(tok, str(out))
            self.assertTrue(out.is_file())
            # The stored hash is stable and matches the in-memory tokenizer.
            reloaded = load_tokenizer(str(out))
            self.assertEqual(spec_hash(reloaded), spec_hash(tok))
            self.assertEqual(spec_hash(reloaded), digest)

    def test_reconstructs_solely_from_json(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            corpus = self._corpus_file(tmp)
            tok = train_bpe([str(corpus)], vocab_size=400)
            out = tmp / "tok.json"
            save_tokenizer(tok, str(out))
            # Reconstruct from bytes on disk only — no access to the original object.
            spec = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("merges", spec)
            reloaded = load_tokenizer(str(out))
            for s in ("the quick brown fox", "a banana is yellow", "sea shells"):
                self.assertEqual(reloaded.decode(reloaded.encode(s)), s)

    def test_encode_decode_identity(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            corpus = self._corpus_file(tmp)
            tok = train_bpe([str(corpus)], vocab_size=400)
            for s in ["hello world", "the farmer walks", "", "123 456"]:
                self.assertEqual(tok.decode(tok.encode(s)), s)


if __name__ == "__main__":
    unittest.main()
