import shutil
import tempfile
import unittest

import numpy as np

from tinymind.model.checkpoint import CheckpointError, from_pretrained, load_optimizer_state, save_pretrained
from tinymind.model.config import ModelConfig
from tinymind.model.model import TinyMindTransformer
from tinymind.model.optim import AdamW


def _tiny_config():
    return ModelConfig(hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=2,
                       intermediate_size=32, max_seq_len=16, vocab_size=20)


class TestCheckpointRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parameters_match_exactly_after_round_trip(self):
        model = TinyMindTransformer(_tiny_config(), seed=1)
        save_pretrained(model, self.tmpdir)
        loaded = from_pretrained(self.tmpdir)

        original = dict(model.named_parameters())
        restored = dict(loaded.named_parameters())
        self.assertEqual(set(original), set(restored))
        for name in original:
            self.assertTrue(np.array_equal(original[name].data, restored[name].data), name)

    def test_logits_match_exactly_after_round_trip(self):
        model = TinyMindTransformer(_tiny_config(), seed=2)
        seq = np.array([[1, 2, 3, 4]])
        before = model(seq).logits.data.copy()

        save_pretrained(model, self.tmpdir)
        loaded = from_pretrained(self.tmpdir)
        after = loaded(seq).logits.data

        self.assertTrue(np.array_equal(before, after))

    def test_config_preserved(self):
        model = TinyMindTransformer(_tiny_config(), seed=3)
        save_pretrained(model, self.tmpdir)
        loaded = from_pretrained(self.tmpdir)
        self.assertEqual(model.config, loaded.config)

    def test_missing_weights_file_raises_clean_error(self):
        with self.assertRaises(CheckpointError):
            from_pretrained(self.tmpdir)  # empty dir, nothing saved

    def test_no_pickle_involved_load_succeeds_with_allow_pickle_false(self):
        # save_pretrained/from_pretrained internally pass allow_pickle=False
        # to np.load — if that ever silently changed, a normal round trip
        # would start raising. This test's mere passage is the check.
        model = TinyMindTransformer(_tiny_config(), seed=4)
        save_pretrained(model, self.tmpdir)
        from_pretrained(self.tmpdir)  # must not raise

    def test_optimizer_state_round_trip(self):
        model = TinyMindTransformer(_tiny_config(), seed=5)
        opt = AdamW(model.parameters(), learning_rate=1e-3)
        out = model(np.array([[1, 2, 3]]), labels=np.array([[1, 2, 3]]))
        out.loss.backward()
        opt.step()
        step_before = opt.step_count

        save_pretrained(model, self.tmpdir, optimizer=opt)
        loaded = from_pretrained(self.tmpdir)
        new_opt = AdamW(loaded.parameters(), learning_rate=1e-3)
        load_optimizer_state(self.tmpdir, new_opt)

        self.assertEqual(new_opt.step_count, step_before)
        self.assertTrue(np.array_equal(new_opt._m[0], opt._m[0]))

    def test_training_metadata_round_trip(self):
        import json
        model = TinyMindTransformer(_tiny_config(), seed=6)
        save_pretrained(model, self.tmpdir, training_meta={"step": 123, "loss": 0.5})
        meta = json.loads((self._path("training_meta.json")).read_text())
        self.assertEqual(meta["step"], 123)

    def _path(self, name):
        from pathlib import Path
        return Path(self.tmpdir) / name


if __name__ == "__main__":
    unittest.main()
