import math
import unittest

from tinymind.training.config import TrainingConfig


class TestTokenBudgetConfig(unittest.TestCase):
    """Brief sections 6, 8: a stage can be defined by a token budget; the horizon
    is derived, and the budget does not change the run's compatibility identity
    (which flows through the derived step count)."""

    def _cfg(self, **kw):
        base = dict(stage="stage1", batch_size=1, gradient_accumulation_steps=16,
                    max_seq_len=1024, warmup_steps=2)
        base.update(kw)
        return TrainingConfig.from_dict(base)

    def test_target_tokens_field_exists_and_validates(self):
        cfg = self._cfg(max_steps=0, target_tokens=60_000_000)
        self.assertEqual(cfg.target_tokens, 60_000_000)

    def test_requires_some_horizon(self):
        with self.assertRaises(Exception):
            self._cfg(max_steps=0, target_tokens=0, epochs=0)

    def test_effective_batch_tokens_and_derivation(self):
        # batch 1 * accum 16 * seq 1024 = 16384 tokens per optimizer step.
        cfg = self._cfg(max_steps=0, target_tokens=16384 * 10)
        eff = cfg.batch_size * cfg.gradient_accumulation_steps * cfg.max_seq_len
        self.assertEqual(eff, 16384)
        # the engine derives total_steps = ceil(target_tokens / eff); check the arithmetic here
        self.assertEqual(math.ceil(cfg.target_tokens / eff), 10)

    def test_target_tokens_excluded_from_compat_hash(self):
        a = self._cfg(max_steps=100)
        b = self._cfg(max_steps=100, target_tokens=123_456)
        # target_tokens is runtime-only: it is excluded from the compatibility
        # identity, which is taken over the DERIVED step horizon. With the same
        # total_steps, two configs differing only in target_tokens hash equally.
        self.assertEqual(a.compat_hash(100), b.compat_hash(100))
        # and the token budget genuinely does not enter compat_dict
        self.assertNotIn("target_tokens", a.compat_dict(100))


if __name__ == "__main__":
    unittest.main()
