"""KV-cache correctness: prefill + incremental decode must produce the same
logits as a full-sequence forward pass, within numerical tolerance. Marked
mandatory by the brief (section 13) — this is not a nice-to-have test.
"""
import unittest

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.generation import (
    ModelGenerationConfig, generate_full_sequence_no_cache, generate_with_cache_ids,
)
from tinymind.model.model import KVCache, TinyMindTransformer

_TOLERANCE = 1e-3  # generous but meaningful: catches a wrong-position or wrong-mask bug immediately


def _make_tiny_model(seed: int = 0) -> TinyMindTransformer:
    config = ModelConfig(hidden_size=24, num_layers=2, num_heads=4, num_kv_heads=2,
                         intermediate_size=48, max_seq_len=32, vocab_size=16)
    return TinyMindTransformer(config, seed=seed)


class TestKVCacheLogitEquivalence(unittest.TestCase):
    def test_prefill_then_decode_matches_full_sequence(self):
        model = _make_tiny_model()
        full_sequence = np.array([[1, 2, 3, 4, 5, 6, 7]])

        # Ground truth: one full-sequence forward pass, no cache.
        full_out = model(full_sequence, use_cache=False)
        full_logits = full_out.logits.data[0]  # [T, vocab_size]

        # Cached path: prefill on the first token, then decode one token at
        # a time for the rest, reusing the same cache throughout.
        cache = KVCache(model.config, batch_size=1)
        prefill_out = model(full_sequence[:, :1], use_cache=True, past_key_values=cache,
                            position_ids=np.array([[0]]))
        cached_logits = [prefill_out.logits.data[0, 0, :]]

        for t in range(1, full_sequence.shape[1]):
            token = full_sequence[:, t:t + 1]
            position_ids = np.array([[cache.length]])
            step_out = model(token, use_cache=True, past_key_values=cache, position_ids=position_ids)
            cached_logits.append(step_out.logits.data[0, 0, :])

        cached_logits = np.stack(cached_logits, axis=0)  # [T, vocab_size]

        self.assertEqual(cached_logits.shape, full_logits.shape)
        max_diff = np.max(np.abs(cached_logits - full_logits))
        self.assertLess(max_diff, _TOLERANCE,
                        f"cached vs. full-sequence logits differ by {max_diff}, expected < {_TOLERANCE}")

    def test_multi_token_prefill_then_decode_matches(self):
        # A prefill longer than one token (the realistic case: the whole
        # prompt at once), then incremental decode for the rest.
        model = _make_tiny_model(seed=7)
        full_sequence = np.array([[2, 5, 1, 9, 3, 8, 4]])
        prefill_len = 4

        full_out = model(full_sequence, use_cache=False)
        full_logits = full_out.logits.data[0]

        cache = KVCache(model.config, batch_size=1)
        prefill_out = model(full_sequence[:, :prefill_len], use_cache=True, past_key_values=cache,
                            position_ids=np.arange(prefill_len)[None, :])
        cached_logits = list(prefill_out.logits.data[0])  # prefill_len entries

        for t in range(prefill_len, full_sequence.shape[1]):
            token = full_sequence[:, t:t + 1]
            position_ids = np.array([[cache.length]])
            step_out = model(token, use_cache=True, past_key_values=cache, position_ids=position_ids)
            cached_logits.append(step_out.logits.data[0, 0, :])

        cached_logits = np.stack(cached_logits, axis=0)
        max_diff = np.max(np.abs(cached_logits - full_logits))
        self.assertLess(max_diff, _TOLERANCE,
                        f"cached vs. full-sequence logits differ by {max_diff}, expected < {_TOLERANCE}")

    def test_greedy_generation_matches_with_and_without_cache(self):
        model = _make_tiny_model(seed=3)
        prompt_ids = np.array([[1, 2, 3]])
        config = ModelGenerationConfig(max_new_tokens=5, do_sample=False)

        with_cache = generate_with_cache_ids(model, prompt_ids, config)
        without_cache = generate_full_sequence_no_cache(model, prompt_ids, config)

        self.assertTrue(np.array_equal(with_cache, without_cache),
                        f"cached greedy generation {with_cache.tolist()} != "
                        f"uncached greedy generation {without_cache.tolist()}")

    def test_cache_overflow_raises_clearly(self):
        model = _make_tiny_model()
        cache = KVCache(model.config, batch_size=1)
        cache.length = model.config.max_seq_len - 1
        too_long = np.array([[1, 2, 3]])  # 3 new tokens, only 1 slot left
        with self.assertRaises(ValueError):
            model(too_long, use_cache=True, past_key_values=cache,
                 position_ids=np.array([[cache.length, cache.length + 1, cache.length + 2]]))


if __name__ == "__main__":
    unittest.main()
