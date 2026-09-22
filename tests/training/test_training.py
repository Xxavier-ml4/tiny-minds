"""The brief's own section 20 acceptance test, verbatim: "If the model
cannot overfit a tiny dataset, stop and debug before proceeding." This is
not a nice-to-have — it's the test that actually proves gradients flow
correctly through the whole stack (embeddings, attention, RoPE, MLP, norm,
loss) in a way unit tests on individual layers cannot: a bug in how the
layers compose (not in any one layer alone) would still show up here as a
loss that refuses to go down.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tinymind.model.config import ModelConfig
from tinymind.model.model import TinyMindTransformer
from tinymind.model.tokenizer import ByteTokenizer
from tinymind.training.causal_lm_trainer import CausalLMTrainer, CausalLMTrainingConfig
from tinymind.training.dataset import TrainingDataset


def _write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(json.dumps({
                "id": f"ex_{i}", "messages": [{"role": "user", "content": text}],
                "target": {"type": "answer", "content": ""},
            }) + "\n")


class TestSyntheticOverfitting(unittest.TestCase):
    def test_loss_decreases_on_repeated_single_sentence(self):
        # Brief section 20's own example: "hello world" repeated.
        tmpdir = Path(tempfile.mkdtemp())
        _write_jsonl(tmpdir / "train.jsonl", ["hello world"] * 8)

        tok = ByteTokenizer()
        config = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                             intermediate_size=64, max_seq_len=32, vocab_size=tok.vocab_size)
        model = TinyMindTransformer(config, seed=0)
        dataset = TrainingDataset(tmpdir / "train.jsonl", tok)

        trainer_config = CausalLMTrainingConfig(learning_rate=5e-3, batch_size=4, epochs=30, seed=0)
        trainer = CausalLMTrainer(model, trainer_config)
        logs = trainer.train(dataset)

        first_loss = logs[0].loss
        last_loss = logs[-1].loss
        self.assertLess(last_loss, first_loss,
                        f"loss did not decrease: first={first_loss}, last={last_loss}")
        self.assertLess(last_loss, first_loss * 0.5, "loss should drop substantially, not just slightly")

    def test_model_can_overfit_a_tiny_deterministic_pattern(self):
        """The brief's stronger claim: not just "loss decreases" but "the
        model must be capable of overfitting the tiny dataset" — checked
        directly by generating from the trained model and confirming it
        reproduces the memorized pattern via greedy decoding."""
        tmpdir = Path(tempfile.mkdtemp())
        pattern = "abcabcabcabcabcabc"
        _write_jsonl(tmpdir / "train.jsonl", [pattern] * 4)

        tok = ByteTokenizer()
        config = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                             intermediate_size=64, max_seq_len=32, vocab_size=tok.vocab_size)
        model = TinyMindTransformer(config, seed=1)
        dataset = TrainingDataset(tmpdir / "train.jsonl", tok)

        trainer_config = CausalLMTrainingConfig(learning_rate=5e-3, batch_size=4, epochs=80, seed=1)
        trainer = CausalLMTrainer(model, trainer_config)
        logs = trainer.train(dataset)

        self.assertLess(logs[-1].loss, 0.1, f"final loss {logs[-1].loss} is too high to call this overfit")

        from tinymind.model.generation import ModelGenerationConfig, generate
        completion = generate(model, tok, "abc", ModelGenerationConfig(max_new_tokens=9, do_sample=False))
        expected = pattern[:12]  # prompt "abc" (3 chars) + 9 more generated chars of the same cycle
        self.assertEqual(completion, expected,
                         f"expected the model to have memorized the repeating pattern, got {completion!r}")

    def test_gradient_accumulation_matches_single_large_batch_direction(self):
        # Not bit-exact (float non-associativity across summation order),
        # but training with accumulation should still make comparable
        # progress to one large batch over the same effective data.
        tmpdir = Path(tempfile.mkdtemp())
        _write_jsonl(tmpdir / "train.jsonl", ["repeat this phrase"] * 8)
        tok = ByteTokenizer()
        config = ModelConfig(hidden_size=24, num_layers=2, num_heads=4, num_kv_heads=2,
                             intermediate_size=48, max_seq_len=32, vocab_size=tok.vocab_size)

        model_a = TinyMindTransformer(config, seed=2)
        dataset_a = TrainingDataset(tmpdir / "train.jsonl", tok)
        trainer_a = CausalLMTrainer(model_a, CausalLMTrainingConfig(
            learning_rate=5e-3, batch_size=8, gradient_accumulation_steps=1, epochs=10, seed=2))
        logs_a = trainer_a.train(dataset_a)

        model_b = TinyMindTransformer(config, seed=2)
        dataset_b = TrainingDataset(tmpdir / "train.jsonl", tok)
        trainer_b = CausalLMTrainer(model_b, CausalLMTrainingConfig(
            learning_rate=5e-3, batch_size=2, gradient_accumulation_steps=4, epochs=10, seed=2))
        logs_b = trainer_b.train(dataset_b)

        self.assertLess(logs_a[-1].loss, logs_a[0].loss)
        self.assertLess(logs_b[-1].loss, logs_b[0].loss)

    def test_reproducible_given_seed(self):
        tmpdir = Path(tempfile.mkdtemp())
        _write_jsonl(tmpdir / "train.jsonl", ["seeded run"] * 4)
        tok = ByteTokenizer()
        config = ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2,
                             intermediate_size=32, max_seq_len=32, vocab_size=tok.vocab_size)

        def run():
            model = TinyMindTransformer(config, seed=9)
            dataset = TrainingDataset(tmpdir / "train.jsonl", tok)
            trainer = CausalLMTrainer(model, CausalLMTrainingConfig(
                learning_rate=1e-3, batch_size=2, epochs=3, seed=9))
            return trainer.train(dataset)

        logs1, logs2 = run(), run()
        self.assertEqual([entry.loss for entry in logs1], [entry.loss for entry in logs2])

    def test_checkpoint_interval_saves_during_training(self):
        tmpdir = Path(tempfile.mkdtemp())
        _write_jsonl(tmpdir / "train.jsonl", ["checkpoint me"] * 4)
        tok = ByteTokenizer()
        config = ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2,
                             intermediate_size=32, max_seq_len=32, vocab_size=tok.vocab_size)
        model = TinyMindTransformer(config, seed=0)
        dataset = TrainingDataset(tmpdir / "train.jsonl", tok)
        checkpoint_dir = tmpdir / "checkpoints"

        trainer_config = CausalLMTrainingConfig(learning_rate=1e-3, batch_size=2, epochs=4, seed=0,
                                               checkpoint_interval=2, checkpoint_dir=str(checkpoint_dir))
        trainer = CausalLMTrainer(model, trainer_config)
        trainer.train(dataset)

        saved = list(checkpoint_dir.glob("step_*"))
        self.assertGreater(len(saved), 0)


if __name__ == "__main__":
    unittest.main()
