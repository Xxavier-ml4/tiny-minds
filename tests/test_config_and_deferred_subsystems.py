import json
import os
import tempfile
import unittest

from tinymind.config import TinyMindConfig
from tinymind.distillation import DistillationExample, VerifierFilter
from tinymind.model.tokenizer import ByteTokenizer
from tinymind.training import CheckpointManager, TrainingDataset


class TestTinyMindConfig(unittest.TestCase):
    def test_defaults(self):
        config = TinyMindConfig.default()
        self.assertEqual(config.confidence.execute_threshold, 0.90)

    def test_partial_yaml_overrides_only_given_fields(self):
        path = tempfile.mktemp(suffix=".yaml")
        with open(path, "w") as f:
            f.write("confidence:\n  execute_threshold: 0.95\n")
        config = TinyMindConfig.from_yaml(path)
        self.assertEqual(config.confidence.execute_threshold, 0.95)
        self.assertEqual(config.confidence.verify_threshold, 0.70)  # untouched default
        self.assertEqual(config.tools.retrieval.top_k, 5)  # untouched default
        os.remove(path)

    def test_round_trip_to_yaml(self):
        path = tempfile.mktemp(suffix=".yaml")
        config = TinyMindConfig.default()
        config.to_yaml(path)
        reloaded = TinyMindConfig.from_yaml(path)
        self.assertEqual(config.to_dict(), reloaded.to_dict())
        os.remove(path)


class TestVerifierFilter(unittest.TestCase):
    def test_accepts_correct_arithmetic(self):
        f = VerifierFilter()
        example = DistillationExample(prompt="17 * 8", target={"type": "answer", "content": "136"})
        self.assertTrue(f.accepts(example))

    def test_rejects_incorrect_arithmetic(self):
        f = VerifierFilter()
        example = DistillationExample(prompt="17 * 8", target={"type": "answer", "content": "999"})
        self.assertFalse(f.accepts(example))

    def test_non_arithmetic_answers_pass_through(self):
        f = VerifierFilter()
        example = DistillationExample(prompt="what color is the sky",
                                      target={"type": "answer", "content": "blue"})
        self.assertTrue(f.accepts(example))


class TestCheckpointManager(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        tmpdir = tempfile.mkdtemp()
        manager = CheckpointManager(tmpdir)
        manager.save("step_1", {"weights": [1, 2, 3]}, step=1)
        state, metadata = manager.load("step_1")
        self.assertEqual(state["weights"], [1, 2, 3])
        self.assertEqual(metadata.step, 1)

    def test_latest_picks_highest_step(self):
        tmpdir = tempfile.mkdtemp()
        manager = CheckpointManager(tmpdir)
        manager.save("a", {}, step=5)
        manager.save("b", {}, step=10)
        self.assertEqual(manager.latest(), "b")


class TestTrainingDataset(unittest.TestCase):
    def test_iterates_tokenized_examples(self):
        path = tempfile.mktemp(suffix=".jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"id": "a1", "messages": [{"role": "user", "content": "hi"}],
                                "target": {"type": "answer", "content": "hello"}}) + "\n")
        dataset = TrainingDataset(path, ByteTokenizer())
        examples = list(dataset)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].target_text, "hello")
        self.assertEqual(dataset.count(), 1)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
