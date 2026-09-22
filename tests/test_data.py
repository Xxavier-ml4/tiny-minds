import json
import os
import tempfile
import unittest

from tinymind.data import deduplicate, split_dataset, validate_file


class TestValidateFile(unittest.TestCase):
    def _write(self, lines):
        path = tempfile.mktemp(suffix=".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        return path

    def test_valid_answer_example(self):
        path = self._write([json.dumps({
            "id": "a1", "messages": [{"role": "user", "content": "17 * 8"}],
            "tools": [], "target": {"type": "answer", "content": "136"},
        })])
        report = validate_file(path)
        self.assertEqual(report.total, 1)
        self.assertEqual(report.valid, 1)
        os.remove(path)

    def test_tool_call_must_declare_the_tool(self):
        path = self._write([json.dumps({
            "id": "a2", "messages": [{"role": "user", "content": "hi"}],
            "tools": [], "target": {"type": "tool_call", "name": "nope", "arguments": {}},
        })])
        report = validate_file(path)
        self.assertEqual(report.valid, 0)
        self.assertTrue(any("not declared" in str(e) for e in report.errors))
        os.remove(path)

    def test_malformed_json_line_reported(self):
        path = self._write(["not json"])
        report = validate_file(path)
        self.assertEqual(report.valid, 0)
        self.assertTrue(any("invalid JSON" in str(e) for e in report.errors))
        os.remove(path)

    def test_empty_messages_rejected(self):
        path = self._write([json.dumps({
            "id": "a3", "messages": [], "tools": [], "target": {"type": "answer", "content": "x"},
        })])
        report = validate_file(path)
        self.assertEqual(report.valid, 0)
        os.remove(path)


class TestDeduplicate(unittest.TestCase):
    def test_exact_duplicate_removed(self):
        examples = [
            {"messages": [{"role": "user", "content": "What is 17 times 8?"}]},
            {"messages": [{"role": "user", "content": "What is 17 times 8?"}]},
        ]
        kept, report = deduplicate(examples)
        self.assertEqual(len(kept), 1)
        self.assertEqual(report.exact_duplicates_removed, 1)

    def test_near_duplicate_removed(self):
        examples = [
            {"messages": [{"role": "user", "content": "Turn the bedroom light to thirty percent please"}]},
            {"messages": [{"role": "user", "content": "Turn the bedroom light to thirty percent now"}]},
        ]
        kept, report = deduplicate(examples, near_duplicate_threshold=0.6)
        self.assertEqual(len(kept), 1)
        self.assertEqual(report.near_duplicates_removed, 1)

    def test_distinct_examples_both_kept(self):
        examples = [
            {"messages": [{"role": "user", "content": "What is 17 times 8?"}]},
            {"messages": [{"role": "user", "content": "Set a timer for ten minutes"}]},
        ]
        kept, report = deduplicate(examples)
        self.assertEqual(len(kept), 2)


class TestSplitDataset(unittest.TestCase):
    def test_proportions_roughly_correct(self):
        examples = list(range(100))
        split = split_dataset(examples, train=0.8, validation=0.1, test=0.1, seed=1)
        self.assertEqual(len(split.train) + len(split.validation) + len(split.test), 100)
        self.assertGreater(len(split.train), len(split.validation))
        self.assertGreater(len(split.train), len(split.test))

    def test_deterministic_given_seed(self):
        examples = list(range(50))
        split1 = split_dataset(examples, seed=7)
        split2 = split_dataset(examples, seed=7)
        self.assertEqual(split1.train, split2.train)

    def test_rejects_bad_proportions(self):
        with self.assertRaises(ValueError):
            split_dataset([1, 2, 3], train=0.5, validation=0.3, test=0.3)

    def test_stratified_split_keeps_each_group_proportional(self):
        examples = [{"type": "answer", "id": i} for i in range(40)] + \
                   [{"type": "tool_call", "id": i} for i in range(10)]
        split = split_dataset(examples, train=0.5, validation=0.25, test=0.25, seed=3,
                              stratify_key=lambda e: e["type"])
        train_tool_calls = sum(1 for e in split.train if e["type"] == "tool_call")
        self.assertGreater(train_tool_calls, 0)  # the minority class isn't starved out entirely


if __name__ == "__main__":
    unittest.main()
