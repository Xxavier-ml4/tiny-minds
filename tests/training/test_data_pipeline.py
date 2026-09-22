"""Phase 3B, Phase B: canonical format, completion-only masking, packing
without leakage, dataset identity and the deterministic data plan."""
import unittest

import numpy as np

from tinymind.model import ModelConfig, TinyMindTransformer
from tinymind.model.tokenizer import ByteTokenizer
from tinymind.training.batching import collate_packed, collate_padded, plan_packed_rows
from tinymind.training.data import (DataPlan, DataSource, DatasetError, TokenizedDataset, sequential_batches,
                                   split_records)
from tinymind.training.render import ChatRenderer, ExampleError, IGNORE_INDEX, tool_call_text

TOK = ByteTokenizer()
R = ChatRenderer(TOK)


def chat(i, q, a, **extra):
    return {"id": f"c{i}", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], **extra}


def ids_of(text):
    return TOK.encode(text)


class TestRenderer(unittest.TestCase):
    def test_text_example_is_bos_text_eos_with_loss_everywhere_but_bos(self):
        ex = R.render({"id": "t", "text": "abc"})
        self.assertEqual(ex.ids.tolist(), [TOK.BOS] + ids_of("abc") + [TOK.EOS])
        self.assertEqual(ex.labels.tolist(), [IGNORE_INDEX] + ids_of("abc") + [TOK.EOS])

    def test_chat_trains_only_the_response_and_eos(self):
        ex = R.render(chat(1, "What is 2+3?", "5"))
        prompt = "user:\nWhat is 2+3?\nassistant:\n"
        expected_ids = [TOK.BOS] + ids_of(prompt) + ids_of("5") + [TOK.EOS]
        self.assertEqual(ex.ids.tolist(), expected_ids)
        trained = np.flatnonzero(ex.labels != IGNORE_INDEX)
        first = 1 + len(ids_of(prompt))
        self.assertEqual(trained.tolist(), list(range(first, len(expected_ids))))  # "5" and EOS, nothing else
        self.assertEqual(ex.labels[trained].tolist(), ids_of("5") + [TOK.EOS])
        self.assertEqual(ex.num_loss_tokens, 2)

    def test_the_response_is_actually_in_the_training_sequence(self):
        # the Phase 3A failure: the target was never tokenized
        ex = R.render(chat(1, "capital of France?", "Paris"))
        self.assertIn("Paris", TOK.decode(ex.ids[1:-1].tolist()))

    def test_legacy_target_shapes(self):
        legacy = {"id": "l1", "messages": [{"role": "user", "content": "Add 2 and 3"}], "tools": [{"name": "calculator", "parameters": {}}],
                  "target": {"type": "tool_call", "name": "calculator", "arguments": {"expr": "2+3"}}}
        ex = R.render(legacy)
        self.assertIn('{"name":"calculator","arguments":{"expr":"2+3"}}', TOK.decode(ex.ids[1:-1].tolist()))
        self.assertIn("tools: calculator\n", TOK.decode(ex.ids.tolist()[1:-1]))
        self.assertEqual(ex.category, "tool_call")
        ans = R.render({"id": "l2", "messages": [{"role": "user", "content": "hi"}], "target": {"type": "answer", "content": "Hello!"}})
        self.assertTrue(TOK.decode(ans.ids[1:-1].tolist()).endswith("assistant:\nHello!"))

    def test_tool_call_text_is_canonical(self):
        self.assertEqual(tool_call_text("t", {"b": 1, "a": "x"}), '{"name":"t","arguments":{"a":"x","b":1}}')

    def test_empty_or_missing_completion_is_rejected_not_silently_trained(self):
        with self.assertRaises(ExampleError):
            R.render(chat(1, "hello", "  "))
        with self.assertRaises(ExampleError):
            R.render({"id": "x", "messages": [{"role": "user", "content": "only a prompt"}]})
        with self.assertRaises(ExampleError):  # the Phase 3A demo shape: empty target
            R.render({"id": "y", "messages": [{"role": "user", "content": "text"}], "target": {"type": "answer", "content": ""}})

    def test_malformed_examples_name_the_example(self):
        bad = [{"messages": []}, {"id": "a", "text": "x", "messages": [{"role": "user", "content": "q"}]},
               {"id": "b", "messages": [{"role": "robot", "content": "q"}]},
               {"id": "c", "messages": [{"role": "user", "content": "q", "train": True}, {"role": "assistant", "content": "a"}]},
               {"id": "d", "messages": [{"role": "user", "content": "q"}], "tools": [{"name": "t"}],
                "target": {"type": "tool_call", "name": "other", "arguments": {}}}, 5, {"id": "e", "text": ""}]
        for record in bad:
            with self.assertRaises(ExampleError, msg=str(record)):
                R.render(record)

    def test_prompt_is_an_exact_prefix_of_the_training_sequence(self):
        cases = [
            chat(1, "hello", "hi there"),
            {"id": "s", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q"},
                                      {"role": "assistant", "content": "a"}], "tools": ["calc", "lookup"]},
        ]
        for record in cases:
            ex = R.render(record)
            msgs = [m for m in record["messages"] if m["role"] != "assistant"]
            prompt = R.render_prompt(msgs, tools=[t for t in record.get("tools", [])])
            self.assertEqual(ex.ids[:len(prompt)].tolist(), prompt)
            self.assertEqual(int(np.flatnonzero(ex.labels != IGNORE_INDEX)[0]), len(prompt))  # loss starts right after the prompt

    def test_multi_turn_and_tool_result_turns(self):
        record = {"id": "m", "messages": [
            {"role": "user", "content": "weather in Rome?"},
            {"role": "assistant", "content": tool_call_text("get_weather", {"city": "Rome"})},
            {"role": "tool", "content": "21C sunny"},
            {"role": "assistant", "content": "It is 21C and sunny in Rome."}], "tools": ["get_weather"]}
        ex = R.render(record)
        runs = R.explain(record)
        self.assertEqual([t for t, trained in runs if trained],
                         [tool_call_text("get_weather", {"city": "Rome"}), "<EOS>", "It is 21C and sunny in Rome.", "<EOS>"])
        self.assertNotIn("21C sunny", "".join(t for t, trained in runs if trained))
        self.assertEqual(ex.num_loss_tokens, sum(len(ids_of(t)) if t != "<EOS>" else 1 for t, tr in runs if tr))

    def test_train_false_assistant_turns_are_context_only(self):
        record = {"id": "m2", "messages": [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "context answer", "train": False},
            {"role": "user", "content": "b"}, {"role": "assistant", "content": "graded answer"}]}
        trained_text = "".join(t for t, tr in R.explain(record) if tr)
        self.assertEqual(trained_text, "graded answer<EOS>")

    def test_decode_completion_stops_at_eos(self):
        self.assertEqual(R.decode_completion(ids_of("done") + [TOK.EOS] + ids_of("junk")), "done")


class TestDatasetIdentity(unittest.TestCase):
    def build(self, records, **kw):
        return TokenizedDataset.from_records(records, R, 128, **kw)

    def test_hash_is_stable_and_sensitive(self):
        a = self.build([chat(1, "q1", "a1"), chat(2, "q2", "a2")])
        b = self.build([chat(1, "q1", "a1"), chat(2, "q2", "a2")])
        c = self.build([chat(1, "q1", "a1"), chat(2, "q2", "a3")])
        self.assertEqual(a.content_hash, b.content_hash)
        self.assertNotEqual(a.content_hash, c.content_hash)

    def test_duplicate_ids_and_overflow_policies(self):
        with self.assertRaises(DatasetError):
            self.build([chat(1, "q", "a"), chat(1, "q2", "a2")])
        long = chat(9, "q" * 200, "a")
        with self.assertRaises(DatasetError):
            self.build([chat(1, "q", "a"), long])
        dropped = self.build([chat(1, "q", "a"), long], overflow="drop")
        self.assertEqual((len(dropped), dropped.dropped["too_long"]), (1, 1))
        text = self.build([{"id": "t", "text": "z" * 500}], overflow="truncate")
        self.assertEqual((len(text.examples[0]), text.dropped["truncated"]), (128, 1))

    def test_split_is_deterministic_disjoint_and_order_independent(self):
        records = [chat(i, f"q{i}", f"a{i}") for i in range(50)]
        train, val = split_records(records, 0.2, seed=3)
        self.assertEqual(len(val), 10)
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in val})
        train2, val2 = split_records(list(reversed(records)), 0.2, seed=3)
        self.assertEqual({r["id"] for r in val}, {r["id"] for r in val2})
        self.assertEqual(split_records(records, 0.0)[1], [])


class TestBatching(unittest.TestCase):
    def examples(self):
        return [R.render(chat(i, "q" * (2 + i), "a" * (1 + i))) for i in range(5)]

    def test_padded_batch_masks_padding_and_counts_loss_tokens(self):
        exs = self.examples()
        b = collate_padded(exs, TOK.pad_token_id, 128)
        self.assertEqual(b.input_ids.shape[0], 5)
        self.assertTrue((b.labels[b.input_ids == TOK.pad_token_id] == IGNORE_INDEX).all())
        self.assertEqual(b.num_loss_tokens, sum(e.num_loss_tokens for e in exs))
        self.assertEqual(b.num_real_tokens, sum(len(e) for e in exs))
        self.assertIsNone(b.segment_ids)

    def test_packing_plan_is_greedy_in_order(self):
        self.assertEqual(plan_packed_rows([3, 4, 5, 2], 8), [[0, 1], [2, 3]])
        self.assertEqual(plan_packed_rows([8, 8], 8), [[0], [1]])
        with self.assertRaises(ValueError):
            plan_packed_rows([9], 8)

    def test_packed_batch_structure(self):
        exs = self.examples()
        rows = [exs[:3], exs[3:]]
        b = collate_packed(rows, TOK.pad_token_id, 128)
        self.assertEqual(b.num_examples, 5)
        self.assertEqual(b.num_loss_tokens, sum(e.num_loss_tokens for e in exs))  # packing loses no targets
        first_row = b.segment_ids[0]
        self.assertEqual(sorted(set(first_row.tolist())) , [0, 1, 2, 3] if first_row.min() == 0 else [1, 2, 3])
        starts = np.flatnonzero(np.r_[True, first_row[1:] != first_row[:-1]])
        self.assertTrue((b.labels[0, starts] == IGNORE_INDEX).all())  # nothing predicts a segment's first token

    def test_packing_reduces_padding(self):
        exs = [R.render(chat(i, "q" * (1 + (i * 7) % 25), "a")) for i in range(16)]  # varied lengths
        padded = collate_padded(exs, 0, 128)
        packed = collate_packed([[exs[i] for i in row] for row in plan_packed_rows([len(e) for e in exs], 128)], 0, 128)
        self.assertEqual(packed.num_real_tokens, padded.num_real_tokens)
        self.assertGreater(packed.num_real_tokens / packed.padded_tokens, padded.num_real_tokens / padded.padded_tokens)


class TestPackingDoesNotLeak(unittest.TestCase):
    """Spec section 9: an example must not be able to leak into another."""

    @classmethod
    def setUpClass(cls):
        cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
                          max_seq_len=160, vocab_size=TOK.vocab_size)
        cls.model = TinyMindTransformer(cfg, seed=11)

    def logits(self, ids, seg=None):
        return self.model(np.asarray(ids)[None], segment_ids=None if seg is None else np.asarray(seg)[None]).logits.data[0]

    def test_earlier_and_later_segments_do_not_influence_each_other(self):
        a = R.render(chat(1, "first question", "first answer")).ids
        b1 = R.render(chat(2, "second one", "alpha")).ids
        b2 = R.render(chat(3, "different!", "omega")).ids[:len(b1)]
        b2 = np.concatenate([b2, np.full(len(b1) - len(b2), 5)]) if len(b2) < len(b1) else b2
        seg = [1] * len(a) + [2] * len(b1)
        one = self.logits(np.concatenate([a, b1]), seg)
        two = self.logits(np.concatenate([a, b2]), seg)
        np.testing.assert_array_equal(one[:len(a)], two[:len(a)])       # A never sees B (causal + block mask)
        alone = self.logits(b1)
        np.testing.assert_allclose(one[len(a):], alone, atol=2e-5)      # B behaves as if A were absent
        c = R.render(chat(4, "x" * 20, "y")).ids
        three = self.logits(np.concatenate([c, b1]), [1] * len(c) + [2] * len(b1))
        np.testing.assert_allclose(three[len(c):], alone, atol=2e-5)    # ... whatever A was

    def test_control_without_segment_ids_the_same_pair_does_leak(self):
        a1, a2 = R.render(chat(1, "aaaa", "bbbb")).ids, R.render(chat(2, "cccc", "dddd")).ids
        b = R.render(chat(3, "second", "x")).ids
        l1 = self.logits(np.concatenate([a1, b]))
        l2 = self.logits(np.concatenate([a2, b]))
        self.assertGreater(float(np.abs(l1[len(a1):] - l2[len(a2):]).max()), 1e-4)  # plain causal attention mixes them

    def test_packed_loss_equals_sum_of_individual_losses(self):
        exs = [R.render(chat(i, f"question {i}", f"answer {i * 7}")) for i in range(3)]
        total = 0.0
        for e in exs:
            total += float(self.model(e.ids[None], labels=e.labels[None], loss_normalizer=1.0).loss.item())
        packed = collate_packed([exs], 0, 160)
        got = float(self.model(packed.input_ids, labels=packed.labels, segment_ids=packed.segment_ids,
                               loss_normalizer=1.0).loss.item())
        self.assertAlmostEqual(got, total, delta=1e-3 * max(1.0, abs(total)))

    def test_label_at_segment_boundary_never_trains_cross_example_prediction(self):
        # The last token of A would, unmasked, be trained to predict B's BOS. The label is -100 there
        # (and test_packed_loss_equals_sum_of_individual_losses proves no such term is in the loss).
        a, b = R.render(chat(1, "q1", "a1")), R.render(chat(2, "q2", "a2"))
        row = collate_packed([[a, b]], 0, 160)
        self.assertEqual(int(row.input_ids[0, len(a)]), TOK.BOS)
        self.assertEqual(int(row.labels[0, len(a)]), IGNORE_INDEX)
        self.assertEqual(int(row.segment_ids[0, len(a) - 1]), 1)
        self.assertEqual(int(row.segment_ids[0, len(a)]), 2)


class TestDataPlan(unittest.TestCase):
    def sources(self, n_a=20, n_b=8):
        a = TokenizedDataset.from_records([chat(i, f"a-question {i}", f"a{i}") for i in range(n_a)], R, 128, name="a")
        b = TokenizedDataset.from_records([chat(1000 + i, f"b-question {i}", f"b{i}") for i in range(n_b)], R, 128, name="b")
        return a, b

    def plan(self, sources, **kw):
        base = dict(seed=5, batch_size=4, max_seq_len=128, packing=False, pad_id=0)
        base.update(kw)
        return DataPlan(sources, **base)

    def test_same_seed_same_batches_and_position_is_stateless(self):
        a, _ = self.sources()
        p1, p2 = self.plan([DataSource("a", a)]), self.plan([DataSource("a", a)])
        for epoch in (0, 1, 2):
            for k in range(p1.num_micro_batches(epoch)):
                np.testing.assert_array_equal(p1.micro_batch(epoch, k).input_ids, p2.micro_batch(epoch, k).input_ids)
        self.assertFalse(np.array_equal(p1.micro_batch(0, 0).input_ids, p1.micro_batch(1, 0).input_ids))
        self.assertFalse(np.array_equal(p1.micro_batch(0, 0).input_ids, self.plan([DataSource("a", a)], seed=6).micro_batch(0, 0).input_ids))

    def test_every_example_appears_once_per_epoch_for_a_single_source(self):
        a, _ = self.sources(n_a=16)
        p = self.plan([DataSource("a", a)])
        seen = [tuple(b.input_ids[i][:6]) for k in range(p.num_micro_batches(0)) for b in [p.micro_batch(0, k)] for i in range(4)]
        self.assertEqual(len(seen), 16)
        self.assertEqual(sorted(p._epoch_order(0)[:, 1].tolist()), list(range(16)))

    def test_mixture_quotas_are_exact_and_reported(self):
        a, b = self.sources(20, 8)
        p = self.plan([DataSource("a", a, 0.75), DataSource("b", b, 0.25)], epoch_examples=40)
        order = p._epoch_order(0)
        self.assertEqual((int((order[:, 0] == 0).sum()), int((order[:, 0] == 1).sum())), (30, 10))
        self.assertEqual(p.repeat_factors(), {"a": 1.5, "b": 1.25})

    def test_other_sources_do_not_reorder_a_source(self):
        a, b = self.sources(20, 8)
        alone = self.plan([DataSource("a", a)], epoch_examples=20)._epoch_order(0)
        mixed = self.plan([DataSource("a", a, 1.0), DataSource("b", b, 1e-9)], epoch_examples=20)._epoch_order(0)
        self.assertEqual(alone[alone[:, 0] == 0][:, 1].tolist(), mixed[mixed[:, 0] == 0][:, 1].tolist())

    def test_identity_hash_pins_data_weights_and_algorithm(self):
        a, b = self.sources()
        base = self.plan([DataSource("a", a, 1.0), DataSource("b", b, 1.0)]).dataset_hash()
        self.assertEqual(base, self.plan([DataSource("a", a, 1.0), DataSource("b", b, 1.0)]).dataset_hash())
        self.assertNotEqual(base, self.plan([DataSource("a", a, 1.0), DataSource("b", b, 2.0)]).dataset_hash())
        self.assertNotEqual(base, self.plan([DataSource("a", a, 1.0)]).dataset_hash())

    def test_drop_last_and_packed_plan(self):
        a, _ = self.sources(n_a=18)
        p = self.plan([DataSource("a", a)])
        self.assertEqual(p.num_micro_batches(0), 18 // 4)
        pk = self.plan([DataSource("a", a)], packing=True, batch_size=2, max_seq_len=128)
        batch = pk.micro_batch(0, 0)
        self.assertIsNotNone(batch.segment_ids)
        self.assertGreater(batch.num_examples, 2)  # several examples per packed row

    def test_sequential_batches_cover_the_set_in_order(self):
        a, _ = self.sources(n_a=10)
        got = [b.num_examples for b in sequential_batches(a, 4, 128, 0)]
        self.assertEqual(got, [4, 4, 2])


if __name__ == "__main__":
    unittest.main()
