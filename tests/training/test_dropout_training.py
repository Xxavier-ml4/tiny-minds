"""Dropout inside the trainer, and the diagnostics that tell you whether memorization is really the problem.

* Dropout masks are seeded from (seed, optimizer step, micro-batch), so a run interrupted and resumed must end
  bit-identical to an uninterrupted one — the property that makes enabling dropout safe on a long CPU run.
* Evaluation never sees dropout.
* ``fit_diagnostics`` / ``memorization_signal`` / ``planned_passes`` measure overfitting instead of assuming it.
"""
import unittest

import numpy as np

from tests.training._helpers import dataset, make_engine, tmpdir
from tinymind.training.engine import MAX_RECOMMENDED_PASSES, memorization_signal


def weights(engine):
    return {n: p.data.copy() for n, p in engine.model.named_parameters()}


def assert_same_weights(test, a, b, msg=""):
    test.assertEqual(a.keys(), b.keys())
    for n in a:
        test.assertTrue(np.array_equal(a[n], b[n]), f"{msg} {n}")


class TestDropoutTraining(unittest.TestCase):
    def train(self, dropout, steps=8, **kw):
        e = make_engine(tmpdir(), model_over=dict(dropout=dropout), train_over=dict(max_steps=steps, checkpoint_interval=1000),
                        **kw)
        e.train()
        return e

    def test_dropout_changes_what_training_learns(self):
        with_dropout, without = weights(self.train(0.2)), weights(self.train(0.0))
        self.assertTrue(with_dropout, "no parameters found")
        self.assertTrue(any(not np.array_equal(with_dropout[n], without[n]) for n in with_dropout))

    def test_same_seed_is_bit_reproducible_with_dropout(self):
        assert_same_weights(self, weights(self.train(0.2)), weights(self.train(0.2)))

    def test_rate_zero_is_bit_identical_to_the_pre_dropout_trainer(self):
        # an engine whose model config never mentioned dropout (default 0.0) and one that sets it explicitly
        explicit = self.train(0.0)
        e = make_engine(tmpdir(), train_over=dict(max_steps=8, checkpoint_interval=1000))
        e.train()
        assert_same_weights(self, weights(explicit), weights(e))

    def test_interrupted_run_resumes_bit_identical_with_dropout(self):
        kw = dict(model_over=dict(dropout=0.2), train_over=dict(max_steps=20, checkpoint_interval=10))
        continuous = make_engine(tmpdir(), **{**kw, "train_over": dict(max_steps=20, checkpoint_interval=1000)})
        continuous.train()

        out, state = tmpdir(), {"n": 0}

        def hook(point):
            state["n"] += point == "after_model"
            if state["n"] == 2 and point == "before_rename":  # die while writing the SECOND checkpoint
                raise KeyboardInterrupt
        interrupted = make_engine(out, fault_hook=hook, **kw)
        with self.assertRaises(KeyboardInterrupt):
            interrupted.train()
        resumed = make_engine(tmpdir(), resume=out / "checkpoints", **kw)
        self.assertEqual(resumed.step, 10)  # it really did restart from the first checkpoint, mid-run
        resumed.train()
        assert_same_weights(self, weights(continuous), weights(resumed), "resume with dropout")

    def test_evaluation_is_deterministic_and_ignores_the_dropout_setting(self):
        drop, plain = make_engine(tmpdir(), model_over=dict(dropout=0.3)), make_engine(tmpdir())
        first = drop.evaluate()
        self.assertEqual(first, drop.evaluate())   # no hidden randomness
        self.assertEqual(first, plain.evaluate())  # same init, same validation data -> same loss: dropout is off here

    def test_dropout_generator_is_a_function_of_step_and_micro_batch_only(self):
        e = make_engine(tmpdir(), model_over=dict(dropout=0.2))
        a, b = e._dropout_rng(0).random(4), e._dropout_rng(0).random(4)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, e._dropout_rng(1).random(4)))   # another micro-batch
        e.step += 1
        self.assertFalse(np.array_equal(a, e._dropout_rng(0).random(4)))   # another step
        self.assertIsNone(make_engine(tmpdir())._dropout_rng(0))           # no dropout -> no generator at all


class TestFitDiagnostics(unittest.TestCase):
    def test_eval_history_records_train_loss_and_the_gap(self):
        e = make_engine(tmpdir(), train_over=dict(max_steps=12, eval_interval=4, checkpoint_interval=1000))
        e.train()
        rows = [r for r in e.val_history if "train_loss_recent" in r]
        self.assertTrue(rows, "evaluations after the first training step must carry the diagnostics")
        for r in rows:
            self.assertAlmostEqual(r["generalization_gap"], r["val_loss"] - r["train_loss_recent"], places=6)

    def test_no_diagnostics_before_any_training_step(self):
        self.assertEqual(make_engine(tmpdir()).fit_diagnostics({"val_loss": 3.0}), {})


class TestMemorizationSignal(unittest.TestCase):
    @staticmethod
    def hist(pairs):
        return [{"step": i, "val_loss": v, "train_loss_recent": t} for i, (v, t) in enumerate(pairs)]

    def test_signals_when_val_rises_and_train_falls_repeatedly(self):
        msg = memorization_signal(self.hist([(3.0, 3.0), (3.1, 2.8), (3.3, 2.5)]))
        self.assertIn("memorizing", msg)

    def test_quiet_when_validation_still_improves(self):
        self.assertIsNone(memorization_signal(self.hist([(3.0, 3.0), (2.8, 2.7), (2.6, 2.4)])))

    def test_quiet_for_a_single_bad_evaluation(self):
        self.assertIsNone(memorization_signal(self.hist([(3.0, 3.0), (2.9, 2.8), (3.0, 2.6)])))

    def test_quiet_when_train_loss_is_not_falling(self):
        self.assertIsNone(memorization_signal(self.hist([(3.0, 3.0), (3.1, 3.1), (3.3, 3.3)])))

    def test_quiet_without_enough_history_or_diagnostic_fields(self):
        self.assertIsNone(memorization_signal(self.hist([(3.0, 3.0), (3.5, 2.0)])))
        self.assertIsNone(memorization_signal([{"step": i, "val_loss": 3.0 + i} for i in range(5)]))

    def test_patience_is_respected(self):
        h = self.hist([(3.0, 3.0), (3.1, 2.8), (3.3, 2.5), (3.4, 2.2)])
        self.assertIsNotNone(memorization_signal(h, patience=3))
        self.assertIsNone(memorization_signal(self.hist([(3.0, 3.0), (3.1, 2.8), (3.3, 2.5)]), patience=3))


class TestDataSufficiencyWarnings(unittest.TestCase):
    def test_a_budget_far_larger_than_the_data_warns_and_says_so_in_the_summary(self):
        # 40 examples / batch 4 = 10 micro-batches per epoch; 100 steps = 10 passes, more than the advisory limit
        e = make_engine(tmpdir(), train_over=dict(max_steps=100, checkpoint_interval=1000, eval_interval=0))
        self.assertGreater(e.planned_passes["train"], MAX_RECOMMENDED_PASSES)
        self.assertEqual(len(e.data_warnings), 1)
        self.assertIn("train", e.data_warnings[0])
        self.assertEqual(e.planned_passes["train"], 10.0)

    def test_enough_data_does_not_warn(self):
        e = make_engine(tmpdir(), train_data=dataset(400), train_over=dict(max_steps=40, checkpoint_interval=1000))
        self.assertEqual(e.data_warnings, [])
        self.assertLess(e.planned_passes["train"], MAX_RECOMMENDED_PASSES)

    def test_summary_carries_planned_passes_and_warnings(self):
        e = make_engine(tmpdir(), train_over=dict(max_steps=60, checkpoint_interval=1000))
        summary = e.train()
        self.assertEqual(summary["dataset"]["planned_passes"], e.planned_passes)
        self.assertEqual(summary["dataset"]["warnings"], e.data_warnings)
        self.assertTrue(summary["dataset"]["warnings"])                    # 60 steps = 6 passes > 4

    def test_warning_is_advisory_it_never_stops_training(self):
        e = make_engine(tmpdir(), train_over=dict(max_steps=60, checkpoint_interval=1000))
        e.train()
        self.assertEqual(e.step, 60)



class TestEveryTrainerAppliesDropout(unittest.TestCase):
    """A ``dropout`` setting that some trainer quietly ignores is the exact "silently ignored config knob" failure this
    repository's Phase 3A audit was about. Every code path that trains must apply it; evaluation must not."""

    @staticmethod
    def legacy_weights(dropout, seed=2):
        import json
        from pathlib import Path

        from tinymind.model import ByteTokenizer, ModelConfig, TinyMindTransformer
        from tinymind.training.causal_lm_trainer import CausalLMTrainer, CausalLMTrainingConfig
        from tinymind.training.dataset import TrainingDataset

        tmp = tmpdir()
        with (tmp / "train.jsonl").open("w") as f:
            for i in range(4):
                f.write(json.dumps({"id": f"ex_{i}", "messages": [{"role": "user", "content": "repeat this phrase"}],
                                    "target": {"type": "answer", "content": ""}}) + "\n")
        tok = ByteTokenizer()
        cfg = ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=32,
                          max_seq_len=32, vocab_size=tok.vocab_size, dropout=dropout)
        model = TinyMindTransformer(cfg, seed=seed)
        CausalLMTrainer(model, CausalLMTrainingConfig(learning_rate=5e-3, batch_size=2, epochs=3, seed=seed)).train(
            TrainingDataset(tmp / "train.jsonl", tok))
        return {n: p.data.copy() for n, p in model.named_parameters()}

    def test_legacy_trainer_applies_dropout(self):
        with_dropout, without = self.legacy_weights(0.3), self.legacy_weights(0.0)
        self.assertTrue(any(not np.array_equal(with_dropout[n], without[n]) for n in with_dropout))

    def test_legacy_trainer_with_dropout_is_reproducible(self):
        assert_same_weights(self, self.legacy_weights(0.3), self.legacy_weights(0.3))

    def test_benchmark_step_runs_with_dropout(self):
        from tinymind.model import ModelConfig
        from tinymind.training.benchmark_50m import measure_shape
        cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
                          max_seq_len=64, vocab_size=64, dropout=0.1)
        m = measure_shape(cfg, batch_size=1, gradient_accumulation=2, seq_len=32, seed=0, warmup=False)
        self.assertGreater(m.forward_seconds, 0.0)
        self.assertGreater(m.backward_seconds, 0.0)

    def test_the_seeding_rule_is_defined_once(self):
        from tinymind.model.tensor import dropout_generator
        e = make_engine(tmpdir(), model_over=dict(dropout=0.2))
        self.assertTrue(np.array_equal(e._dropout_rng(1).random(8), dropout_generator(e.config.seed, e.step, 1).random(8)))


if __name__ == "__main__":
    unittest.main()
