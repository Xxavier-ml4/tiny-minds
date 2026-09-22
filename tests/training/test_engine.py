"""Phase 3B, phases C-E: accumulation correctness, exact resume, time budget,
mismatch rejection, non-finite handling."""
import json
import math
import unittest

import numpy as np

from tinymind.model.config import ModelConfigError
from tinymind.model.optim import AdamW, NonFiniteGradientError, OptimizerStateError
from tinymind.model.tokenizer import ByteTokenizer
from tinymind.training import checkpoint as ck
from tinymind.training.config import TrainingConfig, TrainingConfigError
from tinymind.training.data import DataSource
from tinymind.training.engine import TrainingDivergedError, TrainingEngine
from tinymind.training.schedule import LRSchedule, ScheduleStateError

from tests.training._helpers import (FakeClock, RENDERER, TOK, dataset, make_engine, model_config, records, tmpdir,
                                     train_config)


def weights(engine):
    return {n: p.data.copy() for n, p in engine.model.named_parameters()}


def assert_engines_identical(tc, a, b):
    for (n, x), (_, y) in zip(sorted(weights(a).items()), sorted(weights(b).items())):
        tc.assertTrue(np.array_equal(x, y), f"weights differ at {n}")
    sa, sb = a.optimizer.state_dict(), b.optimizer.state_dict()
    tc.assertEqual((sa["step_count"], sa["lr"], sa["names"]), (sb["step_count"], sb["lr"], sb["names"]))
    for k in sa["arrays"]:
        tc.assertTrue(np.array_equal(sa["arrays"][k], sb["arrays"][k]), f"optimizer moment {k} differs")
    tc.assertEqual(a.schedule.state_dict(), b.schedule.state_dict())
    tc.assertEqual((a.step, a.epoch, a.cursor, a.tokens_processed, a.loss_tokens_processed),
                   (b.step, b.epoch, b.cursor, b.tokens_processed, b.loss_tokens_processed))
    tc.assertEqual(ck.rng_state_to_json(a.rng), ck.rng_state_to_json(b.rng))
    tc.assertEqual(a.recent_losses, b.recent_losses)


class TestAccumulation(unittest.TestCase):
    def one_step_grads(self, batch, accum):
        e = make_engine(tmpdir(), train_over=dict(batch_size=batch, gradient_accumulation_steps=accum, max_steps=1,
                                                  warmup_steps=0, checkpoint_interval=0), val_data=dataset(8, 1000, "v", "val"))
        e.train_step()
        return {n: p.grad.copy() for n, p in e.model.named_parameters()}, e

    def test_batch8_equals_batch4_accum2(self):
        g8, _ = self.one_step_grads(8, 1)
        g42, _ = self.one_step_grads(4, 2)
        worst = 0.0
        for n in g8:
            np.testing.assert_allclose(g8[n], g42[n], rtol=2e-4, atol=1e-6, err_msg=n)
            worst = max(worst, float(np.abs(g8[n] - g42[n]).max()))
        self.assertLess(worst, 1e-5)

    def test_accumulation_is_token_weighted_not_micro_batch_weighted(self):
        # The Phase 3A rule (mean of micro-batch means) differs when micro-batches carry different token counts.
        g_correct, e = self.one_step_grads(4, 2)
        micro = [e.plan.micro_batch(0, j) for j in range(2)]
        self.assertNotEqual(micro[0].num_loss_tokens, micro[1].num_loss_tokens)  # the fixture must exercise the difference
        m = make_engine(tmpdir(), train_over=dict(batch_size=4, gradient_accumulation_steps=2, max_steps=1, warmup_steps=0))
        m.optimizer.zero_grad()
        for b in micro:  # old behaviour: each micro-batch normalised by ITS OWN token count, scaled 1/2
            out = m.model(b.input_ids, labels=b.labels, segment_ids=b.segment_ids)
            (out.loss * 0.5).backward()
        old = {n: p.grad.copy() for n, p in m.model.named_parameters()}
        biggest = max(float(np.abs(old[n] - g_correct[n]).max()) for n in old)
        self.assertGreater(biggest, 1e-3)

    def test_accumulation_with_packing_matches_larger_batch(self):
        def grads(batch, accum):
            e = make_engine(tmpdir(), train_over=dict(batch_size=batch, gradient_accumulation_steps=accum, max_steps=1,
                                                      warmup_steps=0, packing=True, max_seq_len=96))
            e.train_step()
            return {n: p.grad.copy() for n, p in e.model.named_parameters()}
        a, b = grads(2, 2), grads(4, 1)
        # same rows in the same order -> identical token sets regardless of the split
        for n in a:
            np.testing.assert_allclose(a[n], b[n], rtol=2e-4, atol=1e-6, err_msg=n)


class TestExactResume(unittest.TestCase):
    def run_pair(self, split, **over):
        over = dict(max_steps=100, checkpoint_interval=1000, **over)
        cont = make_engine(tmpdir(), train_over=over)
        cont.train()
        out = tmpdir()
        clock = FakeClock(1.0)
        first = make_engine(out, train_over=dict(over, max_runtime_seconds=split + 0.5, safety_margin_seconds=0), clock=clock)
        first.train()
        self.assertEqual((first.step, first._stop_reason, first.last_checkpoint.name), (split, None, f"step-{split:08d}"))
        second = make_engine(out, train_over=over, resume=out / "checkpoints")   # brand-new objects, state only from disk
        self.assertEqual((second.step, second.cursor), (first.step, first.cursor))
        second.train()
        return cont, second

    def test_split_run_matches_continuous_run(self):
        cont, split = self.run_pair(50)
        self.assertEqual(cont.step, 100)
        assert_engines_identical(self, cont, split)  # bitwise: weights, both Adam moments, step, schedule, RNG, loss

    def test_split_mid_epoch_matches_too(self):
        cont, split = self.run_pair(37)
        assert_engines_identical(self, cont, split)

    def test_split_with_packing_and_accumulation_and_several_epochs(self):
        cont, split = self.run_pair(41, packing=True, gradient_accumulation_steps=2, batch_size=2, max_seq_len=96)
        self.assertGreater(cont.epoch, 1)  # the split really crosses epoch boundaries
        assert_engines_identical(self, cont, split)

    def test_data_position_is_restored_batch_for_batch(self):
        def trace(engine):
            seen = []
            real = engine.plan.micro_batch
            engine.plan.micro_batch = lambda e, i: (seen.append((e, i, real(e, i).input_ids.tobytes())) or real(e, i))
            engine.train()
            return seen
        over = dict(max_steps=30, checkpoint_interval=1000)
        cont_trace = trace(make_engine(tmpdir(), train_over=over))
        out = tmpdir()
        make_engine(out, train_over=dict(over, max_runtime_seconds=13.5, safety_margin_seconds=0), clock=FakeClock()).train()
        second = make_engine(out, train_over=over, resume=out / "checkpoints")
        tail = trace(second)
        self.assertEqual(cont_trace[13:], tail)  # the resumed run consumes exactly the remaining batches, in order

    def test_completed_stage_refuses_resume(self):
        out = tmpdir()
        make_engine(out).train()
        with self.assertRaises(ck.ResumeMismatchError) as cm:
            make_engine(out, resume=out / "checkpoints")
        self.assertIn("COMPLETED", str(cm.exception))


class TestMismatchRejection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tmpdir()
        make_engine(cls.out, train_over=dict(max_steps=20, max_runtime_seconds=10.5, safety_margin_seconds=0),
                    clock=FakeClock()).train()
        cls.root = cls.out / "checkpoints"

    def snapshot(self):
        return {str(p.relative_to(self.root)): ck.sha256_file(p) for p in sorted(self.root.rglob("*")) if p.is_file()}

    def expect_rejected(self, needles, **kw):
        before = self.snapshot()
        with self.assertRaises(ck.ResumeMismatchError) as cm:
            make_engine(tmpdir(), resume=self.root, **kw)
        for needle in needles:
            self.assertIn(needle, str(cm.exception))
        self.assertEqual(before, self.snapshot())  # never modified

    def test_baseline_resume_works(self):
        e = make_engine(tmpdir(), resume=self.root, train_over=dict(max_steps=20))
        self.assertEqual(e.step, 10)

    def test_architecture_hidden_size(self):
        self.expect_rejected(["model architecture differs", "hidden_size"], model_over=dict(hidden_size=64, num_heads=4,
                             num_kv_heads=2, intermediate_size=128), train_over=dict(max_steps=20))

    def test_architecture_layers(self):
        self.expect_rejected(["num_layers"], model_over=dict(num_layers=3), train_over=dict(max_steps=20))

    def test_vocabulary_and_tokenizer(self):
        class Wide(ByteTokenizer):
            def __init__(self):
                super().__init__()
                self._vocab_size = 300

            def spec(self):
                return {**super().spec(), "vocab_size": 300}
        from tinymind.training.data import TokenizedDataset
        from tinymind.training.render import ChatRenderer
        wide = Wide()
        r = ChatRenderer(wide)  # datasets must be rendered with the tokenizer the run uses
        tr = TokenizedDataset.from_records(records(40), r, 96, name="train")
        va = TokenizedDataset.from_records(records(8, 1000, "v"), r, 96, name="val")
        with self.assertRaises(ck.ResumeMismatchError) as cm:
            make_engine(tmpdir(), resume=self.root, tokenizer=wide, model_over=dict(vocab_size=300),
                        train_data=tr, val_data=va, train_over=dict(max_steps=20))
        text = str(cm.exception)
        self.assertIn("vocab_size", text)
        self.assertIn("tokenizer differs", text)

    def test_tokenizer_alone(self):
        class Other(ByteTokenizer):
            def spec(self):
                return {**super().spec(), "variant": "other"}
        other = Other()
        from tests.training._helpers import RENDERER as _r
        from tinymind.training.data import TokenizedDataset
        from tinymind.training.render import ChatRenderer
        r2 = ChatRenderer(other)
        tr = TokenizedDataset.from_records(records(40), r2, 96, name="train")
        va = TokenizedDataset.from_records(records(8, 1000, "v"), r2, 96, name="val")
        self.expect_rejected(["tokenizer differs"], tokenizer=other, train_data=tr, val_data=va, train_over=dict(max_steps=20))

    def test_dataset(self):
        self.expect_rejected(["dataset identity differs"], train_data=dataset(40, offset=5), train_over=dict(max_steps=20))

    def test_training_config_and_stage_and_all_problems_reported_together(self):
        self.expect_rejected(["training configuration differs", "learning_rate"], train_over=dict(max_steps=20, learning_rate=1e-3))
        self.expect_rejected(["training configuration differs", "batch_size"], train_over=dict(max_steps=20, batch_size=2))
        self.expect_rejected(["total_steps"], train_over=dict(max_steps=30))  # different schedule length
        self.expect_rejected(["stage differs"], train_over=dict(max_steps=20, stage="stage1"))
        self.expect_rejected(["model architecture differs", "training configuration differs", "dataset identity differs"],
                             model_over=dict(num_layers=3), train_over=dict(max_steps=20, seed=99), train_data=dataset(40, 3))

    def test_runtime_only_settings_may_differ(self):
        e = make_engine(tmpdir(), resume=self.root, train_over=dict(max_steps=20, eval_interval=5, checkpoint_interval=7,
                                                                   log_interval=3, max_runtime_seconds=999))
        self.assertEqual(e.step, 10)

    def test_a_corrupt_named_checkpoint_is_an_error_not_a_fresh_start(self):
        out = tmpdir()
        make_engine(out, train_over=dict(max_steps=20, max_runtime_seconds=10.5, safety_margin_seconds=0), clock=FakeClock()).train()
        target = next((out / "checkpoints").glob("step-*"))
        blob = bytearray((target / "model.npz").read_bytes())
        blob[len(blob) // 2] ^= 0xFF
        (target / "model.npz").write_bytes(bytes(blob))
        with self.assertRaises(ck.CheckpointCorruptError):
            make_engine(tmpdir(), resume=target, train_over=dict(max_steps=20))


class TestTimeBudget(unittest.TestCase):
    def test_saves_exports_and_exits_cleanly_before_the_limit(self):
        exported = {}

        def exporter(engine, out):
            exported["step"] = engine.step
            (out / "model.marker").write_text("exported")
            return {"marker": str(out / "model.marker")}

        out = tmpdir()
        clock = FakeClock(1.0)
        limit, margin = 30.0, 12.0
        e = make_engine(out, train_over=dict(max_steps=1000, max_runtime_seconds=limit, safety_margin_seconds=margin,
                                             checkpoint_interval=0), clock=clock, exporter=exporter)
        summary = e.train()
        self.assertEqual(summary["stop_reason"], "time_budget")
        self.assertFalse(summary["stage_complete"])
        self.assertTrue(0 < e.step < 1000)
        self.assertLessEqual(clock.now - 1, limit - margin + 2)     # stopped no later than the margin allows
        self.assertTrue(ck.verify_checkpoint(e.last_checkpoint).ok)  # the final checkpoint is valid
        self.assertEqual(exported["step"], e.step)                    # export ran at the stopping step
        self.assertEqual(summary["exports"]["marker"], str(out / "model.marker"))
        self.assertTrue((out / "training_summary.json").exists())
        again = make_engine(out, train_over=dict(max_steps=1000, checkpoint_interval=0), resume=out / "checkpoints")
        self.assertEqual(again.step, e.step)

    def test_external_stop_request_behaves_the_same(self):
        e = make_engine(tmpdir(), train_over=dict(max_steps=50, log_interval=1))
        e._log_fn = lambda msg: e.request_stop("sigterm") if "step 7/" in msg else None
        s = e.train()
        self.assertEqual((s["stop_reason"], e.step), ("sigterm", 7))
        self.assertTrue(ck.verify_checkpoint(e.last_checkpoint).ok)


class TestDivergence(unittest.TestCase):
    def test_nan_weight_stops_without_poisoning_the_last_good_checkpoint(self):
        out = tmpdir()
        e = make_engine(out, train_over=dict(max_steps=30, checkpoint_interval=5))
        real = e.train_step
        state = {"n": 0}

        def poisoned():
            state["n"] += 1
            if state["n"] == 8:
                next(iter(e.model.parameters())).data[0, 0] = np.nan
            return real()
        e.train_step = poisoned
        with self.assertRaises(TrainingDivergedError):
            e.train()
        found, _ = ck.find_latest_valid(out / "checkpoints")
        self.assertEqual(found.name, "step-00000005")
        self.assertTrue(ck.verify_checkpoint(found).ok)
        summary = json.loads((out / "training_summary.json").read_text())
        self.assertEqual(summary["stop_reason"], "diverged")

    def test_optimizer_refuses_nonfinite_gradients_before_touching_state(self):
        e = make_engine(tmpdir())
        e.train_step()
        before = e.optimizer.state_dict()
        w = weights(e)
        e.optimizer.zero_grad()
        next(iter(e.model.parameters())).grad = np.full_like(next(iter(e.model.parameters())).data, np.inf)
        with self.assertRaises(NonFiniteGradientError):
            e.optimizer.step()
        after = e.optimizer.state_dict()
        self.assertEqual(before["step_count"], after["step_count"])
        for k in w:
            self.assertTrue(np.array_equal(w[k], weights(e)[k]))


class TestStateObjects(unittest.TestCase):
    def test_scheduler_state_round_trip_and_mismatch(self):
        s = LRSchedule("cosine", 1e-3, 1e-4, 5, 50)
        for _ in range(17):
            s.advance()
        t = LRSchedule("cosine", 1e-3, 1e-4, 5, 50)
        t.load_state_dict(s.state_dict())
        self.assertEqual((t.last_step, t.current_lr()), (17, s.current_lr()))
        with self.assertRaises(ScheduleStateError):
            LRSchedule("cosine", 1e-3, 1e-4, 5, 60).load_state_dict(s.state_dict())
        with self.assertRaises(ScheduleStateError):
            LRSchedule("linear", 1e-3, 1e-4, 5, 50).load_state_dict(s.state_dict())

    def test_schedule_shape(self):
        s = LRSchedule("cosine", 1e-2, 1e-3, 10, 100)
        lrs = [s.lr_at(i) for i in range(100)]
        self.assertAlmostEqual(lrs[9], 1e-2)
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:])))
        self.assertAlmostEqual(lrs[-1], 1e-3, delta=2e-5)

    def test_optimizer_state_round_trip_and_all_or_nothing(self):
        a = make_engine(tmpdir(), train_over=dict(max_steps=5)); a.train_step(); a.train_step()
        b = make_engine(tmpdir(), train_over=dict(max_steps=5))
        b.optimizer.load_state_dict(a.optimizer.state_dict())
        for k, v in a.optimizer.state_dict()["arrays"].items():
            self.assertTrue(np.array_equal(v, b.optimizer.state_dict()["arrays"][k]))
        bad = a.optimizer.state_dict()
        bad["arrays"]["v." + bad["names"][-1]] = bad["arrays"]["v." + bad["names"][-1]][:1]
        c = make_engine(tmpdir(), train_over=dict(max_steps=5))
        with self.assertRaises(OptimizerStateError):
            c.optimizer.load_state_dict(bad)
        self.assertEqual(c.optimizer.step_count, 0)
        self.assertTrue(all((m == 0).all() for m in c.optimizer._m))  # nothing was partially loaded
        wrong_hyper = a.optimizer.state_dict()
        wrong_hyper["hyper"] = {**wrong_hyper["hyper"], "beta2": 0.5}
        with self.assertRaises(OptimizerStateError):
            c.optimizer.load_state_dict(wrong_hyper)

    def test_weight_decay_exempts_norm_gains(self):
        e = make_engine(tmpdir(), train_over=dict(weight_decay=0.5, learning_rate=1e-2, min_learning_rate=1e-3, warmup_steps=0))
        self.assertEqual(e.optimizer.decay_min_ndim, 2)
        norm = dict(e.model.named_parameters())["final_norm.weight"]
        norm.data[...] = 1.0
        e.optimizer.zero_grad()
        for p in e.model.parameters():
            p.grad = np.zeros_like(p.data)
        e.optimizer.step()
        self.assertTrue(np.array_equal(norm.data, np.ones_like(norm.data)))  # zero grad + no decay => untouched

    def test_rng_state_survives_json_round_trip(self):
        rng = np.random.default_rng(np.random.SeedSequence([4, 5]))
        rng.random(7)
        restored = ck.rng_from_json(json.loads(json.dumps(ck.rng_state_to_json(rng))))
        self.assertTrue(np.array_equal(rng.random(5), restored.random(5)))


class TestEngineGuards(unittest.TestCase):
    def test_validation_set_is_mandatory_and_loss_is_token_mean(self):
        with self.assertRaises(TrainingConfigError):
            TrainingEngine(model_config=model_config(), tokenizer=TOK, config=train_config(),
                           sources=[DataSource("train", dataset())], validation=None, output_dir=tmpdir(), log=None)
        e = make_engine(tmpdir())
        ev = e.evaluate()
        manual, tokens = 0.0, 0
        for ex in e.validation.examples:
            out = e.model(ex.ids[None], labels=ex.labels[None], loss_normalizer=1.0)
            manual += float(out.loss.item()); tokens += ex.num_loss_tokens
        self.assertAlmostEqual(ev["val_loss"], manual / tokens, places=4)
        self.assertEqual(ev["val_tokens"], tokens)

    def test_misconfigurations_fail_loudly(self):
        with self.assertRaises(TrainingConfigError):
            make_engine(tmpdir(), model_over=dict(vocab_size=300))          # tokenizer/model vocabulary mismatch
        with self.assertRaises(ModelConfigError):
            make_engine(tmpdir(), model_over=dict(dropout=0.1))              # would be silently ignored otherwise
        with self.assertRaises(Exception):
            make_engine(tmpdir(), train_over=dict(max_seq_len=500))
        with self.assertRaises(TrainingConfigError):
            TrainingConfig.from_dict({"learning_rat": 1})                    # typo
        with self.assertRaises(TrainingConfigError):
            train_config(min_learning_rate=1.0)

    def test_summary_is_generated_from_the_run(self):
        out = tmpdir()
        e = make_engine(out, train_over=dict(max_steps=12, eval_interval=6, checkpoint_interval=6))
        s = e.train()
        on_disk = json.loads((out / "training_summary.json").read_text())
        self.assertEqual(s["final_step"], on_disk["final_step"], 12)
        for key in ("model_config", "parameter_count", "tokenizer", "dataset", "training_config", "initial_step", "final_step",
                    "initial_train_loss", "final_train_loss_mean_last_10_steps", "final_validation", "tokens_processed_total",
                    "wall_clock_seconds_this_run", "tokens_per_second_train_average", "checkpoint", "git_commit", "environment"):
            self.assertIn(key, on_disk)
        self.assertEqual(on_disk["parameter_count"], e.model.count_parameters())
        self.assertLess(on_disk["final_validation"]["val_loss"], on_disk["initial_validation"]["val_loss"])
        self.assertEqual([h["step"] for h in on_disk["validation_history"]], [6, 12])


if __name__ == "__main__":
    unittest.main()
