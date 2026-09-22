"""Phase 3B, phase D: checkpoint contents, integrity, corruption, atomicity
(including a real hard kill), retention, stage promotion."""
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np

from tinymind.training import checkpoint as ck
from tinymind.training.data import DataSource

from tests.training._helpers import FakeClock, dataset, make_engine, model_config, tmpdir

REPO = Path(__file__).resolve().parents[2]


def trained(out=None, **over):
    out = out or tmpdir()
    over = {**dict(max_steps=20, checkpoint_interval=10), **over}
    e = make_engine(out, train_over=over)
    e.train()
    return out, e


class TestContentsAndIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out, cls.engine = trained()
        cls.dir = cls.out / "checkpoints" / "step-00000020"

    def test_manifest_has_the_required_concepts_and_real_hashes(self):
        m = json.loads((self.dir / "manifest.json").read_text())
        for key in ("format_version", "stage", "global_step", "epoch", "model_config_hash", "tokenizer_hash", "dataset_hash",
                    "training_config_hash", "model_parameter_count", "created_at", "git_commit", "files", "stage_complete", "parent"):
            self.assertIn(key, m)
        self.assertEqual((m["global_step"], m["stage"], m["stage_complete"]), (20, "stage0", True))
        self.assertEqual(m["model_parameter_count"], self.engine.model.count_parameters())
        self.assertEqual(sorted(m["files"]), ["model.npz", "optimizer.npz", "state.json"])
        for name, info in m["files"].items():
            self.assertEqual(info["sha256"], ck.sha256_file(self.dir / name))
            self.assertEqual(info["size"], (self.dir / name).stat().st_size)

    def test_state_covers_everything_needed_to_continue(self):
        s = json.loads((self.dir / "state.json").read_text())
        self.assertEqual(s["model_config"], self.engine.model_config.to_dict())
        self.assertEqual(s["tokenizer"]["type"], "byte")
        self.assertEqual(s["renderer"]["template"], "tinymind-chat-v1")
        self.assertEqual({"beta1", "beta2", "eps", "weight_decay", "decay_min_ndim"}, set(s["optimizer"]["hyper"]))
        self.assertEqual(s["optimizer"]["step_count"], 20)
        self.assertEqual(s["scheduler"]["last_step"], 20)
        self.assertEqual({"global_step", "epoch", "cursor", "cumulative_steps", "tokens_processed"} - set(s["progress"]), set())
        self.assertEqual(s["accumulation"]["pending_micro_batches"], 0)
        self.assertEqual(s["rng"]["bit_generator"], "PCG64")
        self.assertEqual(len(s["dataset"]["dataset_hash"]), 64)
        self.assertEqual(s["training_config"]["seed"], 3)
        self.assertEqual(s["architecture"]["id"], "tinymind-transformer-v1")
        self.assertIn("git_commit", s["environment"])
        self.assertTrue((self.dir / "model.npz").exists() and (self.dir / "optimizer.npz").exists())

    def test_files_are_numeric_and_pickle_free(self):
        with np.load(self.dir / "model.npz", allow_pickle=False) as z:
            self.assertTrue(all(z[k].dtype == np.float32 for k in z.files))
        with np.load(self.dir / "optimizer.npz", allow_pickle=False) as z:
            self.assertTrue(all(z[k].dtype == np.float32 for k in z.files))
            self.assertTrue(all(k.startswith(("m.", "v.")) for k in z.files))
        src = (REPO / "tinymind/training/checkpoint.py").read_text()
        self.assertNotIn("import pickle", src)
        self.assertNotIn("allow_pickle=True", src)

    def copy(self):
        import shutil
        dst = tmpdir() / "step-00000020"
        shutil.copytree(self.dir, dst)
        return dst

    def flip(self, path, where=0.5):
        blob = bytearray(path.read_bytes())
        blob[int(len(blob) * where)] ^= 0x01
        path.write_bytes(bytes(blob))

    def test_verify_accepts_the_intact_checkpoint(self):
        self.assertTrue(ck.verify_checkpoint(self.dir).ok)
        info = ck.checkpoint_summary(self.dir)
        self.assertTrue(info["valid"])
        self.assertEqual(info["progress"]["global_step"], 20)

    def test_single_bit_flip_in_each_file_is_detected(self):
        for name in ("model.npz", "optimizer.npz", "state.json", "manifest.json"):
            d = self.copy()
            self.flip(d / name)
            report = ck.verify_checkpoint(d)
            self.assertFalse(report.ok, name)
            with self.assertRaises(ck.CheckpointCorruptError):
                ck.load_checkpoint(d)

    def test_truncation_and_missing_pieces_are_detected(self):
        d = self.copy(); (d / "model.npz").write_bytes((d / "model.npz").read_bytes()[:100])
        self.assertIn("size", " ".join(ck.verify_checkpoint(d).errors))
        d = self.copy(); (d / "manifest.json").unlink()
        self.assertIn("manifest.json missing", ck.verify_checkpoint(d).errors[0])
        d = self.copy(); (d / "optimizer.npz").unlink()
        self.assertFalse(ck.verify_checkpoint(d).ok)

    def test_a_weights_only_directory_is_not_a_training_checkpoint(self):
        from tinymind.model.checkpoint import save_pretrained
        d = tmpdir() / "legacy"
        save_pretrained(self.engine.model, d)
        report = ck.verify_checkpoint(d)
        self.assertFalse(report.ok)
        self.assertIn("manifest.json missing", report.errors[0])

    def test_semantic_corruption_with_valid_hashes_is_still_caught(self):
        # An adversarial/buggy writer that recomputes the manifest: NaN weights, pickled object arrays.
        snap = self.engine._snapshot(True)
        snap.weights = {k: v.copy() for k, v in snap.weights.items()}
        next(iter(snap.weights.values()))[0, 0] = np.nan
        d = ck.save_checkpoint(tmpdir() / "nan", snap)
        rep = ck.verify_checkpoint(d)
        self.assertFalse(rep.ok)
        self.assertIn("NaN/Inf", " ".join(rep.errors))
        d2 = self.copy()
        np.savez(d2 / "model.npz", **{"embed_tokens": np.array([{"x": 1}], dtype=object)})
        m = json.loads((d2 / "manifest.json").read_text())
        m["files"]["model.npz"] = {"sha256": ck.sha256_file(d2 / "model.npz"), "size": (d2 / "model.npz").stat().st_size}
        (d2 / "manifest.json").write_text(json.dumps(m))
        rep2 = ck.verify_checkpoint(d2)
        self.assertFalse(rep2.ok)
        self.assertIn("cannot parse", rep2.errors[0])  # np.load(allow_pickle=False) refuses the object array

    def test_wrong_shape_is_caught_against_the_stored_config(self):
        d = self.copy()
        with np.load(d / "model.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["embed_tokens"] = arrays["embed_tokens"][:, :16]
        np.savez(d / "model.npz", **arrays)
        m = json.loads((d / "manifest.json").read_text())
        m["files"]["model.npz"] = {"sha256": ck.sha256_file(d / "model.npz"), "size": (d / "model.npz").stat().st_size}
        (d / "manifest.json").write_text(json.dumps(m))
        self.assertIn("shape", " ".join(ck.verify_checkpoint(d).errors))


class TestLatestPointerAndFallback(unittest.TestCase):
    def test_fallback_is_never_silent_and_never_a_fresh_start(self):
        out, _ = trained(max_steps=30, checkpoint_interval=10, keep_checkpoints=3)
        root = out / "checkpoints"
        self.assertEqual(json.loads((root / "latest.json").read_text())["checkpoint"], "step-00000030")
        self.assertEqual(json.loads((root / "previous.json").read_text())["checkpoint"], "step-00000020")
        (root / "step-00000030" / "model.npz").write_bytes(b"garbage")
        found, notes = ck.find_latest_valid(root)
        self.assertEqual(found.name, "step-00000020")
        self.assertTrue(any("step-00000030 rejected" in n for n in notes))
        e = make_engine(tmpdir(), train_over=dict(max_steps=30, checkpoint_interval=10), resume=root)
        self.assertEqual(e.step, 20)
        self.assertTrue(any("rejected" in n for n in e.load_notes))  # surfaced to the caller/log
        for d in root.glob("step-*"):
            (d / "state.json").write_text("{}")
        self.assertEqual(ck.find_latest_valid(root)[0], None)
        with self.assertRaises(ck.CheckpointCorruptError):
            ck.resolve_checkpoint(root)
        with self.assertRaises(ck.CheckpointCorruptError):
            make_engine(tmpdir(), train_over=dict(max_steps=30, checkpoint_interval=10), resume=root)

    def test_retention_keeps_the_newest_and_protects_the_pointers(self):
        out, _ = trained(max_steps=50, checkpoint_interval=10, keep_checkpoints=2)
        names = sorted(d.name for d in (out / "checkpoints").glob("step-*"))
        self.assertEqual(names, ["step-00000040", "step-00000050"])
        self.assertTrue(ck.verify_checkpoint(out / "checkpoints" / "step-00000040").ok)

    def test_saving_the_same_step_twice_is_idempotent_and_replaces_a_corrupt_copy(self):
        out, e = trained(max_steps=10, checkpoint_interval=0)
        root = out / "checkpoints"
        before = ck.sha256_file(root / "step-00000010" / "manifest.json")
        e.save_checkpoint(complete=True)
        self.assertEqual(before, ck.sha256_file(root / "step-00000010" / "manifest.json"))  # untouched
        (root / "step-00000010" / "model.npz").write_bytes(b"x")
        e.save_checkpoint(complete=True)
        self.assertTrue(ck.verify_checkpoint(root / "step-00000010").ok)
        self.assertTrue(any(p.name.startswith(".corrupt-") for p in root.iterdir()))


class TestAtomicity(unittest.TestCase):
    POINTS = ["after_model", "after_optimizer", "after_state", "after_manifest", "before_rename", "after_rename", "after_latest"]

    def test_failure_at_every_step_of_the_protocol_leaves_a_usable_latest_checkpoint(self):
        for point in self.POINTS:
            out = tmpdir()
            calls = {"saves": 0}

            def hook(p, point=point):
                if p == "after_model":
                    calls["saves"] += 1
                if calls["saves"] == 2 and p == point:
                    raise RuntimeError(f"injected failure at {p}")
            e = make_engine(out, train_over=dict(max_steps=40, checkpoint_interval=10), fault_hook=hook)
            with self.assertRaises(RuntimeError, msg=point):
                e.train()
            root = out / "checkpoints"
            found, notes = ck.find_latest_valid(root)
            self.assertIsNotNone(found, point)
            self.assertEqual(found.name, "step-00000020" if point == "after_latest" else "step-00000010", point)
            self.assertTrue(ck.verify_checkpoint(found).ok, point)
            self.assertEqual(list(root.glob(".tmp-*")), [], point)  # cleaned up on the way out
            resumed = make_engine(tmpdir(), train_over=dict(max_steps=40, checkpoint_interval=10), resume=root)
            self.assertEqual(resumed.step, 20 if point == "after_latest" else 10, point)
            resumed.train()
            self.assertEqual(resumed.step, 40)

    def test_interrupted_run_resumes_to_the_same_result_as_an_uninterrupted_one(self):
        continuous = make_engine(tmpdir(), train_over=dict(max_steps=40, checkpoint_interval=1000))
        continuous.train()
        out = tmpdir()
        state = {"n": 0}

        def hook(p):
            state["n"] += p == "after_model"
            if state["n"] == 3 and p == "before_rename":
                raise KeyboardInterrupt
        e = make_engine(out, train_over=dict(max_steps=40, checkpoint_interval=10), fault_hook=hook)
        with self.assertRaises(KeyboardInterrupt):
            e.train()
        resumed = make_engine(tmpdir(), train_over=dict(max_steps=40, checkpoint_interval=10), resume=out / "checkpoints")
        resumed.train()
        for (n, a), (_, b) in zip(continuous.model.named_parameters(), resumed.model.named_parameters()):
            self.assertTrue(np.array_equal(a.data, b.data), n)

    def test_real_process_kill_during_checkpoint_write(self):
        out = tmpdir()
        code = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {str(REPO)!r})
            from tests.training._helpers import make_engine
            from pathlib import Path
            n = {{"saves": 0}}
            def hook(p):
                n["saves"] += p == "after_model"
                if n["saves"] == 2 and p == "after_manifest":
                    os._exit(17)          # no cleanup, no finally blocks: like kill -9 / power loss
            make_engine(Path({str(out)!r}), train_over=dict(max_steps=40, checkpoint_interval=10), fault_hook=hook).train()
        """)
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 17, proc.stderr)
        root = out / "checkpoints"
        self.assertTrue(list(root.glob(".tmp-*")), "the dead writer's temp directory should still be there")
        found, _ = ck.find_latest_valid(root)
        self.assertEqual(found.name, "step-00000010")
        resumed = make_engine(tmpdir(), train_over=dict(max_steps=40, checkpoint_interval=10), resume=root)
        self.assertEqual(resumed.step, 10)
        resumed_out = out
        e2 = make_engine(resumed_out, train_over=dict(max_steps=40, checkpoint_interval=10), resume=root)
        e2.train()  # writing again sweeps the stale temp directory
        self.assertEqual(list(root.glob(".tmp-*")), [])
        cont = make_engine(tmpdir(), train_over=dict(max_steps=40, checkpoint_interval=1000))
        cont.train()
        for (n, a), (_, b) in zip(cont.model.named_parameters(), e2.model.named_parameters()):
            self.assertTrue(np.array_equal(a.data, b.data), n)


class TestStagePromotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out, cls.parent = trained(max_steps=20, checkpoint_interval=0)
        cls.parent_root = cls.out / "checkpoints"

    def stage1(self, **kw):
        over = dict(stage="stage1", max_steps=15, learning_rate=1e-3, min_learning_rate=1e-4, checkpoint_interval=0)
        return make_engine(tmpdir(), train_over=over, train_data=dataset(48, offset=200, prefix="s1"), **kw)

    def test_init_from_loads_weights_records_lineage_and_starts_a_fresh_optimizer(self):
        e = self.stage1(init_from=self.parent_root)
        for (n, a), (_, b) in zip(self.parent.model.named_parameters(), e.model.named_parameters()):
            self.assertTrue(np.array_equal(a.data, b.data), n)
        self.assertEqual((e.step, e.optimizer.step_count, e.cumulative_steps), (0, 0, 20))
        m = ck.load_checkpoint(self.parent_root / "step-00000020").manifest
        self.assertEqual(e.parent["manifest_sha256"], ck.sha256_file(self.parent_root / "step-00000020" / "manifest.json"))
        self.assertEqual((e.parent["stage"], e.parent["global_step"]), ("stage0", 20))
        s = e.train()
        self.assertEqual((s["cumulative_steps"], s["parent"]["stage"], s["stage_complete"]), (35, "stage0", True))
        self.assertEqual(json.loads((e.last_checkpoint / "manifest.json").read_text())["parent"]["checkpoint"], "step-00000020")

    def test_optional_optimizer_carry_over(self):
        e = self.stage1(init_from=self.parent_root, carry_optimizer=True)
        self.assertEqual(e.optimizer.step_count, 20)
        self.assertTrue(any(np.abs(m).sum() > 0 for m in e.optimizer._m))

    def test_promotion_is_refused_for_architecture_or_tokenizer_mismatch(self):
        with self.assertRaises(ck.ResumeMismatchError):
            make_engine(tmpdir(), model_over=dict(num_layers=3), train_over=dict(stage="stage1", max_steps=5), init_from=self.parent_root)
        with self.assertRaises(ck.CheckpointCorruptError):
            make_engine(tmpdir(), train_over=dict(stage="stage1", max_steps=5), init_from=self.parent_root / "nope")

    def test_stage_can_itself_be_interrupted_and_resumed(self):
        out = tmpdir()
        over = dict(stage="stage1", max_steps=30, learning_rate=1e-3, min_learning_rate=1e-4, checkpoint_interval=1000,
                    max_runtime_seconds=12.5, safety_margin_seconds=0)
        data = dataset(48, offset=200, prefix="s1")
        first = make_engine(out, train_over=over, train_data=data, init_from=self.parent_root, clock=FakeClock())
        first.train()
        self.assertEqual(first.step, 12)
        over2 = dict(over, max_runtime_seconds=0.0)
        second = make_engine(out, train_over=over2, train_data=data, resume=out / "checkpoints")
        self.assertEqual((second.step, second.parent["stage"]), (12, "stage0"))
        second.train()
        cont = make_engine(tmpdir(), train_over=dict(over, max_runtime_seconds=0.0), train_data=data, init_from=self.parent_root)
        cont.train()
        for (n, a), (_, b) in zip(cont.model.named_parameters(), second.model.named_parameters()):
            self.assertTrue(np.array_equal(a.data, b.data), n)


if __name__ == "__main__":
    unittest.main()
