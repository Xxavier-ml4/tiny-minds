import unittest

from tinymind.data import curriculum_v2 as C2
from tinymind.data.contamination import check_overlap
from tinymind.evaluation.scoring import score


class TestCurriculumV2Structure(unittest.TestCase):
    """Brief sections 3-5: seven stages, configurable replay, per-stage token
    budgets, all summing to the planned total."""

    def test_seven_stages(self):
        self.assertEqual(C2.STAGES, ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"))

    def test_every_mixture_sums_to_one(self):
        for stage in C2.STAGES:
            mix = C2.effective_mixture(stage)
            self.assertAlmostEqual(sum(mix.values()), 1.0, places=5, msg=f"{stage} mixture must sum to 1.0")

    def test_replay_fraction_is_respected(self):
        man = C2.load_manifest()
        for stage in C2.STAGES:
            spec = man["stages"][stage]
            replay = spec.get("replay", {})
            if not replay:
                continue
            mix = C2.effective_mixture(stage, man)
            primary = set(spec.get("primary", {}))
            replay_only = {k: v for k, v in mix.items() if k in replay and k not in primary}
            got = sum(replay_only.values())
            self.assertAlmostEqual(got, spec["replay_fraction"], places=5,
                                   msg=f"{stage} replay share should equal replay_fraction")

    def test_total_token_budget(self):
        man = C2.load_manifest()
        total = sum(man["stages"][s]["target_tokens"] for s in C2.STAGES)
        self.assertEqual(total, man["total_target_tokens"])
        self.assertEqual(total, 345_000_000)


class TestCurriculumV2Data(unittest.TestCase):
    """Brief sections 4, 11: built stages have disjoint train/val/test, verified
    math, and pass the contamination checker."""

    def test_splits_are_disjoint(self):
        for stage in ("stage1", "stage2", "stage5"):
            data = C2.build_stage(stage, seed=0, scale=0.08)
            train_keys = {C2.prompt_key(r) for rs in data["train"].values() for r in rs}
            val_keys = {C2.prompt_key(r) for r in data["val"]}
            test_keys = {C2.prompt_key(r) for r in data["test"]}
            self.assertEqual(val_keys & train_keys, set(), f"{stage}: val overlaps train")
            self.assertEqual(test_keys & train_keys, set(), f"{stage}: held-out test overlaps train")

    def test_all_math_examples_are_verified(self):
        data = C2.build_stage("stage2", seed=1, scale=0.2)
        train_all = [r for rs in data["train"].values() for r in rs]
        math_recs = [r for r in train_all if r["meta"].get("score") == "final_answer"]
        self.assertGreater(len(math_recs), 0)
        for r in math_recs:
            self.assertTrue(C2.verify_record(r), f"unverifiable: {r['id']}")
            # the gold answer must also satisfy the scorer
            self.assertTrue(score(r["messages"][-1]["content"], r["meta"])["ok"])

    def test_no_contamination(self):
        data = C2.build_stage("stage3", seed=0, scale=0.08)
        train_all = [r for rs in data["train"].values() for r in rs]
        report = check_overlap(train_all, data["test"])
        self.assertFalse(report.contaminated)

    def test_categories_are_eval_categories(self):
        data = C2.build_stage("stage1", seed=0, scale=0.08)
        cats = {r["category"] for rs in data["train"].values() for r in rs}
        self.assertTrue(cats.issubset(set(C2.EVAL_CATEGORIES)), cats)


if __name__ == "__main__":
    unittest.main()
