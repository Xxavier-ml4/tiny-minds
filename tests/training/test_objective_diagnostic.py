"""The decoding-mitigated generation diagnostic.

The stage objective judges the model's RAW greedy output. A repetition penalty or n-gram ban hides looping instead
of curing it, so letting one into a gate would let a model that loops pass its own looping check. These tests pin
the design: the diagnostic is reported next to the raw numbers, answers "decoding artifact or broken model?", and
can never change a verdict.
"""
import json
import unittest
from pathlib import Path

import numpy as np

import tinymind.training.objective as O
from tinymind.model import ByteTokenizer, ModelConfig, TinyMindTransformer
from tinymind.training.objective import StageObjective, StageObjectiveError, render_markdown

REPO = Path(__file__).resolve().parents[2]
LOOP = " the town of the town of the town of the town of the town of the town of"
FINE = " A clear and varied sentence about ships, harbours and grain."
DIAG = {"repetition_penalty": 1.15, "no_repeat_ngram_size": 3}


def row(text):
    return {"prompt": "p", "generation": text, "tokens_generated": 20, "finish_reason": "max_new_tokens"}


def objective(diagnostic=None, gated=None, **gen):
    generation = {"prompts": ["a"], **gen}
    if diagnostic is not None:
        generation["diagnostic_decoding"] = diagnostic
    if gated is not None:
        generation["gated_decoding"] = gated
    return StageObjective({"stage": "s1", "generation": generation,
                           "measurements": [{"name": "loss", "metric": "loss.val_loss", "max": 3.0},
                                            {"name": "loops", "metric": "generation.mean_repetition", "max": 0.3}]})


class FakeGeneration:
    """A model that loops under plain greedy decoding but is fine once repetition is mitigated, recording how
    ``generate_continuations`` was called."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self.saved = O.generate_continuations

        def fake(model, tokenizer, prompts, **kw):
            self.calls.append(kw)
            mitigated = (kw.get("no_repeat_ngram_size", 0) > 0 or kw.get("repetition_penalty", 1.0) != 1.0
                         or kw.get("temperature", 0.0) > 0)
            return [row(FINE if mitigated else LOOP)]
        O.generate_continuations = fake
        return self

    def __exit__(self, *exc):
        O.generate_continuations = self.saved


def evaluate(obj, loss=1.0):
    return obj.evaluate_checkpoint(model=None, tokenizer=None, held_out={"val_loss": loss}, step=1, tokens=100,
                                   min_tokens=100)


class TestDiagnosticNeverGates(unittest.TestCase):
    def test_a_looping_model_still_fails_even_though_the_mitigated_decode_is_healthy(self):
        with FakeGeneration():
            report = evaluate(objective(DIAG))
        self.assertFalse(report["objective_met"])  # the raw greedy loop is what is judged
        self.assertTrue(report["verdict"]["reasons"])
        self.assertLess(report["metrics"]["generation_decoded"]["mean_repetition"], 0.1)
        self.assertGreater(report["metrics"]["generation"]["mean_repetition"], 0.5)

    def test_verdict_and_checks_are_identical_with_and_without_the_diagnostic(self):
        with FakeGeneration():
            with_diag, without = evaluate(objective(DIAG)), evaluate(objective())
        for key in ("objective_met", "verdict", "regression", "retained"):
            self.assertEqual(with_diag[key], without[key], key)
        self.assertEqual(with_diag["metrics"]["generation"], without["metrics"]["generation"])

    def test_the_gated_generation_always_uses_plain_greedy_decoding(self):
        with FakeGeneration() as fake:
            evaluate(objective(DIAG))
        raw, mitigated = fake.calls
        self.assertNotIn("repetition_penalty", raw)
        self.assertNotIn("no_repeat_ngram_size", raw)
        self.assertEqual(mitigated["repetition_penalty"], 1.15)
        self.assertEqual(mitigated["no_repeat_ngram_size"], 3)

    def test_no_diagnostic_means_no_second_generation_pass_and_no_extra_keys(self):
        with FakeGeneration() as fake:
            report = evaluate(objective())
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("generation_decoded", report["metrics"])
        self.assertNotIn("generations_decoded", report)

    def test_a_measurement_cannot_reference_the_diagnostic(self):
        with self.assertRaises(StageObjectiveError):
            StageObjective({"stage": "s1", "generation": {"prompts": ["a"], "diagnostic_decoding": DIAG},
                            "measurements": [{"name": "cheat", "metric": "generation_decoded.mean_repetition",
                                              "max": 0.3}]})


class TestDiagnosticConfigValidation(unittest.TestCase):
    def test_bad_configs_are_rejected(self):
        for bad in ({}, {"top_p": 0.9}, {"repetition_penalty": 0.0}, {"no_repeat_ngram_size": -1},
                    {"repetition_penalty": 1.0}, {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0}, "yes"):
            with self.assertRaises(StageObjectiveError, msg=str(bad)):
                objective(bad)

    def test_either_control_alone_is_enough(self):
        self.assertEqual(objective({"no_repeat_ngram_size": 3}).diagnostic_decoding,
                         {"repetition_penalty": 1.0, "no_repeat_ngram_size": 3})
        self.assertEqual(objective({"repetition_penalty": 1.2}).diagnostic_decoding,
                         {"repetition_penalty": 1.2, "no_repeat_ngram_size": 0})


SAMPLING = {"temperature": 0.7, "top_p": 0.9, "seed": 0}


class TestSamplingDecodingConfig(unittest.TestCase):
    def test_sampling_is_normalised_with_a_default_seed(self):
        self.assertEqual(objective({"temperature": 0.7, "top_p": 0.9}).diagnostic_decoding,
                         {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "temperature": 0.7, "top_p": 0.9,
                          "seed": 0})
        self.assertEqual(objective(gated={"temperature": 0.5, "top_k": 40, "seed": 7}).gated_decoding,
                         {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "temperature": 0.5, "top_k": 40,
                          "seed": 7})

    def test_bad_sampling_configs_are_rejected_for_both_blocks(self):
        for bad in ({"top_p": 0.9}, {"top_k": 5}, {"seed": 1}, {"temperature": -1}, {"temperature": 0.7, "top_p": 0},
                    {"temperature": 0.7, "top_p": 1.5}, {"temperature": 0.7, "top_k": 0},
                    {"temperature": 0.7, "seed": -1}, {"temperature": "hot"}):
            with self.assertRaises(StageObjectiveError, msg=str(bad)):
                objective(bad)
            with self.assertRaises(StageObjectiveError, msg=str(bad)):
                objective(gated=bad)

    def test_the_shipped_gate_default_is_unchanged_plain_greedy(self):
        self.assertIsNone(objective().gated_decoding)


class TestGatedDecoding(unittest.TestCase):
    """``generation.gated_decoding`` is an explicit opt-in that changes what the gate means, so it must be applied to
    the gated measurements, be visible in the report, and stay out of the default path."""

    def test_the_gate_uses_the_configured_decoding_and_the_report_says_so(self):
        with FakeGeneration() as fake:
            report = evaluate(objective(gated=SAMPLING))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual((fake.calls[0]["temperature"], fake.calls[0]["top_p"], fake.calls[0]["seed"]), (0.7, 0.9, 0))
        self.assertTrue(report["objective_met"])  # this fake is healthy under sampling, so the gate now passes
        self.assertEqual(report["gated_decoding"]["temperature"], 0.7)
        md = render_markdown(report)
        self.assertIn("fixed prompts, seeded sampling, temperature 0.7, top-p 0.9, seed 0", md)

    def test_without_it_the_same_model_fails_the_gate_and_the_report_says_greedy(self):
        with FakeGeneration():
            report = evaluate(objective())
        self.assertFalse(report["objective_met"])
        self.assertNotIn("gated_decoding", report)
        self.assertIn("fixed prompts, greedy decoding", render_markdown(report))

    def test_a_diagnostic_can_sample_and_is_described_in_the_report(self):
        with FakeGeneration() as fake:
            report = evaluate(objective({"temperature": 0.7, "top_p": 0.9, "no_repeat_ngram_size": 3}))
        raw, sampled = fake.calls
        self.assertNotIn("temperature", raw)
        self.assertEqual((sampled["temperature"], sampled["top_p"], sampled["no_repeat_ngram_size"]), (0.7, 0.9, 3))
        self.assertFalse(report["objective_met"])  # the gate is still greedy; the diagnostic cannot rescue it
        self.assertIn("seeded sampling, temperature 0.7, top-p 0.9, seed 0, no-repeat 3-gram", render_markdown(report))


class TestDiagnosticReporting(unittest.TestCase):
    def test_markdown_shows_the_diagnostic_beside_the_raw_numbers(self):
        with FakeGeneration():
            md = render_markdown(evaluate(objective(DIAG)))
        self.assertIn("NOT gated", md)
        self.assertIn("decoding-mitigated", md)
        self.assertIn("raw greedy (gated)", md)
        self.assertIn("<details>", md)
        self.assertIn(FINE.strip(), md)  # the mitigated generations are shown verbatim too

    def test_markdown_is_unchanged_without_the_diagnostic(self):
        with FakeGeneration():
            md = render_markdown(evaluate(objective()))
        self.assertNotIn("NOT gated", md)
        self.assertNotIn("<details>", md)


class TestRepositoryObjective(unittest.TestCase):
    def test_stage1_ships_the_diagnostic_and_gates_only_on_raw_generation(self):
        cfg = json.loads((REPO / "configs/stages_v2/stage1.objective.json").read_text())
        obj = StageObjective(cfg)
        self.assertEqual(obj.diagnostic_decoding, {"repetition_penalty": 1.0, "no_repeat_ngram_size": 3,
                                                    "temperature": 0.7, "top_p": 0.9, "seed": 0})
        self.assertIsNone(obj.gated_decoding)  # the shipped gate is still plain greedy
        for m in cfg["measurements"]:
            self.assertNotIn("generation_decoded", m["metric"], m["name"])
        looping_gates = [m for m in cfg["measurements"]
                         if m["name"] in ("generation_not_looping", "generation_not_repetitive")]
        self.assertEqual(len(looping_gates), 2)  # the anti-looping gates are still there, untouched


class TestGenerateContinuationsPassThrough(unittest.TestCase):
    """The real function (not the fake): the new arguments reach the generator and the default is unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.model = TinyMindTransformer(ModelConfig(hidden_size=16, num_layers=1, num_heads=4, num_kv_heads=2,
                                                    intermediate_size=32, max_seq_len=96, vocab_size=cls.tok.vocab_size),
                                        seed=0)

    def gen(self, **kw):
        return [r["generation"] for r in O.generate_continuations(self.model, self.tok, ["hi"], max_new_tokens=30, **kw)]

    def test_defaults_equal_explicit_off(self):
        self.assertEqual(self.gen(), self.gen(repetition_penalty=1.0, no_repeat_ngram_size=0))

    def test_the_ban_changes_a_looping_model_s_output(self):
        self.assertNotEqual(self.gen(), self.gen(no_repeat_ngram_size=2))

    def test_temperature_zero_is_still_exactly_greedy(self):
        self.assertEqual(self.gen(), self.gen(temperature=0.0, top_p=None, seed=5))

    def test_sampling_is_reproducible_per_seed_and_varies_across_seeds(self):
        a = self.gen(temperature=0.9, top_p=0.95, seed=1)
        self.assertEqual(a, self.gen(temperature=0.9, top_p=0.95, seed=1))
        self.assertNotEqual(a, self.gen(temperature=0.9, top_p=0.95, seed=2))
        self.assertNotEqual(a, self.gen())  # sampling is not greedy

    def test_each_prompt_gets_its_own_seed(self):
        rows = O.generate_continuations(self.model, self.tok, ["hi", "hi"], max_new_tokens=30, temperature=1.0, seed=3)
        self.assertNotEqual(rows[0]["generation"], rows[1]["generation"])  # identical prompts, seeds 3 and 4


if __name__ == "__main__":
    unittest.main()
