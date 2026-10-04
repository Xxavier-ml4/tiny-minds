"""Stage objective gate: a stage completes because it MET ITS OBJECTIVE, not
because a token budget ran out.

The pre-existing :mod:`tinymind.training.gate` applies *promotion* thresholds
(capability accuracies of the exported package on the held-out test split)
when the next stage starts. This module is the complementary, trainer-side
piece: an **objective the engine evaluates on the live model at every
evaluation checkpoint**, built from several *independent* measurements, so
that

* reaching the token budget with a failing objective leaves the stage
  INCOMPLETE — the budget is a *minimum training chunk*, never a promotion
  criterion on its own;
* every evaluation checkpoint produces a report with the exact fixed prompts
  and the model's raw generations (never replaced by a single score);
* "no promotion on validation loss alone" is structural: the objective is met
  only when the minimum budget is reached AND every configured measurement
  passes AND the retained capabilities of earlier stages still pass AND
  nothing regressed against the previous checkpoint.

Measurements (all on the live model; no package export needed):

``loss``
    held-out LM loss/perplexity on the stage's validation split and, when the
    stage has one, on **natural held-out text** (``text_val_*``), including
    **bits per byte** — tokenizer-independent, so one threshold means the same
    thing for a byte-level 1M model and a 16k-BPE 1B model.
``generation``
    greedy continuations of fixed prompts: non-empty rate, repetition/looping,
    distinct-n diversity, length, EOS rate.
``grammar``
    automatic checks that are reliable on short greedy output: word-like
    tokens, letter share, sentence starts capitalised after a terminator,
    spacing after punctuation, word/sentence length.
``data``
    facts about the training data supplied by the caller (for example how many
    bytes of natural text the stage trains on), so an objective can refuse to
    call language "acquired" from a placeholder corpus.

Thresholds live in ``configs/stages_v2/<stage>.objective.json`` (or any
directory given with ``--objective-dir`` / a profile's ``objective_dir``), not in
code. The mechanism is model-size independent: 50M, 100M, 300M, 500M and 1B
runs use this same code and differ only in thresholds, prompts and data.

``retain`` lists earlier stages whose generation/grammar checks are re-measured
on the live model with *their* prompts and thresholds, so a later stage cannot
pass while it has lost an earlier capability; :func:`compare_reports` also
flags regression against the previous checkpoint and against the level each
retained stage had when it was promoted.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tinymind.evaluation.scoring import repetition

DEFAULT_OBJECTIVE_DIR = Path(__file__).resolve().parents[2] / "configs" / "stages_v2"
OBJECTIVE_FORMAT_VERSION = 1
REPORT_KIND = "tinymind-stage-objective-report"
_SECTIONS = ("loss", "generation", "grammar", "data")
_CAPABILITY_SECTIONS = ("generation", "grammar")
_LOSS_KEYS = ("val_loss", "val_ppl", "val_tokens", "text_val_loss", "text_val_ppl", "text_val_bpb",
              "text_val_tokens", "text_val_bytes")


class StageObjectiveError(ValueError):
    """An objective config is missing, malformed, or refers to something that does not exist."""


# --------------------------------------------------------------------------- generation
def generate_continuations(model: Any, tokenizer: Any, prompts: Sequence[str], *, max_new_tokens: int = 48,
                           repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
                           temperature: float = 0.0, top_k: int | None = None, top_p: float | None = None,
                           seed: int | None = None) -> list[dict[str, Any]]:
    """Greedy (deterministic) continuation of each prompt on the live model; with ``temperature > 0`` a SEEDED
    sample instead (prompt ``i`` uses ``seed + i``, default seed 0, so a measurement is still reproducible).
    Returns, per prompt, the exact prompt, the raw generated text, the number of
    tokens generated and whether generation stopped on EOS or ran out of room —
    kept verbatim for the report.

    Every decoding argument defaults to "off", and a stage's gated measurements use that default (plain greedy)
    unless its objective explicitly sets ``generation.gated_decoding`` (the report then says so in its
    headings). A repetition penalty or n-gram ban hides a model's looping rather than curing it, so they are
    best kept to the non-gating ``generation_decoded`` diagnostic
    (see :meth:`StageObjective.evaluate_checkpoint`)."""
    from tinymind.model.generation import ModelGenerationConfig, generate_with_cache_ids

    rows: list[dict[str, Any]] = []
    eos = tokenizer.eos_token_id
    for i, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_bos=True)
        sampling = ({"do_sample": True, "temperature": temperature, "top_k": top_k, "top_p": top_p,
                     "seed": (0 if seed is None else seed) + i} if temperature > 0 else {"do_sample": False})
        cfg = ModelGenerationConfig(max_new_tokens=max_new_tokens, eos_token_id=eos,
                                    repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
                                    **sampling)
        out = generate_with_cache_ids(model, np.array([ids]), cfg)
        new_ids = [int(t) for t in out[0][len(ids):]]
        stopped = bool(new_ids) and new_ids[-1] == eos
        body = new_ids[:-1] if stopped else new_ids
        rows.append({"prompt": prompt, "generation": tokenizer.decode(body), "tokens_generated": len(new_ids),
                     "finish_reason": "eos" if stopped else "max_new_tokens"})
    return rows


# --------------------------------------------------------------------------- metrics
def _mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _distinct_n(tokens: Sequence[str], n: int) -> float:
    if len(tokens) < n:
        return 1.0 if tokens else 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def generation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Repetition/diversity/length statistics over the raw generations. Each is
    its own number; no composite is formed. Statistics of text quality are taken
    over the non-empty generations (an empty output would otherwise look
    'non-repetitive'); ``non_empty_rate`` reports how many that is."""
    texts = [str(r["generation"]) for r in rows]
    non_empty = [t for t in texts if t.strip()]
    reps = [repetition(t) for t in non_empty]
    return {
        "n": len(rows),
        "non_empty_rate": (len(non_empty) / len(rows)) if rows else 0.0,
        "mean_repetition": _mean(reps),
        "looping_rate": (sum(r > 0.5 for r in reps) / len(reps)) if reps else None,
        "mean_distinct_1": _mean([_distinct_n(t.split(), 1) for t in non_empty]),
        "mean_distinct_2": _mean([_distinct_n(t.split(), 2) for t in non_empty]),
        "mean_chars": _mean([float(len(t.strip())) for t in texts]) if texts else None,
        "mean_tokens_generated": _mean([float(r["tokens_generated"]) for r in rows]),
        "eos_rate": (sum(r["finish_reason"] == "eos" for r in rows) / len(rows)) if rows else None,
    }


_WORD_STRIP = ".,!?;:\"'()[]{}“”‘’…-"
_TERMINATOR_END = re.compile(r"[.!?][\"'”’)\]]*$")
_SENTENCE_START = re.compile(r"[.!?][\"'”’)\]]*\s+[\"'“‘(\[]*([^\W\d_])")
_PUNCT_FOLLOW = re.compile(r"([,;:.!?])(.)", re.S)


def _is_word(token: str) -> bool:
    core = token.strip(_WORD_STRIP)
    return bool(core) and core.replace("'", "").replace("’", "").replace("-", "").isalpha()


def normalize_word(token: str) -> str:
    """The lexicon form of a whitespace token: stripped of surrounding
    punctuation, lower-cased, curly apostrophes made straight."""
    return token.strip(_WORD_STRIP).replace("’", "'").lower()


def grammar_metrics(rows: Sequence[Mapping[str, Any]], lexicon: "set[str] | frozenset[str] | None" = None
                    ) -> dict[str, Any]:
    """Grammar-oriented checks that are reliable on short greedy output. They
    are floors on *language-likeness*, not a grammar judge:

    ``word_like_fraction``         whitespace tokens that are words (letters, inner ' or -), not byte noise
    ``known_word_fraction``        whitespace tokens that are words found in the training corpus lexicon
                                   (catches invented words such as "oollaye"; None without a lexicon)
    ``alpha_fraction``             non-space characters that are letters or ordinary punctuation
    ``sentence_start_capitalized`` sentences that follow a terminator and start with a capital
                                   (pooled over all generations; None when no sentence boundary occurs)
    ``space_after_punctuation``    , ; : . ! ? followed by a space, a closing quote/bracket, more
                                   punctuation, or a digit after a digit (decimals); None when no case occurs
    ``terminator_fraction``        generations that end on . ! or ? (informational: output is length-capped)
    ``mean_word_length`` / ``mean_sentence_words``  plausibility of word and sentence sizes
    """
    texts = [str(r["generation"]).strip() for r in rows if str(r["generation"]).strip()]
    if not texts:
        return {"n": 0, "word_like_fraction": None, "known_word_fraction": None, "alpha_fraction": None,
                "sentence_start_capitalized": None, "space_after_punctuation": None, "terminator_fraction": None,
                "mean_word_length": None, "mean_sentence_words": None}
    word_like, known, alpha, word_lengths, sentence_words = [], [], [], [], []
    starts_upper = starts_total = punct_ok = punct_total = terminated = 0
    for t in texts:
        tokens = t.split()
        words = [w for w in tokens if _is_word(w)]
        word_like.append(len(words) / len(tokens) if tokens else 0.0)
        if lexicon is not None and tokens:
            known.append(sum(normalize_word(w) in lexicon for w in words) / len(tokens))
        chars = [c for c in t if not c.isspace()]
        alpha.append(sum(c.isalpha() or c in ".,!?;:'\"-()’“”" for c in chars) / len(chars) if chars else 0.0)
        word_lengths += [float(len(w.strip(_WORD_STRIP))) for w in words]
        for m in _SENTENCE_START.finditer(t):
            starts_total += 1
            starts_upper += m.group(1).isupper()
        for m in _PUNCT_FOLLOW.finditer(t):
            mark, nxt = m.group(1), m.group(2)
            prev = t[m.start() - 1] if m.start() > 0 else ""
            if mark in ".," and prev.isdigit() and nxt.isdigit():
                continue  # 3.5 / 1,000: not a spacing case
            punct_total += 1
            punct_ok += nxt.isspace() or nxt in "\"')]}”’.,!?;:-"
        terminated += bool(_TERMINATOR_END.search(t))
        pieces = [p for p in re.split(r"[.!?]+", t) if p.strip()]
        sentence_words += [float(len(p.split())) for p in pieces]
    return {
        "n": len(texts),
        "word_like_fraction": _mean(word_like),
        "known_word_fraction": _mean(known) if lexicon is not None else None,
        "alpha_fraction": _mean(alpha),
        "sentence_start_capitalized": (starts_upper / starts_total) if starts_total else None,
        "space_after_punctuation": (punct_ok / punct_total) if punct_total else None,
        "terminator_fraction": terminated / len(texts),
        "mean_word_length": _mean(word_lengths),
        "mean_sentence_words": _mean(sentence_words),
    }


# --------------------------------------------------------------------------- checks
def _dig(root: Mapping[str, Any] | None, dotted: str) -> Any:
    cur: Any = root
    for key in dotted.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _apply(spec: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    value = _dig(metrics, spec["metric"])
    ok = _is_number(value)
    parts: list[str] = []
    if "min" in spec:
        parts.append(f">= {spec['min']}")
        ok = ok and value >= spec["min"]
    if "max" in spec:
        parts.append(f"<= {spec['max']}")
        ok = ok and value <= spec["max"]
    check = {"name": spec.get("name") or spec["metric"], "metric": spec["metric"], "value": value,
             "requirement": " and ".join(parts), "ok": bool(ok)}
    if value is None:
        check["note"] = "not measured (counts as not passing)"
    return check


def _section(metric: str) -> str:
    return metric.split(".", 1)[0]


# --------------------------------------------------------------------------- the objective
class StageObjective:
    """A stage's objective, normally loaded from ``<dir>/<stage>.objective.json``.

    Config keys: ``stage``; ``generation`` {``prompts``, ``max_new_tokens``};
    ``measurements`` [{``name``, ``metric`` (section.key), ``min`` and/or ``max``}];
    ``regression`` {``max_val_loss_increase``, ``checks`` [{``metric``,
    ``max_increase`` | ``max_decrease``}]}; ``retain`` [stage name | inline config].
    """

    def __init__(self, config: Mapping[str, Any], *, source_dir: str | Path | None = None,
                 resolve_retain: bool = True) -> None:
        if not isinstance(config, Mapping):
            raise StageObjectiveError("an objective config must be a JSON object")
        self.config = dict(config)
        self.stage = str(config.get("stage") or "")
        if not self.stage:
            raise StageObjectiveError("an objective config needs a 'stage'")
        gen = config.get("generation") or {}
        self.prompts = [str(p) for p in gen.get("prompts", [])]
        self.max_new_tokens = int(gen.get("max_new_tokens", 48))
        if self.max_new_tokens < 1:
            raise StageObjectiveError(f"objective {self.stage!r}: generation.max_new_tokens must be >= 1")
        self.diagnostic_decoding = self._parse_decoding(gen.get("diagnostic_decoding"), "diagnostic_decoding")
        self.gated_decoding = self._parse_decoding(gen.get("gated_decoding"), "gated_decoding")
        self.measurements = [dict(m) for m in config.get("measurements", [])]
        if not self.measurements:
            raise StageObjectiveError(f"objective {self.stage!r} has no 'measurements'")
        for spec in self.measurements:
            metric = spec.get("metric")
            if not isinstance(metric, str) or _section(metric) not in _SECTIONS:
                raise StageObjectiveError(f"objective {self.stage!r}: measurement metric {metric!r} must start with "
                                          f"one of {list(_SECTIONS)}")
            if "min" not in spec and "max" not in spec:
                raise StageObjectiveError(f"objective {self.stage!r}: measurement {metric!r} needs a 'min' or 'max'")
            for bound in ("min", "max"):
                if bound in spec and not _is_number(spec[bound]):
                    raise StageObjectiveError(f"objective {self.stage!r}: {metric!r} {bound} must be a number")
        if not self.prompts and any(_section(m["metric"]) in _CAPABILITY_SECTIONS for m in self.measurements):
            raise StageObjectiveError(f"objective {self.stage!r}: generation/grammar measurements need "
                                      "generation.prompts")
        self.regression = dict(config.get("regression") or {})
        self.source_dir = Path(source_dir) if source_dir else None
        self.retained: list[StageObjective] = []
        if resolve_retain:
            for item in config.get("retain", []) or []:
                self.retained.append(self._resolve_retained(item))

    def _parse_decoding(self, spec: Any, key: str) -> dict[str, Any] | None:
        """``generation.<key>`` (``diagnostic_decoding`` or ``gated_decoding``), optional: any of
        ``repetition_penalty`` (> 0), ``no_repeat_ngram_size`` (>= 0), ``temperature`` (>= 0; > 0 means seeded
        sampling) and, only with a temperature, ``top_k`` (>= 1), ``top_p`` (in (0, 1]) and ``seed`` (>= 0, default 0).
        Returned normalised: the first two always, the sampling keys only when sampling.

        ``diagnostic_decoding``: every checkpoint also generates from the same prompts this way and reports it under
        ``metrics.generation_decoded`` — informational only; no measurement may reference it.
        ``gated_decoding``: the decoding the GATED measurements use (default: plain greedy). Opting in changes what the
        gate means, so the report names it in its headings."""
        if spec is None:
            return None
        where = f"objective {self.stage!r}: generation.{key}"
        allowed = {"repetition_penalty", "no_repeat_ngram_size", "temperature", "top_k", "top_p", "seed"}
        if not isinstance(spec, Mapping) or not spec or set(spec) - allowed:
            raise StageObjectiveError(f"{where} must be an object with some of {sorted(allowed)}")
        try:
            penalty = float(spec.get("repetition_penalty", 1.0))
            ngram = int(spec.get("no_repeat_ngram_size", 0))
            temperature = float(spec.get("temperature", 0.0))
            top_k = None if spec.get("top_k") is None else int(spec["top_k"])
            top_p = None if spec.get("top_p") is None else float(spec["top_p"])
            seed = None if spec.get("seed") is None else int(spec["seed"])
        except (TypeError, ValueError) as exc:
            raise StageObjectiveError(f"{where}: {exc}") from None
        if penalty <= 0 or ngram < 0 or temperature < 0:
            raise StageObjectiveError(f"{where} needs repetition_penalty > 0, no_repeat_ngram_size >= 0 and temperature >= 0")
        if temperature == 0 and (top_k is not None or top_p is not None or seed is not None):
            raise StageObjectiveError(f"{where}: top_k / top_p / seed only apply when temperature > 0 (sampling)")
        if top_k is not None and top_k < 1 or top_p is not None and not 0 < top_p <= 1 or seed is not None and seed < 0:
            raise StageObjectiveError(f"{where} needs top_k >= 1, 0 < top_p <= 1 and seed >= 0")
        if penalty == 1.0 and ngram == 0 and temperature == 0:
            raise StageObjectiveError(f"{where} changes nothing (penalty 1.0, no n-gram ban, no sampling); remove it "
                                      "or set a real value")
        out: dict[str, Any] = {"repetition_penalty": penalty, "no_repeat_ngram_size": ngram}
        if temperature > 0:
            out["temperature"] = temperature
            if top_k is not None:
                out["top_k"] = top_k
            if top_p is not None:
                out["top_p"] = top_p
            out["seed"] = 0 if seed is None else seed
        return out

    def _resolve_retained(self, item: Any) -> "StageObjective":
        if isinstance(item, Mapping):
            obj = StageObjective(item, source_dir=self.source_dir, resolve_retain=False)
        elif isinstance(item, str):
            directory = self.source_dir or DEFAULT_OBJECTIVE_DIR
            path = directory / f"{item}.objective.json"
            if not path.is_file():
                raise StageObjectiveError(f"objective {self.stage!r} retains {item!r}, but {path} does not exist")
            obj = StageObjective(_load_json(path), source_dir=directory, resolve_retain=False)
        else:
            raise StageObjectiveError(f"objective {self.stage!r}: 'retain' entries are stage names or configs")
        if obj.stage == self.stage:
            raise StageObjectiveError(f"objective {self.stage!r} cannot retain itself")
        if not obj.capability_measurements():
            raise StageObjectiveError(f"objective {self.stage!r} retains {obj.stage!r}, which has no "
                                      "generation/grammar measurements to re-check")
        return obj

    @classmethod
    def from_stage(cls, stage: str, objective_dir: str | Path | None = None) -> "StageObjective | None":
        """Load ``<objective_dir>/<stage>.objective.json`` (default directory:
        ``configs/stages_v2``); None when that stage has no objective config, in
        which case the stage keeps the legacy budget-complete behaviour."""
        directory = Path(objective_dir) if objective_dir else DEFAULT_OBJECTIVE_DIR
        path = directory / f"{stage}.objective.json"
        if not path.is_file():
            return None
        return cls(_load_json(path), source_dir=directory)

    def data_failures(self, data: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        """Measurements on the ``data`` section are decided by the training data,
        which do not change during a run, so they can be checked before training
        starts. Returns the failing ones (the trainer refuses a run whose
        objective could never be met)."""
        metrics = {"data": dict(data or {})}
        return [c for c in (_apply(s, metrics) for s in self.measurements if _section(s["metric"]) == "data")
                if not c["ok"]]

    def capability_measurements(self) -> list[dict[str, Any]]:
        """The measurements a LATER stage re-checks when it retains this one:
        generation and grammar only (this stage's loss and data measurements refer
        to its own held-out data, which a later stage does not train on)."""
        return [m for m in self.measurements if _section(m["metric"]) in _CAPABILITY_SECTIONS]

    # ---- evaluating one checkpoint ------------------------------------------------------------
    def evaluate_checkpoint(self, *, model: Any, tokenizer: Any, held_out: Mapping[str, Any], step: int,
                            tokens: int, min_tokens: int, real_tokens: int | None = None,
                            data: Mapping[str, Any] | None = None, previous: Mapping[str, Any] | None = None,
                            reference: Mapping[str, Any] | None = None, baseline: bool = False,
                            lexicon: "set[str] | frozenset[str] | None" = None,
                            log: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Measure everything on the live model and decide. ``tokens`` is the
        training budget consumed (optimizer steps x effective batch tokens, the
        unit the token budget is defined in); ``min_tokens`` the minimum before
        the objective may promote. ``previous`` is the previous checkpoint's
        report, ``reference`` the metrics each retained stage had when it was
        promoted. A ``baseline`` report (before any training in this run) is
        informational and never promotes. ``lexicon`` is the set of known words
        of the training corpus (for ``grammar.known_word_fraction``)."""
        rows = generate_continuations(model, tokenizer, self.prompts, max_new_tokens=self.max_new_tokens,
                                      **(self.gated_decoding or {})) if self.prompts else []
        metrics = {"loss": {k: held_out.get(k) for k in _LOSS_KEYS if k in held_out},
                   "generation": generation_metrics(rows), "grammar": grammar_metrics(rows, lexicon),
                   "data": dict(data or {})}
        # Diagnostic only: the same prompts under repetition-mitigating decoding. Never gated, never part of
        # the verdict, the regression checks or the retained checks; it answers "is this a decoding artifact
        # or a broken model?" (compare it with the raw greedy numbers above it).
        decoded_rows: list[dict[str, Any]] | None = None
        if self.prompts and self.diagnostic_decoding:
            decoded_rows = generate_continuations(model, tokenizer, self.prompts, max_new_tokens=self.max_new_tokens,
                                                  **self.diagnostic_decoding)
            metrics["generation_decoded"] = {**generation_metrics(decoded_rows), "decoding": dict(self.diagnostic_decoding)}
        checks = [_apply(spec, metrics) for spec in self.measurements]
        retained: dict[str, Any] = {}
        for obj in self.retained:
            r_rows = generate_continuations(model, tokenizer, obj.prompts, max_new_tokens=obj.max_new_tokens,
                                            **(obj.gated_decoding or {}))
            r_metrics = {"generation": generation_metrics(r_rows), "grammar": grammar_metrics(r_rows, lexicon)}
            r_checks = [_apply(spec, r_metrics) for spec in obj.capability_measurements()]
            retained[obj.stage] = {"metrics": r_metrics, "generations": r_rows, "checks": r_checks,
                                   "passed": all(c["ok"] for c in r_checks), "regression": obj.regression}
        budget_reached = tokens >= min_tokens
        measurements_pass = all(c["ok"] for c in checks)
        retained_pass = all(r["passed"] for r in retained.values())
        report: dict[str, Any] = {
            "kind": REPORT_KIND, "format_version": OBJECTIVE_FORMAT_VERSION, "stage": self.stage,
            "step": int(step), "tokens": int(tokens), "real_tokens": real_tokens, "min_tokens": int(min_tokens),
            "baseline": bool(baseline), "metrics": metrics,
            "generations": rows,  # the exact fixed prompts and the model's raw output, verbatim
            "retained": retained,
        }
        if decoded_rows is not None:
            report["generations_decoded"] = decoded_rows  # diagnostic only (see above)
        if self.gated_decoding:
            report["gated_decoding"] = dict(self.gated_decoding)  # the gate is NOT plain greedy: say so in the report
        regression = compare_reports(report, previous, self.regression, reference=reference)
        reasons: list[str] = []
        if baseline:
            reasons.append("baseline measurement before training in this run (never promotes)")
        if not budget_reached:
            reasons.append(f"minimum training budget not reached ({tokens:,} of {min_tokens:,} tokens)")
        reasons += [f"{c['name']}: {c['value']!r} does not satisfy {c['requirement']}" for c in checks if not c["ok"]]
        for name, r in retained.items():
            reasons += [f"retained {name} {c['name']}: {c['value']!r} does not satisfy {c['requirement']}"
                        for c in r["checks"] if not c["ok"]]
        reasons += [f"regression ({c['scope']}) in {c['metric']}: worse by {c['delta_worse']:+.4f}, allowed "
                    f"{c['allowed']}" for c in regression["checks"] if not c["ok"]]
        met = (not baseline) and budget_reached and measurements_pass and retained_pass and not regression["regressed"]
        report["regression"] = regression
        report["verdict"] = {"objective_met": met, "budget_reached": budget_reached,
                             "measurements_pass": measurements_pass, "retained_pass": retained_pass,
                             "regressed": regression["regressed"], "checks": checks, "reasons": reasons}
        report["objective_met"] = met
        if log:
            passed = sum(c["ok"] for c in checks)
            log(f"[objective] {self.stage} step {step}: {'MET' if met else 'not met'} "
                f"({passed}/{len(checks)} measurements pass"
                + (f", retained {'ok' if retained_pass else 'FAILING'}" if retained else "")
                + (", REGRESSED" if regression["regressed"] else "") + ")")
        return report


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageObjectiveError(f"cannot read objective config {path}: {exc}") from exc


# --------------------------------------------------------------------------- regression
def _regression_checks(out: list[dict[str, Any]], current: Mapping[str, Any], previous: Mapping[str, Any],
                       regression: Mapping[str, Any], scope: str, *, include_loss: bool) -> None:
    def add(metric: str, direction: str, allowed: float) -> None:
        cur, prev = _dig(current, metric), _dig(previous, metric)
        if not _is_number(cur) or not _is_number(prev):
            out.append({"scope": scope, "metric": metric, "current": cur, "previous": prev, "ok": True,
                        "note": "not comparable (missing in one report)"})
            return
        # "up": larger is worse (loss, repetition); "down": smaller is worse (diversity, word-likeness).
        delta = (cur - prev) if direction == "up" else (prev - cur)
        out.append({"scope": scope, "metric": metric, "current": cur, "previous": prev, "delta_worse": delta,
                    "allowed": allowed, "direction": direction, "ok": delta <= allowed + 1e-12})

    if include_loss and "max_val_loss_increase" in regression:
        add("loss.val_loss", "up", float(regression["max_val_loss_increase"]))
    for spec in regression.get("checks", []) or []:
        metric = spec["metric"]
        if not include_loss and _section(metric) not in _CAPABILITY_SECTIONS:
            continue
        if "max_increase" in spec:
            add(metric, "up", float(spec["max_increase"]))
        elif "max_decrease" in spec:
            add(metric, "down", float(spec["max_decrease"]))


def compare_reports(current: Mapping[str, Any], previous: Mapping[str, Any] | None,
                    regression: Mapping[str, Any] | None = None, *,
                    reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Regression check of one checkpoint's report.

    * Same stage as ``previous``: this stage's own metrics may worsen by at most
      the configured amounts (``max_val_loss_increase``; per-metric
      ``max_increase``/``max_decrease``).
    * Different stage (``previous`` is the parent stage's final report, after
      ``--init-from``): own metrics are not comparable (other held-out data and
      prompts); only retained capabilities are compared.
    * Every retained stage is compared with the previous checkpoint's
      measurement of it and with ``reference[stage]`` — its metrics when that
      stage was promoted — so slow drift across many checkpoints is caught too.

    Returns ``{"regressed": bool, "checks": [...], "notes": [...]}``."""
    regression = regression or {}
    checks: list[dict[str, Any]] = []
    notes: list[str] = []
    if previous is None:
        notes.append("no previous checkpoint to compare against")
    elif previous.get("stage") == current.get("stage"):
        _regression_checks(checks, current.get("metrics", {}), previous.get("metrics", {}), regression,
                           f"vs previous checkpoint (step {previous.get('step')})", include_loss=True)
    else:
        notes.append(f"previous checkpoint is stage {previous.get('stage')!r}: this stage's own metrics use other "
                     "held-out data and prompts, so only retained capabilities are compared")
        # ...except natural held-out text, when both stages measured the very same held-out split.
        split = _dig(current.get("metrics"), "data.natural_val_sha256")
        if split and split == _dig(previous.get("metrics"), "data.natural_val_sha256"):
            natural = {"checks": [c for c in regression.get("checks", []) or []
                                  if c.get("metric", "").startswith("loss.text_val_")]}
            _regression_checks(checks, current.get("metrics", {}), previous.get("metrics", {}), natural,
                               f"natural held-out text vs {previous.get('stage')} (same held-out split)",
                               include_loss=True)
    for name, ret in (current.get("retained") or {}).items():
        reg = ret.get("regression") or {}
        prev_metrics = None
        if previous is not None:
            prev_metrics = previous.get("metrics") if previous.get("stage") == name else \
                ((previous.get("retained") or {}).get(name) or {}).get("metrics")
        if prev_metrics:
            _regression_checks(checks, ret["metrics"], prev_metrics, reg,
                               f"retained {name} vs previous checkpoint", include_loss=False)
        ref = (reference or {}).get(name)
        if ref and not _same_capabilities(ref, prev_metrics):
            _regression_checks(checks, ret["metrics"], ref, reg, f"retained {name} vs {name} at promotion",
                               include_loss=False)
    return {"regressed": any(not c["ok"] for c in checks), "checks": checks, "notes": notes}


def _same_capabilities(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    if not a or not b:
        return False
    return all(a.get(k) == b.get(k) for k in _CAPABILITY_SECTIONS)


def capability_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    """The generation/grammar metrics of a report — what a later stage keeps as
    the promotion-time reference for this stage."""
    metrics = report.get("metrics") or {}
    return {k: metrics[k] for k in _CAPABILITY_SECTIONS if k in metrics}


# --------------------------------------------------------------------------- rendering
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _visible(text: str) -> str:
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)


def _fenced(text: str) -> str:
    """A fenced code block that shows ``text`` verbatim whatever it contains
    (the fence is longer than any backtick run inside)."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _generation_block(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for i, row in enumerate(rows, start=1):
        how = "stopped at EOS" if row["finish_reason"] == "eos" else "hit max_new_tokens"
        lines += [f"**Prompt {i}** (fixed):", "", _fenced(_visible(row["prompt"])), "",
                  f"**Raw generation {i}** ({row['tokens_generated']} tokens, {how}):", "",
                  _fenced(_visible(row["generation"])) if row["generation"] else "_(empty generation)_", ""]
    return lines


def describe_decoding(d: Mapping[str, Any] | None) -> str:
    """Human-readable decoding settings ("greedy decoding" for ``None``/empty)."""
    if not d:
        return "greedy decoding"
    parts = ([f"seeded sampling, temperature {d['temperature']:g}"] if d.get("temperature", 0) > 0 else ["greedy"])
    if d.get("top_k"):
        parts.append(f"top-k {d['top_k']}")
    if d.get("top_p"):
        parts.append(f"top-p {d['top_p']:g}")
    if d.get("temperature", 0) > 0:
        parts.append(f"seed {d.get('seed', 0)}")
    if d.get("repetition_penalty", 1.0) != 1.0:
        parts.append(f"repetition penalty {d['repetition_penalty']:g}")
    if d.get("no_repeat_ngram_size", 0):
        parts.append(f"no-repeat {d['no_repeat_ngram_size']}-gram")
    return ", ".join(parts)


def render_markdown(report: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> str:
    """GitHub-visible Markdown for one checkpoint: step/tokens, held-out loss and
    perplexity (and natural-text bits per byte), generation statistics, grammar
    checks, every objective measurement with its threshold, retained earlier
    capabilities, regression against the previous checkpoint, and the exact
    fixed prompts with the model's raw generations shown verbatim."""
    m = report["metrics"]
    v = report["verdict"]
    same_prev = previous if previous and previous.get("stage") == report["stage"] else None

    def delta(path: str) -> str:
        if not same_prev:
            return ""
        cur, prev = _dig(m, path), _dig(same_prev.get("metrics"), path)
        return f"{cur - prev:+.4f}" if _is_number(cur) and _is_number(prev) else ""

    if report.get("baseline"):
        status = "baseline (before training in this run; not a promotion decision)"
    else:
        status = "objective MET" if report["objective_met"] else "objective NOT met"
    lines = [f"## Stage `{report['stage']}` @ step {report['step']:,}: {status}", "",
             "| step | budget tokens trained | real tokens | minimum before promotion | measurements | retained | regression |",
             "|---|---|---|---|---|---|---|",
             f"| {report['step']:,} | {report['tokens']:,} | {_fmt(report.get('real_tokens'))} | {report['min_tokens']:,} | "
             f"{sum(c['ok'] for c in v['checks'])}/{len(v['checks'])} pass | "
             f"{'n/a' if not report.get('retained') else ('pass' if v['retained_pass'] else 'FAIL')} | "
             f"{'REGRESSED' if v['regressed'] else 'none'} |", ""]
    if v["reasons"]:
        lines += ["Not met because:", ""] + [f"- {r}" for r in v["reasons"]] + [""]

    loss = m.get("loss", {})
    lines += ["### Held-out language modelling", "", "| metric | value | Δ vs previous checkpoint |", "|---|---|---|"]
    for key, label in (("val_loss", "validation loss (nats/token)"), ("val_ppl", "validation perplexity"),
                       ("text_val_loss", "natural held-out text loss (nats/token)"),
                       ("text_val_ppl", "natural held-out text perplexity"),
                       ("text_val_bpb", "natural held-out text bits per byte")):
        if key in loss:
            lines.append(f"| {label} | {_fmt(loss[key])} | {delta('loss.' + key)} |")
    lines.append("")

    gen = m["generation"]
    gated = report.get("gated_decoding")
    lines += [f"### Generation statistics (fixed prompts, {describe_decoding(gated)})", "",
              "| metric | value | Δ vs previous checkpoint |", "|---|---|---|"]
    for key, label in (("non_empty_rate", "non-empty rate"), ("mean_repetition", "mean repetition (char 6-grams)"),
                       ("looping_rate", "looping rate (repetition > 0.5)"), ("mean_distinct_1", "distinct-1"),
                       ("mean_distinct_2", "distinct-2"), ("mean_chars", "mean characters"),
                       ("mean_tokens_generated", "mean tokens generated"), ("eos_rate", "EOS rate")):
        lines.append(f"| {label} | {_fmt(gen.get(key))} | {delta('generation.' + key)} |")
    lines.append("")

    dec = m.get("generation_decoded")
    if dec:
        d = dec.get("decoding", {})
        lines += [f"### Diagnostic (NOT gated): same prompts, decoding-mitigated ({describe_decoding(d)})",
                  "", "If these are healthy while the gated numbers above loop, the model is fine and the gated "
                  "decoding is the problem; if these are bad too, the model itself is broken.", "",
                  f"| metric | decoding-mitigated | {'gated decoding' if gated else 'raw greedy (gated)'} |",
                  "|---|---|---|"]
        for key, label in (("mean_repetition", "mean repetition (char 6-grams)"),
                           ("looping_rate", "looping rate (repetition > 0.5)"), ("mean_distinct_2", "distinct-2")):
            lines.append(f"| {label} | {_fmt(dec.get(key))} | {_fmt(gen.get(key))} |")
        lines.append("")

    gram = m["grammar"]
    lines += ["### Grammar and language checks", "", "| metric | value | Δ vs previous checkpoint |", "|---|---|---|"]
    for key, label in (("word_like_fraction", "word-like tokens"),
                       ("known_word_fraction", "known words (training-corpus lexicon)"),
                       ("alpha_fraction", "letters/punctuation share"),
                       ("sentence_start_capitalized", "sentence starts capitalised"),
                       ("space_after_punctuation", "space after punctuation"),
                       ("terminator_fraction", "ends on . ! ?"), ("mean_word_length", "mean word length"),
                       ("mean_sentence_words", "mean sentence length (words)")):
        lines.append(f"| {label} | {_fmt(gram.get(key))} | {delta('grammar.' + key)} |")
    lines.append("")

    if m.get("data"):
        lines += ["### Training data", "", "| fact | value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(val)} |" for k, val in sorted(m["data"].items())] + [""]

    lines += ["### Objective measurements (every one must pass; there is no composite score)", "",
              "| measurement | metric | value | requirement | result |", "|---|---|---|---|---|"]
    for c in v["checks"]:
        lines.append(f"| {c['name']} | `{c['metric']}` | {_fmt(c['value'])} | {c['requirement']} | "
                     f"{'pass' if c['ok'] else 'FAIL'} |")
    lines.append("")

    for name, ret in (report.get("retained") or {}).items():
        lines += [f"### Retained capability: `{name}` (re-measured on this checkpoint with {name}'s prompts)", "",
                  "| measurement | value | requirement | result |", "|---|---|---|---|"]
        for c in ret["checks"]:
            lines.append(f"| {c['name']} | {_fmt(c['value'])} | {c['requirement']} | {'pass' if c['ok'] else 'FAIL'} |")
        lines += ["", f"<details><summary>Raw generations for the {len(ret['generations'])} retained {name} prompts"
                      "</summary>", ""] + _generation_block(ret["generations"]) + ["</details>", ""]

    reg = report.get("regression") or {}
    lines += ["### Regression checks", ""]
    notes = [f"_{n}_" for n in reg.get("notes", [])]
    compared = [c for c in reg.get("checks", []) if "delta_worse" in c]
    if notes:
        lines += notes + [""]
    if not notes and not compared:
        lines += ["_No regression checks apply at this checkpoint._", ""]
    if compared:
        lines += ["| compared with | metric | previous | current | worse by | allowed | result |",
                  "|---|---|---|---|---|---|---|"]
        for c in compared:
            lines.append(f"| {c['scope']} | `{c['metric']}` | {_fmt(c['previous'])} | {_fmt(c['current'])} | "
                         f"{c['delta_worse']:+.4f} | {_fmt(c['allowed'])} | {'pass' if c['ok'] else 'FAIL'} |")
    lines.append("")

    lines += ["### Fixed prompts and raw generations (verbatim)", ""] + _generation_block(report["generations"])
    if report.get("generations_decoded"):
        lines += ["<details><summary>Same prompts with decoding mitigation (diagnostic only, not gated)</summary>", ""]
        lines += _generation_block(report["generations_decoded"]) + ["</details>", ""]
    lines += ["---", "",
              "Raw generations are shown exactly as decoded (control characters other than newline and tab appear "
              "as \\xNN escapes); the JSON report next to this file holds the exact text. Promotion is decided by "
              "the objective above: every measurement must pass after the minimum budget, retained capabilities "
              "must pass, and nothing may regress. No single number decides it."]
    return "\n".join(lines) + "\n"


def render_history(rows: Sequence[Mapping[str, Any]]) -> str:
    """One Markdown table row per evaluated checkpoint of a run (from
    ``objective_reports/history.jsonl``)."""
    lines = ["| step | budget tokens | val loss | natural text bits/byte | repetition | known words | objective | regressed |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['step']:,} | {r['tokens']:,} | {_fmt(r.get('val_loss'))} | {_fmt(r.get('text_val_bpb'))} | "
                     f"{_fmt(r.get('mean_repetition'))} | {_fmt(r.get('known_word_fraction'))} | "
                     f"{'baseline' if r.get('baseline') else ('MET' if r.get('objective_met') else 'not met')} | "
                     f"{'yes' if r.get('regressed') else 'no'} |")
    return "\n".join(lines) + "\n"


def history_row(report: Mapping[str, Any]) -> dict[str, Any]:
    m = report["metrics"]
    return {"step": report["step"], "tokens": report["tokens"], "baseline": report.get("baseline", False),
            "objective_met": report["objective_met"], "regressed": report["verdict"]["regressed"],
            "val_loss": m["loss"].get("val_loss"), "text_val_bpb": m["loss"].get("text_val_bpb"),
            "mean_repetition": m["generation"].get("mean_repetition"),
            "word_like_fraction": m["grammar"].get("word_like_fraction"),
            "known_word_fraction": m["grammar"].get("known_word_fraction")}
