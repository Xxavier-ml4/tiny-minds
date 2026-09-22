"""Detect evaluation data that also appears in training data.

Two levels, both on *what the model is asked*, because a leaked question is a
leak whatever the recorded answer says:

* **exact** — SHA-256 of the raw prompt text (every non-assistant turn, in
  order); and of the full example (prompt + assistant turns);
* **normalised** — the same after Unicode NFKC, lower-casing, removing
  sentence punctuation and collapsing whitespace, which catches trivial
  re-typings ("What is 4 + 5?" vs "what is 4+5"); math symbols stay significant.

``check_overlap`` reports; ``assert_no_contamination`` raises
``ContaminationError`` unless explicitly overridden (a test may want to prove
the detector fires). The trainer and ``tinymind data check-contamination``
both call it; a stage does not start on a contaminated eval set.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
import unicodedata
from typing import Any, Iterable


class ContaminationError(RuntimeError):
    pass


def normalize(text: str) -> str:
    """Case, Unicode form, sentence punctuation and spacing are ignored; **math symbols are kept as separate
    tokens** so ``4 + 5`` and ``4+5`` match but ``12 + 3`` and ``1 + 23`` (different problems) do not."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[.,!?;:'\"\u2019\u2018\u201c\u201d]", "", t)
    t = re.sub(r"([+\-*/=()<>%])", r" \1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def texts(record: dict[str, Any]) -> tuple[str, str]:
    """``(prompt_text, full_text)`` of a raw example in any accepted format."""
    if "text" in record:
        return record["text"], record["text"]
    prompt, full = [], []
    for m in record["messages"]:
        full.append(f"{m['role']}:{m['content']}")
        if m["role"] != "assistant":
            prompt.append(f"{m['role']}:{m['content']}")
    if "target" in record:
        full.append(f"assistant:{record['target']}")
    return "\x1f".join(prompt), "\x1f".join(full)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class ContaminationReport:
    train_examples: int
    eval_examples: int
    exact_prompt: list[str]        # eval ids whose exact prompt occurs in train
    exact_full: list[str]
    normalized_prompt: list[str]
    normalized_full: list[str]

    @property
    def contaminated(self) -> bool:
        return bool(self.exact_prompt or self.exact_full or self.normalized_prompt or self.normalized_full)

    def summary(self) -> dict[str, Any]:
        return {"train_examples": self.train_examples, "eval_examples": self.eval_examples,
                "exact_prompt_overlap": len(self.exact_prompt), "exact_full_overlap": len(self.exact_full),
                "normalized_prompt_overlap": len(self.normalized_prompt), "normalized_full_overlap": len(self.normalized_full),
                "contaminated": self.contaminated,
                "examples": sorted(set(self.exact_prompt + self.normalized_prompt + self.exact_full))[:10]}


def check_overlap(train: Iterable[dict[str, Any]], evaluation: Iterable[dict[str, Any]]) -> ContaminationReport:
    tp, tf, tnp, tnf = set(), set(), set(), set()
    n_train = 0
    for r in train:
        p, f = texts(r)
        tp.add(_sha(p)); tf.add(_sha(f)); tnp.add(_sha(normalize(p))); tnf.add(_sha(normalize(f)))
        n_train += 1
    ep, ef, enp, enf = [], [], [], []
    n_eval = 0
    for r in evaluation:
        n_eval += 1
        rid = str(r.get("id", n_eval))
        p, f = texts(r)
        if _sha(p) in tp:
            ep.append(rid)
        if _sha(f) in tf:
            ef.append(rid)
        if _sha(normalize(p)) in tnp:
            enp.append(rid)
        if _sha(normalize(f)) in tnf:
            enf.append(rid)
    return ContaminationReport(n_train, n_eval, ep, ef, enp, enf)


def assert_no_contamination(train: Iterable[dict[str, Any]], evaluation: Iterable[dict[str, Any]], *,
                            allow: bool = False, what: str = "evaluation set") -> ContaminationReport:
    report = check_overlap(train, evaluation)
    if report.contaminated and not allow:
        s = report.summary()
        raise ContaminationError(
            f"{what} overlaps the training data: exact prompt {s['exact_prompt_overlap']}, exact full "
            f"{s['exact_full_overlap']}, normalised prompt {s['normalized_prompt_overlap']}, normalised full "
            f"{s['normalized_full_overlap']} (e.g. {s['examples'][:5]}). Fix the data, or pass allow=True / "
            "--allow-contamination for a deliberate test.")
    return report
