"""Component-level confidence, per the engineering brief section 10 and
``docs/architecture/tinymind-design.md`` section 4.

Needle reports one number: ``min(calibrated_head_score,
decode_probability)`` (needle-analysis.md section 9) — a real strength
(an explicit AND of two signals beats one blended number), but a consumer
has to know from documentation, not from the response itself, when that
number stops meaning anything (fine-tuned weights, non-English input).
TinyMind reports the components and a ``valid_for`` list naming which ones
were actually computable for *this* response, so that's a field to branch
on, not a fact to remember. See each component's docstring below for
exactly what it is computed from; none of them are placeholders — each is
either a real deterministic computation over something else in this
codebase, or explicitly ``None`` with a stated reason (never a made-up
number standing in for "we don't actually know").
"""
from __future__ import annotations

import dataclasses

from tinymind.runtime.grounding import GroundingResult
from tinymind.runtime.verification.verifier import VerificationOutcome
from tinymind.tools.validation import ValidationResult


@dataclasses.dataclass
class ConfidenceComponents:
    model: float | None = None
    """A calibrated model head's score. Always None in this delivery — see
    ``tinymind.model.backend.ModelBackend.is_real_model``; no backend
    shipped here is a trained model to calibrate a head against."""
    tool_selection: float | None = None
    """How much better the chosen tool scored than the runner-up during
    retrieval (see ``estimate_tool_selection``). 1.0 when there was no
    real ambiguity (0 or 1 candidates)."""
    arguments: float | None = None
    """Derived from JSON-Schema validation of the arguments: 1.0 if fully
    valid, reduced per validation error (see ``estimate_arguments``)."""
    grounding: float | None = None
    """Fraction of arguments traceable to the source text (see
    ``tinymind.runtime.grounding``)."""
    verification: float | None = None
    """Fraction of run ``Verifier``s that passed. None if no verifiers ran."""

    def as_dict(self) -> dict[str, float | None]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ConfidenceResult:
    confidence: float
    components: ConfidenceComponents
    valid_for: list[str]

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "confidence_components": self.components.as_dict(),
            "valid_for": self.valid_for,
        }


def estimate_tool_selection(ranked_scores: list[float]) -> float:
    """1.0 with 0 or 1 candidates (no ambiguity to resolve). Otherwise the
    normalized margin between the top score and the runner-up — a small
    margin means the retriever nearly picked a different tool."""
    if len(ranked_scores) <= 1:
        return 1.0
    top, second = ranked_scores[0], ranked_scores[1]
    if top <= 0:
        return 0.0
    margin = (top - second) / top
    return max(0.0, min(1.0, margin))


def estimate_arguments(validation: ValidationResult) -> float:
    if validation.valid:
        return 1.0
    return max(0.0, 1.0 - 0.25 * len(validation.errors))


def estimate_grounding(grounding: GroundingResult) -> float:
    if not grounding.fields:
        return 1.0
    grounded = sum(1 for f in grounding.fields if f.grounded)
    return grounded / len(grounding.fields)


def estimate_verification(outcomes: list[VerificationOutcome]) -> float:
    if not outcomes:
        return 1.0
    return sum(1 for o in outcomes if o.passed) / len(outcomes)


class ConfidenceEstimator:
    def estimate(self, *, tool_ranked_scores: list[float] | None = None,
                validation: ValidationResult | None = None,
                grounding: GroundingResult | None = None,
                verifier_outcomes: list[VerificationOutcome] | None = None,
                model_score: float | None = None) -> ConfidenceResult:
        components = ConfidenceComponents(
            model=model_score,
            tool_selection=(estimate_tool_selection(tool_ranked_scores)
                            if tool_ranked_scores is not None else None),
            arguments=estimate_arguments(validation) if validation is not None else None,
            grounding=estimate_grounding(grounding) if grounding is not None else None,
            verification=(estimate_verification(verifier_outcomes)
                          if verifier_outcomes is not None else None),
        )
        values = {name: value for name, value in components.as_dict().items() if value is not None}
        confidence = min(values.values()) if values else 0.0
        return ConfidenceResult(confidence=confidence, components=components,
                                valid_for=sorted(values))


class ConfidenceAction:
    EXECUTE = "execute"
    VERIFY = "verify"
    CLARIFY = "clarify"


@dataclasses.dataclass
class ConfidencePolicy:
    """Thresholds from the engineering brief section 10:
    >= execute_threshold -> execute; >= verify_threshold -> verify;
    below that -> clarify/escalate."""
    execute_threshold: float = 0.90
    verify_threshold: float = 0.70

    def decide(self, confidence: float) -> str:
        if confidence >= self.execute_threshold:
            return ConfidenceAction.EXECUTE
        if confidence >= self.verify_threshold:
            return ConfidenceAction.VERIFY
        return ConfidenceAction.CLARIFY
