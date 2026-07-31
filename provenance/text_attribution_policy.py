"""Calibrated hierarchical decision policy for open-world text attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


VALID_STATUSES = {"exact", "family_only", "unknown", "abstain"}


@dataclass(frozen=True)
class AttributionDecision:
    """Model-independent source decision returned after calibration."""

    status: str
    model_family: str | None = None
    model_id: str | None = None
    confidence: float = 0.0
    known_probability: float = 0.0
    prediction_set: tuple[str, ...] = ()
    prototype_distance: float | None = None
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unsupported attribution status: {self.status}")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "model_family": self.model_family,
            "model_id": self.model_id,
            "confidence": self.confidence,
            "known_probability": self.known_probability,
            "prediction_set": list(self.prediction_set),
            "prototype_distance": self.prototype_distance,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class HierarchicalAttributionPolicy:
    """Convert calibrated classifier and prototype evidence into safe states."""

    min_chars: int = 50
    min_known_probability: float = 0.65
    max_prototype_distance: float = 0.75
    min_exact_probability: float = 0.80
    min_family_probability: float = 0.70
    min_exact_margin: float = 0.15

    def decide(
        self,
        *,
        text: str,
        exact_probabilities: Mapping[str, float],
        family_probabilities: Mapping[str, float],
        model_to_family: Mapping[str, str],
        known_probability: float,
        prototype_distance: float | None,
    ) -> AttributionDecision:
        """Apply length, knownness, distance, hierarchy and margin gates."""

        known = _clamp_probability(known_probability)
        if len(text.strip()) < self.min_chars:
            return AttributionDecision(
                status="abstain",
                known_probability=known,
                prototype_distance=prototype_distance,
                warnings=("insufficient_text",),
            )

        if known < self.min_known_probability:
            return AttributionDecision(
                status="unknown",
                confidence=1.0 - known,
                known_probability=known,
                prototype_distance=prototype_distance,
                evidence=("known_probability_below_threshold",),
            )
        if prototype_distance is not None and prototype_distance > self.max_prototype_distance:
            return AttributionDecision(
                status="unknown",
                confidence=known,
                known_probability=known,
                prototype_distance=prototype_distance,
                evidence=("prototype_distance_above_threshold",),
            )

        exact_ranked = _rank_probabilities(exact_probabilities)
        family_ranked = _rank_probabilities(family_probabilities)
        if not exact_ranked or not family_ranked:
            return AttributionDecision(
                status="abstain",
                known_probability=known,
                prototype_distance=prototype_distance,
                warnings=("missing_classifier_evidence",),
            )

        top_model, top_exact = exact_ranked[0]
        second_exact = exact_ranked[1][1] if len(exact_ranked) > 1 else 0.0
        inferred_family = model_to_family.get(top_model)
        top_family, top_family_probability = family_ranked[0]

        if inferred_family is None:
            return AttributionDecision(
                status="abstain",
                known_probability=known,
                prototype_distance=prototype_distance,
                warnings=("missing_model_family_mapping",),
            )
        if top_family != inferred_family and top_family_probability >= self.min_family_probability:
            return AttributionDecision(
                status="abstain",
                known_probability=known,
                prototype_distance=prototype_distance,
                prediction_set=(top_model,),
                warnings=("family_exact_conflict",),
            )

        exact_margin = top_exact - second_exact
        if (
            top_exact >= self.min_exact_probability
            and exact_margin >= self.min_exact_margin
            and top_family_probability >= self.min_family_probability
        ):
            return AttributionDecision(
                status="exact",
                model_family=inferred_family,
                model_id=top_model,
                confidence=top_exact,
                known_probability=known,
                prediction_set=(top_model,),
                prototype_distance=prototype_distance,
                evidence=("exact_probability", "exact_margin", "family_agreement"),
            )

        if top_family_probability >= self.min_family_probability:
            same_family_candidates = tuple(
                model
                for model, _ in exact_ranked
                if model_to_family.get(model) == top_family
            )
            return AttributionDecision(
                status="family_only",
                model_family=top_family,
                confidence=top_family_probability,
                known_probability=known,
                prediction_set=same_family_candidates,
                prototype_distance=prototype_distance,
                evidence=("family_probability", "exact_evidence_insufficient"),
            )

        return AttributionDecision(
            status="abstain",
            known_probability=known,
            prototype_distance=prototype_distance,
            warnings=("insufficient_family_confidence",),
        )


def _rank_probabilities(values: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(
        ((str(label), _clamp_probability(probability)) for label, probability in values.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

