"""Immutable records produced by family-attribution experiments."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping


def _normalize_real(value: Any, field: str) -> float:
    """Return a built-in float while rejecting non-numeric JSON ambiguities."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    return float(value)


@dataclass(frozen=True)
class FamilyPrediction:
    """One family-level prediction paired with its source sample metadata."""

    request_key: str
    experiment_id: str
    split: str
    true_family: str
    predicted_family: str
    family_probabilities: Mapping[str, float]
    confidence: float
    model_id: str
    prompt_id: str
    semantic_prompt_group: str
    language: str
    task_type: str
    target_length: str
    temperature: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_probabilities",
            MappingProxyType(
                {
                    str(label): _normalize_real(probability, "family probability")
                    for label, probability in self.family_probabilities.items()
                }
            ),
        )
        object.__setattr__(self, "confidence", _normalize_real(self.confidence, "confidence"))
        object.__setattr__(self, "temperature", _normalize_real(self.temperature, "temperature"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this record."""

        return {
            "request_key": self.request_key,
            "experiment_id": self.experiment_id,
            "split": self.split,
            "true_family": self.true_family,
            "predicted_family": self.predicted_family,
            "family_probabilities": dict(self.family_probabilities),
            "confidence": self.confidence,
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "semantic_prompt_group": self.semantic_prompt_group,
            "language": self.language,
            "task_type": self.task_type,
            "target_length": self.target_length,
            "temperature": self.temperature,
        }
