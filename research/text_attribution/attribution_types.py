"""Immutable records produced by family-attribution experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


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
                {str(label): float(probability) for label, probability in self.family_probabilities.items()}
            ),
        )

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
