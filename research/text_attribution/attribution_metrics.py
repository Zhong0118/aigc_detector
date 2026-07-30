"""Metrics and validation for family-attribution probability predictions."""

from __future__ import annotations

from collections import defaultdict
from math import log
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from research.text_attribution.attribution_types import FamilyPrediction
from research.text_attribution.training_data import AttributionSample


def _validated_labels(family_labels: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(family_labels)
    if not labels:
        raise ValueError("family_labels must be nonempty")
    if len(set(labels)) != len(labels):
        raise ValueError("family_labels must be unique")
    return labels


def build_prediction_records(
    samples: Sequence[AttributionSample],
    probabilities: np.ndarray,
    family_labels: Sequence[str],
    experiment_id: str,
    split: str,
) -> tuple[FamilyPrediction, ...]:
    """Pair model probabilities with samples after enforcing their shared contract."""

    labels = _validated_labels(family_labels)
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape != (len(samples), len(labels)):
        raise ValueError("probabilities must have shape (samples, family_labels)")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if (values < 0).any():
        raise ValueError("probabilities must be nonnegative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")

    records: list[FamilyPrediction] = []
    for sample, row in zip(samples, values, strict=True):
        if sample.family_id not in labels:
            raise ValueError(f"sample true family not in family_labels: {sample.family_id}")
        prediction_index = int(np.argmax(row))
        records.append(
            FamilyPrediction(
                request_key=sample.request_key,
                experiment_id=experiment_id,
                split=split,
                true_family=sample.family_id,
                predicted_family=labels[prediction_index],
                family_probabilities=dict(zip(labels, row.tolist(), strict=True)),
                confidence=float(row[prediction_index]),
                model_id=sample.model_id,
                prompt_id=sample.prompt_id,
                semantic_prompt_group=sample.semantic_prompt_group,
                language=sample.language,
                task_type=sample.task_type,
                target_length=sample.target_length,
                temperature=sample.temperature,
            )
        )
    return tuple(records)


def _validate_records(
    records: Iterable[FamilyPrediction], family_labels: Sequence[str]
) -> tuple[tuple[FamilyPrediction, ...], tuple[str, ...]]:
    labels = _validated_labels(family_labels)
    validated_records = tuple(records)
    if not validated_records:
        raise ValueError("records must be nonempty")
    expected_keys = set(labels)
    for record in validated_records:
        if set(record.family_probabilities) != expected_keys:
            raise ValueError("record family probability keys must match family_labels")
        probabilities = np.asarray(
            [record.family_probabilities[label] for label in labels], dtype=float
        )
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("record probabilities must be finite and nonnegative")
        if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("record probability rows must sum to one")
        if record.true_family not in expected_keys or record.predicted_family not in expected_keys:
            raise ValueError("record families must be in family_labels")
    return validated_records, labels


def expected_calibration_error(records: Iterable[FamilyPrediction], bins: int = 10) -> float:
    """Return top-label expected calibration error, including confidence 1 in the last bin."""

    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    validated_records = tuple(records)
    if not validated_records:
        raise ValueError("records must be nonempty")
    ece = 0.0
    total = len(validated_records)
    for bin_index in range(bins):
        in_bin = [
            record
            for record in validated_records
            if min(int(record.confidence * bins), bins - 1) == bin_index
        ]
        if in_bin:
            mean_confidence = sum(record.confidence for record in in_bin) / len(in_bin)
            accuracy = sum(record.predicted_family == record.true_family for record in in_bin) / len(in_bin)
            ece += len(in_bin) / total * abs(accuracy - mean_confidence)
    return float(ece)


def _subgroup_metrics(records: Sequence[FamilyPrediction], labels: Sequence[str], field: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[FamilyPrediction]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, field))].append(record)
    return {
        value: {
            "support": len(group),
            "accuracy": float(accuracy_score([r.true_family for r in group], [r.predicted_family for r in group])),
            "macro_f1": float(f1_score([r.true_family for r in group], [r.predicted_family for r in group], labels=labels, average="macro", zero_division=0)),
        }
        for value, group in sorted(groups.items())
    }


def evaluate_family_predictions(
    records: Iterable[FamilyPrediction], family_labels: Sequence[str], ece_bins: int = 10
) -> dict[str, Any]:
    """Evaluate prediction records with fixed-label classification and calibration metrics."""

    validated_records, labels = _validate_records(records, family_labels)
    if not isinstance(ece_bins, int) or isinstance(ece_bins, bool) or ece_bins <= 0:
        raise ValueError("ece_bins must be a positive integer")
    truth = [record.true_family for record in validated_records]
    predicted = [record.predicted_family for record in validated_records]
    probability_matrix = np.asarray(
        [[record.family_probabilities[label] for label in labels] for record in validated_records],
        dtype=float,
    )
    report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    per_model: dict[str, dict[str, float | int]] = {}
    for model_id in sorted({record.model_id for record in validated_records}):
        group = [record for record in validated_records if record.model_id == model_id]
        per_model[model_id] = {
            "support": len(group),
            "family_accuracy": float(accuracy_score([r.true_family for r in group], [r.predicted_family for r in group])),
            "mean_confidence": float(np.mean([r.confidence for r in group])),
            "mean_true_family_probability": float(np.mean([r.family_probabilities[r.true_family] for r in group])),
        }
    true_indexes = np.asarray([labels.index(value) for value in truth])
    one_hot = np.eye(len(labels))[true_indexes]
    brier = float(np.mean(np.sum((probability_matrix - one_hot) ** 2, axis=1)))
    nll = float(-np.mean([log(max(row[index], np.finfo(float).tiny)) for row, index in zip(probability_matrix, true_indexes, strict=True)]))
    overall = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)),
        "support": len(validated_records),
    }
    return {
        "overall": overall,
        "per_family": {label: report[label] for label in labels},
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels, normalize="true").tolist(),
        "per_model": per_model,
        "calibration": {
            "expected_calibration_error": expected_calibration_error(validated_records, ece_bins),
            "multiclass_brier_score": brier,
            "negative_log_likelihood": nll,
        },
        "subgroups": {
            field: _subgroup_metrics(validated_records, labels, field)
            for field in ("language", "target_length", "task_type", "temperature")
        },
    }
