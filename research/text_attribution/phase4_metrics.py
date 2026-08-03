"""Extended and selective metrics for Phase4 family attribution."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from research.text_attribution.attribution_metrics import (
    evaluate_family_predictions,
    expected_calibration_error,
)
from research.text_attribution.attribution_types import FamilyPrediction


def evaluate_phase4_predictions(
    records: Iterable[FamilyPrediction],
    family_labels: Sequence[str],
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Return the Phase3 metrics plus macro precision/recall and Top-K."""

    rows = tuple(records)
    labels = _labels(family_labels)
    base = evaluate_family_predictions(rows, labels, ece_bins=ece_bins)
    truth = [row.true_family for row in rows]
    predicted = [row.predicted_family for row in rows]
    probabilities = np.asarray(
        [[row.family_probabilities[label] for label in labels] for row in rows],
        dtype=float,
    )
    overall = dict(base["overall"])
    overall["macro_precision"] = float(
        precision_score(
            truth, predicted, labels=labels, average="macro", zero_division=0
        )
    )
    overall["macro_recall"] = float(
        recall_score(truth, predicted, labels=labels, average="macro", zero_division=0)
    )
    base["overall"] = overall

    true_indexes = np.asarray([labels.index(value) for value in truth], dtype=int)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    base["top_k_accuracy"] = {
        f"top_{k}": float(
            np.mean(
                [
                    true_index in order[index, : min(k, len(labels))]
                    for index, true_index in enumerate(true_indexes)
                ]
            )
        )
        for k in (1, 2, 3)
    }
    return base


def evaluate_selective_predictions(
    records: Sequence[FamilyPrediction],
    accepted_mask: Sequence[bool] | np.ndarray,
    family_labels: Sequence[str],
    *,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Measure correctness among accepted rows and coverage among eligible rows."""

    rows = tuple(records)
    labels = _labels(family_labels)
    mask = np.asarray(accepted_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape != (len(rows),):
        raise ValueError("accepted_mask must have one value per record")
    if not rows:
        raise ValueError("records must be nonempty")

    accepted = tuple(row for row, keep in zip(rows, mask, strict=True) if keep)
    correct = sum(row.predicted_family == row.true_family for row in accepted)
    accepted_precision = None if not accepted else float(correct / len(accepted))
    per_family: dict[str, dict[str, Any]] = {}
    for label in labels:
        eligible_true = [row for row in rows if row.true_family == label]
        accepted_true = [row for row in accepted if row.true_family == label]
        accepted_predicted = [row for row in accepted if row.predicted_family == label]
        true_positive = sum(
            row.true_family == label and row.predicted_family == label for row in accepted
        )
        per_family[label] = {
            "eligible_support": len(eligible_true),
            "accepted_support": len(accepted_true),
            "accepted_predicted_support": len(accepted_predicted),
            "accepted_precision": (
                None
                if not accepted_predicted
                else float(true_positive / len(accepted_predicted))
            ),
            "accepted_recall": (
                None if not eligible_true else float(true_positive / len(eligible_true))
            ),
            "coverage": (
                None
                if not eligible_true
                else float(len(accepted_true) / len(eligible_true))
            ),
            "coverage_warning": not accepted_predicted,
        }
    return {
        "eligible_count": len(rows),
        "accepted_count": len(accepted),
        "accepted_precision": accepted_precision,
        "coverage": float(len(accepted) / len(rows)),
        "selective_risk": (
            None if accepted_precision is None else float(1.0 - accepted_precision)
        ),
        "ece": (
            None
            if not accepted
            else expected_calibration_error(accepted, bins=ece_bins)
        ),
        "per_family": per_family,
    }


def group_bootstrap_interval(
    records: Sequence[FamilyPrediction],
    family_labels: Sequence[str],
    metric_name: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Return a percentile interval while resampling semantic groups."""

    rows = tuple(records)
    labels = _labels(family_labels)
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    grouped: dict[str, list[FamilyPrediction]] = defaultdict(list)
    for row in rows:
        grouped[row.semantic_prompt_group].append(row)
    group_names = tuple(sorted(grouped))
    if len(group_names) < 2:
        raise ValueError("group bootstrap requires at least two semantic groups")

    estimate = _metric(rows, labels, metric_name)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_names = rng.choice(group_names, size=len(group_names), replace=True)
        sampled_rows = tuple(
            row for name in sampled_names for row in grouped[str(name)]
        )
        values[index] = _metric(sampled_rows, labels, metric_name)
    return {
        "metric": metric_name,
        "estimate": float(estimate),
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "resampling_unit": "semantic_prompt_group",
    }


def _metric(
    rows: Sequence[FamilyPrediction], labels: tuple[str, ...], metric_name: str
) -> float:
    truth = [row.true_family for row in rows]
    predicted = [row.predicted_family for row in rows]
    if metric_name == "accuracy":
        return float(np.mean([a == b for a, b in zip(truth, predicted, strict=True)]))
    if metric_name == "macro_f1":
        return float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        )
    if metric_name == "macro_precision":
        return float(
            precision_score(
                truth, predicted, labels=labels, average="macro", zero_division=0
            )
        )
    if metric_name == "macro_recall":
        return float(
            recall_score(
                truth, predicted, labels=labels, average="macro", zero_division=0
            )
        )
    raise ValueError(f"unsupported bootstrap metric: {metric_name}")


def _labels(values: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(values)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("family_labels must be unique and nonempty")
    return labels
