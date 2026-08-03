from __future__ import annotations

import math

import numpy as np
import pytest

from research.text_attribution.attribution_types import FamilyPrediction
from research.text_attribution.phase4_metrics import (
    evaluate_phase4_predictions,
    evaluate_selective_predictions,
    group_bootstrap_interval,
)


LABELS = ("a", "b")


def _records(
    truth: tuple[str, ...],
    probabilities: tuple[tuple[float, float], ...],
    groups: tuple[str, ...],
) -> tuple[FamilyPrediction, ...]:
    rows = []
    for index, (actual, values, group) in enumerate(
        zip(truth, probabilities, groups, strict=True)
    ):
        predicted_index = int(np.argmax(values))
        rows.append(
            FamilyPrediction(
                request_key=f"r{index}",
                experiment_id="fixture",
                split="validation",
                true_family=actual,
                predicted_family=LABELS[predicted_index],
                family_probabilities=dict(zip(LABELS, values, strict=True)),
                confidence=max(values),
                model_id=f"{actual}-model",
                prompt_id=f"p{index}",
                semantic_prompt_group=group,
                language="zh",
                task_type="explanation",
                target_length="medium",
                temperature=0.0,
            )
        )
    return tuple(rows)


def test_extended_metrics_include_precision_recall_and_top_k() -> None:
    records = _records(
        ("a", "a", "b", "b"),
        ((0.9, 0.1), (0.4, 0.6), (0.3, 0.7), (0.8, 0.2)),
        ("g1", "g2", "g1", "g2"),
    )
    result = evaluate_phase4_predictions(records, LABELS, ece_bins=2)
    assert result["overall"]["accuracy"] == 0.5
    assert result["overall"]["macro_precision"] == 0.5
    assert result["overall"]["macro_recall"] == 0.5
    assert result["top_k_accuracy"]["top_1"] == 0.5
    assert result["top_k_accuracy"]["top_2"] == 1.0


def test_selective_metrics_use_fixed_denominators_and_report_undefined() -> None:
    records = _records(
        ("a", "a", "b", "b"),
        ((0.9, 0.1), (0.4, 0.6), (0.3, 0.7), (0.8, 0.2)),
        ("g1", "g2", "g1", "g2"),
    )
    result = evaluate_selective_predictions(
        records, np.asarray([True, False, True, False]), LABELS
    )
    assert result["accepted_precision"] == 1.0
    assert result["coverage"] == 0.5
    assert result["selective_risk"] == 0.0
    assert result["per_family"]["a"]["accepted_precision"] == 1.0

    none = evaluate_selective_predictions(
        records, np.asarray([False, False, False, False]), LABELS
    )
    assert none["accepted_precision"] is None
    assert none["selective_risk"] is None
    assert none["coverage"] == 0.0


def test_group_bootstrap_is_reproducible() -> None:
    records = _records(
        ("a", "b", "a", "b"),
        ((0.8, 0.2), (0.3, 0.7), (0.6, 0.4), (0.6, 0.4)),
        ("g1", "g1", "g2", "g2"),
    )
    first = group_bootstrap_interval(records, LABELS, "macro_f1", 100, 7)
    second = group_bootstrap_interval(records, LABELS, "macro_f1", 100, 7)
    assert first == second
    assert first["resampling_unit"] == "semantic_prompt_group"
    assert math.isfinite(first["estimate"])


def test_group_bootstrap_rejects_single_group() -> None:
    records = _records(("a", "b"), ((0.8, 0.2), (0.2, 0.8)), ("g1", "g1"))
    with pytest.raises(ValueError, match="two.*group"):
        group_bootstrap_interval(records, LABELS, "accuracy", 20, 7)
