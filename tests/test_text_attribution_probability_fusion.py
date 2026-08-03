from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from research.text_attribution.diagnostic_analysis import AlignedBranchProbabilities
from research.text_attribution.probability_fusion import (
    FusionFitRequest,
    apply_probability_fusion,
    assign_group_folds,
    fit_probability_fusion,
)


def _aligned() -> AlignedBranchProbabilities:
    labels = ("a", "b")
    keys = []
    truth = []
    groups = []
    a4 = []
    b0 = []
    b1 = []
    for group_index in range(10):
        for label_index, label in enumerate(labels):
            keys.append(f"g{group_index}-{label}")
            truth.append(label)
            groups.append(f"g{group_index}")
            if group_index % 2 == 0:
                a4.append((0.65, 0.35) if label_index == 0 else (0.55, 0.45))
                b0.append((0.85, 0.15) if label_index == 0 else (0.15, 0.85))
            else:
                a4.append((0.75, 0.25) if label_index == 0 else (0.25, 0.75))
                b0.append((0.55, 0.45) if label_index == 0 else (0.45, 0.55))
            b1.append((0.6, 0.4) if label_index == 0 else (0.4, 0.6))
    return AlignedBranchProbabilities(
        request_keys=tuple(keys),
        true_families=tuple(truth),
        semantic_prompt_groups=tuple(groups),
        family_labels=labels,
        records_by_branch=MappingProxyType({}),
        probabilities_by_branch=MappingProxyType(
            {
                "A4": np.asarray(a4, dtype=float),
                "B0": np.asarray(b0, dtype=float),
                "B1": np.asarray(b1, dtype=float),
            }
        ),
    )


def test_group_fold_assignment_is_deterministic_and_has_no_leakage() -> None:
    aligned = _aligned()
    first = assign_group_folds(
        aligned.semantic_prompt_groups,
        aligned.true_families,
        folds=5,
        seed=7,
        family_labels=aligned.family_labels,
    )
    second = assign_group_folds(
        aligned.semantic_prompt_groups,
        aligned.true_families,
        folds=5,
        seed=7,
        family_labels=aligned.family_labels,
    )
    assert np.array_equal(first, second)
    assert set(first.tolist()) == {0, 1, 2, 3, 4}
    for group in set(aligned.semantic_prompt_groups):
        indexes = [
            index
            for index, value in enumerate(aligned.semantic_prompt_groups)
            if value == group
        ]
        assert len(set(first[index] for index in indexes)) == 1


def test_fusion_returns_complete_oof_and_serializable_artifact() -> None:
    aligned = _aligned()
    result = fit_probability_fusion(
        FusionFitRequest(
            aligned=aligned,
            folds=5,
            fold_seed=7,
            weight_step=0.05,
            max_macro_f1_drop=0.005,
            config_hash="config-hash",
        )
    )
    assert result.oof_probabilities.shape == (len(aligned.request_keys), 2)
    assert np.allclose(result.oof_probabilities.sum(axis=1), 1.0)
    assert set(result.fold_artifacts) == {0, 1, 2, 3, 4}
    assert abs(sum(result.final_artifact.weights.values()) - 1.0) < 1e-9
    assert set(result.final_artifact.weights) == {"A4", "B0", "B1"}
    replay = apply_probability_fusion(
        result.final_artifact, aligned.probabilities_by_branch
    )
    assert replay.shape == result.oof_probabilities.shape
    assert result.final_artifact.to_dict()["schema_version"] == 1


def test_fold_assignment_rejects_too_few_groups() -> None:
    with pytest.raises(ValueError, match="groups"):
        assign_group_folds(
            ("g1", "g1"),
            ("a", "b"),
            folds=5,
            seed=7,
            family_labels=("a", "b"),
        )
