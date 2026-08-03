from __future__ import annotations

from types import MappingProxyType

import numpy as np

from research.text_attribution.attribution_types import FamilyPrediction
from research.text_attribution.selective_policy import (
    AttributionEvidence,
    SelectiveFitRequest,
    SelectivePolicyArtifact,
    decide_attribution,
    fit_selective_policy,
)


LABELS = ("a", "b")


def _records() -> tuple[FamilyPrediction, ...]:
    rows = []
    for index in range(20):
        actual = LABELS[index % 2]
        correct = index not in {5, 11, 17}
        if actual == "a":
            values = (0.9, 0.1) if correct else (0.45, 0.55)
        else:
            values = (0.1, 0.9) if correct else (0.55, 0.45)
        predicted = LABELS[int(values[1] > values[0])]
        rows.append(
            FamilyPrediction(
                request_key=f"r{index}",
                experiment_id="E1",
                split="validation",
                true_family=actual,
                predicted_family=predicted,
                family_probabilities=dict(zip(LABELS, values, strict=True)),
                confidence=max(values),
                model_id=f"{actual}-model",
                prompt_id=f"p{index}",
                semantic_prompt_group=f"g{index // 2}",
                language="zh",
                task_type="explanation",
                target_length="medium",
                temperature=0.0,
            )
        )
    return tuple(rows)


def _fit_request() -> SelectiveFitRequest:
    records = _records()
    fused = np.asarray(
        [[row.family_probabilities[label] for label in LABELS] for row in records]
    )
    a4 = np.clip(fused + np.asarray([0.02, -0.02]), 0.001, 0.999)
    a4 /= a4.sum(axis=1, keepdims=True)
    b0 = fused[:, ::-1] * 0.3 + fused * 0.7
    b0 /= b0.sum(axis=1, keepdims=True)
    b1 = fused.copy()
    return SelectiveFitRequest(
        records=records,
        branch_probabilities=MappingProxyType({"A4": a4, "B0": b0, "B1": b1}),
        fold_ids=np.asarray([index // 4 for index in range(20)]),
        family_labels=LABELS,
        accepted_precision_min=0.9,
        coverage_min=0.4,
        ece_max=0.2,
        config_hash="config",
    )


def test_policy_meets_precision_then_maximizes_coverage() -> None:
    result = fit_selective_policy(_fit_request())
    assert result.oof_metrics["accepted_precision"] >= 0.9
    assert result.oof_metrics["coverage"] >= 0.4
    assert result.final_artifact.max_probability_min > 0.0
    assert set(result.fold_artifacts) == {0, 1, 2, 3, 4}


def test_runtime_abstains_on_conflict_and_never_returns_unknown() -> None:
    artifact = SelectivePolicyArtifact(
        schema_version=1,
        rule_family="confidence-plus-disagreement",
        family_labels=LABELS,
        branch_order=("A4", "B0", "B1"),
        max_probability_min=0.5,
        margin_min=None,
        entropy_max=None,
        js_max=None,
        disagreement_max=0,
        min_effective_characters=32,
        deployment_ready=True,
        config_hash="config",
    )
    conflict = AttributionEvidence(
        fused_probabilities=(0.51, 0.49),
        branch_probabilities={
            "A4": (0.9, 0.1),
            "B0": (0.1, 0.9),
            "B1": (0.5, 0.5),
        },
        effective_character_count=400,
    )
    decision = decide_attribution(artifact, conflict)
    assert decision.status == "abstain"
    assert decision.known_probability is None
    assert "branch_conflict" in decision.warnings


def test_runtime_returns_family_only_for_supported_evidence() -> None:
    artifact = SelectivePolicyArtifact(
        schema_version=1,
        rule_family="confidence-only",
        family_labels=LABELS,
        branch_order=("A4", "B0", "B1"),
        max_probability_min=0.8,
        margin_min=None,
        entropy_max=None,
        js_max=None,
        disagreement_max=None,
        min_effective_characters=32,
        deployment_ready=True,
        config_hash="config",
    )
    evidence = AttributionEvidence(
        fused_probabilities=(0.9, 0.1),
        branch_probabilities={name: (0.9, 0.1) for name in artifact.branch_order},
        effective_character_count=400,
    )
    decision = decide_attribution(artifact, evidence)
    assert decision.status == "family_only"
    assert decision.top_families[0] == ("a", 0.9)
    assert decision.prediction_set == ("a",)


def test_non_deployable_artifact_always_abstains() -> None:
    artifact = SelectivePolicyArtifact(
        schema_version=1,
        rule_family="confidence-only",
        family_labels=LABELS,
        branch_order=("A4", "B0", "B1"),
        max_probability_min=0.1,
        margin_min=None,
        entropy_max=None,
        js_max=None,
        disagreement_max=None,
        min_effective_characters=1,
        deployment_ready=False,
        config_hash="config",
    )
    evidence = AttributionEvidence(
        fused_probabilities=(0.99, 0.01),
        branch_probabilities={name: (0.99, 0.01) for name in artifact.branch_order},
        effective_character_count=400,
    )
    decision = decide_attribution(artifact, evidence)
    assert decision.status == "abstain"
    assert "policy_not_deployable" in decision.warnings
