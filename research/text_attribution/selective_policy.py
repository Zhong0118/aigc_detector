"""Cross-fitted selective attribution and runtime abstention policy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from research.text_attribution.attribution_types import FamilyPrediction
from research.text_attribution.phase4_metrics import evaluate_selective_predictions


_BRANCH_ORDER = ("A4", "B0", "B1")
_PRIMARY_QUANTILES = np.linspace(0.0, 1.0, 21)
_SECONDARY_QUANTILES = np.linspace(0.0, 1.0, 5)
_EPSILON = 1e-12


@dataclass(frozen=True)
class SelectiveFitRequest:
    records: tuple[FamilyPrediction, ...]
    branch_probabilities: Mapping[str, np.ndarray]
    fold_ids: np.ndarray
    family_labels: tuple[str, ...]
    accepted_precision_min: float
    coverage_min: float
    ece_max: float
    config_hash: str


@dataclass(frozen=True)
class SelectivePolicyArtifact:
    schema_version: int
    rule_family: str
    family_labels: tuple[str, ...]
    branch_order: tuple[str, ...]
    max_probability_min: float
    margin_min: float | None
    entropy_max: float | None
    js_max: float | None
    disagreement_max: int | None
    min_effective_characters: int
    deployment_ready: bool
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rule_family": self.rule_family,
            "family_labels": list(self.family_labels),
            "branch_order": list(self.branch_order),
            "max_probability_min": self.max_probability_min,
            "margin_min": self.margin_min,
            "entropy_max": self.entropy_max,
            "js_max": self.js_max,
            "disagreement_max": self.disagreement_max,
            "min_effective_characters": self.min_effective_characters,
            "deployment_ready": self.deployment_ready,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True)
class SelectiveCrossFitResult:
    oof_accepted_mask: np.ndarray
    oof_metrics: Mapping[str, Any]
    fold_artifacts: Mapping[int, SelectivePolicyArtifact]
    final_artifact: SelectivePolicyArtifact


@dataclass(frozen=True)
class AttributionEvidence:
    fused_probabilities: tuple[float, ...]
    branch_probabilities: Mapping[str, tuple[float, ...]]
    effective_character_count: int


@dataclass(frozen=True)
class AttributionDecision:
    schema_version: int
    status: str
    top_families: tuple[tuple[str, float], ...]
    confidence: float | None
    prediction_set: tuple[str, ...]
    known_probability: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Features:
    maximum: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    js: np.ndarray
    disagreement: np.ndarray


def fit_selective_policy(request: SelectiveFitRequest) -> SelectiveCrossFitResult:
    """Cross-fit threshold selection and fit one final validation artifact."""

    _validate_request(request)
    fused = np.asarray(
        [
            [row.family_probabilities[label] for label in request.family_labels]
            for row in request.records
        ],
        dtype=float,
    )
    features = _evidence_features(fused, request.branch_probabilities)
    fold_values = tuple(sorted(set(np.asarray(request.fold_ids, dtype=int).tolist())))
    if fold_values != (0, 1, 2, 3, 4):
        raise ValueError("selective policy requires fold IDs 0 through 4")
    oof_mask = np.zeros(len(request.records), dtype=bool)
    fold_artifacts: dict[int, SelectivePolicyArtifact] = {}
    for fold in fold_values:
        training = np.flatnonzero(request.fold_ids != fold)
        held_out = np.flatnonzero(request.fold_ids == fold)
        artifact = _fit_artifact(request, features, training)
        fold_artifacts[fold] = artifact
        oof_mask[held_out] = _apply_artifact(artifact, features, held_out)
    oof_metrics = evaluate_selective_predictions(
        request.records, oof_mask, request.family_labels
    )
    final_artifact = _fit_artifact(
        request, features, np.arange(len(request.records), dtype=int)
    )
    return SelectiveCrossFitResult(
        oof_accepted_mask=oof_mask,
        oof_metrics=MappingProxyType(dict(oof_metrics)),
        fold_artifacts=MappingProxyType(fold_artifacts),
        final_artifact=final_artifact,
    )


def decide_attribution(
    artifact: SelectivePolicyArtifact, evidence: AttributionEvidence
) -> AttributionDecision:
    """Return a calibrated family set or abstain; Phase4 never returns unknown."""

    try:
        if not artifact.deployment_ready:
            return _abstain("policy_not_deployable")
        fused = np.asarray([evidence.fused_probabilities], dtype=float)
        branches = {
            name: np.asarray([evidence.branch_probabilities[name]], dtype=float)
            for name in artifact.branch_order
        }
        if artifact.schema_version != 1 or artifact.branch_order != _BRANCH_ORDER:
            raise ValueError("artifact contract mismatch")
        _validate_probability_matrix(fused, len(artifact.family_labels))
        for values in branches.values():
            _validate_probability_matrix(values, len(artifact.family_labels))
        features = _evidence_features(fused, branches)
        accepted = bool(_apply_artifact(artifact, features, np.asarray([0]))[0])
    except (AttributeError, KeyError, TypeError, ValueError):
        return _abstain("artifact_mismatch")
    if evidence.effective_character_count < artifact.min_effective_characters:
        return _abstain("short_text")
    if not accepted:
        if (
            artifact.disagreement_max is not None
            and int(features.disagreement[0]) > artifact.disagreement_max
        ):
            return _abstain("branch_conflict")
        return _abstain("low_confidence")

    probabilities = fused[0]
    order = np.argsort(-probabilities, kind="stable")
    top = tuple(
        (artifact.family_labels[index], float(probabilities[index]))
        for index in order[: min(3, len(order))]
    )
    return AttributionDecision(
        schema_version=1,
        status="family_only",
        top_families=top,
        confidence=top[0][1],
        prediction_set=(top[0][0],),
        known_probability=None,
        warnings=("single_text_evidence", "unseen_version_risk"),
    )


def _fit_artifact(
    request: SelectiveFitRequest, features: _Features, indexes: np.ndarray
) -> SelectivePolicyArtifact:
    candidates = _candidate_artifacts(request, features, indexes)
    rows = tuple(request.records[index] for index in indexes)
    ranked = []
    for candidate in candidates:
        mask = _apply_artifact(candidate, features, indexes)
        metrics = evaluate_selective_predictions(rows, mask, request.family_labels)
        precision = metrics["accepted_precision"]
        coverage = float(metrics["coverage"])
        family_recalls = [
            value["accepted_recall"]
            for value in metrics["per_family"].values()
            if value["accepted_recall"] is not None
        ]
        worst_recall = min(family_recalls) if family_recalls else 0.0
        ece = metrics["ece"] if metrics["ece"] is not None else float("inf")
        feasible = (
            precision is not None
            and precision >= request.accepted_precision_min
            and coverage >= request.coverage_min
        )
        if feasible:
            key = (
                0,
                -coverage,
                -worst_recall,
                float(ece),
                _artifact_key(candidate),
            )
        else:
            risk = 1.0 if precision is None else 1.0 - float(precision)
            key = (1, risk, -coverage, float(ece), _artifact_key(candidate))
        ranked.append((key, candidate, metrics))
    _, selected, metrics = min(ranked, key=lambda item: item[0])
    deployment_ready = bool(
        metrics["accepted_precision"] is not None
        and metrics["accepted_precision"] >= request.accepted_precision_min
        and metrics["coverage"] >= request.coverage_min
        and metrics["ece"] is not None
        and metrics["ece"] <= request.ece_max
    )
    return SelectivePolicyArtifact(
        **{**selected.__dict__, "deployment_ready": deployment_ready}
    )


def _candidate_artifacts(
    request: SelectiveFitRequest, features: _Features, indexes: np.ndarray
) -> tuple[SelectivePolicyArtifact, ...]:
    maximums = _quantiles(features.maximum[indexes], _PRIMARY_QUANTILES)
    margin_values = _quantiles(features.margin[indexes], _SECONDARY_QUANTILES)
    entropy_values = _quantiles(features.entropy[indexes], _SECONDARY_QUANTILES)
    js_values = _quantiles(features.js[indexes], _SECONDARY_QUANTILES)

    def artifact(
        rule_family: str,
        maximum: float,
        *,
        margin: float | None = None,
        entropy: float | None = None,
        js: float | None = None,
        disagreement: int | None = None,
    ) -> SelectivePolicyArtifact:
        return SelectivePolicyArtifact(
            schema_version=1,
            rule_family=rule_family,
            family_labels=request.family_labels,
            branch_order=_BRANCH_ORDER,
            max_probability_min=maximum,
            margin_min=margin,
            entropy_max=entropy,
            js_max=js,
            disagreement_max=disagreement,
            min_effective_characters=32,
            deployment_ready=False,
            config_hash=request.config_hash,
        )

    candidates = [artifact("confidence-only", maximum) for maximum in maximums]
    candidates.extend(
        artifact("confidence-plus-margin", maximum, margin=secondary)
        for maximum in maximums
        for secondary in margin_values
    )
    candidates.extend(
        artifact("confidence-plus-entropy", maximum, entropy=secondary)
        for maximum in maximums
        for secondary in entropy_values
    )
    candidates.extend(
        artifact("confidence-plus-js", maximum, js=secondary)
        for maximum in maximums
        for secondary in js_values
    )
    candidates.extend(
        artifact("confidence-plus-disagreement", maximum, disagreement=secondary)
        for maximum in maximums
        for secondary in (0, 1, 2)
    )
    unique = {_artifact_key(value): value for value in candidates}
    return tuple(unique[key] for key in sorted(unique))


def _apply_artifact(
    artifact: SelectivePolicyArtifact, features: _Features, indexes: np.ndarray
) -> np.ndarray:
    accepted = features.maximum[indexes] >= artifact.max_probability_min
    if artifact.margin_min is not None:
        accepted &= features.margin[indexes] >= artifact.margin_min
    if artifact.entropy_max is not None:
        accepted &= features.entropy[indexes] <= artifact.entropy_max
    if artifact.js_max is not None:
        accepted &= features.js[indexes] <= artifact.js_max
    if artifact.disagreement_max is not None:
        accepted &= features.disagreement[indexes] <= artifact.disagreement_max
    return accepted


def _evidence_features(
    fused: np.ndarray, branch_probabilities: Mapping[str, np.ndarray]
) -> _Features:
    _validate_probability_matrix(fused, fused.shape[1])
    if tuple(branch_probabilities) != _BRANCH_ORDER:
        raise ValueError("branch probabilities must use A4, B0, B1 order")
    branches = [np.asarray(branch_probabilities[name], dtype=float) for name in _BRANCH_ORDER]
    for values in branches:
        _validate_probability_matrix(values, fused.shape[1])
        if values.shape != fused.shape:
            raise ValueError("branch and fused probability shapes must match")
    sorted_values = np.sort(fused, axis=1)
    maximum = sorted_values[:, -1]
    margin = sorted_values[:, -1] - sorted_values[:, -2]
    entropy = -np.sum(fused * np.log(np.clip(fused, _EPSILON, 1.0)), axis=1)
    entropy /= np.log(fused.shape[1])
    predictions = np.stack([np.argmax(values, axis=1) for values in branches], axis=1)
    disagreement = np.asarray(
        [len(set(row.tolist())) - 1 for row in predictions], dtype=int
    )
    js_values = []
    for first, second in combinations(branches, 2):
        midpoint = (first + second) / 2.0
        js_values.append(
            0.5
            * np.sum(first * np.log((first + _EPSILON) / (midpoint + _EPSILON)), axis=1)
            + 0.5
            * np.sum(second * np.log((second + _EPSILON) / (midpoint + _EPSILON)), axis=1)
        )
    return _Features(maximum, margin, entropy, np.mean(js_values, axis=0), disagreement)


def _validate_request(request: SelectiveFitRequest) -> None:
    if not request.records:
        raise ValueError("selective request records must be nonempty")
    if tuple(request.branch_probabilities) != _BRANCH_ORDER:
        raise ValueError("selective request branch order mismatch")
    if np.asarray(request.fold_ids).shape != (len(request.records),):
        raise ValueError("fold_ids must align with records")
    for row in request.records:
        if tuple(row.family_probabilities) != request.family_labels:
            raise ValueError("record family probability order mismatch")


def _validate_probability_matrix(values: np.ndarray, label_count: int) -> None:
    if (
        values.ndim != 2
        or values.shape[1] != label_count
        or label_count < 2
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("invalid probability matrix")


def _quantiles(values: np.ndarray, quantiles: np.ndarray) -> tuple[float, ...]:
    return tuple(sorted(set(float(value) for value in np.quantile(values, quantiles))))


def _artifact_key(artifact: SelectivePolicyArtifact) -> tuple[Any, ...]:
    return (
        artifact.rule_family,
        artifact.max_probability_min,
        -1.0 if artifact.margin_min is None else artifact.margin_min,
        -1.0 if artifact.entropy_max is None else artifact.entropy_max,
        -1.0 if artifact.js_max is None else artifact.js_max,
        -1 if artifact.disagreement_max is None else artifact.disagreement_max,
    )


def _abstain(warning: str) -> AttributionDecision:
    return AttributionDecision(
        schema_version=1,
        status="abstain",
        top_families=(),
        confidence=None,
        prediction_set=(),
        known_probability=None,
        warnings=(warning,),
    )
