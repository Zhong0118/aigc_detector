"""Leakage-safe temperature calibration and probability fusion."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from math import exp
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from research.text_attribution.diagnostic_analysis import AlignedBranchProbabilities


_BRANCH_ORDER = ("A4", "B0", "B1")
_EPSILON = 1e-12


@dataclass(frozen=True)
class FusionFitRequest:
    aligned: AlignedBranchProbabilities
    folds: int
    fold_seed: int
    weight_step: float
    max_macro_f1_drop: float
    config_hash: str


@dataclass(frozen=True)
class FusionArtifact:
    schema_version: int
    branch_order: tuple[str, ...]
    temperatures: Mapping[str, float]
    weights: Mapping[str, float]
    family_labels: tuple[str, ...]
    fit_request_keys_hash: str
    config_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "temperatures", MappingProxyType(dict(self.temperatures))
        )
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "branch_order": list(self.branch_order),
            "temperatures": dict(self.temperatures),
            "weights": dict(self.weights),
            "family_labels": list(self.family_labels),
            "fit_request_keys_hash": self.fit_request_keys_hash,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True)
class CrossFitFusionResult:
    oof_probabilities: np.ndarray
    fold_ids: np.ndarray
    fold_artifacts: Mapping[int, FusionArtifact]
    final_artifact: FusionArtifact


def assign_group_folds(
    semantic_prompt_groups: Sequence[str],
    true_families: Sequence[str],
    *,
    folds: int,
    seed: int,
    family_labels: Sequence[str],
) -> np.ndarray:
    """Assign whole semantic groups to balanced deterministic folds."""

    groups = tuple(semantic_prompt_groups)
    truth = tuple(true_families)
    labels = tuple(family_labels)
    if len(groups) != len(truth) or not groups:
        raise ValueError("groups and true_families must be aligned and nonempty")
    if folds <= 1:
        raise ValueError("folds must be greater than one")
    unique_groups = tuple(sorted(set(groups)))
    if len(unique_groups) < folds:
        raise ValueError("not enough semantic groups for requested folds")
    label_to_index = {label: index for index, label in enumerate(labels)}
    if len(label_to_index) != len(labels) or any(value not in label_to_index for value in truth):
        raise ValueError("family labels must uniquely cover true_families")

    indexes_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        indexes_by_group[group].append(index)
    group_counts = {
        group: np.bincount(
            [label_to_index[truth[index]] for index in indexes], minlength=len(labels)
        )
        for group, indexes in indexes_by_group.items()
    }
    ordered_groups = sorted(
        unique_groups,
        key=lambda group: hashlib.sha256(
            f"{seed}:{group}:{group_counts[group].tolist()}".encode("utf-8")
        ).hexdigest(),
    )
    fold_counts = np.zeros((folds, len(labels)), dtype=int)
    fold_sizes = np.zeros(folds, dtype=int)
    group_fold: dict[str, int] = {}
    for group in ordered_groups:
        counts = group_counts[group]
        candidates = []
        for fold in range(folds):
            projected = fold_counts.copy()
            projected[fold] += counts
            family_imbalance = int(np.sum(np.ptp(projected, axis=0)))
            candidates.append(
                (family_imbalance, int(fold_sizes[fold]), fold)
            )
        selected = min(candidates)[-1]
        group_fold[group] = selected
        fold_counts[selected] += counts
        fold_sizes[selected] += len(indexes_by_group[group])
    result = np.asarray([group_fold[group] for group in groups], dtype=int)
    for fold in range(folds):
        held_out = result == fold
        training = ~held_out
        if not held_out.any() or not training.any():
            raise ValueError("group fold assignment produced an empty fold")
        for mask, scope in ((held_out, "held-out"), (training, "training")):
            present = {truth[index] for index in np.flatnonzero(mask)}
            if present != set(labels):
                raise ValueError(f"{scope} fold {fold} does not contain every family")
    return result


def fit_probability_fusion(request: FusionFitRequest) -> CrossFitFusionResult:
    """Generate honest OOF fusion predictions and a final validation artifact."""

    aligned = request.aligned
    _validate_aligned(aligned)
    if request.folds != 5:
        raise ValueError("Fast Track probability fusion requires five folds")
    if not 0.0 < request.weight_step <= 1.0:
        raise ValueError("weight_step must be in (0, 1]")
    fold_ids = assign_group_folds(
        aligned.semantic_prompt_groups,
        aligned.true_families,
        folds=request.folds,
        seed=request.fold_seed,
        family_labels=aligned.family_labels,
    )
    true_indexes = np.asarray(
        [aligned.family_labels.index(value) for value in aligned.true_families],
        dtype=int,
    )
    oof = np.full(
        (len(aligned.request_keys), len(aligned.family_labels)), np.nan, dtype=float
    )
    fold_artifacts: dict[int, FusionArtifact] = {}
    for fold in range(request.folds):
        train_indexes = np.flatnonzero(fold_ids != fold)
        validation_indexes = np.flatnonzero(fold_ids == fold)
        artifact = _fit_artifact(
            aligned,
            train_indexes,
            true_indexes[train_indexes],
            request,
        )
        fold_artifacts[fold] = artifact
        branch_values = {
            branch: values[validation_indexes]
            for branch, values in aligned.probabilities_by_branch.items()
        }
        oof[validation_indexes] = apply_probability_fusion(artifact, branch_values)
    if not np.isfinite(oof).all() or not np.allclose(oof.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("cross-fit fusion did not fill every OOF probability row")
    final_artifact = _fit_artifact(
        aligned,
        np.arange(len(aligned.request_keys)),
        true_indexes,
        request,
    )
    return CrossFitFusionResult(
        oof_probabilities=oof,
        fold_ids=fold_ids,
        fold_artifacts=MappingProxyType(fold_artifacts),
        final_artifact=final_artifact,
    )


def apply_probability_fusion(
    artifact: FusionArtifact,
    branches: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Apply a validated calibration-and-weight artifact."""

    if tuple(artifact.branch_order) != _BRANCH_ORDER:
        raise ValueError("unsupported fusion branch order")
    if set(branches) != set(artifact.branch_order):
        raise ValueError("branch probabilities do not match fusion artifact")
    calibrated = []
    shape = None
    for branch in artifact.branch_order:
        values = np.asarray(branches[branch], dtype=float)
        if values.ndim != 2 or values.shape[1] != len(artifact.family_labels):
            raise ValueError("branch probability shape mismatch")
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            raise ValueError("branch probability row counts must match")
        _validate_probabilities(values)
        calibrated.append(_apply_temperature(values, artifact.temperatures[branch]))
    fused = sum(
        artifact.weights[branch] * values
        for branch, values in zip(artifact.branch_order, calibrated, strict=True)
    )
    return fused / fused.sum(axis=1, keepdims=True)


def _fit_artifact(
    aligned: AlignedBranchProbabilities,
    indexes: np.ndarray,
    truth: np.ndarray,
    request: FusionFitRequest,
) -> FusionArtifact:
    calibrated: dict[str, np.ndarray] = {}
    temperatures: dict[str, float] = {}
    single_macro: list[float] = []
    for branch in _BRANCH_ORDER:
        values = aligned.probabilities_by_branch[branch][indexes]
        temperature = _fit_temperature(values, truth)
        temperatures[branch] = temperature
        calibrated[branch] = _apply_temperature(values, temperature)
        single_macro.append(_macro_f1(calibrated[branch], truth, len(aligned.family_labels)))
    best_single = max(single_macro)
    units = round(1.0 / request.weight_step)
    if abs(units * request.weight_step - 1.0) > 1e-9:
        raise ValueError("weight_step must divide one exactly")
    candidates = []
    for first in range(units + 1):
        for second in range(units - first + 1):
            third = units - first - second
            weights = np.asarray([first, second, third], dtype=float) / units
            fused = sum(
                weights[position] * calibrated[branch]
                for position, branch in enumerate(_BRANCH_ORDER)
            )
            macro = _macro_f1(fused, truth, len(aligned.family_labels))
            if macro < best_single - request.max_macro_f1_drop:
                continue
            predicted = np.argmax(fused, axis=1)
            family_scores = f1_score(
                truth,
                predicted,
                labels=np.arange(len(aligned.family_labels)),
                average=None,
                zero_division=0,
            )
            nll = _nll(fused, truth)
            weight_map = dict(zip(_BRANCH_ORDER, weights.tolist(), strict=True))
            key = (
                round(nll, 6),
                -float(np.min(family_scores)),
                int(np.count_nonzero(weights)),
                -weight_map["A4"],
                tuple(weight_map[name] for name in _BRANCH_ORDER),
            )
            candidates.append((key, weight_map))
    if not candidates:
        raise RuntimeError("no fusion weight candidate passed the Macro-F1 guardrail")
    weights = min(candidates, key=lambda item: item[0])[1]
    fit_keys = tuple(aligned.request_keys[index] for index in indexes)
    return FusionArtifact(
        schema_version=1,
        branch_order=_BRANCH_ORDER,
        temperatures=temperatures,
        weights=weights,
        family_labels=aligned.family_labels,
        fit_request_keys_hash=_hash_strings(fit_keys),
        config_hash=request.config_hash,
    )


def _fit_temperature(probabilities: np.ndarray, truth: np.ndarray) -> float:
    pseudo_logits = np.log(np.clip(probabilities, _EPSILON, 1.0))

    def objective(log_temperature: float) -> float:
        temperature = exp(float(log_temperature))
        values = _softmax(pseudo_logits / temperature)
        return _nll(values, truth)

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("temperature calibration did not converge")
    temperature = exp(float(result.x))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError("temperature calibration produced an invalid value")
    return temperature


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return _softmax(np.log(np.clip(probabilities, _EPSILON, 1.0)) / temperature)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _nll(probabilities: np.ndarray, truth: np.ndarray) -> float:
    return float(
        -np.mean(np.log(np.clip(probabilities[np.arange(len(truth)), truth], _EPSILON, 1.0)))
    )


def _macro_f1(probabilities: np.ndarray, truth: np.ndarray, label_count: int) -> float:
    return float(
        f1_score(
            truth,
            np.argmax(probabilities, axis=1),
            labels=np.arange(label_count),
            average="macro",
            zero_division=0,
        )
    )


def _validate_aligned(aligned: AlignedBranchProbabilities) -> None:
    if tuple(aligned.probabilities_by_branch) != _BRANCH_ORDER:
        raise ValueError(f"fusion branches must be ordered as {_BRANCH_ORDER!r}")
    expected_shape = (len(aligned.request_keys), len(aligned.family_labels))
    for values in aligned.probabilities_by_branch.values():
        if np.asarray(values).shape != expected_shape:
            raise ValueError("aligned branch probability shape mismatch")
        _validate_probabilities(np.asarray(values, dtype=float))


def _validate_probabilities(values: np.ndarray) -> None:
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("probabilities must be finite, bounded, and normalized")


def _hash_strings(values: Sequence[str]) -> str:
    encoded = json.dumps(tuple(values), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
