"""Strict Phase3 prediction loading, alignment, and complementarity analysis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score

from research.text_attribution.attribution_metrics import evaluate_family_predictions
from research.text_attribution.attribution_types import FamilyPrediction


@dataclass(frozen=True)
class ExpectedRun:
    branch_id: str
    seed: int
    view_id: str
    recipe_hash: str | None = None


@dataclass(frozen=True)
class RunIdentity:
    branch_id: str
    seed: int
    view_id: str
    dataset_fingerprint: str
    split_manifest_hash: str
    catalog_hash: str
    recipe_hash: str
    run_fingerprint: str
    test_evaluated: bool
    smoke_test: bool
    preflight_only: bool


@dataclass(frozen=True)
class PredictionRun:
    identity: RunIdentity
    family_labels: tuple[str, ...]
    records: tuple[FamilyPrediction, ...]
    run_dir: Path


@dataclass(frozen=True)
class AlignedBranchProbabilities:
    request_keys: tuple[str, ...]
    true_families: tuple[str, ...]
    semantic_prompt_groups: tuple[str, ...]
    family_labels: tuple[str, ...]
    records_by_branch: Mapping[str, tuple[FamilyPrediction, ...]]
    probabilities_by_branch: Mapping[str, np.ndarray]


def load_prediction_run(run_dir: str | Path, expected: ExpectedRun) -> PredictionRun:
    """Load one completed formal validation run and enforce its identity."""

    root = Path(run_dir)
    required = (
        "run_config.json",
        "resolved_recipe.json",
        "label_maps.json",
        "metrics.json",
        "predictions.jsonl",
        "COMPLETED.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"prediction run missing required files: {missing!r}")
    config = _read_json(root / "run_config.json")
    recipe = _read_json(root / "resolved_recipe.json")
    labels_payload = _read_json(root / "label_maps.json")
    metrics = _read_json(root / "metrics.json")
    completed = _read_json(root / "COMPLETED.json")
    if completed.get("status") != "complete":
        raise ValueError("COMPLETED.json status must be complete")
    if any(str(key).lower() == "test" for key in metrics):
        raise ValueError("metrics.json must not contain test metrics")

    branch = config.get("branch_id", config.get("stage_id"))
    if branch != expected.branch_id:
        raise ValueError(f"branch mismatch: expected {expected.branch_id!r}, got {branch!r}")
    seed = config.get("training_seed", config.get("seed"))
    if seed != expected.seed:
        raise ValueError(f"seed mismatch: expected {expected.seed}, got {seed!r}")
    if config.get("view_id") != expected.view_id:
        raise ValueError("view_id mismatch")
    for field in ("test_evaluated", "smoke_test", "preflight_only"):
        value = config.get(field)
        if value is not False:
            raise ValueError(f"{field} must be false for formal Phase4 input")
    recipe_hash = recipe.get("recipe_hash")
    if not isinstance(recipe_hash, str) or not recipe_hash:
        raise ValueError("resolved_recipe.json requires recipe_hash")
    if expected.recipe_hash is not None and recipe_hash != expected.recipe_hash:
        raise ValueError("recipe hash mismatch")

    raw_labels = labels_payload.get("family_labels")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("label_maps.json requires family_labels")
    labels = tuple(str(label) for label in raw_labels)
    if len(labels) != len(set(labels)):
        raise ValueError("family label order must be unique")
    records = _read_predictions(root / "predictions.jsonl")
    if any(row.split != "validation" for row in records):
        raise ValueError("predictions must be validation-only")
    keys = [row.request_key for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate request_key in predictions")
    evaluate_family_predictions(records, labels)

    identity = RunIdentity(
        branch_id=expected.branch_id,
        seed=expected.seed,
        view_id=expected.view_id,
        dataset_fingerprint=_required_text(
            config, "split_dataset_fingerprint", "dataset fingerprint"
        ),
        split_manifest_hash=_required_text(
            config, "split_manifest_hash", "split manifest hash"
        ),
        catalog_hash=_required_text(config, "catalog_hash", "catalog hash"),
        recipe_hash=recipe_hash,
        run_fingerprint=_required_text(
            config, "run_fingerprint", "run fingerprint"
        ),
        test_evaluated=False,
        smoke_test=False,
        preflight_only=False,
    )
    return PredictionRun(identity, labels, records, root.resolve())


def align_prediction_runs(
    runs: Sequence[PredictionRun],
) -> AlignedBranchProbabilities:
    """Align complete branches by request key without intersection truncation."""

    values = tuple(runs)
    if len(values) < 2:
        raise ValueError("alignment requires at least two branches")
    branch_ids = [run.identity.branch_id for run in values]
    if len(branch_ids) != len(set(branch_ids)):
        raise ValueError("branch IDs must be unique")
    reference = values[0]
    reference_map = {row.request_key: row for row in reference.records}
    reference_keys = tuple(sorted(reference_map))
    records_by_branch: dict[str, tuple[FamilyPrediction, ...]] = {}
    probabilities_by_branch: dict[str, np.ndarray] = {}
    for run in values:
        _same_identity(reference.identity, run.identity)
        if run.family_labels != reference.family_labels:
            raise ValueError("family label order mismatch")
        indexed = {row.request_key: row for row in run.records}
        if set(indexed) != set(reference_keys):
            raise ValueError("request_key sets must match exactly")
        ordered = tuple(indexed[key] for key in reference_keys)
        for key, left, right in zip(
            reference_keys,
            (reference_map[key] for key in reference_keys),
            ordered,
            strict=True,
        ):
            if (
                left.true_family,
                left.model_id,
                left.prompt_id,
                left.semantic_prompt_group,
                left.language,
                left.task_type,
                left.target_length,
                left.temperature,
            ) != (
                right.true_family,
                right.model_id,
                right.prompt_id,
                right.semantic_prompt_group,
                right.language,
                right.task_type,
                right.target_length,
                right.temperature,
            ):
                raise ValueError(f"prediction metadata mismatch for request_key {key!r}")
        records_by_branch[run.identity.branch_id] = ordered
        probabilities_by_branch[run.identity.branch_id] = np.asarray(
            [
                [row.family_probabilities[label] for label in reference.family_labels]
                for row in ordered
            ],
            dtype=float,
        )
    ordered_reference = tuple(reference_map[key] for key in reference_keys)
    return AlignedBranchProbabilities(
        request_keys=reference_keys,
        true_families=tuple(row.true_family for row in ordered_reference),
        semantic_prompt_groups=tuple(
            row.semantic_prompt_group for row in ordered_reference
        ),
        family_labels=reference.family_labels,
        records_by_branch=MappingProxyType(records_by_branch),
        probabilities_by_branch=MappingProxyType(probabilities_by_branch),
    )


def analyze_complementarity(
    aligned: AlignedBranchProbabilities, branch_a: str, branch_b: str
) -> dict[str, Any]:
    """Describe observed pairwise error overlap and a non-deployable oracle."""

    labels = aligned.family_labels
    truth = np.asarray([labels.index(value) for value in aligned.true_families])
    first = aligned.probabilities_by_branch[branch_a]
    second = aligned.probabilities_by_branch[branch_b]
    first_predicted = np.argmax(first, axis=1)
    second_predicted = np.argmax(second, axis=1)
    first_correct = first_predicted == truth
    second_correct = second_predicted == truth
    average = (first + second) / 2.0
    average_predicted = np.argmax(average, axis=1)

    epsilon = 1e-12
    midpoint = (first + second) / 2.0
    js_rows = 0.5 * np.sum(
        first * np.log((first + epsilon) / (midpoint + epsilon)), axis=1
    ) + 0.5 * np.sum(
        second * np.log((second + epsilon) / (midpoint + epsilon)), axis=1
    )
    per_family = {}
    for index, label in enumerate(labels):
        family_mask = truth == index
        a_errors = family_mask & ~first_correct
        b_errors = family_mask & ~second_correct
        per_family[label] = {
            "a_error_count": int(a_errors.sum()),
            "b_error_count": int(b_errors.sum()),
            "a_errors_corrected_by_b": int((a_errors & second_correct).sum()),
            "b_errors_corrected_by_a": int((b_errors & first_correct).sum()),
        }
    return {
        "branch_a": branch_a,
        "branch_b": branch_b,
        "sample_count": len(truth),
        "both_correct": int((first_correct & second_correct).sum()),
        "a_only_correct": int((first_correct & ~second_correct).sum()),
        "b_only_correct": int((~first_correct & second_correct).sum()),
        "both_wrong": int((~first_correct & ~second_correct).sum()),
        "label_disagreement_rate": float(np.mean(first_predicted != second_predicted)),
        "mean_js_divergence": float(np.mean(js_rows)),
        "a_error_correction_rate_by_b": _safe_rate(
            int((~first_correct & second_correct).sum()), int((~first_correct).sum())
        ),
        "b_error_correction_rate_by_a": _safe_rate(
            int((first_correct & ~second_correct).sum()), int((~second_correct).sum())
        ),
        "oracle_top1_accuracy": float(np.mean(first_correct | second_correct)),
        "equal_weight_top1_accuracy": float(np.mean(average_predicted == truth)),
        "equal_weight_macro_f1": float(
            f1_score(
                truth,
                average_predicted,
                labels=np.arange(len(labels)),
                average="macro",
                zero_division=0,
            )
        ),
        "per_family": per_family,
    }


def _same_identity(reference: RunIdentity, candidate: RunIdentity) -> None:
    if reference.seed != candidate.seed or reference.view_id != candidate.view_id:
        raise ValueError("seed/view identity mismatch")
    if reference.dataset_fingerprint != candidate.dataset_fingerprint:
        raise ValueError("dataset fingerprint mismatch")
    if reference.split_manifest_hash != candidate.split_manifest_hash:
        raise ValueError("split manifest hash mismatch")
    if reference.catalog_hash != candidate.catalog_hash:
        raise ValueError("catalog hash mismatch")


def _read_predictions(path: Path) -> tuple[FamilyPrediction, ...]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            rows.append(FamilyPrediction(**payload))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction row at line {line_number}") from exc
    if not rows:
        raise ValueError("predictions.jsonl must be nonempty")
    return tuple(rows)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _required_text(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run_config requires {label}")
    return value


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)
