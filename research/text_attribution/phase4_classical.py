"""Train validation-only classical branches aligned to one Phase3 data view."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import joblib

from research.text_attribution.attribution_metrics import build_prediction_records
from research.text_attribution.classical_models import (
    build_classical_pipeline,
    predict_family_probabilities,
)
from research.text_attribution.diagnostic_analysis import RunIdentity
from research.text_attribution.phase3_data import (
    build_phase3_data_view,
    load_variant_catalog,
)
from research.text_attribution.phase4_metrics import evaluate_phase4_predictions
from research.text_attribution.split_manifest import load_split_manifest
from research.text_attribution.training_data import load_training_dataset


@dataclass(frozen=True)
class ClassicalBranchRequest:
    data_path: Path
    catalog_path: Path
    split_manifest_path: Path
    output_root: Path
    seed: int
    view_id: str
    expected_validation_keys: tuple[str, ...]
    test_request_keys: frozenset[str]
    expected_identity: RunIdentity
    persist_model: bool


@dataclass(frozen=True)
class ClassicalBranchRun:
    branch_id: str
    run_dir: Path
    request_keys: tuple[str, ...]
    sample_count: int


@dataclass(frozen=True)
class ClassicalBranchResult:
    runs: Mapping[str, ClassicalBranchRun]


def train_aligned_classical_branches(
    request: ClassicalBranchRequest,
) -> ClassicalBranchResult:
    """Fit B0/B1 on train rows and publish aligned validation predictions."""

    output_root = Path(request.output_root)
    if output_root.exists():
        raise FileExistsError(f"classical output root already exists: {output_root}")
    if request.seed != request.expected_identity.seed:
        raise ValueError("classical seed must match expected A4 identity")
    if request.view_id != request.expected_identity.view_id:
        raise ValueError("classical view must match expected A4 identity")

    manifest = load_split_manifest(request.split_manifest_path)
    if manifest.dataset_fingerprint != request.expected_identity.dataset_fingerprint:
        raise ValueError("classical dataset fingerprint does not match A4")
    if _sha256_file(request.split_manifest_path) != request.expected_identity.split_manifest_hash:
        raise ValueError("classical split manifest hash does not match A4")
    if _sha256_file(request.catalog_path) != request.expected_identity.catalog_hash:
        raise ValueError("classical catalog hash does not match A4")

    included_groups = frozenset(
        group
        for group, split_name in manifest.group_assignments.items()
        if split_name in {"train", "validation"}
    )
    dataset = load_training_dataset(
        request.data_path,
        request.catalog_path,
        included_semantic_groups=included_groups,
    )
    splits = {
        split_name: tuple(
            sample
            for sample in dataset.samples
            if manifest.group_assignments.get(sample.semantic_prompt_group) == split_name
        )
        for split_name in ("train", "validation")
    }
    present_groups = {sample.semantic_prompt_group for sample in dataset.samples}
    if present_groups != set(included_groups):
        raise ValueError("train/validation groups do not match frozen split manifest")
    view = build_phase3_data_view(
        splits,
        load_variant_catalog(request.catalog_path),
        request.view_id,
        test_request_keys=request.test_request_keys,
    )
    validation_keys = tuple(sorted(row.request_key for row in view.validation_samples))
    if validation_keys != request.expected_validation_keys:
        raise ValueError("classical validation request keys do not match A4")
    labels = tuple(view.family_to_id)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.run-", dir=output_root.parent)
    )
    published: dict[str, ClassicalBranchRun] = {}
    try:
        train_texts = [row.text for row in view.train_samples]
        train_labels = [row.family_id for row in view.train_samples]
        validation_texts = [row.text for row in view.validation_samples]
        for branch_id in ("B0", "B1"):
            branch_dir = staging / branch_id
            branch_dir.mkdir()
            model = build_classical_pipeline(branch_id, seed=request.seed)
            model.fit(train_texts, train_labels)
            probabilities = predict_family_probabilities(
                model, validation_texts, family_labels=labels
            )
            records = build_prediction_records(
                view.validation_samples,
                probabilities,
                labels,
                experiment_id=branch_id,
                split="validation",
            )
            recipe_hash = _canonical_hash(
                {"branch_id": branch_id, "seed": request.seed, "view": request.view_id}
            )
            run_fingerprint = _canonical_hash(
                {
                    "branch_id": branch_id,
                    "seed": request.seed,
                    "view_id": request.view_id,
                    "dataset_fingerprint": manifest.dataset_fingerprint,
                    "split_manifest_hash": request.expected_identity.split_manifest_hash,
                    "catalog_hash": request.expected_identity.catalog_hash,
                    "recipe_hash": recipe_hash,
                }
            )
            _write_json(
                branch_dir / "run_config.json",
                {
                    "schema_version": 1,
                    "branch_id": branch_id,
                    "stage_id": branch_id,
                    "view_id": request.view_id,
                    "training_seed": request.seed,
                    "split_dataset_fingerprint": manifest.dataset_fingerprint,
                    "split_manifest_hash": request.expected_identity.split_manifest_hash,
                    "catalog_hash": request.expected_identity.catalog_hash,
                    "run_fingerprint": run_fingerprint,
                    "test_evaluated": False,
                    "smoke_test": False,
                    "preflight_only": False,
                    "train_samples": len(view.train_samples),
                    "validation_samples": len(view.validation_samples),
                    "model_persisted": request.persist_model,
                },
            )
            _write_json(
                branch_dir / "resolved_recipe.json",
                {
                    "schema_version": 1,
                    "branch_id": branch_id,
                    "recipe_hash": recipe_hash,
                },
            )
            _write_json(
                branch_dir / "label_maps.json",
                {"schema_version": 1, "family_labels": list(labels)},
            )
            _write_json(
                branch_dir / "metrics.json",
                evaluate_phase4_predictions(records, labels),
            )
            _write_predictions(branch_dir / "predictions.jsonl", records)
            if request.persist_model:
                joblib.dump(model, branch_dir / "model.joblib")
            _write_json(
                branch_dir / "COMPLETED.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "test_evaluated": False,
                },
            )
            published[branch_id] = ClassicalBranchRun(
                branch_id=branch_id,
                run_dir=output_root / branch_id,
                request_keys=validation_keys,
                sample_count=len(records),
            )
            del model
        os.replace(staging, output_root)
        return ClassicalBranchResult(MappingProxyType(published))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_predictions(path: Path, records: tuple[Any, ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
