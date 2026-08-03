from __future__ import annotations

import hashlib
from pathlib import Path
import json

from research.text_attribution.diagnostic_analysis import RunIdentity
from research.text_attribution.phase3_data import build_phase3_data_view, load_variant_catalog
from research.text_attribution.phase4_classical import (
    ClassicalBranchRequest,
    train_aligned_classical_branches,
)
from research.text_attribution.split_manifest import (
    apply_split_manifest,
    create_split_manifest,
    save_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


def _request(tmp_path: Path, view_id: str = "closed") -> ClassicalBranchRequest:
    data_path = tmp_path / "data.jsonl"
    catalog_path = tmp_path / "catalog.json"
    split_path = tmp_path / "split.json"
    write_dataset(data_path, groups=20)
    write_model_catalog(catalog_path)
    dataset = load_training_dataset(data_path, catalog_path)
    manifest = create_split_manifest(dataset.samples, seed=42)
    save_split_manifest(split_path, manifest)
    splits = apply_split_manifest(dataset.samples, manifest)
    view = build_phase3_data_view(
        splits,
        load_variant_catalog(catalog_path),
        view_id,
        test_request_keys=frozenset(row.request_key for row in splits["test"]),
    )
    identity = RunIdentity(
        branch_id="A4",
        seed=42,
        view_id=view_id,
        dataset_fingerprint=manifest.dataset_fingerprint,
        split_manifest_hash=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        catalog_hash=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        recipe_hash="a4-recipe",
        run_fingerprint="a4-run",
        test_evaluated=False,
        smoke_test=False,
        preflight_only=False,
    )
    return ClassicalBranchRequest(
        data_path=data_path,
        catalog_path=catalog_path,
        split_manifest_path=split_path,
        output_root=tmp_path / "classical",
        seed=42,
        view_id=view_id,
        expected_validation_keys=tuple(
            sorted(row.request_key for row in view.validation_samples)
        ),
        test_request_keys=frozenset(row.request_key for row in splits["test"]),
        expected_identity=identity,
        persist_model=True,
    )


def test_classical_branches_match_phase3_validation_keys(tmp_path: Path) -> None:
    request = _request(tmp_path, view_id="fast-to-balanced")
    result = train_aligned_classical_branches(request)
    assert set(result.runs) == {"B0", "B1"}
    for run in result.runs.values():
        assert run.request_keys == request.expected_validation_keys
        assert not request.test_request_keys.intersection(run.request_keys)
        assert (run.run_dir / "model.joblib").is_file()
        assert (run.run_dir / "COMPLETED.json").is_file()


def test_nonproduction_seed_discards_models_but_keeps_predictions(tmp_path: Path) -> None:
    original = _request(tmp_path, view_id="balanced-to-fast")
    request = ClassicalBranchRequest(
        **{**original.__dict__, "persist_model": False}
    )
    result = train_aligned_classical_branches(request)
    for run in result.runs.values():
        assert not (run.run_dir / "model.joblib").exists()
        assert (run.run_dir / "predictions.jsonl").is_file()


def test_existing_output_root_is_rejected(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.output_root.mkdir(parents=True)
    try:
        train_aligned_classical_branches(request)
    except FileExistsError as exc:
        assert "output" in str(exc).lower()
    else:
        raise AssertionError("existing classical output root must be rejected")


def test_test_split_text_is_filtered_before_normalization(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    manifest = json.loads(request.split_manifest_path.read_text(encoding="utf-8"))
    assignments = manifest["group_assignments"]
    rows = [json.loads(line) for line in request.data_path.read_text().splitlines()]
    for row in rows:
        prompt_id = row["prompt"]["prompt_id"]
        group = prompt_id.removeprefix("zh-").removeprefix("en-")
        if assignments[group] == "test":
            row["text"] = "DO_NOT_NORMALIZE_FROZEN_TEST_TEXT"
    request.data_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    from research.text_attribution import baseline_data

    original = baseline_data.normalize_text

    def guarded(text: str) -> str:
        if "DO_NOT_NORMALIZE" in text:
            raise AssertionError("frozen test text reached normalization")
        return original(text)

    monkeypatch.setattr(baseline_data, "normalize_text", guarded)
    result = train_aligned_classical_branches(request)
    assert set(result.runs) == {"B0", "B1"}
