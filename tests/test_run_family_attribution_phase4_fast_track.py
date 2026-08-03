from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from research.text_attribution.attribution_metrics import build_prediction_records
from research.text_attribution.phase3_data import build_phase3_data_view, load_variant_catalog
from research.text_attribution.split_manifest import (
    apply_split_manifest,
    create_split_manifest,
    save_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset
from scripts.run_family_attribution_phase4_fast_track import main
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


A4_HASH = "1d4b21b8e2bb1477d4872049805cf6d1ca8e037ec52e2a538f851eb335dd71f2"
VIEWS = ("closed", "fast-to-balanced", "balanced-to-fast")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "data.jsonl"
    catalog = tmp_path / "catalog.json"
    split = tmp_path / "split.json"
    config = tmp_path / "config.json"
    a0_root = tmp_path / "A0"
    a4_root = tmp_path / "A4"
    write_dataset(data, groups=40)
    write_model_catalog(catalog)
    dataset = load_training_dataset(data, catalog)
    manifest = create_split_manifest(dataset.samples, seed=42)
    save_split_manifest(split, manifest)
    splits = apply_split_manifest(dataset.samples, manifest)
    catalog_view = load_variant_catalog(catalog)
    config_payload = json.loads(
        Path("configs/text_attribution/training/family_phase4_fast_track.v1.json").read_text(
            encoding="utf-8"
        )
    )
    config_payload["seeds"] = [42]
    config_payload["bootstrap_resamples"] = 20
    _write_json(config, config_payload)
    split_hash = hashlib.sha256(split.read_bytes()).hexdigest()
    catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
    test_keys = frozenset(row.request_key for row in splits["test"])

    for view_id in VIEWS:
        view = build_phase3_data_view(
            splits, catalog_view, view_id, test_request_keys=test_keys
        )
        labels = tuple(view.family_to_id)
        for branch, root, confidence, recipe_hash in (
            ("A0", a0_root, 0.70, "a0-recipe"),
            ("A4", a4_root, 0.75, A4_HASH),
        ):
            run = root / f"{branch}-{view_id}-seed42"
            run.mkdir(parents=True)
            probabilities = []
            for sample in view.validation_samples:
                true_index = labels.index(sample.family_id)
                row = [(1.0 - confidence) / (len(labels) - 1)] * len(labels)
                row[true_index] = confidence
                probabilities.append(row)
            records = build_prediction_records(
                view.validation_samples,
                probabilities,
                labels,
                branch,
                "validation",
            )
            _write_json(
                run / "run_config.json",
                {
                    "schema_version": 1,
                    "stage_id": branch,
                    "view_id": view_id,
                    "training_seed": 42,
                    "split_dataset_fingerprint": manifest.dataset_fingerprint,
                    "split_manifest_hash": split_hash,
                    "catalog_hash": catalog_hash,
                    "run_fingerprint": f"{branch}-{view_id}-42",
                    "test_evaluated": False,
                    "smoke_test": False,
                    "preflight_only": False,
                },
            )
            _write_json(run / "resolved_recipe.json", {"recipe_hash": recipe_hash})
            _write_json(run / "label_maps.json", {"family_labels": list(labels)})
            _write_json(run / "metrics.json", {"overall": {}})
            _write_json(run / "COMPLETED.json", {"status": "complete"})
            (run / "predictions.jsonl").write_text(
                "".join(json.dumps(record.to_dict()) + "\n" for record in records),
                encoding="utf-8",
            )
    return {
        "data": data,
        "catalog": catalog,
        "split": split,
        "config": config,
        "a0": a0_root,
        "a4": a4_root,
    }


def _args(paths: dict[str, Path], output: Path) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
        "--input",
        str(paths["data"]),
        "--catalog",
        str(paths["catalog"]),
        "--split-manifest",
        str(paths["split"]),
        "--a0-root",
        str(paths["a0"]),
        "--a4-root",
        str(paths["a4"]),
        "--output-dir",
        str(output),
        "--bootstrap-resamples",
        "20",
    ]


def test_dry_run_discovers_matrix_without_creating_output(
    tmp_path: Path, capsys
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result"
    assert main([*_args(paths, output), "--dry-run"]) == 0
    assert not output.exists()
    assert '"test_evaluated": false' in capsys.readouterr().out


def test_end_to_end_writes_direct_model_result_and_artifacts(
    tmp_path: Path, capsys
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result"
    assert main(_args(paths, output)) == 0
    summary = json.loads(
        (output / "phase4_fast_track_summary.json").read_text(encoding="utf-8")
    )
    assert summary["selection_scope"] == "validation_only"
    assert summary["test_evaluated"] is False
    assert set(summary["systems"]) == {"A0", "A4", "E1", "E1-selective"}
    assert (output / "artifacts/seed42-fusion.json").is_file()
    assert (output / "artifacts/seed42-selective-policy.json").is_file()
    assert (output / "phase4_fast_track_report.md").is_file()
    assert json.loads((output / "COMPLETED.json").read_text())["status"] == "complete"
    manifest = json.loads(
        (output / "artifacts/artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["source_commit"]) == 40
    assert manifest["family_labels"]
    assert manifest["cross_fit"]["fold_seed"] == 20260803
    console = capsys.readouterr().out
    assert "Starting Phase4 cell 1/3" in console
    assert "Completed Phase4 cell 3/3" in console


def test_existing_output_directory_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "result"
    output.mkdir()
    try:
        main(_args(paths, output))
    except FileExistsError as exc:
        assert "output" in str(exc).lower()
    else:
        raise AssertionError("existing output directory must be rejected")


def test_confirmation_seed_roots_are_discovered_by_validated_identity(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["config"].read_text(encoding="utf-8"))
    payload["seeds"] = [42, 12]
    _write_json(paths["config"], payload)
    a0_confirmation = tmp_path / "confirmation" / "winner"
    a4_confirmation = tmp_path / "confirmation" / "comparator"
    for view in VIEWS:
        for branch, source_root, target_root, prefix in (
            ("A0", paths["a0"], a0_confirmation, "promoted_recipe"),
            ("A4", paths["a4"], a4_confirmation, "resolved_recipe"),
        ):
            source = source_root / f"{branch}-{view}-seed42"
            target = target_root / f"{prefix}-{view}-seed12"
            shutil.copytree(source, target)
            run_config = json.loads((target / "run_config.json").read_text())
            run_config["training_seed"] = 12
            run_config["run_fingerprint"] = f"{branch}-{view}-12"
            _write_json(target / "run_config.json", run_config)
    args = _args(paths, tmp_path / "result")
    args.extend(
        [
            "--a0-confirmation-root",
            str(a0_confirmation),
            "--a4-confirmation-root",
            str(a4_confirmation),
            "--dry-run",
        ]
    )
    assert main(args) == 0


def test_cross_cell_identity_mismatch_is_rejected_before_training(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    for branch, root in (("A0", paths["a0"]), ("A4", paths["a4"])):
        config_path = root / f"{branch}-fast-to-balanced-seed42/run_config.json"
        payload = json.loads(config_path.read_text())
        payload["split_dataset_fingerprint"] = "different-dataset"
        _write_json(config_path, payload)
    try:
        main([*_args(paths, tmp_path / "result"), "--dry-run"])
    except ValueError as exc:
        assert "cross-cell" in str(exc)
    else:
        raise AssertionError("cross-cell identity mismatch must fail")


def test_formal_run_rejects_reduced_bootstrap_override(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["config"].read_text(encoding="utf-8"))
    payload["bootstrap_resamples"] = 2000
    _write_json(paths["config"], payload)
    args = _args(paths, tmp_path / "result")
    bootstrap_index = args.index("--bootstrap-resamples") + 1
    args[bootstrap_index] = "20"
    try:
        main(args)
    except ValueError as exc:
        assert "formal" in str(exc).lower()
    else:
        raise AssertionError("formal bootstrap override must be rejected")
