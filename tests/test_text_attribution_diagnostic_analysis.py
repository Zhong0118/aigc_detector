from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.text_attribution.diagnostic_analysis import (
    ExpectedRun,
    align_prediction_runs,
    analyze_complementarity,
    load_prediction_run,
)


LABELS = ("a", "b")


def _write_run(
    root: Path,
    branch: str,
    *,
    request_keys: tuple[str, ...] = ("r1", "r2", "r3", "r4"),
    truth: tuple[str, ...] = ("a", "a", "b", "b"),
    probabilities: tuple[tuple[float, float], ...] = (
        (0.8, 0.2),
        (0.7, 0.3),
        (0.4, 0.6),
        (0.2, 0.8),
    ),
    test_evaluated: bool = False,
    dataset_fingerprint: str = "dataset-hash",
) -> Path:
    run = root / branch
    run.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "stage_id": branch,
        "branch_id": branch,
        "view_id": "closed",
        "training_seed": 42,
        "split_dataset_fingerprint": dataset_fingerprint,
        "split_manifest_hash": "split-hash",
        "catalog_hash": "catalog-hash",
        "run_fingerprint": f"run-{branch}",
        "test_evaluated": test_evaluated,
        "smoke_test": False,
        "preflight_only": False,
    }
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "resolved_recipe.json").write_text(
        json.dumps({"recipe_hash": f"recipe-{branch}"}), encoding="utf-8"
    )
    (run / "label_maps.json").write_text(
        json.dumps({"family_labels": list(LABELS)}), encoding="utf-8"
    )
    (run / "metrics.json").write_text(json.dumps({"overall": {}}), encoding="utf-8")
    (run / "COMPLETED.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    rows = []
    for index, (key, actual, values) in enumerate(
        zip(request_keys, truth, probabilities, strict=True)
    ):
        predicted = LABELS[0 if values[0] >= values[1] else 1]
        rows.append(
            {
                "request_key": key,
                "experiment_id": branch,
                "split": "validation",
                "true_family": actual,
                "predicted_family": predicted,
                "family_probabilities": dict(zip(LABELS, values, strict=True)),
                "confidence": max(values),
                "model_id": f"{actual}-model",
                "prompt_id": f"p{index}",
                "semantic_prompt_group": f"g{index % 2}",
                "language": "zh",
                "task_type": "explanation",
                "target_length": "medium",
                "temperature": 0.0,
            }
        )
    (run / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return run


def test_loads_only_matching_validation_only_formal_run(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "A4")
    run = load_prediction_run(
        run_dir, ExpectedRun(branch_id="A4", seed=42, view_id="closed")
    )
    assert run.identity.test_evaluated is False
    assert run.identity.smoke_test is False
    assert len(run.records) == 4


def test_rejects_test_and_fingerprint_mismatch(tmp_path: Path) -> None:
    test_run = _write_run(tmp_path / "one", "A4", test_evaluated=True)
    with pytest.raises(ValueError, match="test_evaluated"):
        load_prediction_run(
            test_run, ExpectedRun(branch_id="A4", seed=42, view_id="closed")
        )

    a4 = load_prediction_run(
        _write_run(tmp_path / "two", "A4"),
        ExpectedRun(branch_id="A4", seed=42, view_id="closed"),
    )
    b0 = load_prediction_run(
        _write_run(tmp_path / "three", "B0", dataset_fingerprint="other"),
        ExpectedRun(branch_id="B0", seed=42, view_id="closed"),
    )
    with pytest.raises(ValueError, match="dataset fingerprint"):
        align_prediction_runs((a4, b0))


def test_alignment_rejects_missing_request_key(tmp_path: Path) -> None:
    a4 = load_prediction_run(
        _write_run(tmp_path / "a", "A4"),
        ExpectedRun(branch_id="A4", seed=42, view_id="closed"),
    )
    b0 = load_prediction_run(
        _write_run(
            tmp_path / "b",
            "B0",
            request_keys=("r1", "r2", "r3"),
            truth=("a", "a", "b"),
            probabilities=((0.8, 0.2), (0.7, 0.3), (0.4, 0.6)),
        ),
        ExpectedRun(branch_id="B0", seed=42, view_id="closed"),
    )
    with pytest.raises(ValueError, match="request_key"):
        align_prediction_runs((a4, b0))


def test_complementarity_counts_corrections_and_oracle(tmp_path: Path) -> None:
    a4 = load_prediction_run(
        _write_run(
            tmp_path / "a",
            "A4",
            probabilities=((0.8, 0.2), (0.8, 0.2), (0.7, 0.3), (0.7, 0.3)),
        ),
        ExpectedRun("A4", 42, "closed"),
    )
    b0 = load_prediction_run(
        _write_run(
            tmp_path / "b",
            "B0",
            probabilities=((0.8, 0.2), (0.3, 0.7), (0.3, 0.7), (0.7, 0.3)),
        ),
        ExpectedRun("B0", 42, "closed"),
    )
    result = analyze_complementarity(align_prediction_runs((a4, b0)), "A4", "B0")
    assert result["both_correct"] == 1
    assert result["a_only_correct"] == 1
    assert result["b_only_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["oracle_top1_accuracy"] == 0.75
    assert 0.0 <= result["oracle_macro_f1"] <= 1.0
