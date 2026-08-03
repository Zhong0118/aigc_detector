#!/usr/bin/env python3
"""Run Phase4 diagnostics, aligned classical branches, fusion, and abstention."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from research.text_attribution.diagnostic_analysis import (
    ExpectedRun,
    PredictionRun,
    align_prediction_runs,
    analyze_complementarity,
    load_prediction_run,
)
from research.text_attribution.phase4_classical import (
    ClassicalBranchRequest,
    train_aligned_classical_branches,
)
from research.text_attribution.phase4_config import (
    Phase4FastTrackConfig,
    canonical_config_hash,
    load_phase4_fast_track_config,
)
from research.text_attribution.phase4_metrics import (
    evaluate_phase4_predictions,
    evaluate_selective_predictions,
)
from research.text_attribution.phase4_reporting import (
    aggregate_system,
    build_probability_records,
    render_fast_track_report,
)
from research.text_attribution.probability_fusion import (
    FusionFitRequest,
    fit_probability_fusion,
)
from research.text_attribution.selective_policy import (
    SelectiveFitRequest,
    fit_selective_policy,
)
from research.text_attribution.split_manifest import apply_split_manifest, load_split_manifest
from research.text_attribution.training_data import load_training_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/text_attribution/training/family_phase4_fast_track.v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run validation-only Phase4 attribution Fast Track."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--a0-root", required=True)
    parser.add_argument("--a4-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_phase4_fast_track_config(args.config)
    bootstrap_resamples = (
        config.bootstrap_resamples
        if args.bootstrap_resamples is None
        else args.bootstrap_resamples
    )
    if bootstrap_resamples <= 0 or bootstrap_resamples > config.bootstrap_resamples:
        raise ValueError(
            "bootstrap-resamples must be positive and no greater than configured"
        )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Phase4 output directory already exists: {output_dir}")
    a0_root = Path(args.a0_root).resolve()
    a4_root = Path(args.a4_root).resolve()
    matrix = _load_matrix(config, a0_root, a4_root)
    preflight = {
        "schema_version": 1,
        "run_id": config.run_id,
        "selection_scope": "validation_only",
        "test_evaluated": False,
        "seeds": list(config.seeds),
        "views": list(config.views),
        "matrix_cells": len(config.seeds) * len(config.views),
        "bootstrap_resamples": bootstrap_resamples,
        "output_dir": str(output_dir),
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    data_path = Path(args.input).resolve()
    catalog_path = Path(args.catalog).resolve()
    split_path = Path(args.split_manifest).resolve()
    dataset = load_training_dataset(data_path, catalog_path)
    manifest = load_split_manifest(split_path)
    splits = apply_split_manifest(dataset.samples, manifest)
    test_request_keys = frozenset(row.request_key for row in splits["test"])
    del dataset, splits

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.run-", dir=output_dir.parent)
    )
    try:
        summary = _run_fast_track(
            config=config,
            matrix=matrix,
            data_path=data_path,
            catalog_path=catalog_path,
            split_path=split_path,
            test_request_keys=test_request_keys,
            staging=staging,
            bootstrap_resamples=bootstrap_resamples,
        )
        _write_json(staging / "phase4_fast_track_summary.json", summary)
        (staging / "phase4_fast_track_report.md").write_text(
            render_fast_track_report(summary), encoding="utf-8"
        )
        _write_json(
            staging / "COMPLETED.json",
            {
                "schema_version": 1,
                "status": "complete",
                "test_evaluated": False,
            },
        )
        os.replace(staging, output_dir)
    except Exception as exc:
        _write_json(
            staging / "FAILED.json",
            {
                "schema_version": 1,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if staging.exists() and (staging / "FAILED.json").exists() is False:
            shutil.rmtree(staging)
    return 0


def _load_matrix(
    config: Phase4FastTrackConfig, a0_root: Path, a4_root: Path
) -> dict[tuple[int, str], tuple[PredictionRun, PredictionRun]]:
    matrix = {}
    for seed in config.seeds:
        for view in config.views:
            a0 = load_prediction_run(
                a0_root / f"A0-{view}-seed{seed}", ExpectedRun("A0", seed, view)
            )
            a4 = load_prediction_run(
                a4_root / f"A4-{view}-seed{seed}",
                ExpectedRun("A4", seed, view, config.a4_recipe_hash),
            )
            align_prediction_runs((a0, a4))
            matrix[(seed, view)] = (a0, a4)
    return matrix


def _run_fast_track(
    *,
    config: Phase4FastTrackConfig,
    matrix: Mapping[tuple[int, str], tuple[PredictionRun, PredictionRun]],
    data_path: Path,
    catalog_path: Path,
    split_path: Path,
    test_request_keys: frozenset[str],
    staging: Path,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    for name in ("E0-A4", "classical", "E1-fusion", "selective", "diagnostics", "artifacts"):
        (staging / name).mkdir(parents=True, exist_ok=True)
    _write_json(
        staging / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "config_hash": canonical_config_hash(config),
            "selection_scope": "validation_only",
            "test_evaluated": False,
            "bootstrap_resamples": bootstrap_resamples,
        },
    )

    records_by_system_view: dict[str, dict[str, list[Any]]] = {
        system: {view: [] for view in config.views} for system in ("A0", "A4", "E1")
    }
    metrics_by_system_view: dict[str, dict[str, list[Any]]] = {
        system: {view: [] for view in config.views} for system in ("A0", "A4", "E1")
    }
    selective_closed_records = []
    selective_closed_masks = []
    complementarity_cells = []
    production = None
    config_hash = canonical_config_hash(config)

    for seed in config.seeds:
        for view in config.views:
            cell = f"{view}-seed{seed}"
            a0, a4 = matrix[(seed, view)]
            expected_keys = tuple(sorted(row.request_key for row in a4.records))
            classical = train_aligned_classical_branches(
                ClassicalBranchRequest(
                    data_path=data_path,
                    catalog_path=catalog_path,
                    split_manifest_path=split_path,
                    output_root=staging / "classical" / cell,
                    seed=seed,
                    view_id=view,
                    expected_validation_keys=expected_keys,
                    test_request_keys=test_request_keys,
                    expected_identity=a4.identity,
                    persist_model=seed == config.production_seed and view == "closed",
                )
            )
            b0 = load_prediction_run(
                classical.runs["B0"].run_dir, ExpectedRun("B0", seed, view)
            )
            b1 = load_prediction_run(
                classical.runs["B1"].run_dir, ExpectedRun("B1", seed, view)
            )
            aligned = align_prediction_runs((a4, b0, b1))
            complementarity = {
                f"{left}-{right}": analyze_complementarity(aligned, left, right)
                for left, right in (("A4", "B0"), ("A4", "B1"), ("B0", "B1"))
            }
            complementarity_cells.append(
                {"seed": seed, "view": view, "pairs": complementarity}
            )
            diagnostic_dir = staging / "diagnostics" / cell
            diagnostic_dir.mkdir()
            _write_json(diagnostic_dir / "complementarity.json", complementarity)

            fusion = fit_probability_fusion(
                FusionFitRequest(
                    aligned=aligned,
                    folds=config.cross_fit_folds,
                    fold_seed=config.fold_seed,
                    weight_step=config.weight_step,
                    max_macro_f1_drop=config.max_single_branch_macro_f1_drop,
                    config_hash=config_hash,
                )
            )
            e1_records = build_probability_records(
                aligned.records_by_branch["A4"],
                fusion.oof_probabilities,
                aligned.family_labels,
                "E1",
            )
            policy = fit_selective_policy(
                SelectiveFitRequest(
                    records=e1_records,
                    branch_probabilities=aligned.probabilities_by_branch,
                    fold_ids=fusion.fold_ids,
                    family_labels=aligned.family_labels,
                    accepted_precision_min=config.accepted_precision_min,
                    coverage_min=config.coverage_min,
                    ece_max=config.ece_max,
                    config_hash=config_hash,
                )
            )
            cell_metrics = {
                "A0": evaluate_phase4_predictions(a0.records, a0.family_labels),
                "A4": evaluate_phase4_predictions(a4.records, a4.family_labels),
                "E1": evaluate_phase4_predictions(e1_records, aligned.family_labels),
            }
            for system, records in (
                ("A0", a0.records),
                ("A4", a4.records),
                ("E1", e1_records),
            ):
                records_by_system_view[system][view].extend(records)
                metrics_by_system_view[system][view].append(cell_metrics[system])
            if view == "closed":
                selective_closed_records.extend(e1_records)
                selective_closed_masks.extend(policy.oof_accepted_mask.tolist())

            e0_dir = staging / "E0-A4" / cell
            e0_dir.mkdir()
            _write_json(e0_dir / "metrics.json", cell_metrics["A4"])
            _write_predictions(e0_dir / "predictions.jsonl", a4.records)
            e1_dir = staging / "E1-fusion" / cell
            e1_dir.mkdir()
            _write_json(e1_dir / "metrics.json", cell_metrics["E1"])
            _write_predictions(e1_dir / "predictions.jsonl", e1_records)
            _write_json(e1_dir / "fusion.json", fusion.final_artifact.to_dict())
            selective_dir = staging / "selective" / cell
            selective_dir.mkdir()
            _write_json(selective_dir / "metrics.json", dict(policy.oof_metrics))
            _write_json(
                selective_dir / "policy.json", policy.final_artifact.to_dict()
            )
            if seed == config.production_seed and view == "closed":
                production = (fusion, policy, classical, a4)

    if production is None:
        raise RuntimeError("production seed closed-view artifact was not generated")
    systems = {}
    labels = production[3].family_labels
    for system in ("A0", "A4", "E1"):
        systems[system] = {}
        for offset, view in enumerate(config.views):
            systems[system][view] = aggregate_system(
                metrics_by_system_view[system][view],
                records_by_system_view[system][view],
                labels,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=config.bootstrap_seed + offset,
            )
    selective_metrics = evaluate_selective_predictions(
        tuple(selective_closed_records),
        np.asarray(selective_closed_masks, dtype=bool),
        labels,
    )
    systems["E1-selective"] = {"closed": selective_metrics}
    paired_gains = {
        view: systems["E1"][view]["macro_f1"]["mean"]
        - systems["A4"][view]["macro_f1"]["mean"]
        for view in config.views
    }
    gates = {
        "closed_macro_f1_gain": paired_gains["closed"]
        >= config.e1_closed_macro_f1_min_gain,
        "fast_to_balanced_guardrail": paired_gains["fast-to-balanced"]
        >= -config.holdout_max_drop,
        "balanced_to_fast_guardrail": paired_gains["balanced-to-fast"]
        >= -config.holdout_max_drop,
        "worst_family_guardrail": systems["E1"]["closed"]["worst_family_f1"]["mean"]
        >= systems["A4"]["closed"]["worst_family_f1"]["mean"],
    }
    zero_recall = any(
        row["eligible_support"] > 0 and row["accepted_recall"] == 0.0
        for row in selective_metrics["per_family"].values()
    )
    selective_ready = bool(
        selective_metrics["accepted_precision"] is not None
        and selective_metrics["accepted_precision"] >= config.accepted_precision_min
        and selective_metrics["coverage"] >= config.coverage_min
        and selective_metrics["ece"] is not None
        and selective_metrics["ece"] <= config.ece_max
        and not zero_recall
    )
    e1_promoted = all(gates.values())
    oracle_gain = _mean_oracle_gain(complementarity_cells, systems)
    if e1_promoted and selective_ready:
        next_action = "冻结 E1 Fast Track，并补推理 API 与多文本聚合。"
    elif e1_promoted:
        next_action = "保留 E1 研究模型，补充 calibration/OOD 数据后再部署。"
    elif oracle_gain >= 0.02:
        next_action = "A4 保留；仅增加一个严格对齐的第二神经分支 E2。"
    else:
        next_action = "停止堆融合，进入学习曲线与 V6 扩数。"

    fusion, policy, classical, a4 = production
    artifacts = staging / "artifacts"
    _write_json(artifacts / "seed42-fusion.json", fusion.final_artifact.to_dict())
    _write_json(
        artifacts / "seed42-calibrators.json",
        {
            "schema_version": 1,
            "temperatures": dict(fusion.final_artifact.temperatures),
        },
    )
    _write_json(
        artifacts / "seed42-selective-policy.json", policy.final_artifact.to_dict()
    )
    model_hashes = {
        branch: _sha256_file(run.run_dir / "model.joblib")
        for branch, run in classical.runs.items()
    }
    _write_json(
        artifacts / "artifact_manifest.json",
        {
            "schema_version": 1,
            "production_seed": config.production_seed,
            "view_id": "closed",
            "config_hash": config_hash,
            "dataset_fingerprint": a4.identity.dataset_fingerprint,
            "split_manifest_hash": a4.identity.split_manifest_hash,
            "catalog_hash": a4.identity.catalog_hash,
            "a4_run_fingerprint": a4.identity.run_fingerprint,
            "classical_model_hashes": model_hashes,
            "fusion_hash": _canonical_hash(fusion.final_artifact.to_dict()),
            "policy_hash": _canonical_hash(policy.final_artifact.to_dict()),
            "test_evaluated": False,
        },
    )
    return {
        "schema_version": 1,
        "selection_scope": "validation_only",
        "test_evaluated": False,
        "seeds": list(config.seeds),
        "views": list(config.views),
        "systems": systems,
        "paired_gains": paired_gains,
        "per_family": {
            "A4": systems["A4"]["closed"]["per_family"],
            "E1": systems["E1"]["closed"]["per_family"],
        },
        "complementarity": {
            "cell_count": len(complementarity_cells),
            "a4_b0_oracle_gain_mean": oracle_gain,
        },
        "selective": selective_metrics,
        "decision": {
            "evidence_valid": True,
            "e1_promoted": e1_promoted,
            "selective_deployment_ready": selective_ready,
            "gates": gates,
            "next_action": next_action,
        },
    }


def _mean_oracle_gain(
    cells: Sequence[Mapping[str, Any]], systems: Mapping[str, Any]
) -> float:
    oracle = [cell["pairs"]["A4-B0"]["oracle_top1_accuracy"] for cell in cells]
    a4_accuracy = [
        systems["A4"][view]["accuracy"]["mean"]
        for view in ("closed", "fast-to-balanced", "balanced-to-fast")
    ]
    return float(np.mean(oracle) - np.mean(a4_accuracy))


def _write_predictions(path: Path, records: Sequence[Any]) -> None:
    path.write_text(
        "".join(
            json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON output contains non-finite number")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("JSON output contains non-finite number")
        return normalized
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
