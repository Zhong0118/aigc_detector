"""Train and evaluate leakage-safe family-attribution baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

import joblib

from research.text_attribution.attribution_metrics import (
    build_prediction_records,
    evaluate_family_predictions,
)
from research.text_attribution.attribution_reporting import write_attribution_report
from research.text_attribution.classical_models import (
    EXPERIMENT_IDS,
    build_classical_pipeline,
    predict_family_probabilities,
)
from research.text_attribution.split_manifest import (
    SplitManifest,
    apply_split_manifest,
    create_split_manifest,
    load_split_manifest,
)
from research.text_attribution.training_data import (
    AttributionSample,
    load_training_dataset,
)


DEFAULT_CONFIG_PATH = (
    "configs/text_attribution/training/family_baselines.v1.json"
)
_CONFIG_KEYS = {
    "schema_version",
    "run_id",
    "experiments",
    "seed",
    "split",
    "model_catalog",
}
_SPLIT_KEYS = {"train_ratio", "validation_ratio", "test_ratio"}
_MAX_SKLEARN_SEED = 2**32 - 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe family attribution baselines."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENT_IDS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate-selected-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid family attribution config JSON: {path}") from error
    if not isinstance(payload, dict) or set(payload) != _CONFIG_KEYS:
        raise ValueError(
            "Family attribution config must be an object with exactly the "
            f"required config fields: {sorted(_CONFIG_KEYS)}"
        )
    if (
        isinstance(payload["schema_version"], bool)
        or not isinstance(payload["schema_version"], int)
        or payload["schema_version"] != 1
    ):
        raise ValueError("Family attribution config schema_version must be 1")
    if not isinstance(payload["run_id"], str) or not payload["run_id"].strip():
        raise ValueError("Family attribution config run_id must be a nonempty string")
    _validate_training_seed(payload["seed"], source="config")
    _validate_experiments(payload["experiments"], source="config")
    _validate_split_config(payload["split"])
    if (
        not isinstance(payload["model_catalog"], str)
        or not payload["model_catalog"].strip()
    ):
        raise ValueError(
            "Family attribution config model_catalog must be a nonempty string"
        )
    return payload


def _validate_experiments(value: object, *, source: str) -> tuple[str, ...]:
    if not isinstance(value, list) and not isinstance(value, tuple):
        raise ValueError(f"{source} experiments must be a nonempty list")
    experiments = tuple(value)
    if (
        not experiments
        or any(not isinstance(item, str) for item in experiments)
        or len(set(experiments)) != len(experiments)
    ):
        raise ValueError(f"{source} experiments must be nonempty unique strings")
    unknown = sorted(set(experiments) - set(EXPERIMENT_IDS))
    if unknown:
        raise ValueError(f"Unknown {source} experiment(s): {', '.join(unknown)}")
    return experiments


def _validate_training_seed(value: object, *, source: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_SKLEARN_SEED
    ):
        raise ValueError(
            f"Family attribution {source} seed must be an integer in "
            f"[0, {_MAX_SKLEARN_SEED}]"
        )
    return value


def _validate_split_config(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _SPLIT_KEYS:
        raise ValueError(
            "Family attribution config split must contain exactly "
            "train_ratio, validation_ratio, and test_ratio"
        )
    ratios: dict[str, float] = {}
    for key in sorted(_SPLIT_KEYS):
        raw_value = value[key]
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, Real)
            or not math.isfinite(float(raw_value))
            or not 0.0 < float(raw_value) < 1.0
        ):
            raise ValueError(f"Family attribution config {key} must be in (0, 1)")
        ratios[key] = float(raw_value)
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Family attribution config split ratios must sum to 1")
    return ratios


def _manifest_ratios_from_config(
    split_config: Mapping[str, float],
) -> dict[str, float]:
    return {
        "train": float(split_config["train_ratio"]),
        "validation": float(split_config["validation_ratio"]),
        "test": float(split_config["test_ratio"]),
    }


def _validate_requested_split(
    manifest: SplitManifest,
    *,
    seed: int,
    split_config: Mapping[str, float],
) -> None:
    if manifest.seed != seed:
        raise ValueError(
            "Existing split manifest seed does not match the requested split "
            f"seed: existing={manifest.seed}, requested={seed}"
        )
    requested_ratios = _manifest_ratios_from_config(split_config)
    if any(
        not math.isclose(
            float(manifest.ratios[name]),
            requested_ratios[name],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name in requested_ratios
    ):
        raise ValueError(
            "Existing split manifest ratios do not match the requested split "
            f"ratios: existing={dict(manifest.ratios)}, "
            f"requested={requested_ratios}"
        )


def _split_manifest_payload(manifest: SplitManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "seed": manifest.seed,
        "ratios": dict(manifest.ratios),
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "group_assignments": dict(manifest.group_assignments),
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publish_split_manifest_exclusive(
    path: Path,
    manifest: SplitManifest,
) -> bool:
    """Atomically publish a complete manifest unless another creator won."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            json.dump(
                _split_manifest_payload(manifest),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def _create_or_load_split_manifest(
    path: Path,
    samples: Sequence[AttributionSample],
    *,
    seed: int,
    split_config: Mapping[str, float],
    persist: bool,
) -> SplitManifest:
    if path.exists():
        manifest = load_split_manifest(path)
        _validate_requested_split(
            manifest,
            seed=seed,
            split_config=split_config,
        )
        return manifest

    candidate = create_split_manifest(
        samples,
        seed=seed,
        train_ratio=float(split_config["train_ratio"]),
        validation_ratio=float(split_config["validation_ratio"]),
        test_ratio=float(split_config["test_ratio"]),
    )
    if not persist:
        return candidate
    if _publish_split_manifest_exclusive(path, candidate):
        return candidate

    manifest = load_split_manifest(path)
    _validate_requested_split(
        manifest,
        seed=seed,
        split_config=split_config,
    )
    return manifest


def _validate_b3_training_support(
    experiments: Sequence[str],
    train_samples: Sequence[AttributionSample],
    family_labels: Sequence[str],
) -> None:
    if "B3" not in experiments:
        return
    counts = Counter(sample.family_id for sample in train_samples)
    insufficient = {
        family: counts[family] for family in family_labels if counts[family] < 3
    }
    if insufficient:
        raise ValueError(
            "B3 requires at least 3 training samples per family for "
            f"CalibratedClassifierCV cv=3; insufficient={insufficient}"
        )


def _replace_path(source: str | Path, target: str | Path) -> None:
    os.replace(source, target)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_run_directory(staging: Path, output_dir: Path) -> None:
    backup = output_dir.with_name(
        f".{output_dir.name}.backup-{uuid4().hex}"
    )
    had_existing = output_dir.exists()
    if had_existing:
        _replace_path(output_dir, backup)
    try:
        _replace_path(staging, output_dir)
    except BaseException:
        if had_existing and backup.exists() and not output_dir.exists():
            _replace_path(backup, output_dir)
        raise
    _fsync_directory(output_dir.parent)
    if backup.exists():
        _remove_path(backup)


def _selection_key(
    experiment_id: str, metrics: Mapping[str, Mapping[str, Any]]
) -> tuple[float, float]:
    validation = metrics[experiment_id]["validation"]
    worst_family_f1 = min(
        float(row["f1-score"]) for row in validation["per_family"].values()
    )
    return float(validation["overall"]["macro_f1"]), worst_family_f1


def _select_experiment(
    experiments: Sequence[str], metrics: Mapping[str, Mapping[str, Any]]
) -> tuple[str, float, float]:
    ranked = sorted(
        (
            (-macro_f1, -worst_family_f1, experiment_id)
            for experiment_id in experiments
            for macro_f1, worst_family_f1 in [_selection_key(experiment_id, metrics)]
        )
    )
    negative_macro, negative_worst, selected = ranked[0]
    return selected, -negative_macro, -negative_worst


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _load_config(config_path)
    catalog_path = (
        Path(args.catalog)
        if args.catalog is not None
        else (config_path.parent / config["model_catalog"]).resolve()
    )
    experiments = _validate_experiments(
        args.experiments if args.experiments is not None else config["experiments"],
        source="CLI" if args.experiments is not None else "config",
    )
    seed = _validate_training_seed(
        args.seed if args.seed is not None else config["seed"],
        source="CLI" if args.seed is not None else "config",
    )
    split_config = _validate_split_config(config["split"])

    dataset = load_training_dataset(args.input, catalog_path)
    split_path = Path(args.split_manifest)
    if args.create_split:
        manifest = _create_or_load_split_manifest(
            split_path,
            dataset.samples,
            seed=seed,
            split_config=split_config,
            persist=not args.dry_run,
        )
    else:
        if not split_path.is_file():
            raise FileNotFoundError(
                "Frozen split manifest is required; pass --create-split once "
                f"to create it: {split_path}"
            )
        manifest = load_split_manifest(split_path)
    splits = apply_split_manifest(dataset.samples, manifest)
    _validate_b3_training_support(
        experiments,
        splits["train"],
        dataset.family_labels,
    )

    split_summary = {name: len(rows) for name, rows in splits.items()}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "input": str(args.input),
        "clean_samples": len(dataset.samples),
        "families": list(dataset.family_labels),
        "models": list(dataset.model_labels),
        "splits": split_summary,
        "split_manifest": str(split_path),
        "split_dataset_fingerprint": manifest.dataset_fingerprint,
        "split_seed": manifest.seed,
        "split_ratios": dict(manifest.ratios),
        "experiments": list(experiments),
        "training_seed": seed,
        "config": str(config_path),
        "catalog": str(catalog_path),
        "evaluate_selected_test": args.evaluate_selected_test,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output-dir must be a directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.run-",
            dir=output_dir.parent,
        )
    )
    model_dir = staging / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_samples = splits["train"]
    train_texts = [sample.text for sample in train_samples]
    train_labels = [sample.family_id for sample in train_samples]
    all_records = []
    all_metrics: dict[str, Any] = {}
    trained_models = {}

    try:
        for experiment_id in experiments:
            model = build_classical_pipeline(experiment_id, seed=seed)
            model.fit(train_texts, train_labels)
            joblib.dump(model, model_dir / f"{experiment_id}.joblib")
            trained_models[experiment_id] = model
            validation_samples = splits["validation"]
            probabilities = predict_family_probabilities(
                model,
                [sample.text for sample in validation_samples],
                family_labels=dataset.family_labels,
            )
            records = build_prediction_records(
                validation_samples,
                probabilities,
                family_labels=dataset.family_labels,
                experiment_id=experiment_id,
                split="validation",
            )
            all_records.extend(records)
            all_metrics[experiment_id] = {
                "validation": evaluate_family_predictions(
                    records,
                    family_labels=dataset.family_labels,
                )
            }

        selected_experiment, selected_macro_f1, selected_worst_family_f1 = (
            _select_experiment(experiments, all_metrics)
        )
        all_metrics["selection"] = {
            "selected_experiment": selected_experiment,
            "selection_split": "validation",
            "selection_metric": "macro_f1_then_worst_family_f1",
            "validation_macro_f1": selected_macro_f1,
            "validation_worst_family_f1": selected_worst_family_f1,
            "tie_break": "lexicographically_smallest_experiment_id",
        }
        summary["selected_experiment"] = selected_experiment

        if args.evaluate_selected_test:
            test_samples = splits["test"]
            test_probabilities = predict_family_probabilities(
                trained_models[selected_experiment],
                [sample.text for sample in test_samples],
                family_labels=dataset.family_labels,
            )
            test_records = build_prediction_records(
                test_samples,
                test_probabilities,
                family_labels=dataset.family_labels,
                experiment_id=selected_experiment,
                split="test",
            )
            all_records.extend(test_records)
            all_metrics[selected_experiment]["test"] = evaluate_family_predictions(
                test_records,
                family_labels=dataset.family_labels,
            )

        write_attribution_report(
            output_dir=staging,
            run_config=summary,
            dataset_audit=dataset.audit,
            split_summary=split_summary,
            metrics=all_metrics,
            predictions=all_records,
        )
        _publish_run_directory(staging, output_dir)
    finally:
        if staging.exists():
            _remove_path(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
