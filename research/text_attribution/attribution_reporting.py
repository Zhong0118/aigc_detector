"""Durable report artifacts for family-attribution experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from research.text_attribution.attribution_types import FamilyPrediction


_CALIBRATION_KEYS = (
    "expected_calibration_error",
    "multiclass_brier_score",
    "negative_log_likelihood",
)
_CALIBRATION_LABELS = ("ECE", "Brier", "NLL")


def _json_safe(value: Any, *, path: str = "value") -> Any:
    """Return built-in JSON values, rejecting unsupported or non-finite data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{path} must contain only finite numeric values")
        return normalized
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} mapping keys must be strings")
            normalized_mapping[key] = _json_safe(item, path=f"{path}.{key}")
        return normalized_mapping
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path=path)
    raise ValueError(f"{path} contains a non-JSON-safe value of type {type(value).__name__}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _require_number(mapping: Mapping[str, Any], key: str, path: str) -> float:
    if key not in mapping:
        raise ValueError(f"{path} is missing required metric {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path}.{key} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{path}.{key} must be a finite number")
    return normalized


def _require_bounded_metric(
    mapping: Mapping[str, Any], key: str, path: str
) -> float:
    value = _require_number(mapping, key, path)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{path}.{key} metric must be within [0, 1]")
    return value


def _require_nonnegative_metric(
    mapping: Mapping[str, Any], key: str, path: str
) -> float:
    value = _require_number(mapping, key, path)
    if value < 0.0:
        raise ValueError(f"{path}.{key} metric must be nonnegative")
    return value


def _require_count(mapping: Mapping[str, Any], key: str, path: str) -> int:
    if key not in mapping:
        raise ValueError(f"{path} is missing required support count")
    value = mapping[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not float(value).is_integer()
        or int(value) < 0
    ):
        raise ValueError(f"{path}.{key} support must be a nonnegative integer")
    return int(value)


def _metric_rows(
    metrics: Mapping[str, Any],
) -> list[tuple[str, str, Mapping[str, Any]]]:
    """Flatten experiment/split metrics while excluding selection metadata."""

    rows: list[tuple[str, str, Mapping[str, Any]]] = []
    for experiment_id, experiment_metrics in metrics.items():
        if experiment_id == "selection":
            continue
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("metric experiment identifiers must be nonempty strings")
        experiment_mapping = _require_mapping(
            experiment_metrics, f"metrics.{experiment_id}"
        )
        for split_name, split_metrics in experiment_mapping.items():
            if not isinstance(split_name, str) or not split_name:
                raise ValueError("metric split identifiers must be nonempty strings")
            rows.append(
                (
                    experiment_id,
                    split_name,
                    _require_mapping(
                        split_metrics,
                        f"metrics.{experiment_id}.{split_name}",
                    ),
                )
            )
    if not rows:
        raise ValueError("metrics must contain at least one experiment/split metric row")
    return rows


def _validate_metric_rows(
    rows: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> tuple[str, ...]:
    fixed_family_order: tuple[str, ...] | None = None
    for experiment, split, row in rows:
        path = f"metrics.{experiment}.{split}"
        overall = _require_mapping(row.get("overall"), f"{path}.overall")
        for key in ("accuracy", "macro_f1", "weighted_f1"):
            _require_bounded_metric(overall, key, f"{path}.overall")
        _require_count(overall, "support", f"{path}.overall")

        calibration = _require_mapping(
            row.get("calibration"), f"{path}.calibration"
        )
        _require_bounded_metric(
            calibration,
            "expected_calibration_error",
            f"{path}.calibration",
        )
        brier = _require_number(
            calibration,
            "multiclass_brier_score",
            f"{path}.calibration",
        )
        if not 0.0 <= brier <= 2.0:
            raise ValueError(
                f"{path}.calibration.multiclass_brier_score metric must be "
                "within [0, 2]"
            )
        _require_nonnegative_metric(
            calibration,
            "negative_log_likelihood",
            f"{path}.calibration",
        )

        per_family = _require_mapping(row.get("per_family"), f"{path}.per_family")
        if not per_family:
            raise ValueError(f"{path}.per_family must not be empty")
        labels = tuple(per_family)
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError(f"{path}.per_family labels must be nonempty strings")
        if fixed_family_order is None:
            fixed_family_order = labels
        elif labels != fixed_family_order:
            raise ValueError("all metric rows must share one fixed family order")
        for family, diagnostic_value in per_family.items():
            diagnostic = _require_mapping(
                diagnostic_value, f"{path}.per_family.{family}"
            )
            for key in ("precision", "recall", "f1-score"):
                _require_bounded_metric(
                    diagnostic, key, f"{path}.per_family.{family}"
                )
            _require_count(diagnostic, "support", f"{path}.per_family.{family}")

        matrix = np.asarray(row.get("confusion_matrix"), dtype=float)
        expected_shape = (len(labels), len(labels))
        if (
            matrix.shape != expected_shape
            or not np.isfinite(matrix).all()
            or (matrix < 0.0).any()
            or (matrix > 1.0).any()
        ):
            raise ValueError(
                f"{path}.confusion_matrix must be a finite {expected_shape} "
                "matrix with confusion entries within [0, 1]"
            )
        row_sums = matrix.sum(axis=1)
        if not np.logical_or(
            np.isclose(row_sums, 0.0, rtol=0.0, atol=1e-6),
            np.isclose(row_sums, 1.0, rtol=0.0, atol=1e-6),
        ).all():
            raise ValueError(
                f"{path}.confusion_matrix row sums must be approximately 1 or 0"
            )

        per_model = _require_mapping(row.get("per_model"), f"{path}.per_model")
        for model_id, diagnostic_value in per_model.items():
            if not isinstance(model_id, str) or not model_id:
                raise ValueError(f"{path}.per_model identifiers must be nonempty strings")
            diagnostic = _require_mapping(
                diagnostic_value, f"{path}.per_model.{model_id}"
            )
            _require_count(
                diagnostic, "support", f"{path}.per_model.{model_id}"
            )
            for key in ("accuracy", "mean_true_family_probability"):
                _require_bounded_metric(
                    diagnostic, key, f"{path}.per_model.{model_id}"
                )
            if "mean_confidence" in diagnostic:
                _require_bounded_metric(
                    diagnostic,
                    "mean_confidence",
                    f"{path}.per_model.{model_id}",
                )

        if "subgroups" in row:
            subgroups = _require_mapping(row["subgroups"], f"{path}.subgroups")
            for dimension, group_value in subgroups.items():
                groups = _require_mapping(
                    group_value, f"{path}.subgroups.{dimension}"
                )
                for group_name, group_metrics_value in groups.items():
                    group_metrics = _require_mapping(
                        group_metrics_value,
                        f"{path}.subgroups.{dimension}.{group_name}",
                    )
                    _require_count(
                        group_metrics,
                        "support",
                        f"{path}.subgroups.{dimension}.{group_name}",
                    )
                    for key in ("accuracy", "macro_f1"):
                        _require_bounded_metric(
                            group_metrics,
                            key,
                            f"{path}.subgroups.{dimension}.{group_name}",
                        )
    assert fixed_family_order is not None
    return fixed_family_order


def _normalize_prediction(value: FamilyPrediction | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, FamilyPrediction):
        payload = value.to_dict()
    elif isinstance(value, Mapping):
        try:
            payload = FamilyPrediction(**dict(value)).to_dict()
        except (TypeError, ValueError) as error:
            raise ValueError(f"malformed prediction mapping: {error}") from error
    else:
        raise ValueError("predictions must contain FamilyPrediction objects or mappings")

    normalized = _json_safe(payload, path="prediction")
    probabilities = _require_mapping(
        normalized.get("family_probabilities"),
        "prediction.family_probabilities",
    )
    if not probabilities:
        raise ValueError("prediction.family_probabilities must not be empty")
    probability_values = np.asarray(list(probabilities.values()), dtype=float)
    if (
        not np.isfinite(probability_values).all()
        or (probability_values < 0.0).any()
        or (probability_values > 1.0).any()
        or not np.isclose(probability_values.sum(), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(
            "prediction family probabilities must be finite, bounded, and sum to one"
        )
    if normalized["true_family"] not in probabilities:
        raise ValueError("prediction true_family must be in family_probabilities")
    maximum_index = int(np.argmax(probability_values))
    expected_prediction = list(probabilities)[maximum_index]
    if normalized["predicted_family"] != expected_prediction:
        raise ValueError("prediction predicted_family must equal probability argmax")
    if not np.isclose(
        normalized["confidence"],
        probability_values[maximum_index],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("prediction confidence must equal maximum probability")
    return normalized


def _validate_prediction_population(
    predictions: Sequence[Mapping[str, Any]],
    rows: Sequence[tuple[str, str, Mapping[str, Any]]],
    family_order: Sequence[str],
) -> None:
    expected_rows = {(experiment, split) for experiment, split, _ in rows}
    observed_rows: set[tuple[str, str]] = set()
    observed_keys: set[tuple[str, str, str]] = set()
    counts: dict[tuple[str, str], int] = {}

    for prediction in predictions:
        experiment = prediction["experiment_id"]
        split = prediction["split"]
        request_key = prediction["request_key"]
        if not all(
            isinstance(value, str) and value
            for value in (experiment, split, request_key)
        ):
            raise ValueError(
                "prediction experiment_id, split, and request_key must be "
                "nonempty strings"
            )
        row_key = (experiment, split)
        unique_key = (experiment, split, request_key)
        if unique_key in observed_keys:
            raise ValueError(
                "duplicate prediction key "
                f"(experiment_id, split, request_key)={unique_key!r}"
            )
        observed_keys.add(unique_key)
        observed_rows.add(row_key)
        counts[row_key] = counts.get(row_key, 0) + 1

        probability_order = tuple(prediction["family_probabilities"])
        if probability_order != tuple(family_order):
            raise ValueError(
                "prediction family probability key order must match the fixed "
                "metric family order"
            )

    if observed_rows != expected_rows:
        raise ValueError(
            "prediction experiment/split rows must correspond exactly to metric rows"
        )
    for experiment, split, row in rows:
        expected_count = int(row["overall"]["support"])
        observed_count = counts[(experiment, split)]
        if observed_count != expected_count:
            raise ValueError(
                f"prediction count must equal overall support for "
                f"{experiment}/{split}: expected {expected_count}, "
                f"observed {observed_count}"
            )


def _write_text_durable(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, payload: Any) -> None:
    _write_text_durable(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n",
    )


def _sync_file(path: Path) -> None:
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _subplot_grid(count: int, *, width: float = 5.0, height: float = 4.2):
    columns = min(3, max(1, count))
    rows = max(1, math.ceil(count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width * columns, height * rows),
        squeeze=False,
    )
    return figure, list(axes.flat)


def _plot_overall(
    rows: Sequence[tuple[str, str, Mapping[str, Any]]], path: Path
) -> None:
    labels = [f"{experiment}\n{split}" for experiment, split, _ in rows]
    positions = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=(max(7.0, len(rows) * 1.6), 4.8))
    try:
        for offset, (key, label) in enumerate(
            (
                ("accuracy", "Accuracy"),
                ("macro_f1", "Macro-F1"),
                ("weighted_f1", "Weighted-F1"),
            )
        ):
            values = [float(row["overall"][key]) for _, _, row in rows]
            axis.bar(positions + (offset - 1) * width, values, width, label=label)
        axis.set_xticks(positions, labels)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Score")
        axis.set_title("Family attribution performance")
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)


def _plot_confusions(
    rows: Sequence[tuple[str, str, Mapping[str, Any]]], path: Path
) -> None:
    figure, axes = _subplot_grid(len(rows), width=5.3, height=4.8)
    try:
        for axis, (experiment, split, row) in zip(axes, rows, strict=False):
            labels = list(row["per_family"])
            sns.heatmap(
                np.asarray(row["confusion_matrix"], dtype=float),
                annot=len(labels) <= 10,
                fmt=".2f",
                vmin=0.0,
                vmax=1.0,
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                cbar=True,
                ax=axis,
            )
            axis.set_title(f"{experiment} / {split}")
            axis.set_xlabel("Predicted family")
            axis.set_ylabel("True family")
            axis.tick_params(axis="x", rotation=55)
            axis.tick_params(axis="y", rotation=0)
        for axis in axes[len(rows) :]:
            axis.remove()
        figure.suptitle("Normalized family confusion matrices")
        figure.tight_layout()
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)


def _plot_per_model(
    rows: Sequence[tuple[str, str, Mapping[str, Any]]], path: Path
) -> None:
    figure, axes = _subplot_grid(len(rows), width=5.5, height=4.5)
    try:
        for axis, (experiment, split, row) in zip(axes, rows, strict=False):
            diagnostics = row["per_model"]
            models = list(diagnostics)
            values = [float(diagnostics[model]["accuracy"]) for model in models]
            axis.bar(models, values, color=sns.color_palette("deep", 1))
            axis.set_ylim(0.0, 1.0)
            axis.tick_params(axis="x", rotation=60)
            axis.set_ylabel("Family accuracy")
            axis.set_title(f"{experiment} / {split}")
        for axis in axes[len(rows) :]:
            axis.remove()
        figure.suptitle(
            "Per-model diagnostics of family classification "
            "(not exact-version attribution)"
        )
        figure.tight_layout()
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)


def _plot_calibration(
    rows: Sequence[tuple[str, str, Mapping[str, Any]]], path: Path
) -> None:
    labels = [f"{experiment}\n{split}" for experiment, split, _ in rows]
    positions = np.arange(len(rows))
    figure, axes_array = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    try:
        for axis, key, title in zip(
            axes_array[0], _CALIBRATION_KEYS, _CALIBRATION_LABELS, strict=True
        ):
            values = [float(row["calibration"][key]) for _, _, row in rows]
            axis.bar(positions, values)
            axis.set_xticks(positions, labels)
            upper = max(values, default=0.0)
            axis.set_ylim(0.0, max(upper * 1.1, 1e-6))
            axis.set_title(title)
            axis.set_ylabel("Score")
        figure.suptitle("Family probability calibration")
        figure.tight_layout()
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)


def _plot_subgroups(
    rows: Sequence[tuple[str, str, Mapping[str, Any]]], path: Path
) -> None:
    panels = [
        (experiment, split, dimension, groups)
        for experiment, split, row in rows
        for dimension, groups in row.get("subgroups", {}).items()
        if groups
    ]
    figure, axes = _subplot_grid(max(1, len(panels)), width=5.2, height=4.2)
    try:
        if not panels:
            axes[0].text(
                0.5,
                0.5,
                "No subgroup metrics were supplied for this run.",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_axis_off()
        else:
            for axis, (experiment, split, dimension, groups) in zip(
                axes, panels, strict=False
            ):
                names = list(groups)
                values = [float(groups[name]["macro_f1"]) for name in names]
                axis.bar(names, values)
                axis.set_ylim(0.0, 1.0)
                axis.tick_params(axis="x", rotation=45)
                axis.set_title(f"{experiment} / {split}: {dimension}")
                axis.set_ylabel("Family Macro-F1")
        for axis in axes[len(panels) if panels else 1 :]:
            axis.remove()
        figure.suptitle("Family attribution subgroup diagnostics")
        figure.tight_layout()
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)


def _format_number(value: Any) -> str:
    return f"{float(value):.4f}"


def _markdown_report(
    run_config: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    rows: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> str:
    experiments = run_config.get("experiments", [])
    experiment_text = ", ".join(str(value) for value in experiments)
    training_seed = run_config.get(
        "training_seed",
        run_config.get("seed", "unspecified"),
    )
    lines = [
        "# Family Attribution Baseline Report",
        "",
        "## Run",
        "",
        f"- Run ID: `{run_config.get('run_id', 'unspecified')}`",
        f"- Training seed: `{training_seed}`",
        f"- Experiments: `{experiment_text}`",
        f"- Clean samples: `{dataset_audit.get('kept_records', 0)}`",
    ]
    if "split_seed" in run_config:
        lines.append(f"- Split seed: `{run_config['split_seed']}`")
    split_ratios = run_config.get("split_ratios")
    if isinstance(split_ratios, Mapping):
        ratio_text = ", ".join(
            f"{split_name}={ratio}" for split_name, ratio in split_ratios.items()
        )
        lines.append(f"- Split ratios: `{ratio_text}`")
    lines.append("")
    selection = metrics.get("selection")
    if isinstance(selection, Mapping):
        lines.extend(
            [
                "## Validation selection",
                "",
                f"- Selected experiment: `{selection.get('selected_experiment', 'unspecified')}`",
                f"- Selection split: `{selection.get('selection_split', 'validation')}`",
                f"- Selection metric: `{selection.get('selection_metric', 'unspecified')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Frozen split",
            "",
            "| Split | Samples |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| {split_name} | {count} |"
        for split_name, count in split_summary.items()
    )
    lines.extend(
        [
            "",
            "## Family-level metrics",
            "",
            "| Experiment | Split | Accuracy | Macro-F1 | Weighted-F1 | ECE | Brier | NLL |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for experiment, split, row in rows:
        overall = row["overall"]
        calibration = row["calibration"]
        lines.append(
            f"| {experiment} | {split} | {_format_number(overall['accuracy'])} | "
            f"{_format_number(overall['macro_f1'])} | "
            f"{_format_number(overall['weighted_f1'])} | "
            f"{_format_number(calibration['expected_calibration_error'])} | "
            f"{_format_number(calibration['multiclass_brier_score'])} | "
            f"{_format_number(calibration['negative_log_likelihood'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-model/version family diagnostics",
            "",
            "These views diagnose family classification after grouping examples by "
            "their true generating model ID; they are not exact-version attribution "
            "results.",
            "",
            "| Experiment | Split | True model ID | Support | Family accuracy | Mean true-family probability |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for experiment, split, row in rows:
        for model_id, diagnostic in row["per_model"].items():
            lines.append(
                f"| {experiment} | {split} | {model_id} | "
                f"{diagnostic['support']} | "
                f"{_format_number(diagnostic['accuracy'])} | "
                f"{_format_number(diagnostic['mean_true_family_probability'])} |"
            )
    return "\n".join(lines) + "\n"


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "run_config": root / "run_config.json",
        "dataset_audit": root / "dataset_audit.json",
        "metrics": root / "metrics.json",
        "predictions": root / "predictions.jsonl",
        "report": root / "report.md",
        "overall_metrics_png": root / "overall_metrics.png",
        "confusion_matrices_png": root / "confusion_matrices.png",
        "per_model_diagnostics_png": root / "per_model_diagnostics.png",
        "calibration_png": root / "calibration.png",
        "subgroup_metrics_png": root / "subgroup_metrics.png",
    }


def _validate_staged_artifacts(
    paths: Mapping[str, Path], expected_predictions: int
) -> None:
    for key in ("run_config", "dataset_audit", "metrics"):
        try:
            json.loads(paths[key].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"staged {key} artifact is not valid JSON") from error

    try:
        prediction_rows = [
            json.loads(line)
            for line in paths["predictions"].read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("staged predictions artifact is not valid JSONL") from error
    if len(prediction_rows) != expected_predictions:
        raise RuntimeError("staged predictions artifact has an unexpected row count")

    if not paths["report"].read_text(encoding="utf-8").strip():
        raise RuntimeError("staged Markdown report is empty")

    for key in (
        "overall_metrics_png",
        "confusion_matrices_png",
        "per_model_diagnostics_png",
        "calibration_png",
        "subgroup_metrics_png",
    ):
        payload = paths[key].read_bytes()
        if len(payload) <= 8 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"staged {key} artifact is not a valid PNG")
        _sync_file(paths[key])


def _publish_staged_directory(staging: Path, target: Path) -> None:
    """Swap a complete staged directory into place, restoring on failure."""

    backup = staging.with_name(f"{staging.name}.backup")
    had_existing_target = target.exists()
    if had_existing_target:
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        if had_existing_target and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def write_attribution_report(
    *,
    output_dir: str | Path,
    run_config: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    split_summary: Mapping[str, int],
    metrics: Mapping[str, Any],
    predictions: Sequence[FamilyPrediction | Mapping[str, Any]],
) -> dict[str, str]:
    """Write JSON, Markdown, and PNG artifacts for family-level experiments."""

    normalized_run_config = _json_safe(
        _require_mapping(run_config, "run_config"), path="run_config"
    )
    normalized_audit = _json_safe(
        _require_mapping(dataset_audit, "dataset_audit"), path="dataset_audit"
    )
    normalized_split_summary = _json_safe(
        _require_mapping(split_summary, "split_summary"), path="split_summary"
    )
    normalized_metrics = _json_safe(
        _require_mapping(metrics, "metrics"), path="metrics"
    )
    metric_rows = _metric_rows(normalized_metrics)
    family_order = _validate_metric_rows(metric_rows)
    normalized_predictions = [
        _normalize_prediction(prediction) for prediction in predictions
    ]
    if not normalized_predictions:
        raise ValueError("predictions must contain at least one record")
    _validate_prediction_population(
        normalized_predictions,
        metric_rows,
        family_order,
    )

    root = Path(output_dir)
    if root.exists() and not root.is_dir():
        raise ValueError(f"output_dir must be a directory: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent)
    )
    try:
        if root.exists():
            shutil.copytree(root, staging, dirs_exist_ok=True)
        staged_paths = _artifact_paths(staging)
        _write_json(staged_paths["run_config"], normalized_run_config)
        _write_json(staged_paths["dataset_audit"], normalized_audit)
        _write_json(staged_paths["metrics"], normalized_metrics)
        _write_text_durable(
            staged_paths["predictions"],
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=False,
                    allow_nan=False,
                )
                + "\n"
                for row in normalized_predictions
            )
        )
        _write_text_durable(
            staged_paths["report"],
            _markdown_report(
                normalized_run_config,
                normalized_audit,
                normalized_split_summary,
                normalized_metrics,
                metric_rows,
            ),
        )
        _plot_overall(metric_rows, staged_paths["overall_metrics_png"])
        _plot_confusions(metric_rows, staged_paths["confusion_matrices_png"])
        _plot_per_model(metric_rows, staged_paths["per_model_diagnostics_png"])
        _plot_calibration(metric_rows, staged_paths["calibration_png"])
        _plot_subgroups(metric_rows, staged_paths["subgroup_metrics_png"])
        _validate_staged_artifacts(staged_paths, len(normalized_predictions))
        _publish_staged_directory(staging, root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return {
        name: str(path)
        for name, path in _artifact_paths(root).items()
    }
