"""Artifact and visualization generation for attribution baseline runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

from .baseline_data import BaselineSample
from .baseline_models import EvaluationResult, PredictionRow


_DISPLAY_NAMES = {
    "char_lr": "Char TF-IDF + LR",
    "tfidf_svm": "Word/Char TF-IDF + SVM",
    "frozen_e5": "Frozen E5 + Linear Probe",
}
_METRICS = ("accuracy", "macro_f1", "weighted_f1")
_SUBGROUP_ATTRIBUTES = ("language", "temperature", "target_length")


def write_baseline_report(
    *,
    samples: Sequence[BaselineSample],
    audit: dict[str, Any],
    results: Sequence[EvaluationResult],
    output_dir: str | Path,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Write all machine-readable, textual, and visual run artifacts."""

    if not results:
        raise ValueError("At least one successful method result is required")
    root = Path(output_dir)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dataset": {
            "samples": len(samples),
            "labels": sorted({sample.label for sample in samples}),
            "semantic_prompt_groups": len(
                {sample.semantic_prompt_group for sample in samples}
            ),
        },
        "methods": {
            result.method: _result_summary(result)
            for result in results
        },
    }

    run_config_path = root / "run_config.json"
    audit_path = root / "dataset_audit.json"
    metrics_path = root / "metrics.json"
    metrics_csv_path = root / "metrics.csv"
    predictions_path = root / "predictions.jsonl"
    report_path = root / "report.md"
    _write_json(run_config_path, run_config)
    _write_json(audit_path, audit)
    _write_json(metrics_path, metrics)
    _write_metrics_csv(metrics_csv_path, results, metrics)
    _write_predictions(predictions_path, results)
    report_path.write_text(
        _markdown_report(samples, audit, results, metrics),
        encoding="utf-8",
    )

    figure_pairs = [
        _plot_overall_metrics(results, figures_dir / "overall_metrics"),
        _plot_fold_metrics(results, figures_dir / "fold_metrics"),
        _plot_confusion_matrices(results, figures_dir / "confusion_matrices"),
        _plot_subgroup_metrics(metrics, figures_dir / "subgroup_metrics"),
    ]
    return {
        "run_config_json": str(run_config_path),
        "dataset_audit_json": str(audit_path),
        "metrics_json": str(metrics_path),
        "metrics_csv": str(metrics_csv_path),
        "predictions_jsonl": str(predictions_path),
        "report_markdown": str(report_path),
        "figures": figure_pairs,
    }


def _result_summary(result: EvaluationResult) -> dict[str, Any]:
    fold_summary: dict[str, dict[str, float]] = {}
    for metric in _METRICS:
        values = [float(row[metric]) for row in result.fold_metrics]
        fold_summary[metric] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)),
        }
    return {
        "overall": result.oof_metrics,
        "folds": result.fold_metrics,
        "fold_summary": fold_summary,
        "labels": result.labels,
        "per_class": result.per_class,
        "confusion_matrix_normalized": result.confusion_matrix,
        "subgroups": {
            attribute: _subgroup_metrics(result.predictions, attribute)
            for attribute in _SUBGROUP_ATTRIBUTES
        },
    }


def _subgroup_metrics(
    predictions: Sequence[PredictionRow],
    attribute: str,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[PredictionRow]] = {}
    for row in predictions:
        value = getattr(row, attribute)
        key = f"{value:g}" if isinstance(value, float) else str(value)
        grouped.setdefault(key, []).append(row)
    output: dict[str, dict[str, float | int]] = {}
    for key, rows in sorted(grouped.items()):
        true_labels = [row.true_label for row in rows]
        predicted_labels = [row.predicted_label for row in rows]
        output[key] = {
            "samples": len(rows),
            "accuracy": float(accuracy_score(true_labels, predicted_labels)),
            "macro_f1": float(
                f1_score(
                    true_labels,
                    predicted_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    true_labels,
                    predicted_labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_predictions(
    path: Path,
    results: Sequence[EvaluationResult],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            for row in result.predictions:
                handle.write(
                    json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )


def _write_metrics_csv(
    path: Path,
    results: Sequence[EvaluationResult],
    metrics: dict[str, Any],
) -> None:
    fieldnames = [
        "method",
        "scope",
        "group",
        "fold",
        "samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "method": result.method,
                "scope": "overall",
                **result.oof_metrics,
                "samples": len(result.predictions),
            }
        )
        for fold_row in result.fold_metrics:
            rows.append(
                {
                    "method": result.method,
                    "scope": "fold",
                    **fold_row,
                }
            )
        for label, values in result.per_class.items():
            rows.append(
                {
                    "method": result.method,
                    "scope": "class",
                    "group": label,
                    **values,
                }
            )
        method_metrics = metrics["methods"][result.method]
        for dimension, groups in method_metrics["subgroups"].items():
            for group, values in groups.items():
                rows.append(
                    {
                        "method": result.method,
                        "scope": f"subgroup:{dimension}",
                        "group": group,
                        **values,
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(
    samples: Sequence[BaselineSample],
    audit: dict[str, Any],
    results: Sequence[EvaluationResult],
    metrics: dict[str, Any],
) -> str:
    lines = [
        "# Attribution Pilot Baseline Report",
        "",
        "## Dataset",
        "",
        f"- Raw records: {audit.get('raw_records', 'unknown')}",
        f"- Clean records used: {len(samples)}",
        f"- Labels: {len(metrics['dataset']['labels'])}",
        (
            "- Semantic prompt groups: "
            f"{metrics['dataset']['semantic_prompt_groups']}"
        ),
        "",
        "## Out-of-fold results",
        "",
        "| Method | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        overall = result.oof_metrics
        lines.append(
            f"| {_DISPLAY_NAMES.get(result.method, result.method)} "
            f"| {overall['accuracy']:.4f} "
            f"| {overall['macro_f1']:.4f} "
            f"| {overall['weighted_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "This pilot currently has one model per family. Therefore the "
                "reported family classification performance is confounded with "
                "the specific routed model version and must not yet be interpreted "
                "as generalization to unseen models from the same family."
            ),
            "",
            (
                "All scores are out-of-fold. Chinese and English versions of the "
                "same semantic prompt are assigned to the same fold to prevent "
                "prompt-pair leakage."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _plot_overall_metrics(
    results: Sequence[EvaluationResult],
    base_path: Path,
) -> dict[str, str]:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axis = plt.subplots(figsize=(9, 5.2))
    names = [_DISPLAY_NAMES.get(result.method, result.method) for result in results]
    positions = np.arange(len(names))
    width = 0.24
    palette = sns.color_palette("colorblind", len(_METRICS))
    for offset, metric in enumerate(_METRICS):
        axis.bar(
            positions + (offset - 1) * width,
            [result.oof_metrics[metric] for result in results],
            width,
            label=metric.replace("_", " ").title(),
            color=palette[offset],
        )
    axis.set_xticks(positions, names, rotation=10, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("OOF score")
    axis.set_title("Attribution baseline performance")
    axis.legend(frameon=False)
    return _save_figure(fig, base_path)


def _plot_fold_metrics(
    results: Sequence[EvaluationResult],
    base_path: Path,
) -> dict[str, str]:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axis = plt.subplots(figsize=(9, 5.2))
    palette = sns.color_palette("colorblind", len(results))
    for color, result in zip(palette, results, strict=True):
        folds = [int(row["fold"]) for row in result.fold_metrics]
        values = [float(row["macro_f1"]) for row in result.fold_metrics]
        axis.plot(
            folds,
            values,
            marker="o",
            linewidth=2,
            color=color,
            label=_DISPLAY_NAMES.get(result.method, result.method),
        )
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Held-out semantic-prompt fold")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Fold-to-fold generalization")
    axis.legend(frameon=False)
    return _save_figure(fig, base_path)


def _plot_confusion_matrices(
    results: Sequence[EvaluationResult],
    base_path: Path,
) -> dict[str, str]:
    sns.set_theme(style="white", context="notebook")
    columns = min(3, len(results))
    rows = math.ceil(len(results) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 5.2 * rows),
        squeeze=False,
    )
    flattened = axes.ravel()
    for axis, result in zip(flattened, results, strict=False):
        sns.heatmap(
            np.asarray(result.confusion_matrix),
            ax=axis,
            cmap="Blues",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            xticklabels=result.labels,
            yticklabels=result.labels,
            cbar=False,
        )
        axis.set_title(_DISPLAY_NAMES.get(result.method, result.method))
        axis.set_xlabel("Predicted family")
        axis.set_ylabel("True family")
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
    for axis in flattened[len(results) :]:
        axis.set_visible(False)
    fig.suptitle("Normalized out-of-fold confusion matrices", y=1.01)
    return _save_figure(fig, base_path)


def _plot_subgroup_metrics(
    metrics: dict[str, Any],
    base_path: Path,
) -> dict[str, str]:
    sns.set_theme(style="whitegrid", context="notebook")
    dimensions = list(_SUBGROUP_ATTRIBUTES)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), squeeze=False)
    palette = sns.color_palette("colorblind", len(metrics["methods"]))
    for axis, dimension in zip(axes.ravel(), dimensions, strict=True):
        all_groups = sorted(
            {
                group
                for method in metrics["methods"].values()
                for group in method["subgroups"][dimension]
            }
        )
        positions = np.arange(len(all_groups))
        width = 0.8 / max(1, len(metrics["methods"]))
        for method_index, (method_name, method) in enumerate(
            metrics["methods"].items()
        ):
            values = [
                method["subgroups"][dimension].get(group, {}).get(
                    "macro_f1",
                    np.nan,
                )
                for group in all_groups
            ]
            offset = (method_index - (len(metrics["methods"]) - 1) / 2) * width
            axis.bar(
                positions + offset,
                values,
                width,
                color=palette[method_index],
                label=_DISPLAY_NAMES.get(method_name, method_name),
            )
        axis.set_xticks(positions, all_groups, rotation=30, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(dimension.replace("_", " ").title())
        axis.set_ylabel("Macro-F1")
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Subgroup attribution performance", y=1.02)
    return _save_figure(fig, base_path)


def _save_figure(fig: plt.Figure, base_path: Path) -> dict[str, str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}
