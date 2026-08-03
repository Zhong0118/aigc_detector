"""Aggregation and answer-first reporting for Phase4 Fast Track."""

from __future__ import annotations

from dataclasses import replace
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from research.text_attribution.attribution_types import FamilyPrediction
from research.text_attribution.phase4_metrics import group_bootstrap_interval


def build_probability_records(
    reference_records: Sequence[FamilyPrediction],
    probabilities: np.ndarray,
    family_labels: Sequence[str],
    experiment_id: str,
) -> tuple[FamilyPrediction, ...]:
    labels = tuple(family_labels)
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(reference_records), len(labels)):
        raise ValueError("probabilities do not align with reference records")
    rows = []
    for reference, row in zip(reference_records, values, strict=True):
        index = int(np.argmax(row))
        rows.append(
            replace(
                reference,
                experiment_id=experiment_id,
                predicted_family=labels[index],
                family_probabilities=dict(zip(labels, row.tolist(), strict=True)),
                confidence=float(row[index]),
            )
        )
    return tuple(rows)


def aggregate_system(
    cell_metrics: Sequence[Mapping[str, Any]],
    pooled_records: Sequence[FamilyPrediction],
    family_labels: Sequence[str],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not cell_metrics:
        raise ValueError("system aggregation requires cell metrics")
    metric_paths = {
        "accuracy": ("overall", "accuracy"),
        "macro_precision": ("overall", "macro_precision"),
        "macro_recall": ("overall", "macro_recall"),
        "macro_f1": ("overall", "macro_f1"),
        "weighted_f1": ("overall", "weighted_f1"),
        "nll": ("calibration", "negative_log_likelihood"),
        "ece": ("calibration", "expected_calibration_error"),
        "brier": ("calibration", "multiclass_brier_score"),
    }
    result = {
        name: _summary([_path(metrics, path) for metrics in cell_metrics])
        for name, path in metric_paths.items()
    }
    worst = [
        min(float(row["f1-score"]) for row in metrics["per_family"].values())
        for metrics in cell_metrics
    ]
    result["worst_family_f1"] = _summary(worst)
    result["macro_f1_group_bootstrap_95"] = group_bootstrap_interval(
        tuple(pooled_records),
        tuple(family_labels),
        "macro_f1",
        bootstrap_resamples,
        bootstrap_seed,
    )
    per_family: dict[str, Any] = {}
    for label in family_labels:
        per_family[label] = {
            metric: _summary(
                [float(cell["per_family"][label][source]) for cell in cell_metrics]
            )
            for metric, source in (
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1", "f1-score"),
            )
        }
    result["per_family"] = per_family
    return result


def render_fast_track_report(summary: Mapping[str, Any]) -> str:
    decision = summary["decision"]
    if decision["evidence_valid"] is False:
        answer = "证据无效：输入矩阵或身份校验未通过。"
    elif decision["e1_promoted"]:
        answer = "结论：E1 融合超过 A4，并通过预设晋级门槛。"
    else:
        answer = "结论：E1 未通过预设晋级门槛，继续保留 A4。"
    lines = [
        "# Phase4 Fast Track Result",
        "",
        answer,
        "",
        f"选择性部署就绪：{'是' if decision['selective_deployment_ready'] else '否'}。",
        "",
        "## 三视图 Macro-F1",
        "",
        "| System | closed | fast-to-balanced | balanced-to-fast |",
        "| --- | ---: | ---: | ---: |",
    ]
    for system in ("A0", "A4", "E1"):
        values = summary["systems"][system]
        lines.append(
            "| "
            + system
            + " | "
            + " | ".join(
                f"{values[view]['macro_f1']['mean']:.6f}"
                for view in (
                    "closed",
                    "fast-to-balanced",
                    "balanced-to-fast",
                )
            )
            + " |"
        )
    selective = summary["selective"]
    lines.extend(
        [
            "",
            "## 选择性输出",
            "",
            f"- accepted precision：{_format_optional(selective['accepted_precision'])}",
            f"- coverage：{selective['coverage']:.4f}",
            f"- ECE：{_format_optional(selective['ece'])}",
            "",
            "## Paired gains（E1 - A4）",
            "",
        ]
    )
    for view, gain in summary["paired_gains"].items():
        lines.append(f"- {view}: {gain:+.6f}")
    lines.extend(["", "## Gate results", ""])
    for gate, passed in decision["gates"].items():
        lines.append(f"- {gate}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", f"下一步：{decision['next_action']}", ""])
    return "\n".join(lines)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "mean": mean(normalized),
        "sample_std": stdev(normalized) if len(normalized) > 1 else 0.0,
        "min": min(normalized),
        "max": max(normalized),
    }


def _path(payload: Mapping[str, Any], path: tuple[str, str]) -> float:
    return float(payload[path[0]][path[1]])


def _format_optional(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"
