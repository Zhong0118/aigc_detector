"""Exploratory analysis and figures for text-attribution JSONL collections."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


_ENGLISH_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def analyze_jsonl(
    input_path: str | Path,
    *,
    expected_per_model: int | None = None,
    expected_models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load one JSONL collection and return deterministic EDA statistics."""

    path = Path(input_path)
    rows = list(_read_jsonl(path))
    flat = [_flatten_row(row) for row in rows]
    return _analyze_flat_rows(
        flat,
        input_path=path,
        expected_per_model=expected_per_model,
        expected_models=expected_models,
    )


def write_eda_artifacts(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    expected_per_model: int | None = None,
    expected_models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write JSON/Markdown summaries plus four PNG/PDF diagnostic figures."""

    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = list(_read_jsonl(source))
    flat = [_flatten_row(row) for row in rows]
    analysis = _analyze_flat_rows(
        flat,
        input_path=source,
        expected_per_model=expected_per_model,
        expected_models=expected_models,
    )

    analysis_json = destination / "pilot_eda.analysis.json"
    report_markdown = destination / "pilot_eda.report.md"
    analysis_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_markdown.write_text(
        _render_markdown(analysis),
        encoding="utf-8",
    )

    figure_stems = _create_figures(
        flat,
        analysis,
        destination,
        expected_per_model=expected_per_model,
    )
    return {
        "analysis_json": str(analysis_json),
        "report_markdown": str(report_markdown),
        "figures": [str(path.with_suffix(".png")) for path in figure_stems],
        "figure_pdfs": [str(path.with_suffix(".pdf")) for path in figure_stems],
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield row


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = row.get("prompt") if isinstance(row.get("prompt"), dict) else {}
    parameters = (
        row.get("parameters") if isinstance(row.get("parameters"), dict) else {}
    )
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )
    text = str(row.get("text", ""))
    language = str(prompt.get("language", "unknown"))
    length_value, length_unit = _language_length(text, language)
    finish_reason = row.get("finish_reason")
    return {
        "request_key": str(row.get("request_key", "")),
        "family": str(row.get("family_id", "unknown")),
        "model": str(row.get("request_model", "unknown")),
        "prompt_id": str(prompt.get("prompt_id", "unknown")),
        "language": language,
        "task_type": str(prompt.get("task_type", "unknown")),
        "target_length": str(prompt.get("target_length", "unknown")),
        "temperature": parameters.get("temperature"),
        "text": text,
        "text_length": length_value,
        "text_length_unit": length_unit,
        "finish_reason": finish_reason,
        "truncated": finish_reason in {"length", "max_tokens"},
        "content_filtered": finish_reason == "content_filter",
        "language_mismatch": _language_mismatch(text, language),
        "route_mismatch": bool(row.get("route_mismatch")),
        "elapsed_seconds": _number(row.get("elapsed_seconds")),
        "completion_tokens": _number(usage.get("completion_tokens")),
        "prompt_tokens": _number(usage.get("prompt_tokens")),
        "reasoning_tokens": _number(details.get("reasoning_tokens")) or 0.0,
        "collected_at": _parse_datetime(row.get("collected_at")),
    }


def _analyze_flat_rows(
    rows: list[dict[str, Any]],
    *,
    input_path: Path,
    expected_per_model: int | None,
    expected_models: Mapping[str, str] | None,
) -> dict[str, Any]:
    expected_model_families = dict(expected_models or {})
    families = sorted(
        {row["family"] for row in rows} | set(expected_model_families.values())
    )
    models = sorted(
        {row["model"] for row in rows} | set(expected_model_families)
    )
    prompts = sorted({row["prompt_id"] for row in rows})
    duplicate_counts = Counter(row["text"].strip() for row in rows if row["text"].strip())
    duplicate_excess = sum(count - 1 for count in duplicate_counts.values() if count > 1)
    low_temperature_duplicate_excess = 0
    for text, count in duplicate_counts.items():
        if count <= 1:
            continue
        matches = [row for row in rows if row["text"].strip() == text]
        if all(row["temperature"] == 0.0 for row in matches):
            low_temperature_duplicate_excess += count - 1
    collected = sorted(
        row["collected_at"] for row in rows if row["collected_at"] is not None
    )
    duration = (
        (collected[-1] - collected[0]).total_seconds()
        if len(collected) >= 2
        else None
    )

    by_family: dict[str, Any] = {}
    for family in families:
        subset = [row for row in rows if row["family"] == family]
        expected_family_records = None
        if expected_per_model is not None:
            if expected_model_families:
                expected_family_records = expected_per_model * sum(
                    expected_family == family
                    for expected_family in expected_model_families.values()
                )
            else:
                expected_family_records = expected_per_model
        texts = Counter(row["text"].strip() for row in subset if row["text"].strip())
        family_duplicates = sum(count - 1 for count in texts.values() if count > 1)
        truncated = sum(row["truncated"] for row in subset)
        zh_lengths = [
            row["text_length"] for row in subset if row["language"] == "zh"
        ]
        en_lengths = [
            row["text_length"] for row in subset if row["language"] == "en"
        ]
        latencies = _present(subset, "elapsed_seconds")
        completion_tokens = _present(subset, "completion_tokens")
        by_family[family] = {
            "records": len(subset),
            "expected_records": expected_family_records,
            "missing_to_expected": (
                max(expected_family_records - len(subset), 0)
                if expected_family_records is not None
                else None
            ),
            "coverage_rate": (
                round(len(subset) / expected_family_records, 6)
                if expected_family_records
                else None
            ),
            "truncated_count": truncated,
            "truncated_rate": _ratio(truncated, len(subset)),
            "content_filtered_count": sum(
                row["content_filtered"] for row in subset
            ),
            "language_mismatch_count": sum(
                row["language_mismatch"] for row in subset
            ),
            "duplicate_excess_count": family_duplicates,
            "duplicate_excess_rate": _ratio(family_duplicates, len(subset)),
            "zh_chars_p50": _percentile(zh_lengths, 50),
            "zh_chars_p95": _percentile(zh_lengths, 95),
            "en_words_p50": _percentile(en_lengths, 50),
            "en_words_p95": _percentile(en_lengths, 95),
            "latency_p50_seconds": _percentile(latencies, 50),
            "latency_p95_seconds": _percentile(latencies, 95),
            "completion_tokens_p50": _percentile(completion_tokens, 50),
            "completion_tokens_p95": _percentile(completion_tokens, 95),
            "reasoning_token_rows": sum(
                row["reasoning_tokens"] > 0 for row in subset
            ),
        }

    by_prompt_family: dict[str, dict[str, int]] = {}
    for prompt in prompts:
        counts = Counter(
            row["family"] for row in rows if row["prompt_id"] == prompt
        )
        by_prompt_family[prompt] = {
            family: int(counts.get(family, 0)) for family in families
        }

    length_by_language = {}
    for language in sorted({row["language"] for row in rows}):
        values = [
            row["text_length"] for row in rows if row["language"] == language
        ]
        length_by_language[language] = _distribution(values)

    length_by_target: dict[str, dict[str, Any]] = {}
    for language in sorted({row["language"] for row in rows}):
        length_by_target[language] = {}
        for target in ("short", "medium", "long", "unknown"):
            values = [
                row["text_length"]
                for row in rows
                if row["language"] == language and row["target_length"] == target
            ]
            if values:
                length_by_target[language][target] = _distribution(values)

    duplicate_groups = []
    for text, count in duplicate_counts.most_common():
        if count <= 1:
            continue
        matches = [row for row in rows if row["text"].strip() == text]
        duplicate_groups.append(
            {
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "count": count,
                "families": sorted({row["family"] for row in matches}),
                "models": sorted({row["model"] for row in matches}),
                "prompt_ids": sorted({row["prompt_id"] for row in matches}),
                "temperatures": sorted(
                    {
                        str(row["temperature"])
                        for row in matches
                        if row["temperature"] is not None
                    }
                ),
                "excerpt": text[:120].replace("\n", " "),
            }
        )

    expected_total = None
    if expected_per_model is not None:
        expected_model_count = (
            len(expected_model_families)
            if expected_model_families
            else len(models)
        )
        expected_total = expected_per_model * expected_model_count

    return {
        "input": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "format": "JSON Lines",
        },
        "summary": {
            "records": len(rows),
            "families": len(families),
            "models": len(models),
            "prompts": len(prompts),
            "expected_per_model": expected_per_model,
            "expected_total": expected_total,
            "missing_to_expected": (
                expected_total - len(rows)
                if expected_total is not None
                else None
            ),
            "empty_text_count": sum(not row["text"].strip() for row in rows),
            "truncated_count": sum(row["truncated"] for row in rows),
            "truncated_rate": _ratio(
                sum(row["truncated"] for row in rows),
                len(rows),
            ),
            "content_filtered_count": sum(
                row["content_filtered"] for row in rows
            ),
            "language_mismatch_count": sum(
                row["language_mismatch"] for row in rows
            ),
            "route_mismatch_count": sum(row["route_mismatch"] for row in rows),
            "duplicate_excess_count": duplicate_excess,
            "duplicate_excess_rate": _ratio(duplicate_excess, len(rows)),
            "low_temperature_only_duplicate_excess_count": (
                low_temperature_duplicate_excess
            ),
            "other_duplicate_excess_count": (
                duplicate_excess - low_temperature_duplicate_excess
            ),
            "collection_started_at": (
                collected[0].isoformat() if collected else None
            ),
            "collection_finished_at": (
                collected[-1].isoformat() if collected else None
            ),
            "collection_duration_seconds": (
                round(duration, 3) if duration is not None else None
            ),
            "latency_seconds": _distribution(
                _present(rows, "elapsed_seconds")
            ),
            "completion_tokens": _distribution(
                _present(rows, "completion_tokens")
            ),
        },
        "by_family": by_family,
        "by_prompt_family": by_prompt_family,
        "length_by_language": length_by_language,
        "length_by_target": length_by_target,
        "duplicate_groups": duplicate_groups,
    }


def _language_length(text: str, language: str) -> tuple[int, str]:
    if language == "en":
        return len(_ENGLISH_WORD.findall(text)), "words"
    return len(re.sub(r"\s+", "", text)), "characters"


def _language_mismatch(text: str, language: str) -> bool:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    total = cjk_count + latin_count
    if total == 0:
        return False
    cjk_ratio = cjk_count / total
    if language == "en":
        return cjk_ratio > 0.35
    if language == "zh":
        return cjk_ratio < 0.15 and latin_count > 100
    return False


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _present(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
    ]


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _percentile(values: Sequence[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    result = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(result, 6)


def _distribution(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": round(min(numeric), 6),
        "median": round(float(median(numeric)), 6),
        "p95": _percentile(numeric, 95),
        "max": round(max(numeric), 6),
    }


def _render_markdown(analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    lines = [
        "# Text-attribution pilot EDA report",
        "",
        "## Metadata",
        "",
        f"- Source: `{analysis['input']['path']}`",
        f"- Format: {analysis['input']['format']}",
        f"- Size: {analysis['input']['size_bytes']:,} bytes",
        f"- Records: {summary['records']:,}",
        f"- Families / models / prompts: {summary['families']} / "
        f"{summary['models']} / {summary['prompts']}",
        "",
        "## Quality summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Expected total | {summary['expected_total']} |",
        f"| Missing | {summary['missing_to_expected']} |",
        f"| Empty text | {summary['empty_text_count']} |",
        f"| Truncated | {summary['truncated_count']} "
        f"({summary['truncated_rate']:.2%}) |",
        f"| Content filtered | {summary['content_filtered_count']} |",
        f"| Language mismatch suspects | {summary['language_mismatch_count']} |",
        f"| Exact duplicate excess | {summary['duplicate_excess_count']} "
        f"({summary['duplicate_excess_rate']:.2%}) |",
        "| └ Low-temperature-only duplicate excess | "
        f"{summary['low_temperature_only_duplicate_excess_count']} |",
        f"| └ Other duplicate excess | {summary['other_duplicate_excess_count']} |",
        f"| Route mismatch | {summary['route_mismatch_count']} |",
        "",
        "## Per-family diagnostics",
        "",
        "| Family | n | Missing | Truncated | Exact duplicate excess | "
        "Latency p50 / p95 (s) | Completion tokens p50 / p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, values in analysis["by_family"].items():
        lines.append(
            f"| {family} | {values['records']} | "
            f"{values['missing_to_expected']} | "
            f"{values['truncated_count']} ({values['truncated_rate']:.1%}) | "
            f"{values['duplicate_excess_count']} | "
            f"{_pair(values['latency_p50_seconds'], values['latency_p95_seconds'])} | "
            f"{_pair(values['completion_tokens_p50'], values['completion_tokens_p95'])} |"
        )
    lines.extend(
        [
            "",
            "## Length distributions",
            "",
            "Chinese length is measured as non-whitespace characters; English "
            "length is measured as regex-tokenized words.",
            "",
            "| Language | n | Min | Median | P95 | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for language, values in analysis["length_by_language"].items():
        lines.append(
            f"| {language} | {values['count']} | {values['min']} | "
            f"{values['median']} | {values['p95']} | {values['max']} |"
        )
    lines.extend(
        [
            "",
            "## Key findings",
            "",
        ]
    )
    if summary["missing_to_expected"]:
        lines.append(
            f"- The collection is incomplete: {summary['missing_to_expected']} "
            "planned cells are missing."
        )
    if summary["truncated_count"]:
        lines.append(
            f"- {summary['truncated_count']} rows ended because an output-token limit "
            "was reached (`length` or `max_tokens`); "
            "these should not enter a formal baseline unchanged."
        )
    if summary["content_filtered_count"]:
        lines.append(
            f"- {summary['content_filtered_count']} rows were content-filtered and "
            "should be regenerated or excluded."
        )
    if summary["language_mismatch_count"]:
        lines.append(
            f"- {summary['language_mismatch_count']} rows appear to answer in the "
            "wrong language and require manual confirmation."
        )
    if summary["duplicate_excess_count"]:
        lines.append(
            f"- {summary['duplicate_excess_count']} rows are exact duplicates beyond "
            "the first occurrence. "
            f"{summary['low_temperature_only_duplicate_excess_count']} of them occur "
            "only at temperature 0 and mainly reflect deterministic repeat generation; "
            "the remaining duplicates need closer review."
        )
    lines.extend(
        [
            "- Group train/test splits by `prompt_id`; never randomly split individual rows.",
            "- Treat this collection as a pilot until missing cells, truncation, and "
            "duplicate handling are resolved.",
            "",
            "## Figures",
            "",
            "- `01_coverage_quality.png`: per-family coverage and quality failures.",
            "- `02_prompt_family_coverage.png`: prompt-by-family coverage matrix.",
            "- `03_length_distribution.png`: Chinese character and English word distributions.",
            "- `04_latency_tokens.png`: latency and completion-token distributions.",
            "",
        ]
    )
    return "\n".join(lines)


def _pair(first: Any, second: Any) -> str:
    return f"{first} / {second}"


def _create_figures(
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
    *,
    expected_per_model: int | None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    palette = sns.color_palette("colorblind", n_colors=10)
    frame = pd.DataFrame(rows)
    families = list(analysis["by_family"])
    colors = {family: palette[index % len(palette)] for index, family in enumerate(families)}

    stems: list[Path] = []

    coverage_stem = output_dir / "01_coverage_quality"
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    counts = [analysis["by_family"][family]["records"] for family in families]
    truncated_rates = [
        analysis["by_family"][family]["truncated_rate"] * 100 for family in families
    ]
    duplicate_counts = [
        analysis["by_family"][family]["duplicate_excess_count"]
        for family in families
    ]
    axes[0].barh(families, counts, color=[colors[value] for value in families])
    expected_family_counts = {
        values["expected_records"]
        for values in analysis["by_family"].values()
        if values["expected_records"] is not None
    }
    if len(expected_family_counts) == 1:
        axes[0].axvline(
            expected_family_counts.pop(),
            color="black",
            linestyle="--",
            linewidth=1,
        )
    axes[0].set_xlabel("Successful records")
    axes[0].set_title("A  Coverage")
    axes[1].barh(families, truncated_rates, color=[colors[value] for value in families])
    axes[1].set_xlabel("Truncated rows (%)")
    axes[1].set_title("B  Truncation")
    axes[2].barh(families, duplicate_counts, color=[colors[value] for value in families])
    axes[2].set_xlabel("Exact duplicate excess")
    axes[2].set_title("C  Exact duplicates")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, coverage_stem)
    stems.append(coverage_stem)

    heatmap_stem = output_dir / "02_prompt_family_coverage"
    matrix = pd.DataFrame.from_dict(
        analysis["by_prompt_family"],
        orient="index",
    ).reindex(columns=families)
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="g",
        cmap="viridis",
        vmin=0,
        vmax=max(4, int(matrix.to_numpy().max(initial=0))),
        linewidths=0.35,
        cbar_kws={"label": "Successful records"},
        ax=ax,
    )
    ax.set_xlabel("Model family")
    ax.set_ylabel("Prompt ID")
    ax.set_title("Prompt × family coverage")
    fig.tight_layout()
    _save_figure(fig, heatmap_stem)
    stems.append(heatmap_stem)

    length_stem = output_dir / "03_length_distribution"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = [
        ("zh", "Chinese output length (non-whitespace characters)"),
        ("en", "English output length (words)"),
    ]
    for axis, (language, ylabel) in zip(axes, panels):
        subset = frame[frame["language"] == language]
        if not subset.empty:
            sns.boxplot(
                data=subset,
                x="family",
                y="text_length",
                hue="family",
                palette=colors,
                legend=False,
                showfliers=False,
                ax=axis,
            )
            sns.stripplot(
                data=subset,
                x="family",
                y="text_length",
                color="black",
                alpha=0.18,
                size=1.8,
                jitter=0.22,
                ax=axis,
            )
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_title("A  Chinese")
    axes[1].set_title("B  English")
    fig.tight_layout()
    _save_figure(fig, length_stem)
    stems.append(length_stem)

    latency_stem = output_dir / "04_latency_tokens"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    latency_frame = frame.dropna(subset=["elapsed_seconds"])
    token_frame = frame.dropna(subset=["completion_tokens"])
    if not latency_frame.empty:
        sns.boxplot(
            data=latency_frame,
            x="family",
            y="elapsed_seconds",
            hue="family",
            palette=colors,
            legend=False,
            showfliers=False,
            ax=axes[0],
        )
        axes[0].set_yscale("log")
    if not token_frame.empty:
        sns.boxplot(
            data=token_frame,
            x="family",
            y="completion_tokens",
            hue="family",
            palette=colors,
            legend=False,
            showfliers=False,
            ax=axes[1],
        )
    axes[0].set_ylabel("Latency (seconds, log scale)")
    axes[1].set_ylabel("Completion tokens")
    for index, axis in enumerate(axes):
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(("A  Latency", "B  Token usage")[index])
    fig.tight_layout()
    _save_figure(fig, latency_stem)
    stems.append(latency_stem)

    plt.close("all")
    return stems


def _save_figure(figure: Any, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.clear()
