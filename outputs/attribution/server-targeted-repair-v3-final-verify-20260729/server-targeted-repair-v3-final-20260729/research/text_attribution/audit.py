"""Deterministic quality and coverage audit for attribution JSONL data."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from .production import build_plan_from_config, request_key
from .run_config import ResolvedRunConfig


def audit_collection(
    config: ResolvedRunConfig,
    *,
    input_path: str | Path,
    error_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = list(_read_jsonl(Path(input_path)))
    error_rows = (
        list(_read_jsonl(Path(error_path)))
        if error_path is not None and Path(error_path).exists()
        else []
    )
    plan = build_plan_from_config(config)
    expected = {request_key(item): item for item in plan}
    successful_keys = {
        str(row.get("request_key"))
        for row in rows
        if row.get("request_key") in expected
    }
    missing_keys = [key for key in expected if key not in successful_keys]
    texts = [
        str(row.get("text", "")).strip()
        for row in rows
        if str(row.get("text", "")).strip()
    ]
    errors_by_status = Counter(
        str(row.get("http_status"))
        if row.get("http_status") is not None
        else "none"
        for row in error_rows
    )
    return {
        "run_id": config.run_id,
        "expected_requests": config.request_count,
        "records": len(rows),
        "unique_successes": len(successful_keys),
        "error_records": len(error_rows),
        "missing_requests": len(missing_keys),
        "coverage_rate": round(
            len(successful_keys) / config.request_count if config.request_count else 0,
            6,
        ),
        "empty_text_count": sum(
            not str(row.get("text", "")).strip() for row in rows
        ),
        "truncated_count": sum(
            row.get("finish_reason") in {"length", "max_tokens"} for row in rows
        ),
        "route_mismatch_count": sum(
            bool(row.get("route_mismatch")) for row in rows
        ),
        "duplicate_text_count": len(texts) - len(set(texts)),
        "counts_by_model": _counter(rows, "request_model"),
        "counts_by_family": _counter(rows, "family_id"),
        "counts_by_language": _nested_counter(rows, "prompt", "language"),
        "counts_by_task": _nested_counter(rows, "prompt", "task_type"),
        "counts_by_repeat": _counter(rows, "repeat_index"),
        "counts_by_sampling": dict(
            sorted(
                Counter(_sampling_key(row) for row in rows).items(),
                key=lambda item: item[0],
            )
        ),
        "errors_by_status": dict(sorted(errors_by_status.items())),
        "missing_cells": [
            {
                "request_key": key,
                "model": expected[key].model,
                "prompt_id": expected[key].prompt.prompt_id,
                "temperature": expected[key].parameters.temperature,
                "top_p": expected[key].parameters.top_p,
                "repeat_index": expected[key].repeat_index,
            }
            for key in missing_keys
        ],
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
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


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key)) for row in rows)
    return dict(sorted(counts.items()))


def _nested_counter(
    rows: list[dict[str, Any]],
    parent: str,
    key: str,
) -> dict[str, int]:
    counts = Counter(
        str(row.get(parent, {}).get(key))
        for row in rows
        if isinstance(row.get(parent), dict)
    )
    return dict(sorted(counts.items()))


def _sampling_key(row: dict[str, Any]) -> str:
    parameters = row.get("parameters")
    if not isinstance(parameters, dict):
        return "missing"
    return (
        f"temperature={parameters.get('temperature')}|"
        f"top_p={parameters.get('top_p')}|"
        f"max_tokens={parameters.get('max_tokens')}"
    )
