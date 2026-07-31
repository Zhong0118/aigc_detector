"""Strict assembly of an immutable source plus targeted replacement batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .production import build_plan_from_config, request_key
from .repair import ResolvedRepairManifest
from .run_config import ResolvedRunConfig


@dataclass(frozen=True)
class AssemblyResult:
    records: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def assemble_v3_records(
    *,
    config: ResolvedRunConfig,
    repair_manifest: ResolvedRepairManifest,
    base_records: Iterable[dict[str, Any]],
    replacement_records: Iterable[dict[str, Any]],
    repair_records: Iterable[dict[str, Any]],
    retired_models: set[str] | frozenset[str],
    replacement_models: set[str] | frozenset[str] | None = None,
) -> AssemblyResult:
    """Assemble records only when they exactly cover the final logical plan."""

    expected_plan = build_plan_from_config(config)
    expected_by_key = {request_key(item): item for item in expected_plan}
    if len(expected_by_key) != len(expected_plan):
        raise ValueError("Final logical plan contains duplicate request keys")
    model_by_id = {model.model_id: model for model in config.models}
    superseded_by_key = {
        cell.supersedes_request_key: cell for cell in repair_manifest.cells
    }
    repair_physical_by_key = {
        request_key(item): cell
        for item, cell in zip(repair_manifest.plan, repair_manifest.cells)
    }

    base_rows = list(base_records)
    replacement_rows = list(replacement_records)
    repair_rows = list(repair_records)
    observed_replacement_models = {
        _required_text(row, "request_model") for row in replacement_rows
    }
    replacement_models = (
        set(replacement_models)
        if replacement_models is not None
        else observed_replacement_models
    )
    unexpected_observed_models = observed_replacement_models - replacement_models
    if unexpected_observed_models:
        raise ValueError(
            "replacement records contain models outside replacement config: "
            f"{sorted(unexpected_observed_models)}"
        )
    unknown_replacement_models = replacement_models - set(model_by_id)
    if unknown_replacement_models:
        raise ValueError(
            "replacement records contain unknown models: "
            f"{sorted(unknown_replacement_models)}"
        )

    expected_replacement_keys = {
        key
        for key, item in expected_by_key.items()
        if item.model in replacement_models
    }
    replacement_by_key = _index_unique(replacement_rows, "replacement")
    replacement_missing = expected_replacement_keys - set(replacement_by_key)
    replacement_extra = set(replacement_by_key) - expected_replacement_keys
    if replacement_missing:
        raise ValueError(
            f"replacement batch missing {len(replacement_missing)} logical rows"
        )
    if replacement_extra:
        raise ValueError(
            f"replacement batch has {len(replacement_extra)} unexpected rows"
        )

    repair_by_key = _index_unique(repair_rows, "repair")
    repair_missing = set(repair_physical_by_key) - set(repair_by_key)
    repair_extra = set(repair_by_key) - set(repair_physical_by_key)
    if repair_missing:
        raise ValueError(f"repair batch missing {len(repair_missing)} physical rows")
    if repair_extra:
        raise ValueError(
            f"repair batch has {len(repair_extra)} unexpected physical rows"
        )

    expected_base_keys = (
        set(expected_by_key)
        - expected_replacement_keys
        - set(superseded_by_key)
    )
    base_by_key: dict[str, dict[str, Any]] = {}
    retired_count = 0
    superseded_count = 0
    ignored_count = 0
    for row in base_rows:
        model_id = _required_text(row, "request_model")
        key = _required_text(row, "request_key")
        if model_id in retired_models:
            retired_count += 1
            continue
        if key in superseded_by_key:
            superseded_count += 1
            continue
        if key not in expected_base_keys:
            ignored_count += 1
            continue
        if key in base_by_key:
            raise ValueError(f"base records contain duplicate request_key: {key}")
        base_by_key[key] = row

    base_missing = expected_base_keys - set(base_by_key)
    if base_missing:
        raise ValueError(f"base records missing {len(base_missing)} logical rows")

    final_by_key: dict[str, dict[str, Any]] = {}
    for key, row in base_by_key.items():
        final_by_key[key] = _normalize_row(
            row, config=config, model_by_id=model_by_id
        )
    for key, row in replacement_by_key.items():
        _validate_quality(row, source="replacement")
        final_by_key[key] = _normalize_row(
            row, config=config, model_by_id=model_by_id
        )
    for physical_key, row in repair_by_key.items():
        cell = repair_physical_by_key[physical_key]
        _validate_quality(row, source="repair")
        normalized = _normalize_row(
            row, config=config, model_by_id=model_by_id
        )
        normalized["request_key"] = cell.supersedes_request_key
        normalized["repair"] = {
            "repair_id": cell.repair_id,
            "reason": cell.reason,
            "supersedes_request_key": cell.supersedes_request_key,
            "collection_request_key": physical_key,
        }
        final_by_key[cell.supersedes_request_key] = normalized

    missing_final = set(expected_by_key) - set(final_by_key)
    extra_final = set(final_by_key) - set(expected_by_key)
    if missing_final or extra_final:
        raise ValueError(
            "assembled logical key set mismatch: "
            f"missing={len(missing_final)}, extra={len(extra_final)}"
        )
    ordered_records = tuple(
        final_by_key[request_key(item)] for item in expected_plan
    )
    text_owner: dict[str, str] = {}
    for row in ordered_records:
        _validate_quality(row, source="final")
        normalized_text = str(row["text"]).strip()
        previous_key = text_owner.get(normalized_text)
        if previous_key is not None:
            raise ValueError(
                "final records contain exact duplicate text for request keys "
                f"{previous_key} and {row['request_key']}"
            )
        text_owner[normalized_text] = str(row["request_key"])

    report = {
        "run_id": config.run_id,
        "expected_records": len(expected_plan),
        "assembled_records": len(ordered_records),
        "kept_from_base": len(base_by_key),
        "added_replacements": len(replacement_by_key),
        "added_repairs": len(repair_by_key),
        "removed_retired_rows": retired_count,
        "removed_superseded_rows": superseded_count,
        "ignored_non_final_rows": ignored_count,
        "ignored_non_v3_rows": ignored_count,
        "replacement_models": sorted(replacement_models),
        "retired_models": sorted(retired_models),
        "status": "valid",
    }
    return AssemblyResult(records=ordered_records, report=report)


def _index_unique(
    records: Iterable[dict[str, Any]], source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in records:
        key = _required_text(row, "request_key")
        if key in indexed:
            raise ValueError(f"{source} records contain duplicate request_key: {key}")
        indexed[key] = row
    return indexed


def _normalize_row(
    row: dict[str, Any],
    *,
    config: ResolvedRunConfig,
    model_by_id: dict[str, Any],
) -> dict[str, Any]:
    model_id = _required_text(row, "request_model")
    model = model_by_id.get(model_id)
    if model is None:
        raise ValueError(f"row uses a model outside the V3 catalog: {model_id}")
    normalized = dict(row)
    normalized["source_run_id"] = str(
        row.get("source_run_id") or row.get("run_id") or "unknown"
    )
    normalized["run_id"] = config.run_id
    normalized["family_id"] = model.family_id
    normalized["model_variant_role"] = model.variant_role
    return normalized


def _validate_quality(row: dict[str, Any], *, source: str) -> None:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{source} row contains empty text")
    if bool(row.get("route_mismatch", False)):
        raise ValueError(f"{source} row contains a route mismatch")
    finish_reason = str(row.get("finish_reason") or "").lower()
    if finish_reason in {"length", "max_tokens"}:
        raise ValueError(f"{source} row is truncated")


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty string")
    return value.strip()
