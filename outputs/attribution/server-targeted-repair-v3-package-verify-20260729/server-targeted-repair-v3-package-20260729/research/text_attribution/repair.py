"""Validated, explicit collection plans for replacing individual bad samples."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from .collection import CollectionParameters, PlannedRequest
from .run_config import (
    ModelSelection,
    ResolvedRunConfig,
    load_model_catalog,
    load_prompt_manifest,
)


@dataclass(frozen=True)
class RepairCell:
    repair_id: str
    model_id: str
    prompt_id: str
    temperature: float
    top_p: float | None
    max_tokens: int
    repeat_index: int
    supersedes_request_key: str
    reason: str


@dataclass(frozen=True)
class ResolvedRepairManifest:
    source_path: Path
    run_id: str
    gateway: str
    cells: tuple[RepairCell, ...]
    plan: tuple[PlannedRequest, ...]
    collection_config: ResolvedRunConfig


def load_repair_manifest(path: str | Path) -> ResolvedRepairManifest:
    """Load a repair manifest and resolve every cell to one physical request."""

    source = Path(path).resolve()
    payload = _load_json(source)
    run_id = _required_string(payload, "run_id")
    gateway = _required_string(payload, "gateway")
    catalog_path = _resolve_reference(source, payload, "model_catalog")
    prompts_path = _resolve_reference(source, payload, "prompt_manifest")
    catalog = load_model_catalog(catalog_path)
    prompts = load_prompt_manifest(prompts_path)
    model_by_id = {model.model_id: model for model in catalog}
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}

    rows = payload.get("cells")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Repair manifest must contain a non-empty `cells` list")

    cells: list[RepairCell] = []
    plan: list[PlannedRequest] = []
    selected_models: list[ModelSelection] = []
    selected_prompts = []
    seen_repairs: set[str] = set()
    seen_superseded: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every repair cell must be an object")
        cell = _load_cell(row)
        if cell.repair_id in seen_repairs:
            raise ValueError(f"Duplicate repair_id: {cell.repair_id}")
        if cell.supersedes_request_key in seen_superseded:
            raise ValueError(
                "Duplicate supersedes_request_key: "
                f"{cell.supersedes_request_key}"
            )
        model = model_by_id.get(cell.model_id)
        if model is None:
            raise ValueError(f"Unknown repair model_id: {cell.model_id}")
        prompt = prompt_by_id.get(cell.prompt_id)
        if prompt is None:
            raise ValueError(f"Unknown repair prompt_id: {cell.prompt_id}")

        physical_prompt = replace(prompt, max_output_tokens=cell.max_tokens)
        request = PlannedRequest(
            gateway_name=gateway,
            model=cell.model_id,
            prompt=physical_prompt,
            parameters=CollectionParameters(
                temperature=cell.temperature,
                top_p=cell.top_p,
                max_tokens=cell.max_tokens,
                sampling_id="repair-v3",
            ),
            repeat_index=cell.repeat_index,
        )
        cells.append(cell)
        plan.append(request)
        seen_repairs.add(cell.repair_id)
        seen_superseded.add(cell.supersedes_request_key)
        if model not in selected_models:
            selected_models.append(model)
        if physical_prompt not in selected_prompts:
            selected_prompts.append(physical_prompt)

    collection_config = ResolvedRunConfig(
        source_path=source,
        run_id=run_id,
        gateway=gateway,
        models=tuple(selected_models),
        prompts=tuple(selected_prompts),
        sampling=(),
        repetitions=None,
        max_requests=len(plan),
    )
    return ResolvedRepairManifest(
        source_path=source,
        run_id=run_id,
        gateway=gateway,
        cells=tuple(cells),
        plan=tuple(plan),
        collection_config=collection_config,
    )


def _load_cell(row: dict[str, Any]) -> RepairCell:
    top_p_value = row.get("top_p")
    if top_p_value is not None:
        top_p_value = float(top_p_value)
        if not 0 < top_p_value <= 1:
            raise ValueError("top_p must be null or in (0, 1]")
    temperature = float(row.get("temperature", 0.0))
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    max_tokens = int(row.get("max_tokens", 0))
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    repeat_index = int(row.get("repeat_index", -1))
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")
    reason = _required_string(row, "reason")
    if reason not in {"truncated", "exact_duplicate"}:
        raise ValueError(f"Unsupported repair reason: {reason}")
    return RepairCell(
        repair_id=_required_string(row, "repair_id"),
        model_id=_required_string(row, "model_id"),
        prompt_id=_required_string(row, "prompt_id"),
        temperature=temperature,
        top_p=top_p_value,
        max_tokens=max_tokens,
        repeat_index=repeat_index,
        supersedes_request_key=_required_string(row, "supersedes_request_key"),
        reason=reason,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve_reference(source: Path, payload: dict[str, Any], key: str) -> Path:
    value = _required_string(payload, key)
    referenced = Path(value)
    if not referenced.is_absolute():
        referenced = source.parent / referenced
    return referenced.resolve()


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty string")
    return value.strip()
