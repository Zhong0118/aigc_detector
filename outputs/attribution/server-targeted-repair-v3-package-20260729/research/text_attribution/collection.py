"""Deterministic request planning and secret-safe JSONL publication."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .prompts import PromptSpec


SECRET_FIELD_MARKERS = ("api_key", "authorization", "secret", "bearer")


@dataclass(frozen=True)
class CollectionParameters:
    temperature: float
    top_p: float | None
    max_tokens: int
    sampling_id: str = "legacy"


@dataclass(frozen=True)
class PlannedRequest:
    gateway_name: str
    model: str
    prompt: PromptSpec
    parameters: CollectionParameters
    repeat_index: int = 0


def build_request_plan(
    *,
    gateway_name: str,
    models: Iterable[str],
    prompts: Iterable[PromptSpec],
    parameters: Iterable[CollectionParameters],
    max_requests: int,
    repetitions: int = 1,
) -> list[PlannedRequest]:
    """Build a deterministic Cartesian request plan with a mandatory hard cap."""

    if max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    plan: list[PlannedRequest] = []
    for model in models:
        if not str(model).strip():
            raise ValueError("model IDs must not be blank")
        for prompt in prompts:
            for parameter_set in parameters:
                for repeat_index in range(repetitions):
                    plan.append(
                        PlannedRequest(
                            gateway_name=gateway_name,
                            model=str(model),
                            prompt=prompt,
                            parameters=parameter_set,
                            repeat_index=repeat_index,
                        )
                    )
                    if len(plan) >= max_requests:
                        return plan
    return plan


def write_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append one allow-listed record and force it to disk."""

    secret_path = _find_secret_field(record)
    if secret_path:
        raise ValueError(f"Refusing to write secret-shaped field: {secret_path}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _find_secret_field(value: Any, path: str = "") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}" if path else key_name
            lowered = key_name.lower()
            if any(marker in lowered for marker in SECRET_FIELD_MARKERS):
                return child_path
            found = _find_secret_field(item, child_path)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_secret_field(item, f"{path}[{index}]")
            if found:
                return found
    return None
