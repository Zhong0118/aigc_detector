from __future__ import annotations

import json
from pathlib import Path


FAMILIES = ("claude", "openai")
MODELS = {
    "claude-haiku-test": "claude",
    "claude-sonnet-test": "claude",
    "gpt-fast-test": "openai",
    "gpt-balanced-test": "openai",
}


def write_model_catalog(path: Path) -> None:
    models = [
        {
            "family_id": family_id,
            "model_id": model_id,
            "enabled": True,
            "variant_role": "fast" if index % 2 == 0 else "balanced",
        }
        for index, (model_id, family_id) in enumerate(MODELS.items())
    ]
    path.write_text(
        json.dumps(
            {"schema_version": 5, "catalog_id": "test-v5", "models": models},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def raw_row(
    model_id: str,
    group_index: int,
    family_id: str | None = None,
) -> dict[str, object]:
    family = family_id if family_id is not None else MODELS[model_id]
    language = "zh" if group_index % 2 == 0 else "en"
    text = (
        f"{model_id} 在第 {group_index} 组给出了完整、独特的中文回答。"
        if language == "zh"
        else f"{model_id} provides a complete and unique English answer for group {group_index}."
    )
    return {
        "request_key": f"{model_id}-{group_index}",
        "family_id": family,
        "request_model": model_id,
        "text": text,
        "finish_reason": "stop",
        "route_mismatch": False,
        "repeat_index": 0,
        "prompt": {
            "prompt_id": f"{language}-task-{group_index:03d}",
            "language": language,
            "task_type": "explanation",
            "target_length": "medium",
        },
        "parameters": {"temperature": 0},
    }


def write_dataset(path: Path, groups: int = 12) -> None:
    rows = [
        raw_row(model_id, group_index)
        for group_index in range(groups)
        for model_id in MODELS
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
