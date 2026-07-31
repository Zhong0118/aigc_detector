"""Strictly assemble one attribution dataset from base and targeted batches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from research.text_attribution.assembly import assemble_v3_records
from research.text_attribution.repair import load_repair_manifest
from research.text_attribution.run_config import load_run_config


RETIRED_MODELS = (
    "gpt-4.1-nano-2025-04-14",
    "deepseek-v4-pro",
    "glm-4.5-air",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly assemble a base dataset and targeted batches."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--replacement-config", type=Path)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacements", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.set_defaults(retired_models=RETIRED_MODELS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_run_config(args.config)
    selected_models = {model.model_id for model in config.models}
    retired_models = frozenset(args.retired_models)
    if args.base_config:
        base_config = load_run_config(args.base_config)
        retired_models = frozenset(
            model.model_id
            for model in base_config.models
            if model.model_id not in selected_models
        )
    replacement_models = None
    if args.replacement_config:
        replacement_config = load_run_config(args.replacement_config)
        replacement_models = {
            model.model_id for model in replacement_config.models
        }
    repair_manifest = load_repair_manifest(args.repair_manifest)
    result = assemble_v3_records(
        config=config,
        repair_manifest=repair_manifest,
        base_records=_read_jsonl(args.base),
        replacement_records=_read_jsonl(args.replacements),
        repair_records=_read_jsonl(args.repairs),
        retired_models=retired_models,
        replacement_models=replacement_models,
    )
    report_path = args.report or args.output.with_suffix(".assembly.json")
    _write_jsonl_atomic(args.output, result.records)
    _write_json_atomic(report_path, result.report)
    print(
        json.dumps(
            {
                **result.report,
                "output": str(args.output),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            records.append(row)
    return records


def _write_jsonl_atomic(
    path: Path, records: Sequence[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
