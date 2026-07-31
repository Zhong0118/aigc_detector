"""Generate EDA reports and figures for one attribution JSONL collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from research.text_attribution.eda import write_eda_artifacts
from research.text_attribution.run_config import load_run_config


def _resolve_expected_universe(
    config_path: Path | None,
    expected_per_model: int | None,
) -> tuple[int | None, dict[str, str]]:
    if config_path is None:
        return expected_per_model, {}
    resolved = load_run_config(config_path)
    configured_per_model = resolved.requests_per_model
    if (
        expected_per_model is not None
        and expected_per_model != configured_per_model
    ):
        raise ValueError(
            "--expected-per-model does not match the selected run config: "
            f"{expected_per_model} != {configured_per_model}"
        )
    return configured_per_model, {
        model.model_id: model.family_id for model in resolved.models
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze attribution JSONL quality and generate figures."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-per-model", type=int)
    args = parser.parse_args(argv)

    expected_per_model, expected_models = _resolve_expected_universe(
        args.config,
        args.expected_per_model,
    )
    artifacts = write_eda_artifacts(
        args.input,
        args.output_dir,
        expected_per_model=expected_per_model,
        expected_models=expected_models,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
