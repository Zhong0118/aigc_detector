#!/usr/bin/env python3
"""Write the frozen eight-seed Phase3 winner/comparator validation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from research.text_attribution.phase3_config import load_phase3_config
from research.text_attribution.phase3_selection import (
    build_phase3_confirmation_summary,
)
from scripts.run_family_attribution_phase3 import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine the frozen 3 screening + 5 confirmation seeds."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--winner-screening-root", required=True)
    parser.add_argument("--winner-confirmation-root", required=True)
    parser.add_argument("--comparator-screening-root", required=True)
    parser.add_argument("--comparator-confirmation-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_phase3_confirmation_summary(
        config=load_phase3_config(args.config),
        winner_screening_root=Path(args.winner_screening_root).resolve(),
        winner_confirmation_root=Path(args.winner_confirmation_root).resolve(),
        comparator_screening_root=Path(args.comparator_screening_root).resolve(),
        comparator_confirmation_root=Path(args.comparator_confirmation_root).resolve(),
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"confirmation summary already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
