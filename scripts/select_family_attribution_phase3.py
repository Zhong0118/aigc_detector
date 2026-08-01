#!/usr/bin/env python3
"""Select and promote one complete Phase3 validation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from research.text_attribution.phase3_config import load_phase3_config
from research.text_attribution.phase3_selection import (
    Phase3SelectionRequest,
    select_phase3_stage,
    write_phase3_stage_decision,
)
from scripts.run_family_attribution_phase3 import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the frozen nine Phase3 validation promotion gates."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--stage", choices=("A1", "A2", "A3", "A4"), required=True)
    parser.add_argument("--parent-root")
    parser.add_argument("--candidate-root", action="append", default=[])
    parser.add_argument("--upstream-decision")
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    is_a2_skip_probe = args.stage == "A2" and args.upstream_decision is not None
    if not is_a2_skip_probe and not args.parent_root:
        parser.error("--parent-root is required for matrix selection")
    if not is_a2_skip_probe and not args.candidate_root:
        parser.error("at least one --candidate-root is required")
    if args.stage != "A1" and len(args.candidate_root) > 1:
        parser.error(f"--stage {args.stage} accepts exactly one candidate matrix")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request = Phase3SelectionRequest(
        config=load_phase3_config(args.config),
        stage_id=args.stage,
        parent_run_root=(
            None if args.parent_root is None else Path(args.parent_root).resolve()
        ),
        candidate_run_roots=tuple(
            Path(path).resolve() for path in args.candidate_root
        ),
        output_dir=Path(args.output_dir).resolve(),
        upstream_decision=(
            None
            if args.upstream_decision is None
            else Path(args.upstream_decision).resolve()
        ),
    )
    decision = select_phase3_stage(request)
    paths = write_phase3_stage_decision(decision, request.output_dir)
    print(
        json.dumps(
            {
                "stage_id": decision.stage_id,
                "status": decision.status,
                "promoted": decision.promoted,
                "selected_recipe_id": decision.selected_recipe_id,
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
