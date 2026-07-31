"""Audit coverage and quality of one attribution collection run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from research.text_attribution.audit import audit_collection
from research.text_attribution.run_config import load_run_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit attribution JSONL data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--errors", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on missing, empty, truncated, route-mismatched, or "
            "exact-duplicate responses."
        ),
    )
    parser.add_argument(
        "--minimum-model-coverage",
        type=float,
        help=(
            "Accept an incomplete training collection when every model has at "
            "least this fraction of usable planned records."
        ),
    )
    args = parser.parse_args(argv)
    if (
        args.minimum_model_coverage is not None
        and not 0 < args.minimum_model_coverage <= 1
    ):
        parser.error("--minimum-model-coverage must be in (0, 1]")

    config = load_run_config(args.config)
    report = audit_collection(
        config,
        input_path=args.input,
        error_path=args.errors,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.minimum_model_coverage is None:
        failed = report["missing_requests"] != 0
    else:
        failed = (
            report["minimum_model_usable_coverage"]
            < args.minimum_model_coverage
        )
    if args.strict:
        failed = failed or any(
            report[key] != 0
            for key in (
                "empty_text_count",
                "truncated_count",
                "route_mismatch_count",
                "duplicate_text_count",
            )
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
