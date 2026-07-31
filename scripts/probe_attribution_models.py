"""Probe candidate concrete model routes before changing the production catalog."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from research.text_attribution.collection import write_jsonl_record
from research.text_attribution.config import load_local_environment, resolve_gateway
from research.text_attribution.gateway import OpenAICompatibleGateway
from research.text_attribution.model_probe import (
    load_candidate_manifest,
    probe_candidates,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short bilingual checks against candidate model routes."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_candidate_manifest(args.config)
    preview = {
        "gateway": manifest.gateway,
        "candidate_count": len(manifest.candidates),
        "request_count": len(manifest.candidates) * 2,
        "max_workers": args.max_workers,
        "max_tokens": args.max_tokens,
        "candidates": [
            {
                "family_id": candidate.family_id,
                "model_id": candidate.model_id,
                "replacement_for": candidate.replacement_for,
                "variant_role": candidate.variant_role,
            }
            for candidate in manifest.candidates
        ],
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    if not args.confirm:
        raise SystemExit("Real model probes require --confirm")

    environment = load_local_environment(args.env_file)
    gateway_config = replace(
        resolve_gateway(manifest.gateway, environ=environment),
        timeout_seconds=args.timeout_seconds,
    )
    results = probe_candidates(
        manifest,
        gateway_factory=lambda: OpenAICompatibleGateway(gateway_config),
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
    )

    output = args.output or Path(
        f"outputs/attribution/model-probes/{args.config.stem}.jsonl"
    )
    summary_output = args.summary_output or output.with_suffix(".summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    for result in results:
        write_jsonl_record(output, result)

    eligible = [row["model_id"] for row in results if row["eligible"]]
    summary = {
        **preview,
        "eligible_count": len(eligible),
        "ineligible_count": len(results) - len(eligible),
        "eligible_models": eligible,
        "output": str(output),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(eligible) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
