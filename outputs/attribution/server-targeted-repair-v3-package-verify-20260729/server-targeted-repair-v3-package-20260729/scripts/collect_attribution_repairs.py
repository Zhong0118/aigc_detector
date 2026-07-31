"""Collect only the explicitly listed attribution repair cells."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from research.text_attribution.config import load_local_environment, resolve_gateway
from research.text_attribution.gateway import OpenAICompatibleGateway
from research.text_attribution.production import request_key, run_collection
from research.text_attribution.repair import (
    ResolvedRepairManifest,
    load_repair_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect an explicit set of attribution repair cells."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--empty-max-attempts", type=int, default=2)
    parser.add_argument("--empty-circuit-threshold", type=int, default=2)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args(argv)


def build_preview(manifest: ResolvedRepairManifest) -> dict:
    return {
        "run_id": manifest.run_id,
        "gateway": manifest.gateway,
        "request_count": len(manifest.plan),
        "repairs": [
            {
                "repair_id": cell.repair_id,
                "reason": cell.reason,
                "model_id": cell.model_id,
                "prompt_id": cell.prompt_id,
                "repeat_index": cell.repeat_index,
                "max_tokens": cell.max_tokens,
                "supersedes_request_key": cell.supersedes_request_key,
                "collection_request_key": request_key(item),
            }
            for cell, item in zip(manifest.cells, manifest.plan)
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_repair_manifest(args.manifest)
    preview = build_preview(manifest)
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    if not args.confirm:
        raise SystemExit("Real repair collection requires --confirm")

    output = args.output or Path(
        f"data/attribution/raw/{manifest.run_id}.jsonl"
    )
    errors = args.error_output or Path(
        f"data/attribution/errors/{manifest.run_id}.errors.jsonl"
    )
    environment = load_local_environment(args.env_file)
    gateway_config = replace(
        resolve_gateway(manifest.gateway, environ=environment),
        timeout_seconds=args.timeout_seconds,
    )
    gateway = OpenAICompatibleGateway(gateway_config)
    summary = run_collection(
        manifest.collection_config,
        gateway=gateway,
        gateway_factory=lambda: OpenAICompatibleGateway(gateway_config),
        output_path=output,
        error_path=errors,
        plan=manifest.plan,
        sleep_seconds=args.sleep_seconds,
        max_attempts=args.max_attempts,
        empty_max_attempts=args.empty_max_attempts,
        empty_circuit_threshold=args.empty_circuit_threshold,
        retry_base_seconds=args.retry_base_seconds,
        max_workers=args.max_workers,
        progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True),
    )
    print(
        json.dumps(
            {
                "status": "complete" if not summary.stopped_reason else "stopped",
                "run_id": manifest.run_id,
                "planned": summary.planned,
                "written": summary.written,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "circuit_skipped": summary.circuit_skipped,
                "stopped_reason": summary.stopped_reason,
                "output": str(output),
                "error_output": str(errors),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if summary.stopped_reason:
        return 2
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
