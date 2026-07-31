"""Run a validated, resumable text-attribution collection profile."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from research.text_attribution.config import load_local_environment, resolve_gateway
from research.text_attribution.gateway import OpenAICompatibleGateway
from research.text_attribution.production import build_plan_from_config, run_collection
from research.text_attribution.run_config import load_run_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an attribution collection profile.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--empty-max-attempts", type=int, default=2)
    parser.add_argument("--empty-circuit-threshold", type=int, default=2)
    parser.add_argument("--transient-circuit-threshold", type=int, default=2)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args(argv)


def build_preview(config, plan) -> dict:
    return {
        "run_id": config.run_id,
        "gateway": config.gateway,
        "request_count": len(plan),
        "models": [
            {
                "family_id": model.family_id,
                "model_id": model.model_id,
                "variant_role": model.variant_role,
            }
            for model in config.models
        ],
        "prompt_count": len(config.prompts),
        "sampling_profiles": [
            {
                "sampling_id": profile.sampling_id,
                "temperature": profile.temperature,
                "top_p": profile.top_p,
                "repetitions": profile.repetitions,
                "prompt_ids": (
                    list(profile.prompt_ids)
                    if profile.prompt_ids is not None
                    else ["*"]
                ),
            }
            for profile in config.sampling
        ],
        "requests_per_model": config.requests_per_model,
        "output_budgets": dict(
            sorted(
                Counter(
                    str(item.parameters.max_tokens)
                    for item in plan
                ).items()
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_run_config(args.config)
    plan = build_plan_from_config(config)
    preview = build_preview(config, plan)
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    if not args.confirm:
        raise SystemExit("Real collection requires --confirm")

    output = args.output or Path(f"data/attribution/raw/{config.run_id}.jsonl")
    errors = args.error_output or Path(
        f"data/attribution/errors/{config.run_id}.errors.jsonl"
    )
    environment = load_local_environment(args.env_file)
    gateway_config = replace(
        resolve_gateway(config.gateway, environ=environment),
        timeout_seconds=args.timeout_seconds,
    )
    gateway = OpenAICompatibleGateway(gateway_config)
    summary = run_collection(
        config,
        gateway=gateway,
        gateway_factory=lambda: OpenAICompatibleGateway(gateway_config),
        output_path=output,
        error_path=errors,
        plan=plan,
        sleep_seconds=args.sleep_seconds,
        max_attempts=args.max_attempts,
        empty_max_attempts=args.empty_max_attempts,
        empty_circuit_threshold=args.empty_circuit_threshold,
        transient_circuit_threshold=args.transient_circuit_threshold,
        retry_base_seconds=args.retry_base_seconds,
        max_workers=args.max_workers,
        progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True),
    )
    print(
        json.dumps(
            {
                "status": "complete" if not summary.stopped_reason else "stopped",
                "run_id": config.run_id,
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
