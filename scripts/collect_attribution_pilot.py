"""List gateway models and collect a strictly limited attribution pilot.

Examples:

    python -m scripts.collect_attribution_pilot --gateway closeai --list-models
    python -m scripts.collect_attribution_pilot \
        --gateway closeai \
        --models model-a,model-b \
        --prompt-limit 4 \
        --max-requests 8 \
        --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from research.text_attribution.collection import (
    CollectionParameters,
    PlannedRequest,
    build_request_plan,
    write_jsonl_record,
)
from research.text_attribution.config import load_local_environment, resolve_gateway
from research.text_attribution.gateway import OpenAICompatibleGateway
from research.text_attribution.prompts import PILOT_PROMPTS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a small, paired LLM attribution pilot.")
    parser.add_argument("--gateway", choices=("closeai", "zhizengzeng"), required=True)
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--models", default="", help="Comma-separated exact model IDs.")
    parser.add_argument("--prompt-limit", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--sampling",
        default="0.0:1.0,0.8:0.95",
        help="Comma-separated temperature:top_p pairs.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSONL output; defaults to data/attribution/raw/pilot-<gateway>.jsonl.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-large-run",
        action="store_true",
        help="Required when max-requests exceeds the local safety limit of 100.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.prompt_limit <= 0 or args.prompt_limit > len(PILOT_PROMPTS):
        raise SystemExit(f"--prompt-limit must be between 1 and {len(PILOT_PROMPTS)}")
    if args.max_requests <= 0:
        raise SystemExit("--max-requests must be positive")
    if args.max_requests > 100 and not args.confirm_large_run:
        raise SystemExit("Runs above 100 requests require --confirm-large-run")

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    parameter_sets = parse_sampling(args.sampling, max_tokens=args.max_tokens)

    if args.dry_run:
        if args.list_models:
            raise SystemExit("--dry-run cannot be combined with --list-models")
        if not models:
            raise SystemExit("--models is required for collection or dry-run")
        plan = _build_plan(args, models, parameter_sets)
        print(json.dumps(_preview(plan, args.gateway), ensure_ascii=False, indent=2))
        return 0

    environment = load_local_environment(args.env_file)
    config = resolve_gateway(args.gateway, environ=environment)
    gateway = OpenAICompatibleGateway(config)
    if args.list_models:
        print(json.dumps({"gateway": args.gateway, "models": gateway.list_models()}, ensure_ascii=False, indent=2))
        return 0
    if not models:
        raise SystemExit("--models is required unless --list-models is used")

    plan = _build_plan(args, models, parameter_sets)
    output = Path(args.output or f"data/attribution/raw/pilot-{args.gateway}.jsonl")
    completed = _load_completed_keys(output)
    written = 0
    for item in plan:
        request_key = _request_key(item)
        if request_key in completed:
            continue
        result = gateway.generate(
            model=item.model,
            user_prompt=item.prompt.user_prompt,
            system_prompt=item.prompt.system_prompt,
            temperature=item.parameters.temperature,
            top_p=item.parameters.top_p,
            max_tokens=item.parameters.max_tokens,
        )
        record = {
            "request_key": request_key,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "prompt": {
                "prompt_id": item.prompt.prompt_id,
                "version": item.prompt.version,
                "language": item.prompt.language,
                "domain": item.prompt.domain,
                "task_type": item.prompt.task_type,
                "target_length": item.prompt.target_length,
                "user_prompt": item.prompt.user_prompt,
                "system_prompt": item.prompt.system_prompt,
            },
            "parameters": {
                "temperature": item.parameters.temperature,
                "top_p": item.parameters.top_p,
                "max_tokens": item.parameters.max_tokens,
            },
            **result.to_record(),
        }
        write_jsonl_record(output, record)
        completed.add(request_key)
        written += 1
        print(
            json.dumps(
                {
                    "status": "written",
                    "gateway": args.gateway,
                    "request_model": item.model,
                    "response_model": result.response_model,
                    "prompt_id": item.prompt.prompt_id,
                    "written": written,
                    "planned": len(plan),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    return 0


def parse_sampling(value: str, *, max_tokens: int) -> list[CollectionParameters]:
    if max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    parameters: list[CollectionParameters] = []
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise SystemExit("--sampling entries must use temperature:top_p")
        temperature = float(parts[0])
        top_p = float(parts[1])
        if temperature < 0 or not 0 < top_p <= 1:
            raise SystemExit("temperature must be >= 0 and top_p must be in (0, 1]")
        parameters.append(
            CollectionParameters(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        )
    if not parameters:
        raise SystemExit("--sampling must contain at least one pair")
    return parameters


def _build_plan(
    args: argparse.Namespace,
    models: list[str],
    parameter_sets: list[CollectionParameters],
) -> list[PlannedRequest]:
    return build_request_plan(
        gateway_name=args.gateway,
        models=models,
        prompts=PILOT_PROMPTS[: args.prompt_limit],
        parameters=parameter_sets,
        max_requests=args.max_requests,
    )


def _preview(plan: list[PlannedRequest], gateway_name: str) -> dict[str, object]:
    return {
        "gateway": gateway_name,
        "request_count": len(plan),
        "requests": [
            {
                "model": item.model,
                "prompt_id": item.prompt.prompt_id,
                "language": item.prompt.language,
                "domain": item.prompt.domain,
                "task_type": item.prompt.task_type,
                "temperature": item.parameters.temperature,
                "top_p": item.parameters.top_p,
                "max_tokens": item.parameters.max_tokens,
            }
            for item in plan
        ],
    }


def _request_key(item: PlannedRequest) -> str:
    payload = {
        "gateway": item.gateway_name,
        "model": item.model,
        "prompt_id": item.prompt.prompt_id,
        "prompt_version": item.prompt.version,
        "temperature": item.parameters.temperature,
        "top_p": item.parameters.top_p,
        "max_tokens": item.parameters.max_tokens,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("request_key"):
            completed.add(str(record["request_key"]))
    return completed


if __name__ == "__main__":
    raise SystemExit(main())

