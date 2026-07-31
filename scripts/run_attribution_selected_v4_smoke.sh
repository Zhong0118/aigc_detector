#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/run_attribution_selected_v4_smoke.sh [ENV_FILE]" >&2
  exit 64
fi

ENV_FILE="${1:-.env.local}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_WORKERS="${MAX_WORKERS:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
SMOKE_JSONL="data/attribution/raw/zhizengzeng-attribution-user-selected-smoke-v4.jsonl"
SMOKE_ERRORS="data/attribution/raw/zhizengzeng-attribution-user-selected-smoke-v4.errors.jsonl"
SMOKE_AUDIT="outputs/attribution/user-selected-smoke-v4/audit.json"
SMOKE_EDA="outputs/attribution/user-selected-smoke-v4/eda"

"$PYTHON_BIN" -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/smoke_v4.json \
  --env-file "$ENV_FILE" \
  --dry-run

"$PYTHON_BIN" -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/smoke_v4.json \
  --env-file "$ENV_FILE" \
  --confirm \
  --max-workers "$MAX_WORKERS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --max-attempts 2 \
  --empty-max-attempts 2 \
  --empty-circuit-threshold 2 \
  --transient-circuit-threshold 2 \
  --retry-base-seconds 3 \
  --sleep-seconds 0.25

"$PYTHON_BIN" -u -m scripts.audit_attribution_data \
  --config configs/text_attribution/runs/smoke_v4.json \
  --input "$SMOKE_JSONL" \
  --errors "$SMOKE_ERRORS" \
  --output "$SMOKE_AUDIT" \
  --strict

"$PYTHON_BIN" -u -m scripts.analyze_attribution_pilot \
  --input "$SMOKE_JSONL" \
  --output-dir "$SMOKE_EDA" \
  --config configs/text_attribution/runs/smoke_v4.json

echo "Smoke gate passed."
echo "Smoke data: $SMOKE_JSONL"
echo "Smoke errors: $SMOKE_ERRORS"
echo "Smoke audit: $SMOKE_AUDIT"
echo "Smoke EDA: $SMOKE_EDA"
