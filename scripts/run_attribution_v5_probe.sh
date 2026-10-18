#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/run_attribution_v5_probe.sh [ENV_FILE]" >&2
  exit 64
fi

ENV_FILE="${1:-.env.local}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="data/attribution/raw/zhizengzeng-attribution-replacement-probe-v5.jsonl"
ERRORS="data/attribution/errors/zhizengzeng-attribution-replacement-probe-v5.errors.jsonl"
AUDIT="outputs/attribution/v5-probe/audit.json"

"$PYTHON_BIN" -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/replacement_probe_v5.json \
  --env-file "$ENV_FILE" \
  --dry-run

"$PYTHON_BIN" -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/replacement_probe_v5.json \
  --env-file "$ENV_FILE" \
  --output "$OUTPUT" \
  --error-output "$ERRORS" \
  --confirm \
  --max-workers 2 \
  --timeout-seconds 420 \
  --max-attempts 2 \
  --empty-max-attempts 2 \
  --empty-circuit-threshold 2 \
  --transient-circuit-threshold 2 \
  --retry-base-seconds 3 \
  --sleep-seconds 0.25

"$PYTHON_BIN" -u -m scripts.audit_attribution_data \
  --config configs/text_attribution/runs/replacement_probe_v5.json \
  --input "$OUTPUT" \
  --errors "$ERRORS" \
  --output "$AUDIT" \
  --strict

echo "V5 replacement probe passed: kimi-k2.5 and mimo-v2-pro"

