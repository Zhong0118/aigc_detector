#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_attribution_v5_6k.sh [ENV_FILE] [START_SHARD] [END_SHARD]" >&2
  exit 64
fi

ENV_FILE="${1:-.env.local}"
START_SHARD="${2:-1}"
END_SHARD="${3:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_WORKERS="${MAX_WORKERS:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-420}"

if ! [[ "$START_SHARD" =~ ^[0-9]+$ && "$END_SHARD" =~ ^[0-9]+$ ]]; then
  echo "Shard numbers must be integers." >&2
  exit 64
fi
if (( START_SHARD < 1 || END_SHARD > 10 || START_SHARD > END_SHARD )); then
  echo "Shard range must satisfy 1 <= START_SHARD <= END_SHARD <= 10." >&2
  exit 64
fi

OUTPUT="data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl"
ERRORS="data/attribution/errors/zhizengzeng-attribution-server-6k-v5.errors.jsonl"
REPORT_DIR="outputs/attribution/server-6k-v5"

"$PYTHON_BIN" -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/server_6k_v5.json \
  --env-file "$ENV_FILE" \
  --dry-run

for shard_number in $(seq "$START_SHARD" "$END_SHARD"); do
  shard="$(printf '%02d' "$shard_number")"
  config="configs/text_attribution/runs/v5_shards/shard_${shard}.json"
  audit="$REPORT_DIR/shard_${shard}.audit.json"

  echo "=== V5 shard ${shard}/10 ==="
  set +e
  "$PYTHON_BIN" -u -m scripts.collect_attribution_run \
    --config "$config" \
    --env-file "$ENV_FILE" \
    --output "$OUTPUT" \
    --error-output "$ERRORS" \
    --confirm \
    --max-workers "$MAX_WORKERS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --max-attempts 3 \
    --empty-max-attempts 3 \
    --empty-circuit-threshold 3 \
    --transient-circuit-threshold 3 \
    --retry-base-seconds 3 \
    --sleep-seconds 0.25
  collection_status=$?
  set -e

  if (( collection_status == 2 )); then
    echo "Collection stopped because authentication failed." >&2
    exit 2
  fi
  if (( collection_status != 0 )); then
    echo "Shard ${shard} had failed requests; writing the shard audit."
  fi

  if ! "$PYTHON_BIN" -u -m scripts.audit_attribution_data \
    --config "$config" \
    --input "$OUTPUT" \
    --errors "$ERRORS" \
    --output "$audit" \
    --minimum-model-coverage 0.95; then
    echo "Warning: shard ${shard} is below 95% usable coverage for at least one model." >&2
    echo "Coverage is recorded as a warning; continuing collection." >&2
  fi
done

if (( END_SHARD == 10 )); then
  if ! "$PYTHON_BIN" -u -m scripts.audit_attribution_data \
    --config configs/text_attribution/runs/server_6k_v5.json \
    --input "$OUTPUT" \
    --errors "$ERRORS" \
    --output "$REPORT_DIR/final_audit.json" \
    --minimum-model-coverage 0.95; then
    echo "Warning: final model coverage is below 95%; the final audit was still written." >&2
  fi

  "$PYTHON_BIN" -u -m scripts.analyze_attribution_pilot \
    --input "$OUTPUT" \
    --output-dir "$REPORT_DIR/eda" \
    --config configs/text_attribution/runs/server_6k_v5.json

  echo "Collection pass finished; review final_audit.json before training."
  echo "Data: $OUTPUT"
  echo "Audit: $REPORT_DIR/final_audit.json"
  echo "EDA: $REPORT_DIR/eda"
else
  echo "Completed V5 shards ${START_SHARD}-${END_SHARD}."
fi
