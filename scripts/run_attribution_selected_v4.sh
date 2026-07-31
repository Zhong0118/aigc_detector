#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_attribution_selected_v4.sh BASE_V2_JSONL [ENV_FILE]" >&2
  exit 64
fi

BASE_V2_JSONL="$1"
ENV_FILE="${2:-.env.local}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_WORKERS="${MAX_WORKERS:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

if [[ ! -f "$BASE_V2_JSONL" ]]; then
  echo "Base V2 JSONL not found: $BASE_V2_JSONL" >&2
  exit 66
fi

INCREMENTAL_JSONL="data/attribution/raw/zhizengzeng-attribution-incremental-batch-480-v4.jsonl"
REPAIR_JSONL="data/attribution/raw/zhizengzeng-attribution-quality-repairs-v4.jsonl"
FINAL_JSONL="data/attribution/curated/zhizengzeng-attribution-server-600-v4.jsonl"
FINAL_REPORT="data/attribution/curated/zhizengzeng-attribution-server-600-v4.assembly.json"
EDA_DIR="outputs/attribution/server-600-v4-eda"

# Resolve and validate every local configuration before making a paid API call.
"$PYTHON_BIN" -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/incremental_batch_480_v4.json \
  --env-file "$ENV_FILE" \
  --dry-run

"$PYTHON_BIN" -u -m scripts.collect_attribution_repairs \
  --manifest configs/text_attribution/repairs/server_600_v4_quality_repairs.json \
  --env-file "$ENV_FILE" \
  --dry-run

# Collection is append-only and resumes from request_key, so rerunning this
# script skips successful rows instead of paying for them again.
"$PYTHON_BIN" -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/incremental_batch_480_v4.json \
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

"$PYTHON_BIN" -u -m scripts.collect_attribution_repairs \
  --manifest configs/text_attribution/repairs/server_600_v4_quality_repairs.json \
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

"$PYTHON_BIN" -u -m scripts.assemble_attribution_v4 \
  --config configs/text_attribution/runs/server_600_v4.json \
  --base-config configs/text_attribution/runs/server_600_v2.json \
  --replacement-config configs/text_attribution/runs/incremental_batch_480_v4.json \
  --repair-manifest configs/text_attribution/repairs/server_600_v4_quality_repairs.json \
  --base "$BASE_V2_JSONL" \
  --replacements "$INCREMENTAL_JSONL" \
  --repairs "$REPAIR_JSONL" \
  --output "$FINAL_JSONL" \
  --report "$FINAL_REPORT"

"$PYTHON_BIN" -u -m scripts.analyze_attribution_pilot \
  --input "$FINAL_JSONL" \
  --output-dir "$EDA_DIR" \
  --config configs/text_attribution/runs/server_600_v4.json

echo "V4 dataset: $FINAL_JSONL"
echo "Assembly report: $FINAL_REPORT"
echo "EDA outputs: $EDA_DIR"
