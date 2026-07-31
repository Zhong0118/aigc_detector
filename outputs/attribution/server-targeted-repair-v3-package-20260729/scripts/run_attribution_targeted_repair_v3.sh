#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_attribution_targeted_repair_v3.sh BASE_V2_JSONL [ENV_FILE]" >&2
  exit 64
fi

BASE_V2_JSONL="$1"
ENV_FILE="${2:-.env.local}"
PYTHON_BIN="${PYTHON_BIN:-python}"

REPLACEMENT_JSONL="data/attribution/raw/zhizengzeng-attribution-replacement-batch-90-v3.jsonl"
REPAIR_JSONL="data/attribution/raw/zhizengzeng-attribution-quality-repairs-v3.jsonl"
FINAL_JSONL="data/attribution/curated/zhizengzeng-attribution-server-600-v3.jsonl"
FINAL_REPORT="data/attribution/curated/zhizengzeng-attribution-server-600-v3.assembly.json"
EDA_DIR="outputs/attribution/server-600-v3-eda"

"$PYTHON_BIN" -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/replacement_batch_90_v3.json \
  --env-file "$ENV_FILE" \
  --confirm \
  --max-workers 4 \
  --timeout-seconds 600 \
  --max-attempts 3 \
  --empty-max-attempts 2 \
  --empty-circuit-threshold 2 \
  --retry-base-seconds 3 \
  --sleep-seconds 0.5

"$PYTHON_BIN" -u -m scripts.collect_attribution_repairs \
  --manifest configs/text_attribution/repairs/server_600_v2_quality_repairs.v3.json \
  --env-file "$ENV_FILE" \
  --confirm \
  --max-workers 4 \
  --timeout-seconds 600 \
  --max-attempts 3 \
  --empty-max-attempts 2 \
  --empty-circuit-threshold 2 \
  --retry-base-seconds 3 \
  --sleep-seconds 0.5

"$PYTHON_BIN" -u -m scripts.assemble_attribution_v3 \
  --config configs/text_attribution/runs/server_600_v3.json \
  --repair-manifest configs/text_attribution/repairs/server_600_v2_quality_repairs.v3.json \
  --base "$BASE_V2_JSONL" \
  --replacements "$REPLACEMENT_JSONL" \
  --repairs "$REPAIR_JSONL" \
  --output "$FINAL_JSONL" \
  --report "$FINAL_REPORT"

"$PYTHON_BIN" -u -m scripts.analyze_attribution_pilot \
  --input "$FINAL_JSONL" \
  --output-dir "$EDA_DIR" \
  --config configs/text_attribution/runs/server_600_v3.json

echo "V3 dataset: $FINAL_JSONL"
echo "Assembly report: $FINAL_REPORT"
echo "EDA outputs: $EDA_DIR"
