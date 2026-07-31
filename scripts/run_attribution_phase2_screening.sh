#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_attribution_phase2_screening.sh \
    DATA_JSONL SPLIT_MANIFEST HF_CACHE OUTPUT_ROOT SEED

Environment overrides:
  PYTHON_BIN         Python executable (default: python)
  PHASE2_CONFIG      Neural config JSON path
  PHASE2_LOG_DIR     Log directory (default: logs/attribution-phase2)
EOF
}

if [[ "$#" -ne 5 ]]; then
  usage >&2
  exit 64
fi

readonly DATA_JSONL="$1"
readonly SPLIT_MANIFEST="$2"
readonly HF_CACHE="$3"
readonly OUTPUT_ROOT="$4"
readonly SEED="$5"
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly PHASE2_CONFIG="${PHASE2_CONFIG:-configs/text_attribution/training/family_neural_screening.v1.json}"
readonly LOG_DIR="${PHASE2_LOG_DIR:-logs/attribution-phase2}"

if [[ ! -f "$DATA_JSONL" ]]; then
  echo "Missing dataset: $DATA_JSONL" >&2
  exit 66
fi
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  echo "Missing frozen split manifest: $SPLIT_MANIFEST" >&2
  exit 66
fi
if [[ ! -f "$PHASE2_CONFIG" ]]; then
  echo "Missing Phase2 config: $PHASE2_CONFIG" >&2
  exit 66
fi

mkdir -p "$LOG_DIR" "$HF_CACHE" "$OUTPUT_ROOT"

run_experiment() {
  local experiment="$1"
  local output_dir="${OUTPUT_ROOT}/${experiment}-seed${SEED}"
  local log_path="${LOG_DIR}/${experiment}-seed${SEED}.log"

  echo "[$(date --iso-8601=seconds)] Starting ${experiment}; visible GPU=${CUDA_VISIBLE_DEVICES}; log=${log_path}"
  "$PYTHON_BIN" -u -m scripts.run_family_attribution_neural \
    --config "$PHASE2_CONFIG" \
    --experiment "$experiment" \
    --input "$DATA_JSONL" \
    --split-manifest "$SPLIT_MANIFEST" \
    --output-dir "$output_dir" \
    --seed "$SEED" \
    --device "cuda:0" \
    --hf-cache "$HF_CACHE" \
    2>&1 | tee "$log_path"
}

child_pids=()
on_signal() {
  local pid
  echo "Interrupted; stopping Phase2 child jobs..." >&2
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  exit 130
}
trap on_signal INT TERM

CUDA_VISIBLE_DEVICES="0" run_experiment "N0" &
n0_pid=$!
CUDA_VISIBLE_DEVICES="1" run_experiment "N1" &
n1_pid=$!
child_pids=("$n0_pid" "$n1_pid")

parallel_status=0
if wait "$n0_pid"; then
  echo "N0 completed successfully."
else
  n0_status=$?
  echo "N0 failed with exit code ${n0_status}; see ${LOG_DIR}/N0-seed${SEED}.log" >&2
  parallel_status="$n0_status"
fi
if wait "$n1_pid"; then
  echo "N1 completed successfully."
else
  n1_status=$?
  echo "N1 failed with exit code ${n1_status}; see ${LOG_DIR}/N1-seed${SEED}.log" >&2
  if [[ "$parallel_status" -eq 0 ]]; then
    parallel_status="$n1_status"
  fi
fi
child_pids=()

if [[ "$parallel_status" -ne 0 ]]; then
  echo "N2 was not started because N0 or N1 failed." >&2
  exit "$parallel_status"
fi

echo "N0 and N1 are complete; starting N2 on physical GPU 0."
if CUDA_VISIBLE_DEVICES="0" run_experiment "N2"; then
  echo "N2 completed successfully."
else
  n2_status=$?
  echo "N2 failed with exit code ${n2_status}; see ${LOG_DIR}/N2-seed${SEED}.log" >&2
  exit "$n2_status"
fi

echo "Phase2 validation-only screening completed for seed ${SEED}."
