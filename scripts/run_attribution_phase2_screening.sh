#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "__phase2_run_one" ]]; then
  shift
  if [[ "$#" -ne 10 ]]; then
    echo "Invalid internal Phase2 runner invocation." >&2
    exit 64
  fi
  readonly experiment="$1"
  readonly physical_gpu="$2"
  readonly python_bin="$3"
  readonly phase2_config="$4"
  readonly data_jsonl="$5"
  readonly split_manifest="$6"
  readonly hf_cache="$7"
  readonly output_dir="$8"
  readonly seed="$9"
  readonly log_path="${10}"
  export CUDA_VISIBLE_DEVICES="$physical_gpu"

  echo "[$(date --iso-8601=seconds)] Starting ${experiment}; physical GPU=${physical_gpu}; process device=cuda:0; log=${log_path}"
  set +e
  "$python_bin" -u -m scripts.run_family_attribution_neural \
    --config "$phase2_config" \
    --experiment "$experiment" \
    --input "$data_jsonl" \
    --split-manifest "$split_manifest" \
    --output-dir "$output_dir" \
    --seed "$seed" \
    --device "cuda:0" \
    --hf-cache "$hf_cache" \
    2>&1 | tee "$log_path"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    exit "${pipeline_status[0]}"
  fi
  exit "${pipeline_status[1]}"
fi

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
readonly SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
readonly BASH_BIN="${BASH:-bash}"

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
if ! command -v setsid >/dev/null 2>&1; then
  echo "GNU/Linux setsid is required for safe child-process cleanup." >&2
  exit 69
fi

mkdir -p "$LOG_DIR" "$HF_CACHE" "$OUTPUT_ROOT"

child_pids=()
terminate_children() {
  local exit_code="$1"
  local pid
  trap - INT TERM
  echo "Interrupted; stopping Phase2 python|tee process groups..." >&2
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${child_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$exit_code"
}
trap 'terminate_children 130' INT
trap 'terminate_children 143' TERM

started_pid=""
start_experiment() {
  local experiment="$1"
  local physical_gpu="$2"
  local output_dir="${OUTPUT_ROOT}/${experiment}-seed${SEED}"
  local log_path="${LOG_DIR}/${experiment}-seed${SEED}.log"

  setsid "$BASH_BIN" "$SCRIPT_PATH" __phase2_run_one \
    "$experiment" \
    "$physical_gpu" \
    "$PYTHON_BIN" \
    "$PHASE2_CONFIG" \
    "$DATA_JSONL" \
    "$SPLIT_MANIFEST" \
    "$HF_CACHE" \
    "$output_dir" \
    "$SEED" \
    "$log_path" &
  started_pid=$!
}

start_experiment "N0" "0"
n0_pid="$started_pid"
start_experiment "N1" "1"
n1_pid="$started_pid"
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
start_experiment "N2" "0"
n2_pid="$started_pid"
child_pids=("$n2_pid")
if wait "$n2_pid"; then
  child_pids=()
  echo "N2 completed successfully."
else
  n2_status=$?
  child_pids=()
  echo "N2 failed with exit code ${n2_status}; see ${LOG_DIR}/N2-seed${SEED}.log" >&2
  exit "$n2_status"
fi

echo "Phase2 validation-only screening completed for seed ${SEED}."
