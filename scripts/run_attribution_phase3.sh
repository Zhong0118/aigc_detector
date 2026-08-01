#!/usr/bin/env bash

set -u
set -o pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 STAGE_OR_RECIPE DATA_JSONL SPLIT_MANIFEST HF_CACHE OUTPUT_ROOT SEEDS" >&2
  echo "Example seeds: 42,52,62" >&2
  exit 2
fi

stage_or_recipe=$1
data_jsonl=$2
split_manifest=$3
hf_cache=$4
output_root=$5
seed_csv=$6

python_bin=${PYTHON_BIN:-python}
setsid_bin=${PHASE3_SETSID_BIN:-setsid}

if [ -f "$stage_or_recipe" ]; then
  mode_flag=--recipe
  mode_value=$stage_or_recipe
  run_label=$(basename "$stage_or_recipe")
  run_label=${run_label%.json}
else
  case "$stage_or_recipe" in
    A0|A1-alpha01|A1-alpha03)
      mode_flag=--stage
      mode_value=$stage_or_recipe
      run_label=$stage_or_recipe
      ;;
    A2|A3|A4)
      echo "Promoted stages must be scheduled with their promoted_recipe.json path." >&2
      exit 2
      ;;
    *)
      echo "Unknown canonical stage or missing recipe file: $stage_or_recipe" >&2
      exit 2
      ;;
  esac
fi

if ! command -v "$setsid_bin" >/dev/null 2>&1 && [ ! -x "$setsid_bin" ]; then
  echo "Linux setsid is required for Phase3 process-group cleanup: $setsid_bin" >&2
  exit 2
fi

IFS=',' read -r -a seeds <<< "$seed_csv"
if [ "${#seeds[@]}" -eq 0 ]; then
  echo "At least one seed is required." >&2
  exit 2
fi
for seed in "${seeds[@]}"; do
  case "$seed" in
    12|22|32|42|52|62|72|82) ;;
    *)
      echo "Unsupported Phase3 seed: $seed" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$output_root/logs"

views=(closed fast-to-balanced balanced-to-fast)
task_seeds=()
task_views=()
for seed in "${seeds[@]}"; do
  for view in "${views[@]}"; do
    task_seeds+=("$seed")
    task_views+=("$view")
  done
done

pids=("" "")
task_names=("" "")
active=0
next_task=0
failed=0
task_count=${#task_seeds[@]}

cleanup_process_groups() {
  trap - INT TERM HUP
  echo "Interrupt received; terminating Phase3 Python/tee process groups." >&2
  for slot in 0 1; do
    pid=${pids[$slot]}
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- -"$pid" 2>/dev/null || true
    fi
  done
  for slot in 0 1; do
    pid=${pids[$slot]}
    if [ -n "$pid" ]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit 130
}
trap cleanup_process_groups INT TERM HUP

launch_task() {
  slot=$1
  seed=$2
  view=$3
  gpu=$slot
  task_name="${run_label}-${view}-seed${seed}"
  run_output="$output_root/$task_name"
  log_path="$output_root/logs/$task_name.log"

  echo "Starting $task_name on physical GPU $gpu"
  "$setsid_bin" bash -c '
    set -o pipefail
    log_path=$1
    shift
    "$@" 2>&1 | tee "$log_path"
  ' phase3-child "$log_path" \
    env CUDA_VISIBLE_DEVICES="$gpu" \
    "$python_bin" -m scripts.run_family_attribution_phase3 \
    "$mode_flag" "$mode_value" \
    --input "$data_jsonl" \
    --split-manifest "$split_manifest" \
    --hf-cache "$hf_cache" \
    --output-dir "$run_output" \
    --view "$view" \
    --seed "$seed" \
    --device cuda:0 &

  pids[$slot]=$!
  task_names[$slot]=$task_name
  active=$((active + 1))
}

while [ "$active" -gt 0 ] || { [ "$failed" -eq 0 ] && [ "$next_task" -lt "$task_count" ]; }; do
  if [ "$failed" -eq 0 ]; then
    for slot in 0 1; do
      if [ "$next_task" -ge "$task_count" ]; then
        break
      fi
      if [ -z "${pids[$slot]}" ]; then
        launch_task \
          "$slot" \
          "${task_seeds[$next_task]}" \
          "${task_views[$next_task]}"
        next_task=$((next_task + 1))
      fi
    done
  fi

  if [ "$active" -eq 0 ]; then
    continue
  fi

  observed_completion=0
  for slot in 0 1; do
    pid=${pids[$slot]}
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        status=0
      else
        status=$?
      fi
      name=${task_names[$slot]}
      pids[$slot]=""
      task_names[$slot]=""
      active=$((active - 1))
      observed_completion=1
      if [ "$status" -ne 0 ]; then
        echo "Phase3 task failure: $name exited with status $status" >&2
        failed=1
      else
        echo "Completed $name"
      fi
    fi
  done
  if [ "$observed_completion" -eq 0 ]; then
    sleep 1
  fi
done

trap - INT TERM HUP
if [ "$failed" -ne 0 ]; then
  echo "Phase3 scheduler stopped after failure; no new tasks were dispatched." >&2
  exit 1
fi

echo "Phase3 matrix completed successfully: $output_root"
