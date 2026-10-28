#!/usr/bin/env bash
# Serial GPU0 queue for one registered follow-up config. Usage: config EXP:ACTION ...
# Waits while physical GPU0 uses >1024 MiB; skips completed outputs; retries only launches
# refused before an output directory exists (GPU busy or research lock held). Any created
# run that fails, or an incomplete existing output, stops the queue with outputs preserved.
set -uo pipefail
cd "$(dirname "$0")/.."
config=${1:?registered follow-up config}; shift
(( $# )) || { echo 'no queue items'; exit 2; }
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="$PWD/.venv-image/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
py=.venv-image/bin/python
root=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "$config")
logs="$root-logs"; mkdir -p "$logs"
status() { "$py" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$1" 2>/dev/null; }
echo "queue=started time=$(date -Is) pid=$$ config=$config items=$*"
for item in "$@"; do
  experiment=${item%%:*}; action=${item##*:}; out="$root/$experiment/$action"
  if [[ "$(status "$out/status.json")" == completed ]]; then echo "item=$item skipped=completed"; continue; fi
  if [[ -e "$out" ]]; then echo "queue=needs_review item=$item reason=incomplete_existing_output time=$(date -Is)"; exit 4; fi
  while :; do
    used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( used > 1024 )); then echo "item=$item waiting gpu0_used_mib=$used time=$(date -Is)"; sleep 300; continue; fi
    echo "item=$item start time=$(date -Is)"
    "$py" -u scripts/run_image_followup_registered.py --config "$config" --experiment "$experiment" \
      --action "$action" --execute >> "$logs/$experiment-$action.log" 2>&1
    code=$?
    if (( code == 0 )); then echo "item=$item completed time=$(date -Is)"; break; fi
    if [[ ! -e "$out" ]] && tail -n 5 "$logs/$experiment-$action.log" | grep -q -E 'GPU0 busy|BlockingIOError'; then
      echo "item=$item refused_before_output code=$code retry_in=300s time=$(date -Is)"; sleep 300; continue
    fi
    echo "queue=failed item=$item code=$code time=$(date -Is)"; exit 1
  done
done
echo "queue=completed time=$(date -Is)"
