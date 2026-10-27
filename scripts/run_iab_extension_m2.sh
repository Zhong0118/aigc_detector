#!/usr/bin/env bash
# Run only after successful acquisition; all output directories are exclusive.
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="$PWD/.venv-image/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=.venv-image/bin/python
data=${1:-outputs/image_experiments/iab-balanced-v1}
suffix=${2:-v1}
summary=${3:-outputs/image_experiments/iab-m2-summary.json}
parent_manifest=${4:?Frozen parent manifest required}
epochs=${5:-30}
gpu_index=${6:-1}
[[ -f "$parent_manifest" && "$epochs" == 30 && "$gpu_index" =~ ^[01]$ ]] || { echo 'Extension requires a frozen parent, 30 epochs and registered physical GPU 0/1'; exit 2; }
[[ "$epochs" =~ ^[1-9][0-9]*$ && "$gpu_index" =~ ^[0-9]+$ ]] || { echo "Invalid epochs or GPU index"; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu_index"
assets=outputs/image_experiments/assets
logs=outputs/iab-${suffix}-logs
mkdir -p "$logs" outputs/image_experiments
exec 9>outputs/image_experiments/gpu0-research.lock
flock -n 9 || { echo 'GPU research lock busy'; exit 3; }
"$py" -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); s=json.load(open(d/"download_status.json")); a=json.load(open(d/"selection.json")); assert s["status"]=="completed" and s["selected"]==sum(map(len,a["entries"].values()))>0 and s["selected_by_cell"]==a["audit"]["selected_by_cell"]' "$data"
if [[ ! -f "$data/manifest.jsonl" ]]; then
  prepare_args=(--dataset "$data" --seed 42)
  if [[ -n "$parent_manifest" ]]; then prepare_args+=(--parent-manifest "$parent_manifest"); fi
  "$py" -u scripts/prepare_iab_extension_manifest.py "${prepare_args[@]}" > "$logs/iab-prepare.log" 2>&1
fi
if [[ -n "$parent_manifest" ]]; then
  "$py" -u scripts/prepare_iab_extension_manifest.py --dataset "$data" --seed 42 --parent-manifest "$parent_manifest" --check-existing >> "$logs/iab-parent-check.log" 2>&1
fi
used=$(nvidia-smi --id="$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 1024 )); then echo "GPU $gpu_index is busy ($used MiB); retry later"; exit 3; fi
for task in detection attribution ofa; do
  case "$task" in
    detection) run=d0-resnet50-iab-${suffix};;
    attribution) run=a0-resnet50-iab-${suffix};;
    ofa) run=ofa-iab-${suffix};;
  esac
  output="outputs/image_experiments/$run"
  if [[ -f "$output/status.json" ]]; then
    "$py" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); assert json.load(open(p/"status.json"))["status"]=="completed"; assert sys.argv[2]=="ofa" or json.load(open(p/"resolved_config.json"))["epochs"]==int(sys.argv[3]), "existing completed model epoch count differs; use a new output suffix"' "$output" "$task" "$epochs"
    echo "stage=$task already completed; preserving outputs"
    continue
  fi
  if [[ -d "$output" ]]; then echo "Existing incomplete output: $output; manual review required"; exit 4; fi
  echo "stage=$task time=$(date -Is)"
  if [[ "$task" == ofa ]]; then
    "$py" -u scripts/run_image_experiment.py --manifest "$data/manifest.jsonl" --output "$output" --task ofa --weights "$assets/ofa_fingerprint.pth" --denoiser "$assets/model_best.th" --device cuda --batch-size 32 --workers 4 > "$logs/iab-$task.log" 2>&1
  else
    "$py" -u scripts/run_image_experiment.py --manifest "$data/manifest.jsonl" --output "$output" --task "$task" --weights "$assets/resnet50-0676ba61.pth" --device cuda --batch-size 32 --epochs "$epochs" --workers 4 > "$logs/iab-$task.log" 2>&1
  fi
done
if [[ ! -f "$summary" ]]; then
  "$py" scripts/summarize_iab_m2.py --dataset "$data" --run-suffix "$suffix" --output "$summary"
fi
echo "stage=completed time=$(date -Is)"
