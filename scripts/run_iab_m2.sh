#!/usr/bin/env bash
# Run only after successful acquisition; all output directories are exclusive.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$PWD/.venv-image/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py=.venv-image/bin/python
data=outputs/image_experiments/iab-balanced-v1
assets=outputs/image_experiments/assets
logs=outputs/migration-20261019
"$py" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["status"]=="completed" and s["selected"]==22000' "$data/download_status.json"
if [[ ! -f "$data/manifest.jsonl" ]]; then
  "$py" -u scripts/prepare_iab_manifest.py --dataset "$data" --seed 42 > "$logs/iab-prepare.log" 2>&1
fi
exec 9>outputs/image_experiments/gpu0-research.lock
flock -n 9 || { echo 'GPU research lock busy'; exit 3; }
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used > 1024 )); then echo "GPU 0 is busy ($used MiB); retry later"; exit 3; fi
for task in detection attribution ofa; do
  case "$task" in
    detection) run=d0-resnet50-iab-v1;;
    attribution) run=a0-resnet50-iab-v1;;
    ofa) run=ofa-iab-v1;;
  esac
  output="outputs/image_experiments/$run"
  if [[ -f "$output/status.json" ]]; then
    "$py" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="completed"' "$output/status.json"
    echo "stage=$task already completed; preserving outputs"
    continue
  fi
  echo "stage=$task time=$(date -Is)"
  if [[ "$task" == ofa ]]; then
    "$py" -u scripts/run_image_experiment.py --manifest "$data/manifest.jsonl" --output "$output" --task ofa --weights "$assets/ofa_fingerprint.pth" --denoiser "$assets/model_best.th" --device cuda --batch-size 32 --workers 4 > "$logs/iab-$task.log" 2>&1
  else
    "$py" -u scripts/run_image_experiment.py --manifest "$data/manifest.jsonl" --output "$output" --task "$task" --weights "$assets/resnet50-0676ba61.pth" --device cuda --batch-size 32 --epochs 20 --workers 4 > "$logs/iab-$task.log" 2>&1
  fi
done
echo "stage=completed time=$(date -Is)"
