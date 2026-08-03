# Attribution Phase4 Fast Track Runbook

This workflow reuses completed A0/A4 validation predictions, trains CPU-only B0/B1 branches, cross-fits E1 fusion and abstention, and never evaluates the frozen test split.

## 1. Build the server overlay on Mac

```bash
python -m scripts.package_attribution_phase4_fast_track \
  --project-root "$PWD" \
  --output "$HOME/Desktop/attribution-phase4-fast-track-server.zip"
```

Upload that ZIP to the server project account.

## 2. Extract and install CPU dependencies on the server

```bash
cd /home/zhongxu/projects/aigc_detector
unzip -o ~/attribution-phase4-fast-track-server.zip
python -m pip install -r requirements-attribution-phase4-fast-track.txt
```

Restore the Phase3 variables if this is a new shell:

```bash
export P3_DATA=/home/zhongxu/projects/aigc_detector/data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl
export P3_SPLIT=/home/zhongxu/projects/aigc_detector/data/attribution/splits/family-attribution-v5-seed42.json
export P3_ROOT=/home/zhongxu/projects/aigc_detector/outputs/attribution/family-phase3
export P4_ROOT=/home/zhongxu/projects/aigc_detector/outputs/attribution/family-phase4
```

## 3. Preflight the complete 8-seed × 3-view matrix

```bash
python -u -m scripts.run_family_attribution_phase4_fast_track \
  --input "$P3_DATA" \
  --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json \
  --split-manifest "$P3_SPLIT" \
  --a0-root "$P3_ROOT/A0" \
  --a4-root "$P3_ROOT/A4" \
  --a0-confirmation-root "$P3_ROOT/confirmation/winner" \
  --a4-confirmation-root "$P3_ROOT/confirmation/comparator" \
  --output-dir "$P4_ROOT/fast-track-v1" \
  --dry-run
```

Preflight must report `test_evaluated=false` and `matrix_cells=24`. If any A0/A4 run is missing or mismatched, repair that input before continuing.

## 4. Run Fast Track in tmux

```bash
mkdir -p "$P4_ROOT"
tmux new-session -d -s attribution-phase4-fast-track \
  "cd /home/zhongxu/projects/aigc_detector && python -u -m scripts.run_family_attribution_phase4_fast_track --input '$P3_DATA' --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json --split-manifest '$P3_SPLIT' --a0-root '$P3_ROOT/A0' --a4-root '$P3_ROOT/A4' --a0-confirmation-root '$P3_ROOT/confirmation/winner' --a4-confirmation-root '$P3_ROOT/confirmation/comparator' --output-dir '$P4_ROOT/fast-track-v1' 2>&1 | tee '$P4_ROOT/fast-track-v1.log'"
```

Monitor:

```bash
tmux attach -t attribution-phase4-fast-track
```

Detach with `Ctrl-b`, then `d`.

## 5. Check completion and the direct answer

```bash
jq '{status, test_evaluated}' "$P4_ROOT/fast-track-v1/COMPLETED.json"
jq '{decision, paired_gains, selective}' "$P4_ROOT/fast-track-v1/phase4_fast_track_summary.json"
```

The run is complete only when `status` is `complete` and `test_evaluated` is `false`.

## 6. Return the result

For the first analysis, copy only:

```text
phase4_fast_track_summary.json
phase4_fast_track_report.md
```

For a reproducibility review, build the compact archive:

```bash
tar -czf ~/attribution-phase4-fast-track-result.tar.gz \
  -C "$P4_ROOT/fast-track-v1" \
  phase4_fast_track_summary.json \
  phase4_fast_track_report.md \
  diagnostics \
  E1-fusion \
  selective \
  artifacts \
  COMPLETED.json
```

Do not include Phase3 model weights, checkpoints, Hugging Face cache, raw V5 text, `.env`, API keys, or failed staging directories.

## 7. Recovery

The formal output directory is immutable. If the process stops before `COMPLETED.json`, inspect the adjacent hidden `.fast-track-v1.run-*` staging directory and its `FAILED.json`. After fixing the recorded cause, choose a new output directory such as `fast-track-v1-retry1`; do not overwrite or merge partial artifacts.
