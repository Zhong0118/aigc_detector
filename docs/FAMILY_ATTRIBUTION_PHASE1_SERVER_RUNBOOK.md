# Family Attribution Phase 1: Server Runbook

This runbook trains and evaluates the Phase 1 model-family attribution baselines
against:

`data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl`

The task is ten-family classification. The two enabled versions in each family
are retained as diagnostics, but the user-facing target is the model family.
Every command below must be run from the repository root.

## Files to upload

Preserve these paths exactly when copying the package to the server:

```text
requirements-attribution-training.txt
configs/text_attribution/training/family_baselines.v1.json
configs/text_attribution/zhizengzeng_model_catalog.v5.json
research/text_attribution/baseline_data.py
research/text_attribution/training_data.py
research/text_attribution/split_manifest.py
research/text_attribution/attribution_types.py
research/text_attribution/attribution_metrics.py
research/text_attribution/style_features.py
research/text_attribution/classical_models.py
research/text_attribution/attribution_reporting.py
scripts/run_family_attribution_baselines.py
```

Also place the V5 JSONL dataset at:

```text
data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl
```

Training is offline. It does not call the generation gateway and does not
require an API key or `.env.local`.

## Install and verify

Activate the intended Conda environment, then install the minimal runtime:

```bash
python -m pip install -r requirements-attribution-training.txt
python -m scripts.run_family_attribution_baselines --help
```

Verify the default configuration and V5 catalog before starting a long run:

```bash
python -c "from pathlib import Path; from scripts.run_family_attribution_baselines import _load_config; from research.text_attribution.training_data import load_model_catalog; p=Path('configs/text_attribution/training/family_baselines.v1.json'); c=_load_config(p); q=(p.parent/c['model_catalog']).resolve(); m=load_model_catalog(q); assert len(m)==20 and len({x.family_id for x in m})==10 and {x.variant_role for x in m}=={'fast','balanced'}; assert any(x.model_id=='mimo-v2.5-pro' for x in m); print(q, len(m), 'models, catalog OK')"
```

Create the parent directories used below:

```bash
mkdir -p data/attribution/splits outputs/attribution/family-phase1
```

## 1. Read-only dry run

This validates the cleaned dataset contract, default catalog, experiment list,
and a candidate seed-42 group split:

```bash
python -u -m scripts.run_family_attribution_baselines \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --create-split \
  --output-dir outputs/attribution/family-phase1/dry-run \
  --dry-run
```

The dry run writes neither the split manifest nor the output directory. It only
prints the resolved dataset and split summary.

## 2. Freeze the split and run the B0 smoke baseline

Run this command once. It atomically creates the seed-42 semantic-prompt split
manifest and evaluates only the inexpensive character n-gram baseline:

```bash
python -u -m scripts.run_family_attribution_baselines \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --create-split \
  --experiments B0 \
  --output-dir outputs/attribution/family-phase1/b0-smoke
```

Do not delete or regenerate
`data/attribution/splits/family-attribution-v5-seed42.json`. All subsequent
experiments must reuse this frozen manifest so their scores are comparable and
the paired Chinese/English semantic prompt groups cannot leak across splits.

## 3. Run all validation-only baselines

The default Phase 1 comparison evaluates B0, B1, B2, and B3 on validation data
and selects a winner from validation metrics:

```bash
python -u -m scripts.run_family_attribution_baselines \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --experiments B0 B1 B2 B3 \
  --output-dir outputs/attribution/family-phase1/validation
```

This default full run never evaluates the test split. Inspect
`outputs/attribution/family-phase1/validation/report.md` and
`metrics.json` before authorizing the final test.

## 4. One-time winner-only test evaluation

After the experiment set and selection rule are frozen, run:

```bash
python -u -m scripts.run_family_attribution_baselines \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --experiments B0 B1 B2 B3 \
  --output-dir outputs/attribution/family-phase1/final \
  --evaluate-selected-test
```

`--evaluate-selected-test` evaluates the test split only for the winner chosen
on validation Macro-F1 and worst-family F1. Treat it as a one-time final
measurement; do not use the resulting test score to retune features,
hyperparameters, or model selection.

## Expected artifacts

Each non-dry run writes a transactional output directory containing:

```text
models/B0.joblib
models/B1.joblib                 # all-baseline runs
models/B2.joblib                 # all-baseline runs
models/B3.joblib                 # all-baseline runs
run_config.json
dataset_audit.json
metrics.json
predictions.jsonl
report.md
overall_metrics.png
confusion_matrices.png
per_model_diagnostics.png
calibration.png
subgroup_metrics.png
```

The B0 smoke directory contains only `models/B0.joblib`. The validation run's
predictions contain validation rows only. The final run adds test rows for the
single validation-selected experiment; it does not test the losing
experiments.
