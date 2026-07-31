# Family Attribution Phase2 Neural Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified, leakage-safe and resumable neural training pipeline for N0 XLM-R, N1 mDeBERTa and N2 CANINE family attribution experiments.

**Architecture:** Reuse the existing canonical training dataset, frozen semantic-group split, metrics and reporting modules. Add strict neural experiment configuration, a tokenized PyTorch dataset with model-balanced sampling, a Hugging Face Trainer-based engine, one single-experiment CLI, a validation-only selector and a two-GPU server runner.

**Tech Stack:** Python 3.10+, PyTorch 2.6, Transformers 4.50, Accelerate, scikit-learn metrics, pytest/unittest, Bash.

---

## File map

| File | Responsibility |
| --- | --- |
| `configs/text_attribution/training/family_neural_screening.v1.json` | Versioned N0/N1/N2 hyperparameters |
| `research/text_attribution/neural_config.py` | Strict config parsing and CLI overrides |
| `research/text_attribution/neural_data.py` | Label maps, tokenized dataset, balanced weights and smoke subsets |
| `research/text_attribution/neural_training.py` | Model loading, Trainer lifecycle, probability prediction and checkpoint saving |
| `research/text_attribution/neural_selection.py` | Validation-only result ranking |
| `scripts/run_family_attribution_neural.py` | Dry-run, smoke and full single-experiment entry point |
| `scripts/select_family_attribution_neural.py` | N0/N1/N2 summary CLI |
| `scripts/run_attribution_phase2_screening.sh` | Two-GPU server orchestration |
| `scripts/package_attribution_phase2_server.py` | Secret-free project-root overlay ZIP builder |
| `docs/ATTRIBUTION_PHASE2_NEURAL_RUNBOOK.md` | Upload, environment and server commands |
| `requirements-attribution-phase2.txt` | Minimal Phase2 server dependencies |
| `tests/test_text_attribution_neural_config.py` | Configuration contract |
| `tests/test_text_attribution_neural_data.py` | Dataset and sampling contract |
| `tests/test_text_attribution_neural_training.py` | Tiny local model training contract |
| `tests/test_run_family_attribution_neural.py` | CLI output and test-access discipline |
| `tests/test_text_attribution_neural_selection.py` | Validation-only ranking contract |

---

### Task 1: Strict neural experiment configuration

**Files:**
- Create: `configs/text_attribution/training/family_neural_screening.v1.json`
- Create: `research/text_attribution/neural_config.py`
- Create: `tests/test_text_attribution_neural_config.py`

- [ ] **Step 1: Write failing config tests**

Create tests that require all three experiment profiles, reject unknown keys, reject
invalid types and resolve the catalog path relative to the config file:

```python
from pathlib import Path
import json
import tempfile
import unittest

from research.text_attribution.neural_config import load_neural_screening_config


class NeuralConfigTests(unittest.TestCase):
    def test_loads_three_profiles_and_resolves_catalog(self) -> None:
        config = load_neural_screening_config(
            "configs/text_attribution/training/family_neural_screening.v1.json"
        )
        self.assertEqual(tuple(config.experiments), ("N0", "N1", "N2"))
        self.assertEqual(config.experiments["N0"].max_length, 512)
        self.assertEqual(config.experiments["N2"].max_length, 1024)
        self.assertTrue(config.model_catalog.is_absolute())

    def test_rejects_unknown_top_level_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            payload = {
                "schema_version": 1,
                "run_id": "x",
                "seed": 42,
                "model_catalog": "catalog.json",
                "experiments": {},
                "unknown": True,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_neural_screening_config(path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_config.py -q
```

Expected: collection fails because `neural_config` does not exist.

- [ ] **Step 3: Add the versioned JSON configuration**

The config contains exactly:

```json
{
  "schema_version": 1,
  "run_id": "family-attribution-neural-screening-v1",
  "seed": 42,
  "model_catalog": "../zhizengzeng_model_catalog.v5.json",
  "common": {
    "learning_rate": 0.00002,
    "num_train_epochs": 6,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "label_smoothing_factor": 0.05,
    "gradient_accumulation_steps": 2,
    "early_stopping_patience": 2,
    "save_total_limit": 2,
    "bf16": true
  },
  "experiments": {
    "N0": {
      "model_name": "FacebookAI/xlm-roberta-base",
      "max_length": 512,
      "per_device_train_batch_size": 16,
      "gradient_accumulation_steps": 2
    },
    "N1": {
      "model_name": "microsoft/mdeberta-v3-base",
      "max_length": 512,
      "per_device_train_batch_size": 16,
      "gradient_accumulation_steps": 2
    },
    "N2": {
      "model_name": "google/canine-c",
      "max_length": 1024,
      "per_device_train_batch_size": 8,
      "gradient_accumulation_steps": 4
    }
  }
}
```

- [ ] **Step 4: Implement immutable config dataclasses and strict validation**

Expose:

```python
@dataclass(frozen=True)
class NeuralExperimentConfig:
    experiment_id: str
    model_name: str
    max_length: int
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    warmup_ratio: float
    label_smoothing_factor: float
    early_stopping_patience: int
    save_total_limit: int
    bf16: bool


@dataclass(frozen=True)
class NeuralScreeningConfig:
    schema_version: int
    run_id: str
    seed: int
    model_catalog: Path
    experiments: Mapping[str, NeuralExperimentConfig]
```

`load_neural_screening_config(path)` must require exactly `N0`, `N1`, `N2`,
merge common values with experiment overrides and reject booleans where integer
or float values are required.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_config.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add configs/text_attribution/training/family_neural_screening.v1.json research/text_attribution/neural_config.py tests/test_text_attribution_neural_config.py
git commit -m "feat: define neural attribution screening config"
```

---

### Task 2: Neural dataset, label map and model-balanced sampling

**Files:**
- Create: `research/text_attribution/neural_data.py`
- Create: `tests/test_text_attribution_neural_data.py`

- [ ] **Step 1: Write failing data tests**

Use a recording tokenizer and canonical `AttributionSample` fixtures:

```python
class RecordingTokenizer:
    def __call__(self, text, *, truncation, max_length, padding):
        return {
            "input_ids": [len(text), 1],
            "attention_mask": [1, 1],
        }


def test_dataset_returns_only_model_inputs_and_family_label():
    dataset = NeuralAttributionDataset(
        samples=(sample("a", family="claude", model="c1"),),
        tokenizer=RecordingTokenizer(),
        family_to_id={"claude": 0},
        max_length=512,
    )
    row = dataset[0]
    assert row == {
        "input_ids": [1, 1],
        "attention_mask": [1, 1],
        "labels": 0,
    }


def test_model_balanced_weights_are_inverse_model_frequency():
    samples = (
        sample("a", family="claude", model="c1"),
        sample("b", family="claude", model="c1"),
        sample("c", family="glm", model="g1"),
    )
    assert model_balanced_weights(samples) == (0.5, 0.5, 1.0)
```

Also test deterministic smoke sampling, family coverage and fixed label ordering.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_data.py -q
```

Expected: import fails because `neural_data` does not exist.

- [ ] **Step 3: Implement the minimal dataset and helpers**

Expose:

```python
class NeuralAttributionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[AttributionSample],
        tokenizer: Any,
        family_to_id: Mapping[str, int],
        max_length: int,
    ) -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, Any]: ...


def build_family_label_maps(
    family_labels: Sequence[str],
) -> tuple[dict[str, int], dict[int, str]]: ...


def model_balanced_weights(
    samples: Sequence[AttributionSample],
) -> tuple[float, ...]: ...


def select_smoke_samples(
    samples: Sequence[AttributionSample],
    *,
    per_model: int,
    seed: int,
) -> tuple[AttributionSample, ...]: ...
```

`__getitem__` must return only tokenizer outputs and `labels`; metadata remains in
the original sample tuple and never enters the model.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_data.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add research/text_attribution/neural_data.py tests/test_text_attribution_neural_data.py
git commit -m "feat: add neural attribution data pipeline"
```

---

### Task 3: Trainer engine and tiny local-model integration

**Files:**
- Modify: `requirements.txt`
- Create: `research/text_attribution/neural_training.py`
- Create: `tests/test_text_attribution_neural_training.py`

- [ ] **Step 1: Add explicit Accelerate dependency**

Add:

```text
accelerate>=1.0,<2
```

immediately after the Transformers dependency.

- [ ] **Step 2: Write failing unit and local integration tests**

The test module must create a temporary BERT vocabulary and a tiny local model:

```python
config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=24,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    num_labels=10,
)
BertForSequenceClassification(config).save_pretrained(model_dir)
BertTokenizerFast(vocab_file=str(vocab_path)).save_pretrained(model_dir)
```

Tests must require:

```python
def test_softmax_probabilities_follow_fixed_label_order():
    logits = np.asarray([[0.0, 1.0, 2.0]])
    probabilities = logits_to_probabilities(logits)
    assert probabilities.shape == (1, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_smoke_training_saves_reloadable_model_and_tokenizer(tmp_path):
    result = train_neural_experiment(
        request=tiny_training_request(tmp_path, max_steps=2)
    )
    assert (result.model_dir / "config.json").is_file()
    assert (result.tokenizer_dir / "tokenizer_config.json").is_file()
    AutoModelForSequenceClassification.from_pretrained(result.model_dir)
```

The integration test uses CPU, BF16 disabled, ten family labels, two optimizer
steps and no network access.

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_training.py -q
```

Expected: import fails because `neural_training` does not exist.

- [ ] **Step 4: Implement the Trainer engine**

Expose:

```python
@dataclass(frozen=True)
class NeuralTrainingRequest:
    experiment: NeuralExperimentConfig
    train_samples: tuple[AttributionSample, ...]
    validation_samples: tuple[AttributionSample, ...]
    family_labels: tuple[str, ...]
    output_dir: Path
    seed: int
    hf_cache: Path | None
    bf16: bool
    smoke_test: bool
    resume_from_checkpoint: Path | None


@dataclass(frozen=True)
class NeuralTrainingResult:
    model_dir: Path
    tokenizer_dir: Path
    validation_logits: np.ndarray
    validation_probabilities: np.ndarray
    trainer_history: tuple[Mapping[str, Any], ...]
    best_checkpoint: str | None
```

Implement `ModelBalancedTrainer.get_train_dataloader()` with
`WeightedRandomSampler(model_balanced_weights(...), replacement=True)`.

Use `TrainingArguments` with:

```python
evaluation_strategy="epoch"
save_strategy="epoch"
load_best_model_at_end=True
metric_for_best_model="macro_f1"
greater_is_better=True
report_to=[]
remove_unused_columns=True
```

In smoke mode override `max_steps=2`, disable early stopping and use epoch
strategies `"no"` for evaluation/saving during training; run one explicit
validation prediction after training.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_training.py -q
```

Expected: all tests pass on CPU without network access.

- [ ] **Step 6: Run config and data regressions**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_config.py tests/test_text_attribution_neural_data.py tests/test_text_attribution_neural_training.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt research/text_attribution/neural_training.py tests/test_text_attribution_neural_training.py
git commit -m "feat: train neural family attribution models"
```

---

### Task 4: Single-experiment CLI, reporting and test discipline

**Files:**
- Create: `scripts/run_family_attribution_neural.py`
- Create: `tests/test_run_family_attribution_neural.py`
- Modify only if necessary: `research/text_attribution/attribution_reporting.py`

- [ ] **Step 1: Write failing CLI tests**

Tests must cover:

```python
def test_dry_run_validates_without_loading_transformers_or_writing_output():
    exit_code = main([
        "--config", str(config_path),
        "--experiment", "N0",
        "--input", str(data_path),
        "--catalog", str(catalog_path),
        "--split-manifest", str(split_path),
        "--output-dir", str(output_dir),
        "--dry-run",
    ])
    assert exit_code == 0
    assert not output_dir.exists()


def test_default_run_reports_validation_only():
    with patch(
        "scripts.run_family_attribution_neural.train_neural_experiment",
        side_effect=fake_training_result,
    ):
        main([...])
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert set(metrics["N0"]) == {"validation"}
```

The fake training boundary is acceptable here because Task 3 already tests the
real local Trainer integration. CLI tests verify orchestration, not Transformers.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_run_family_attribution_neural.py -q
```

Expected: import fails because the CLI does not exist.

- [ ] **Step 3: Implement CLI arguments**

Required arguments and controls:

```text
--config
--experiment {N0,N1,N2}
--input
--catalog
--split-manifest
--output-dir
--seed
--device
--hf-cache
--no-bf16
--smoke-test
--resume-from-checkpoint
--evaluate-test
--dry-run
```

The CLI must never create or mutate a split manifest. It only loads an existing
manifest and fails on fingerprint mismatch.

- [ ] **Step 4: Implement orchestration and artifacts**

The CLI:

1. loads config and dataset;
2. applies the frozen split;
3. prints a JSON summary;
4. exits before Transformers imports in dry-run;
5. calls the training engine;
6. creates validation prediction records with existing
   `build_prediction_records`;
7. evaluates with `evaluate_family_predictions`;
8. writes the existing report artifacts;
9. saves `trainer_history.json`;
10. atomically moves a staging directory into the requested output path.

`--evaluate-test` is explicit and recorded in `run_config.json`. Smoke runs must
reject `--evaluate-test`.

- [ ] **Step 5: Run the CLI tests**

Run:

```bash
python -m pytest tests/test_run_family_attribution_neural.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run Phase1 and Phase2 regressions**

Run:

```bash
python -m pytest \
  tests/test_text_attribution_training_data.py \
  tests/test_text_attribution_split_manifest.py \
  tests/test_text_attribution_metrics_v2.py \
  tests/test_text_attribution_reporting_v2.py \
  tests/test_run_family_attribution_baselines.py \
  tests/test_text_attribution_neural_config.py \
  tests/test_text_attribution_neural_data.py \
  tests/test_text_attribution_neural_training.py \
  tests/test_run_family_attribution_neural.py \
  -q
```

Expected: all tests and subtests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_family_attribution_neural.py tests/test_run_family_attribution_neural.py research/text_attribution/attribution_reporting.py
git commit -m "feat: run neural family attribution experiments"
```

---

### Task 5: Validation-only experiment selector

**Files:**
- Create: `research/text_attribution/neural_selection.py`
- Create: `scripts/select_family_attribution_neural.py`
- Create: `tests/test_text_attribution_neural_selection.py`

- [ ] **Step 1: Write failing selection tests**

Require rejection of smoke outputs and any candidate lacking validation:

```python
def test_ranks_by_macro_then_worst_family_then_nll(tmp_path):
    write_result(tmp_path / "N0", macro=0.60, worst=0.20, nll=1.2)
    write_result(tmp_path / "N1", macro=0.60, worst=0.30, nll=1.4)
    write_result(tmp_path / "N2", macro=0.55, worst=0.40, nll=1.0)
    summary = select_neural_experiments(
        [tmp_path / "N0", tmp_path / "N1", tmp_path / "N2"]
    )
    assert summary.ranking == ("N1", "N0", "N2")


def test_rejects_smoke_result(tmp_path):
    path = write_result(tmp_path / "N0", smoke=True)
    with pytest.raises(ValueError, match="smoke"):
        select_neural_experiments([path])
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_selection.py -q
```

Expected: import fails because selector modules do not exist.

- [ ] **Step 3: Implement selector and CLI**

`select_neural_experiments(paths)` returns an immutable summary with:

- ordered experiment IDs;
- source output paths;
- Macro-F1;
- worst-family F1;
- NLL;
- selected top two.

The CLI writes `neural_screening_summary.json` and
`neural_screening_summary.md`. It must not read predictions or test metrics.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_text_attribution_neural_selection.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add research/text_attribution/neural_selection.py scripts/select_family_attribution_neural.py tests/test_text_attribution_neural_selection.py
git commit -m "feat: select neural attribution backbones"
```

---

### Task 6: Two-GPU server runner and runbook

**Files:**
- Create: `scripts/run_attribution_phase2_screening.sh`
- Create: `scripts/package_attribution_phase2_server.py`
- Create: `docs/ATTRIBUTION_PHASE2_NEURAL_RUNBOOK.md`
- Create: `requirements-attribution-phase2.txt`
- Create: `tests/test_attribution_phase2_server_package.py`

- [ ] **Step 1: Write failing package contract test**

The test verifies that the shell script references all three experiments, does
not include `--evaluate-test`, creates logs and propagates non-zero failures:

```python
def test_server_runner_is_validation_only():
    text = Path("scripts/run_attribution_phase2_screening.sh").read_text()
    assert "--experiment N0" in text
    assert "--experiment N1" in text
    assert "--experiment N2" in text
    assert "--evaluate-test" not in text
    assert "mkdir -p logs" in text
    assert "set -euo pipefail" in text
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
python -m pytest tests/test_attribution_phase2_server_package.py -q
```

Expected: failure because the shell script does not exist.

- [ ] **Step 3: Implement server runner**

The script interface is:

```bash
bash scripts/run_attribution_phase2_screening.sh \
  data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  data/attribution/splits/family-attribution-v5-seed42.json \
  models/huggingface \
  outputs/attribution/family-phase2 \
  42
```

It starts N0 on physical GPU 0 and N1 on physical GPU 1, waits for both PIDs,
then starts N2 on GPU 0. Each job writes a distinct log. A failed child makes
the script return non-zero after recording which experiment failed.

- [ ] **Step 4: Write the runbook**

The runbook includes:

- exact files to upload;
- dependency installation;
- dry-run and N0 smoke commands;
- tmux launch/detach/attach commands;
- full screening command;
- checkpoint resume command;
- output tree;
- validation result download list;
- no-test warning.

- [ ] **Step 5: Build the uploadable server package**

The cross-platform Python packager creates a project-root overlay ZIP containing
the necessary research modules, Phase2 scripts, configs, docs, requirements and
tests. It writes a manifest with source commit, file sizes and SHA256 hashes.
It rejects secret-like content and never includes `.git`, `.env*`, raw data,
outputs, caches, models, checkpoints or weight files.

Run:

```bash
python -m scripts.package_attribution_phase2_server \
  --output dist/aigc-detector-attribution-phase2-server.zip
```

The generated ZIP is a delivery artifact and must not be committed.

- [ ] **Step 6: Run package test**

Run:

```bash
python -m pytest tests/test_attribution_phase2_server_package.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Run full relevant suite**

Run:

```bash
python -m pytest \
  tests/test_text_attribution_training_data.py \
  tests/test_text_attribution_split_manifest.py \
  tests/test_text_attribution_metrics_v2.py \
  tests/test_text_attribution_reporting_v2.py \
  tests/test_run_family_attribution_baselines.py \
  tests/test_text_attribution_neural_config.py \
  tests/test_text_attribution_neural_data.py \
  tests/test_text_attribution_neural_training.py \
  tests/test_run_family_attribution_neural.py \
  tests/test_text_attribution_neural_selection.py \
  tests/test_attribution_phase2_server_package.py \
  -q
```

Expected: all tests and subtests pass with zero failures.

- [ ] **Step 8: Run static verification**

Run:

```bash
python -m compileall research/text_attribution scripts
git diff --check
```

Expected: compilation succeeds and `git diff --check` reports nothing.

- [ ] **Step 9: Commit**

```bash
git add scripts/run_attribution_phase2_screening.sh scripts/package_attribution_phase2_server.py requirements-attribution-phase2.txt docs/ATTRIBUTION_PHASE2_NEURAL_RUNBOOK.md tests/test_attribution_phase2_server_package.py
git commit -m "docs: package phase2 neural screening for server"
```

---

## Implementation completion gate

Before calling the implementation complete:

- [ ] Review every requirement in the Phase2 design against a concrete test.
- [ ] Run the full relevant suite from Task 6.
- [ ] Run `python -m compileall research/text_attribution scripts`.
- [ ] Run `git diff --check`.
- [ ] Confirm `git status --short` contains no unintended files.
- [ ] Confirm no CLI default evaluates test.
- [ ] Confirm no model input contains gateway or request metadata.
- [ ] Confirm local integration tests use only a temporary local model.
- [ ] Confirm server runner uses validation only.
