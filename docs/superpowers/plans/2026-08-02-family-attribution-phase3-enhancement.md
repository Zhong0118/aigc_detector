# Family Attribution Phase3 Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe, resumable Phase3 pipeline that evaluates A0–A4 on XLM-R with closed and bidirectional version-holdout validation, applies explicit promotion gates, and ships as a server overlay ZIP.

**Architecture:** Keep Phase2 immutable and add focused Phase3 modules behind three deep interfaces: recipe resolution, single-run training, and stage selection. Each run owns exactly one resolved recipe, seed, and data view; stage selection combines a complete 3-seed × 3-view matrix and emits the next immutable recipe.

**Tech Stack:** Python 3.10+, PyTorch 2.6, Transformers 4.50, Accelerate, scikit-learn, NumPy, safetensors/Hugging Face save formats, pytest/unittest, Bash.

## Global Constraints

- The only Phase3 backbone is `FacebookAI/xlm-roberta-base`.
- Phase3 code must not change Phase2 N0/N1/N2 behavior.
- Training data are V5 with the existing frozen semantic-group split; Phase3 never reads test rows.
- Formal output is always 10-family probabilities; auxiliary model probabilities are diagnostic only.
- Canonical screening seeds are exactly 42, 52, and 62.
- Final confirmation seeds are exactly 12, 22, 32, 72, and 82 in addition to the screening seeds.
- Data views are exactly `closed`, `fast-to-balanced`, and `balanced-to-fast`.
- A0–A3 use batch 16 and gradient accumulation 2; A4 uses batch 4 and accumulation 8.
- A4 may use the explicit low-memory override batch 2 and accumulation 16; no silent fallback is allowed.
- Formal runs require CUDA; CPU is restricted to dry-run and smoke.
- Unknown configuration keys and mismatched recipe/checkpoint hashes fail before training.
- Server packages exclude data, `.env`, credentials, outputs, logs, Hugging Face cache, and model/checkpoint weights.

---

## File Map

| File | Responsibility |
| --- | --- |
| `configs/text_attribution/training/family_phase3_enhancement.v1.json` | Versioned Phase3 hyperparameters, stage transforms, seeds, and gates |
| `research/text_attribution/phase3_config.py` | Strict config parsing, immutable recipes, canonical hashing, parent transforms |
| `research/text_attribution/phase3_data.py` | Catalog roles, data views, label maps, windows, datasets, structured batches |
| `research/text_attribution/attribution_losses.py` | Family/model CE, hierarchical KL, supervised contrastive loss |
| `research/text_attribution/hierarchical_model.py` | Shared encoder, optional heads, multi-window pooling, save/load |
| `research/text_attribution/phase3_training.py` | Trainer lifecycle, sampler selection, loss logging, checkpoint validation |
| `research/text_attribution/phase3_selection.py` | Complete-matrix loading, paired comparisons, gates, promotion/fallback |
| `scripts/run_family_attribution_phase3.py` | One recipe × seed × data-view CLI |
| `scripts/select_family_attribution_phase3.py` | Stage selection CLI |
| `scripts/run_attribution_phase3.sh` | Two-GPU stage scheduler |
| `scripts/package_attribution_phase3_server.py` | Deterministic secret-free overlay package |
| `requirements-attribution-phase3.txt` | Phase3 server dependencies excluding PyTorch |
| `docs/ATTRIBUTION_PHASE3_RUNBOOK.md` | Server upload, smoke, stages, recovery, selection, download |
| `tests/test_text_attribution_phase3_config.py` | Config and recipe contract |
| `tests/test_text_attribution_phase3_data.py` | Views, labels, windows, sampler contract |
| `tests/test_text_attribution_attribution_losses.py` | Loss values, gradients, disabled paths |
| `tests/test_text_attribution_hierarchical_model.py` | A0–A4 forward and save/reload contract |
| `tests/test_text_attribution_phase3_training.py` | Tiny local training, recovery, loss history |
| `tests/test_run_family_attribution_phase3.py` | CLI modes, output, test isolation |
| `tests/test_text_attribution_phase3_selection.py` | Matrix validation and promotion gates |
| `tests/test_attribution_phase3_server_package.py` | Package manifest, hash, secret and weight exclusion |

---

### Task 1: Strict Phase3 Configuration and Immutable Recipes

**Files:**
- Create: `configs/text_attribution/training/family_phase3_enhancement.v1.json`
- Create: `research/text_attribution/phase3_config.py`
- Create: `tests/test_text_attribution_phase3_config.py`

**Interfaces:**
- Consumes: `Path`, parent `stage_decision.json` files.
- Produces: `Phase3Config`, `ResolvedPhase3Recipe`, `load_phase3_config(path)`, `resolve_phase3_recipe(config, stage, parent_decision=None)`, `load_resolved_recipe(path)`.

- [ ] **Step 1: Write failing tests for strict config and canonical recipes**

```python
def test_resolves_canonical_a1_recipe_and_stable_hash():
    config = load_phase3_config(CONFIG_PATH)
    first = resolve_phase3_recipe(config, "A1-alpha01")
    second = resolve_phase3_recipe(config, "A1-alpha01")
    assert first.auxiliary_weight == 0.1
    assert first.consistency_weight == 0.0
    assert first.recipe_hash == second.recipe_hash
    assert len(first.recipe_hash) == 64


def test_a2_requires_promoted_a1_parent(tmp_path):
    decision = write_decision(tmp_path, promoted=False, fallback="A0")
    with pytest.raises(ValueError, match="A1.*promoted"):
        resolve_phase3_recipe(CONFIG, "A2", decision)
```

Also require exact top-level keys, fixed XLM-R, fixed seed lists, bounded weights, A4 batch/accumulation pairs, canonical JSON hashing, replay hash verification, and mutually valid parent stages.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_text_attribution_phase3_config.py -q
```

Expected: collection fails because `phase3_config` does not exist.

- [ ] **Step 3: Add the exact versioned configuration**

The JSON defines public/common training fields, canonical stages `A0`, `A1-alpha01`, `A1-alpha03`, transforms `A2`, `A3`, `A4`, screening/final seeds, three data views, and these gates:

```json
{
  "closed_macro_f1_min_gain": 0.01,
  "closed_seed_wins_min": 2,
  "worst_family_min_gain": 0.0,
  "mimo_min_gain": 0.0,
  "glm_min_gain": 0.0,
  "holdout_combined_min_gain": 0.0,
  "holdout_direction_min_gain": -0.01,
  "forbid_zero_family_f1": true
}
```

- [ ] **Step 4: Implement frozen dataclasses, exact-field validation, transforms, and hashing**

The recipe interface is:

```python
@dataclass(frozen=True)
class ResolvedPhase3Recipe:
    stage_id: str
    recipe_id: str
    parent_recipe_id: str | None
    backbone_model_id: str
    max_length: int
    auxiliary_weight: float
    consistency_weight: float
    contrastive_weight: float
    contrastive_temperature: float
    projection_dimension: int
    window_mode: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    epochs: int
    weight_decay: float
    warmup_ratio: float
    label_smoothing_factor: float
    early_stopping_patience: int
    save_total_limit: int
    bf16: bool
    recipe_hash: str
```

Hash the normalized dictionary without `recipe_hash` using UTF-8 JSON with sorted keys and compact separators.

- [ ] **Step 5: Run tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_phase3_config.py -q
```

- [ ] **Step 6: Commit**

```bash
git add configs/text_attribution/training/family_phase3_enhancement.v1.json research/text_attribution/phase3_config.py tests/test_text_attribution_phase3_config.py
git commit -m "feat: define phase3 attribution recipes"
```

---

### Task 2: Catalog Roles, Data Views, and Train-Visible Labels

**Files:**
- Create: `research/text_attribution/phase3_data.py`
- Create: `tests/test_text_attribution_phase3_data.py`

**Interfaces:**
- Consumes: `AttributionSample`, V5 catalog JSON, applied frozen split mappings.
- Produces: `Phase3DataView`, `VariantCatalog`, `load_variant_catalog(path)`, `build_phase3_data_view(...)`, `build_train_visible_model_maps(...)`.

- [ ] **Step 1: Write failing role and view tests**

```python
def test_fast_to_balanced_filters_train_and_validation_by_catalog_role():
    view = build_phase3_data_view(
        splits=synthetic_splits(),
        catalog=synthetic_catalog(),
        view_id="fast-to-balanced",
        test_request_keys=frozenset({"test-secret"}),
    )
    assert {row.model_id for row in view.train_samples} == {"a-fast", "b-fast"}
    assert {row.model_id for row in view.validation_samples} == {
        "a-balanced", "b-balanced"
    }
    assert set(view.train_visible_model_to_id) == {"a-fast", "b-fast"}
```

Add tests that closed has 20 train-visible model labels, holdout has 10, each family has exactly one fast and balanced model, all families exist in both sides, and any test request key is rejected.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_phase3_data.py -q
```

Expected: import failure for `phase3_data`.

- [ ] **Step 3: Implement immutable catalog and view construction**

```python
@dataclass(frozen=True)
class Phase3DataView:
    view_id: str
    train_samples: tuple[AttributionSample, ...]
    validation_samples: tuple[AttributionSample, ...]
    family_to_id: Mapping[str, int]
    id_to_family: Mapping[int, str]
    train_visible_model_to_id: Mapping[str, int]
    id_to_train_visible_model: Mapping[int, str]
    model_to_family_id: Mapping[str, int]
```

Use catalog order for labels, reject duplicate roles, and never infer roles from names.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_phase3_data.py -q
```

- [ ] **Step 5: Commit**

```bash
git add research/text_attribution/phase3_data.py tests/test_text_attribution_phase3_data.py
git commit -m "feat: add phase3 attribution data views"
```

---

### Task 3: Single-Window and Multi-Window Dataset Views

**Files:**
- Modify: `research/text_attribution/phase3_data.py`
- Modify: `tests/test_text_attribution_phase3_data.py`

**Interfaces:**
- Consumes: `Phase3DataView`, tokenizer, `ResolvedPhase3Recipe`.
- Produces: `Phase3AttributionDataset`, `tokenize_phase3_windows(tokenizer, text, max_length, window_mode)`.

- [ ] **Step 1: Write failing window tests**

```python
def test_three_window_long_text_uses_head_center_tail_and_mask():
    encoded = tokenize_phase3_windows(
        RecordingWindowTokenizer(range(30)),
        "long",
        max_length=8,
        window_mode="head-center-tail",
    )
    assert encoded["window_mask"] == [1, 1, 1]
    assert encoded["input_ids"][0] != encoded["input_ids"][1]
    assert encoded["input_ids"][1] != encoded["input_ids"][2]


def test_short_text_is_not_duplicated_across_three_windows():
    encoded = tokenize_phase3_windows(TOKENIZER, "short", 16, "head-center-tail")
    assert encoded["window_mask"] == [1, 0, 0]
```

Add exact special-token, attention-mask, duplicate-window removal, metadata isolation, family label, optional train-visible model label, semantic-group ID, and validation unseen-model behavior tests.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_phase3_data.py -q
```

- [ ] **Step 3: Implement token windows and the dataset**

`Phase3AttributionDataset.__getitem__` returns only model tensors/labels:

```python
{
    "input_ids": list[list[int]],
    "attention_mask": list[list[int]],
    "window_mask": list[int],
    "family_labels": int,
    "model_labels": int,
    "contrastive_family_labels": int,
    "contrastive_group_ids": int,
    "contrastive_model_ids": int,
}
```

Validation rows whose concrete model is held out use `model_labels=-100`; the diagnostic head is not scored against unseen labels.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_phase3_data.py -q
```

- [ ] **Step 5: Commit**

```bash
git add research/text_attribution/phase3_data.py tests/test_text_attribution_phase3_data.py
git commit -m "feat: add phase3 multi-window inputs"
```

---

### Task 4: Deterministic Structured Contrastive Batch Sampler

**Files:**
- Modify: `research/text_attribution/phase3_data.py`
- Modify: `tests/test_text_attribution_phase3_data.py`

**Interfaces:**
- Consumes: train samples and seed.
- Produces: `StructuredContrastiveBatchSampler`, `summarize_contrastive_batches(...)`.

- [ ] **Step 1: Write failing structured-batch tests**

```python
def test_every_structured_batch_has_cross_group_positive_and_hard_negative():
    sampler = StructuredContrastiveBatchSampler(SAMPLES, batch_size=16, seed=42)
    for indices in sampler:
        rows = [SAMPLES[index] for index in indices]
        assert every_anchor_has_same_family_different_group(rows)
        assert every_anchor_has_same_group_different_family(rows)
```

Require order independence, deterministic seed behavior, family rotation, closed-view different-model preference, holdout fallback, resumable epoch state, and clear failure when constraints are impossible.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_phase3_data.py -q
```

- [ ] **Step 3: Implement the batch sampler and audit**

The sampler exposes `set_epoch(epoch: int)` and derives its RNG from `(seed, epoch)`. It emits batches of exactly 16 indices and drops an incomplete final batch rather than weakening constraints.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_phase3_data.py -q
```

- [ ] **Step 5: Commit**

```bash
git add research/text_attribution/phase3_data.py tests/test_text_attribution_phase3_data.py
git commit -m "feat: add structured attribution batches"
```

---

### Task 5: Hierarchical and Contrastive Losses

**Files:**
- Create: `research/text_attribution/attribution_losses.py`
- Create: `tests/test_text_attribution_attribution_losses.py`

**Interfaces:**
- Consumes: family/model logits, projections, labels, model-to-family mapping, recipe weights.
- Produces: `Phase3LossOutput`, `compute_phase3_loss(...)`, `hierarchical_consistency_loss(...)`, `supervised_contrastive_loss(...)`.

- [ ] **Step 1: Write failing numerical and gradient tests**

```python
def test_consistency_is_zero_when_family_and_aggregated_model_probs_match():
    family_logits = torch.log(torch.tensor([[0.7, 0.3]]))
    model_logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]]))
    loss = hierarchical_consistency_loss(
        family_logits, model_logits, torch.tensor([0, 0, 1, 1])
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_supcon_rejects_batch_without_valid_positive():
    with pytest.raises(ValueError, match="valid anchor"):
        supervised_contrastive_loss(
            projections=torch.eye(3),
            family_labels=torch.tensor([0, 1, 2]),
            group_ids=torch.tensor([0, 1, 2]),
            temperature=0.07,
        )
```

Add tests for label smoothing, ignored unseen model labels, finite gradients, disabled weights producing exact scalar zero, self-pair exclusion, and strong-positive masks.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_attribution_losses.py -q
```

- [ ] **Step 3: Implement stable loss functions**

```python
@dataclass(frozen=True)
class Phase3LossOutput:
    total: torch.Tensor
    family: torch.Tensor
    model: torch.Tensor
    consistency: torch.Tensor
    contrastive: torch.Tensor
```

Use `log_softmax`, `scatter_add`, epsilon clamp, and `torch.isfinite` guards. Raise on a non-finite enabled loss.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_attribution_losses.py -q
```

- [ ] **Step 5: Commit**

```bash
git add research/text_attribution/attribution_losses.py tests/test_text_attribution_attribution_losses.py
git commit -m "feat: add phase3 attribution losses"
```

---

### Task 6: Hierarchical XLM-R Model and Reloadable Artifacts

**Files:**
- Create: `research/text_attribution/hierarchical_model.py`
- Create: `tests/test_text_attribution_hierarchical_model.py`

**Interfaces:**
- Consumes: backbone directory/name, recipe, family/model label maps, token tensors.
- Produces: `HierarchicalAttributionModel`, `Phase3ModelOutput`, `save_phase3_model(...)`, `load_phase3_model(...)`.

- [ ] **Step 1: Write failing tiny-model tests**

```python
def test_a0_has_only_family_logits(tiny_backbone, batch):
    model = build_model(tiny_backbone, recipe("A0"), model_labels=())
    output = model(**batch)
    assert output.family_logits.shape == (2, 10)
    assert output.model_logits is None
    assert output.projections is None


def test_saved_model_reloads_with_identical_logits(tmp_path, tiny_backbone, batch):
    model = build_model(tiny_backbone, recipe("A3"), model_labels=MODEL_LABELS)
    before = model.eval()(**batch).family_logits.detach()
    save_phase3_model(model, tmp_path / "model")
    after = load_phase3_model(tmp_path / "model").eval()(**batch).family_logits.detach()
    torch.testing.assert_close(before, after)
```

Cover A1 model logits, A3 normalized projections, A4 masked mean pooling, recipe/label-map mismatch, and no metadata kwargs.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_hierarchical_model.py -q
```

- [ ] **Step 3: Implement the model and deterministic save/load**

```python
@dataclass
class Phase3ModelOutput:
    family_logits: torch.Tensor
    model_logits: torch.Tensor | None
    projections: torch.Tensor | None
    pooled_representation: torch.Tensor
```

Flatten `(batch, windows, sequence)` for encoding, restore the window dimension, apply the masked mean, and use an XLM-R-compatible dense/tanh/dropout classification head.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_hierarchical_model.py -q
```

- [ ] **Step 5: Commit**

```bash
git add research/text_attribution/hierarchical_model.py tests/test_text_attribution_hierarchical_model.py
git commit -m "feat: add hierarchical attribution model"
```

---

### Task 7: Unified Phase3 Training Lifecycle

**Files:**
- Create: `research/text_attribution/phase3_training.py`
- Create: `tests/test_text_attribution_phase3_training.py`

**Interfaces:**
- Consumes: resolved recipe, `Phase3DataView`, tokenizer/model source, output directory, seed, runtime flags.
- Produces: `Phase3TrainingRequest`, `Phase3TrainingResult`, `train_phase3_experiment(request)`.

- [ ] **Step 1: Write failing two-step training and recovery tests**

```python
def test_two_step_a3_training_writes_loss_components_and_reloadable_model(tmp_path):
    result = train_phase3_experiment(tiny_request(tmp_path, stage="A3", max_steps=2))
    assert result.validation_probabilities.shape[1] == 10
    assert (tmp_path / "run" / "loss_components.json").is_file()
    assert json.loads((tmp_path / "run" / "COMPLETED.json").read_text())["status"] == "complete"


def test_resume_rejects_recipe_hash_mismatch(tmp_path):
    checkpoint = write_checkpoint(tmp_path, recipe_hash="a" * 64)
    with pytest.raises(ValueError, match="recipe hash"):
        train_phase3_experiment(request_with_checkpoint(checkpoint, "b" * 64))
```

Cover ordinary balanced sampler for A0–A2/A4, structured sampler for A3 descendants, validation sequential sampling, component-loss logging, finite logits, BF16 rules, A4 preflight, and best-checkpoint reload.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_phase3_training.py -q
```

- [ ] **Step 3: Implement `Phase3Trainer` and the lifecycle**

Override `compute_loss` to call the model and `compute_phase3_loss`, log detached loss components, and override the train sampler/batch sampler according to the recipe. Publish model/tokenizer/history/COMPLETED atomically.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_phase3_training.py -q
```

- [ ] **Step 5: Run Phase2 neural regression tests**

```bash
python -m pytest tests/test_text_attribution_neural_config.py tests/test_text_attribution_neural_data.py tests/test_text_attribution_neural_training.py -q
```

- [ ] **Step 6: Commit**

```bash
git add research/text_attribution/phase3_training.py tests/test_text_attribution_phase3_training.py
git commit -m "feat: train phase3 attribution recipes"
```

---

### Task 8: Single-Run CLI, Reports, and Test Isolation

**Files:**
- Create: `scripts/run_family_attribution_phase3.py`
- Create: `tests/test_run_family_attribution_phase3.py`
- Modify: `research/text_attribution/attribution_metrics.py`
- Modify: `research/text_attribution/attribution_reporting.py`

**Interfaces:**
- Consumes: canonical/promoted/replay CLI modes and existing V5 paths.
- Produces: one complete run directory using the Phase3 output contract.

- [ ] **Step 1: Write failing CLI mode and output tests**

```python
def test_parser_does_not_expose_test_evaluation():
    parser = build_parser()
    assert "--evaluate-test" not in parser.format_help()


def test_promoted_a3_requires_parent_decision():
    with pytest.raises(SystemExit):
        parse_args(["--stage", "A3", *REQUIRED_ARGS])
```

Add dry-run no-transformers-import, canonical/promoted/replay mutual exclusion, output-directory ownership, fingerprint, smoke/preflight flags, view-specific model diagnostics, and failed-run preservation tests.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_run_family_attribution_phase3.py -q
```

- [ ] **Step 3: Extend metrics/reporting without changing Phase2 contracts**

Add optional Phase3 diagnostic inputs behind new functions rather than changing the existing function signatures used by Phase2. Report family metrics, current-view auxiliary metrics, calibration, subgroup metrics, valid window count, and view metadata.

- [ ] **Step 4: Implement the CLI and atomic run output**

The CLI loads data/split/catalog, builds a data view, resolves or loads a recipe, prints a dry-run summary, delegates training, evaluates only the selected validation view, and publishes reports atomically.

- [ ] **Step 5: Run tests and verify GREEN**

```bash
python -m pytest tests/test_run_family_attribution_phase3.py tests/test_text_attribution_metrics_v2.py tests/test_text_attribution_reporting_v2.py -q
```

- [ ] **Step 6: Commit**

```bash
git add scripts/run_family_attribution_phase3.py research/text_attribution/attribution_metrics.py research/text_attribution/attribution_reporting.py tests/test_run_family_attribution_phase3.py
git commit -m "feat: run phase3 attribution experiments"
```

---

### Task 9: Complete-Matrix Stage Selection and Promotion

**Files:**
- Create: `research/text_attribution/phase3_selection.py`
- Create: `scripts/select_family_attribution_phase3.py`
- Create: `tests/test_text_attribution_phase3_selection.py`

**Interfaces:**
- Consumes: parent/candidate run roots, config, stage.
- Produces: `Phase3SelectionRequest`, `Phase3StageDecision`, `select_phase3_stage(request)`, selection JSON/Markdown, promoted recipe.

- [ ] **Step 1: Write failing gate tests**

```python
def test_promotes_candidate_only_when_all_nine_gates_pass(tmp_path):
    parent = write_matrix(tmp_path / "parent", macro_f1=0.50)
    candidate = write_matrix(tmp_path / "candidate", macro_f1=0.52)
    decision = select_phase3_stage(selection_request(parent, candidate))
    assert decision.promoted is True
    assert all(gate.passed for gate in decision.gates)


def test_zero_family_f1_forces_fallback_even_with_higher_macro_f1(tmp_path):
    candidate = write_matrix(tmp_path / "candidate", macro_f1=0.60, mimo_seed42=0.0)
    decision = select_phase3_stage(selection_request(PARENT, candidate))
    assert decision.promoted is False
    assert decision.fallback_recipe_id == "A0"
```

Cover missing/duplicate seed-view runs, smoke/test rejection, fingerprint mismatch, paired seed wins, two holdout directions, A1 tie order, A2 skipped gate, and canonical output serialization.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_phase3_selection.py -q
```

- [ ] **Step 3: Implement selection and exact output schemas**

Use sample standard deviation for multi-seed summaries and preserve every raw input metric. `promoted_recipe.json` is either the candidate recipe when promoted or the parent recipe on fallback.

- [ ] **Step 4: Add selection CLI and Markdown rendering**

The CLI writes into a new output directory and refuses overwrite. A2 selection can emit `skipped_by_gate` only from an A1 decision that explicitly fell back to A0.

- [ ] **Step 5: Run tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_phase3_selection.py -q
```

- [ ] **Step 6: Commit**

```bash
git add research/text_attribution/phase3_selection.py scripts/select_family_attribution_phase3.py tests/test_text_attribution_phase3_selection.py
git commit -m "feat: select phase3 attribution recipes"
```

---

### Task 10: Two-GPU Stage Scheduler

**Files:**
- Create: `scripts/run_attribution_phase3.sh`
- Create: `tests/test_run_attribution_phase3_scheduler.py`

**Interfaces:**
- Consumes: stage/recipe, data paths, cache, output root, comma-free seed list.
- Produces: independently logged run matrix across physical GPUs 0 and 1.

- [ ] **Step 1: Write failing scheduler tests with a fake Python runner**

Require exactly one command per seed/view, no test flag, correct output names, both GPUs used, no new task after a failure, in-flight task awaited, and interrupt cleanup of Python/tee process groups.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_run_attribution_phase3_scheduler.py -q
```

- [ ] **Step 3: Implement a Linux Bash scheduler**

Usage is:

```bash
bash scripts/run_attribution_phase3.sh \
  STAGE_OR_RECIPE DATA_JSONL SPLIT_MANIFEST HF_CACHE OUTPUT_ROOT \
  "42,52,62"
```

Build the Cartesian task list in stable seed-major/view order and maintain at most two process groups. Every child invokes the single-run CLI with one visible GPU and process device `cuda:0`.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python -m pytest tests/test_run_attribution_phase3_scheduler.py -q
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_attribution_phase3.sh tests/test_run_attribution_phase3_scheduler.py
git commit -m "feat: schedule phase3 attribution runs"
```

---

### Task 11: Server Overlay Package and Dependency Contract

**Files:**
- Create: `requirements-attribution-phase3.txt`
- Create: `scripts/package_attribution_phase3_server.py`
- Create: `tests/test_attribution_phase3_server_package.py`

**Interfaces:**
- Consumes: project root.
- Produces: deterministic `dist/aigc-detector-attribution-phase3-server.zip`, `PHASE3_SERVER_MANIFEST.json`, build and verify commands.

- [ ] **Step 1: Write failing package tests**

Require exact source entries, all Phase3 modules/tests/docs, manifest hashes, executable shell mode, reproducible bytes, project-root layout, and rejection of symlinks, raw data, `.env`, credential patterns, model weights, outputs, caches, and traversal paths.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest tests/test_attribution_phase3_server_package.py -q
```

- [ ] **Step 3: Add the dependency list**

Include the same bounded Transformers, Accelerate, NumPy, scikit-learn, matplotlib, seaborn, and pytest dependencies required by Phase2, while continuing to exclude PyTorch because the server CUDA wheel is environment-specific.

- [ ] **Step 4: Implement deterministic package build and verify**

Use a Phase3-specific package kind/version and exact allowlist. Include reused Phase2 source modules required at runtime, but exclude every runtime artifact.

- [ ] **Step 5: Run tests and verify GREEN**

```bash
python -m pytest tests/test_attribution_phase3_server_package.py -q
```

- [ ] **Step 6: Commit**

```bash
git add requirements-attribution-phase3.txt scripts/package_attribution_phase3_server.py tests/test_attribution_phase3_server_package.py
git commit -m "feat: package phase3 attribution server"
```

---

### Task 12: Server Runbook, End-to-End Validation, and Delivery ZIP

**Files:**
- Create: `docs/ATTRIBUTION_PHASE3_RUNBOOK.md`
- Modify: `scripts/package_attribution_phase3_server.py`
- Modify: `tests/test_attribution_phase3_server_package.py`

**Interfaces:**
- Consumes: all implemented Phase3 commands.
- Produces: audited runbook and final server overlay ZIP.

- [ ] **Step 1: Write the runbook with exact commands**

Cover build/verify, upload/unzip, environment snapshot, data checks, dry-run, A0–A4 smoke, A4 preflight, A0/A1 scheduling, stage selection, promoted recipes, A2 skipped behavior, recovery, monitoring, final-seed confirmation, and result download without weights.

- [ ] **Step 2: Add the runbook and plan/spec to the package allowlist**

Update the package test first so it fails until all three documents are included.

- [ ] **Step 3: Run focused Phase3 tests**

```bash
python -m pytest \
  tests/test_text_attribution_phase3_config.py \
  tests/test_text_attribution_phase3_data.py \
  tests/test_text_attribution_attribution_losses.py \
  tests/test_text_attribution_hierarchical_model.py \
  tests/test_text_attribution_phase3_training.py \
  tests/test_run_family_attribution_phase3.py \
  tests/test_text_attribution_phase3_selection.py \
  tests/test_run_attribution_phase3_scheduler.py \
  tests/test_attribution_phase3_server_package.py \
  -q
```

- [ ] **Step 4: Run the complete repository test suite**

```bash
python -m pytest -q
```

- [ ] **Step 5: Run static validation**

```bash
python -m compileall research/text_attribution scripts tests
ruff check research/text_attribution scripts tests
```

- [ ] **Step 6: Build and verify the server ZIP**

```bash
python -m scripts.package_attribution_phase3_server \
  --output dist/aigc-detector-attribution-phase3-server.zip

python -m scripts.package_attribution_phase3_server \
  --verify dist/aigc-detector-attribution-phase3-server.zip
```

- [ ] **Step 7: Inspect the final archive contract**

```bash
unzip -l dist/aigc-detector-attribution-phase3-server.zip
```

Verify the archive contains no data, outputs, logs, `.env`, caches, weights, tokenizers, or checkpoints.

- [ ] **Step 8: Commit**

```bash
git add docs/ATTRIBUTION_PHASE3_RUNBOOK.md scripts/package_attribution_phase3_server.py tests/test_attribution_phase3_server_package.py
git commit -m "docs: deliver phase3 attribution workflow"
```

---

## Plan Self-Review Checklist

- Every design requirement maps to at least one task.
- A0–A4, version holdout, promotion gates, test isolation, recovery, packaging, and server instructions are covered.
- Every production module is introduced after a failing test.
- Interfaces and field names are consistent across tasks.
- No task depends on an unspecified future function or file.
- Phase2 regression coverage is explicit.
- The final artifact is a verified server overlay ZIP plus a copy/run checklist.

