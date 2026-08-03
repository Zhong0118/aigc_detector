# Family Attribution Phase4 Fast Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重新训练 A4 神经模型、不读取 frozen test 的前提下，用一个服务器命令完成 Phase3 自动诊断、B0/B1 轻量分支训练、E1 概率融合、选择性拒判和 8-seed × 3-view 汇总，直接产出“是否超过 A4”的模型结果与部署 artifact。

**Architecture:** Fast Track 把诊断作为流水线前置校验而不是独立交付。CLI 读取已有 A0/A4 正式运行，按相同 V5 split 为每个 seed/view 训练字符 TF-IDF LR 与风格 LR，严格按 `request_key` 对齐概率；随后使用语义提示词组 5 折交叉拟合温度、融合权重和拒判阈值，输出 OOF 指标、跨 seed 汇总及 seed 42 部署 artifact。完整第二神经模型、Unknown/OOD 和 LoRA 不阻塞本计划，待 E1 结果证明有收益后另立实施计划。

**Tech Stack:** Python 3.10、NumPy 1.24–1.x、SciPy（由 scikit-learn 依赖提供）、scikit-learn 1.5.2、joblib、pytest、标准库 JSON/ZIP/Bash；A4 结果读取不需要 PyTorch 或 Transformers。

## Global Constraints

- 输入数据固定为 V5 约 6K 数据、V5 catalog 与现有 frozen split manifest。
- 家族标签固定为 OpenAI、Claude、Gemini、DeepSeek、Qwen、GLM、MiniMax、Kimi、豆包、Mimo 对应的十个内部 `family_id`，顺序必须来自 A4 `label_maps.json`。
- 正式 seed 固定为 `42,52,62,12,22,32,72,82`，生产 artifact seed 预登记为 42。
- 验证视图固定为 `closed`、`fast-to-balanced`、`balanced-to-fast`。
- A4 recipe hash 固定为 `1d4b21b8e2bb1477d4872049805cf6d1ca8e037ec52e2a538f851eb335dd71f2`。
- 所有输入必须满足相同 dataset fingerprint、split manifest hash、catalog hash、标签顺序与逐样本真值。
- `test_evaluated` 必须严格为 `false`；正式输入不得是 smoke 或 preflight。
- 不允许读取 split 中的 test 文本，不允许输出 test 指标，不允许按 test 选择权重或阈值。
- 交叉拟合单位固定为 `semantic_prompt_group`，外层折数固定为 5。
- Bootstrap 固定 2,000 次，以语义提示词组为重采样单位，分析 seed 固定为 20260803。
- E1 分支固定为 A4、B0 字符 TF-IDF LR、B1 显式风格 LR；第一版不加入深层 stacking。
- 融合权重是非负、和为 1、步长 0.05 的确定性单纯形网格。
- 临时选择性门槛固定为 accepted precision ≥ 0.90、coverage ≥ 0.40、ECE ≤ 0.05。
- 所有正式输出先写 staging 目录，校验通过后原子发布；已存在正式输出目录立即失败。
- Phase2、Phase3 的代码、结果和历史 `stage_decision.json` 保持不变。

---

## Deliverable and Time Box

本计划不是“只做分析”的切片。完成后必须得到以下可运行结果：

```text
outputs/attribution/family-phase4/fast-track-v1/
├── run_manifest.json
├── E0-A4/
├── classical/
├── E1-fusion/
├── selective/
├── diagnostics/
├── artifacts/
│   ├── seed42-fusion.json
│   ├── seed42-calibrators.json
│   ├── seed42-selective-policy.json
│   └── artifact_manifest.json
├── phase4_fast_track_summary.json
├── phase4_fast_track_report.md
└── COMPLETED.json
```

服务器主命令必须是：

```bash
python -u -m scripts.run_family_attribution_phase4_fast_track \
  --input "$P3_DATA" \
  --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json \
  --split-manifest "$P3_SPLIT" \
  --a0-root "$P3_ROOT/A0" \
  --a4-root "$P3_ROOT/A4" \
  --output-dir "$P3_ROOT/../family-phase4/fast-track-v1"
```

成功后只需发送：

```text
phase4_fast_track_summary.json
phase4_fast_track_report.md
```

若需要复核逐样本结果，再补充 `diagnostics/` 与 `E1-fusion/`，不需要打包 A4 权重、checkpoint 或原始文本。

---

## File Map

| File | Responsibility |
| --- | --- |
| `configs/text_attribution/training/family_phase4_fast_track.v1.json` | 固定 seeds、views、recipe hash、Bootstrap、融合网格、门槛和生产 seed |
| `research/text_attribution/phase4_config.py` | 严格解析配置、冻结 dataclass、规范化哈希 |
| `research/text_attribution/phase4_metrics.py` | Macro-P/R/F1、Top-K、校准、选择性指标和组级 Bootstrap |
| `research/text_attribution/diagnostic_analysis.py` | Phase3 run 发现、身份校验、预测读取、请求键对齐和互补性 |
| `research/text_attribution/phase4_classical.py` | 按 Phase3 view 训练 B0/B1 并生成对齐验证概率 |
| `research/text_attribution/probability_fusion.py` | 5 折组级交叉拟合、温度校准、单纯形权重搜索和最终 artifact |
| `research/text_attribution/selective_policy.py` | 拒判特征、阈值网格、OOF policy 与运行时状态决策 |
| `research/text_attribution/phase4_reporting.py` | seed/view 汇总、晋级判断、JSON/Markdown 与 artifact manifest |
| `scripts/run_family_attribution_phase4_fast_track.py` | 单命令端到端编排与原子输出 |
| `scripts/package_attribution_phase4_fast_track.py` | 无数据、无权重、无密钥的可复现服务器 overlay ZIP |
| `requirements-attribution-phase4-fast-track.txt` | CPU Fast Track 依赖 |
| `docs/ATTRIBUTION_PHASE4_FAST_TRACK_RUNBOOK.md` | Mac 打包、服务器上传、运行、恢复和下载命令 |
| `tests/test_text_attribution_phase4_config.py` | 配置与哈希合同 |
| `tests/test_text_attribution_phase4_metrics.py` | 扩展指标、Bootstrap 与选择性指标 |
| `tests/test_text_attribution_diagnostic_analysis.py` | run 校验、对齐和互补性 |
| `tests/test_text_attribution_phase4_classical.py` | B0/B1 视图训练与 test 隔离 |
| `tests/test_text_attribution_probability_fusion.py` | 校准、折分、权重、tie-break 与 OOF |
| `tests/test_text_attribution_selective_policy.py` | risk-coverage、阈值与状态不变量 |
| `tests/test_run_family_attribution_phase4_fast_track.py` | 端到端 CLI、原子目录和结果合同 |
| `tests/test_attribution_phase4_fast_track_package.py` | ZIP manifest、路径、密钥、数据与权重排除 |

---

### Task 1: Freeze Fast Track Configuration and Core Contracts

**Files:**
- Create: `configs/text_attribution/training/family_phase4_fast_track.v1.json`
- Create: `research/text_attribution/phase4_config.py`
- Test: `tests/test_text_attribution_phase4_config.py`

**Interfaces:**
- Consumes: versioned JSON configuration.
- Produces: `Phase4FastTrackConfig`, `load_phase4_fast_track_config(path)`, `canonical_config_hash(config)`.

- [ ] **Step 1: Write the failing strict-config tests**

```python
from dataclasses import FrozenInstanceError

import pytest

from research.text_attribution.phase4_config import (
    canonical_config_hash,
    load_phase4_fast_track_config,
)


def test_loads_frozen_fast_track_contract():
    config = load_phase4_fast_track_config(
        "configs/text_attribution/training/family_phase4_fast_track.v1.json"
    )
    assert config.seeds == (42, 52, 62, 12, 22, 32, 72, 82)
    assert config.views == (
        "closed",
        "fast-to-balanced",
        "balanced-to-fast",
    )
    assert config.production_seed == 42
    assert config.bootstrap_resamples == 2000
    assert len(canonical_config_hash(config)) == 64
    with pytest.raises(FrozenInstanceError):
        config.production_seed = 52


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":1,"unexpected":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown.*unexpected"):
        load_phase4_fast_track_config(path)
```

- [ ] **Step 2: Run the config tests and verify RED**

Run:

```bash
python -m pytest tests/test_text_attribution_phase4_config.py -q
```

Expected: test collection fails because `phase4_config` does not exist.

- [ ] **Step 3: Add the exact JSON configuration**

```json
{
  "schema_version": 1,
  "run_id": "family-attribution-phase4-fast-track-v1",
  "a4_recipe_hash": "1d4b21b8e2bb1477d4872049805cf6d1ca8e037ec52e2a538f851eb335dd71f2",
  "seeds": [42, 52, 62, 12, 22, 32, 72, 82],
  "views": ["closed", "fast-to-balanced", "balanced-to-fast"],
  "production_seed": 42,
  "branches": ["A4", "B0", "B1"],
  "cross_fit_folds": 5,
  "fold_seed": 20260803,
  "bootstrap_resamples": 2000,
  "bootstrap_seed": 20260803,
  "ece_bins": 10,
  "weight_step": 0.05,
  "max_single_branch_macro_f1_drop": 0.005,
  "e1_closed_macro_f1_min_gain": 0.01,
  "holdout_max_drop": 0.01,
  "accepted_precision_min": 0.9,
  "coverage_min": 0.4,
  "ece_max": 0.05
}
```

- [ ] **Step 4: Implement exact-key parsing and frozen dataclasses**

```python
@dataclass(frozen=True)
class Phase4FastTrackConfig:
    schema_version: int
    run_id: str
    a4_recipe_hash: str
    seeds: tuple[int, ...]
    views: tuple[str, ...]
    production_seed: int
    branches: tuple[str, ...]
    cross_fit_folds: int
    fold_seed: int
    bootstrap_resamples: int
    bootstrap_seed: int
    ece_bins: int
    weight_step: float
    max_single_branch_macro_f1_drop: float
    e1_closed_macro_f1_min_gain: float
    holdout_max_drop: float
    accepted_precision_min: float
    coverage_min: float
    ece_max: float
```

Validate exact keys, unique seeds/views/branches, `production_seed in seeds`, the three exact view names, `branches == ("A4", "B0", "B1")`, 64-character lowercase SHA-256 recipe hash, five folds, positive Bootstrap count, `1 / weight_step` integral within `1e-9`, and all thresholds in `[0,1]`. Hash the JSON-serializable dataclass mapping with sorted keys and compact UTF-8 separators.

- [ ] **Step 5: Run the config tests and verify GREEN**

```bash
python -m pytest tests/test_text_attribution_phase4_config.py -q
```

- [ ] **Step 6: Commit the configuration contract**

```bash
git add configs/text_attribution/training/family_phase4_fast_track.v1.json research/text_attribution/phase4_config.py tests/test_text_attribution_phase4_config.py
git commit -m "feat: define phase4 fast track contract"
```

---

### Task 2: Add Extended Metrics and Group Bootstrap

**Files:**
- Create: `research/text_attribution/phase4_metrics.py`
- Test: `tests/test_text_attribution_phase4_metrics.py`

**Interfaces:**
- Consumes: fixed label order and immutable `FamilyPrediction` records.
- Produces: `evaluate_phase4_predictions(records, labels, ece_bins)`, `group_bootstrap_interval(records, labels, metric_name, resamples, seed)`, `evaluate_selective_predictions(records, accepted_mask, labels)`.

- [ ] **Step 1: Write failing metric tests with exact expected values**

```python
import numpy as np

from research.text_attribution.phase4_metrics import (
    evaluate_phase4_predictions,
    evaluate_selective_predictions,
    group_bootstrap_interval,
)
from tests.text_attribution_test_utils import make_family_predictions


def test_extended_metrics_include_precision_recall_and_top_k():
    records = make_family_predictions(
        truth=("a", "a", "b", "b"),
        probabilities=((0.9, 0.1), (0.4, 0.6), (0.3, 0.7), (0.8, 0.2)),
        groups=("g1", "g2", "g1", "g2"),
        labels=("a", "b"),
    )
    result = evaluate_phase4_predictions(records, ("a", "b"), ece_bins=2)
    assert result["overall"]["accuracy"] == 0.5
    assert result["overall"]["macro_precision"] == 0.5
    assert result["overall"]["macro_recall"] == 0.5
    assert result["top_k_accuracy"]["top_1"] == 0.5
    assert result["top_k_accuracy"]["top_2"] == 1.0


def test_selective_precision_and_coverage_use_fixed_denominators():
    records = make_family_predictions(
        truth=("a", "a", "b", "b"),
        probabilities=((0.9, 0.1), (0.4, 0.6), (0.3, 0.7), (0.8, 0.2)),
        groups=("g1", "g2", "g1", "g2"),
        labels=("a", "b"),
    )
    result = evaluate_selective_predictions(
        records, np.asarray([True, False, True, False]), ("a", "b")
    )
    assert result["accepted_precision"] == 1.0
    assert result["coverage"] == 0.5
    assert result["selective_risk"] == 0.0


def test_group_bootstrap_is_reproducible():
    records = make_family_predictions(
        truth=("a", "b", "a", "b"),
        probabilities=((0.8, 0.2), (0.3, 0.7), (0.6, 0.4), (0.6, 0.4)),
        groups=("g1", "g1", "g2", "g2"),
        labels=("a", "b"),
    )
    first = group_bootstrap_interval(records, ("a", "b"), "macro_f1", 100, 7)
    second = group_bootstrap_interval(records, ("a", "b"), "macro_f1", 100, 7)
    assert first == second
    assert first["resampling_unit"] == "semantic_prompt_group"
```

Extend `tests/text_attribution_test_utils.py` only if `make_family_predictions` is absent; the helper must create real `FamilyPrediction` values with deterministic request keys and metadata.

- [ ] **Step 2: Run the metric tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_phase4_metrics.py -q
```

Expected: import failure for `phase4_metrics`.

- [ ] **Step 3: Implement extended full-set and selective metrics**

`evaluate_phase4_predictions` must call the existing `evaluate_family_predictions` and add:

```python
macro_precision = precision_score(
    truth, predicted, labels=labels, average="macro", zero_division=0
)
macro_recall = recall_score(
    truth, predicted, labels=labels, average="macro", zero_division=0
)
top_k = {
    f"top_{k}": float(top_k_accuracy_score(
        truth_indexes,
        probabilities,
        k=min(k, len(labels)),
        labels=np.arange(len(labels)),
    ))
    for k in (1, 2, 3)
}
```

For selective metrics, `accepted_precision = correct accepted Top-1 / accepted_count`, `coverage = accepted_count / all eligible records`, and `selective_risk = 1 - accepted_precision`. A family with no accepted prediction must use JSON `null` for undefined accepted precision and emit `coverage_warning=true`.

- [ ] **Step 4: Implement semantic-group Bootstrap**

Use `np.random.default_rng(seed)` to sample the sorted unique semantic groups with replacement. Include every record belonging to each sampled group, preserving multiplicity. Return `estimate`, `lower_95`, `upper_95`, `resamples`, `seed`, and `resampling_unit`; percentiles are 2.5 and 97.5. Reject fewer than two unique groups and nonpositive resample counts.

- [ ] **Step 5: Run the metric tests and existing metric regression tests**

```bash
python -m pytest tests/test_text_attribution_phase4_metrics.py tests/test_text_attribution_metrics_v2.py -q
```

- [ ] **Step 6: Commit the metrics**

```bash
git add research/text_attribution/phase4_metrics.py tests/test_text_attribution_phase4_metrics.py tests/text_attribution_test_utils.py
git commit -m "feat: add phase4 attribution metrics"
```

---

### Task 3: Load, Validate, Align, and Diagnose Phase3 Runs

**Files:**
- Create: `research/text_attribution/diagnostic_analysis.py`
- Test: `tests/test_text_attribution_diagnostic_analysis.py`

**Interfaces:**
- Consumes: one formal Phase3 run directory per branch/seed/view.
- Produces: `RunIdentity`, `PredictionRun`, `AlignedBranchProbabilities`, `load_prediction_run(run_dir, expected)`, `align_prediction_runs(runs)`, `analyze_complementarity(aligned, branch_a, branch_b)`.

- [ ] **Step 1: Write failing formal-run and alignment tests**

```python
import pytest

from research.text_attribution.diagnostic_analysis import (
    ExpectedRun,
    align_prediction_runs,
    load_prediction_run,
)
from tests.text_attribution_test_utils import write_prediction_run


def test_loads_only_matching_validation_only_formal_run(tmp_path):
    run_dir = write_prediction_run(
        tmp_path,
        branch_id="A4",
        seed=42,
        view_id="closed",
        recipe_hash="1d4b21b8e2bb1477d4872049805cf6d1ca8e037ec52e2a538f851eb335dd71f2",
    )
    run = load_prediction_run(
        run_dir,
        ExpectedRun(branch_id="A4", seed=42, view_id="closed"),
    )
    assert run.identity.test_evaluated is False
    assert run.identity.smoke_test is False
    assert len(run.records) == 4


def test_rejects_test_smoke_and_fingerprint_mismatch(tmp_path):
    run_dir = write_prediction_run(
        tmp_path,
        branch_id="A4",
        seed=42,
        view_id="closed",
        test_evaluated=True,
    )
    with pytest.raises(ValueError, match="test_evaluated"):
        load_prediction_run(
            run_dir,
            ExpectedRun(branch_id="A4", seed=42, view_id="closed"),
        )


def test_alignment_rejects_missing_request_key(tmp_path):
    a4 = load_fixture_run(tmp_path, "A4", request_keys=("r1", "r2"))
    b0 = load_fixture_run(tmp_path, "B0", request_keys=("r1",))
    with pytest.raises(ValueError, match="request_key"):
        align_prediction_runs((a4, b0))
```

The helper must also cover duplicate request keys, probability order mismatch, true-family mismatch, semantic-group mismatch, absent `COMPLETED.json`, wrong recipe hash, missing metrics, and a non-complete status.

- [ ] **Step 2: Run diagnostic tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_diagnostic_analysis.py -q
```

- [ ] **Step 3: Implement immutable run contracts**

```python
@dataclass(frozen=True)
class ExpectedRun:
    branch_id: str
    seed: int
    view_id: str


@dataclass(frozen=True)
class RunIdentity:
    branch_id: str
    seed: int
    view_id: str
    dataset_fingerprint: str
    split_manifest_hash: str
    catalog_hash: str
    recipe_hash: str
    run_fingerprint: str
    test_evaluated: bool
    smoke_test: bool
    preflight_only: bool


@dataclass(frozen=True)
class PredictionRun:
    identity: RunIdentity
    family_labels: tuple[str, ...]
    records: tuple[FamilyPrediction, ...]
    run_dir: Path


@dataclass(frozen=True)
class AlignedBranchProbabilities:
    request_keys: tuple[str, ...]
    true_families: tuple[str, ...]
    semantic_prompt_groups: tuple[str, ...]
    family_labels: tuple[str, ...]
    records_by_branch: Mapping[str, tuple[FamilyPrediction, ...]]
    probabilities_by_branch: Mapping[str, np.ndarray]
```

Read `run_config.json`, `resolved_recipe.json`, `label_maps.json`, `metrics.json`, `predictions.jsonl`, and `COMPLETED.json`. Sort by `request_key` only after proving keys are unique. The identity comparison must include dataset/split/catalog hashes, seed, view, label order, truth, semantic group, language, task, length and temperature.

- [ ] **Step 4: Implement complementarity calculations**

For branch pair A/B, return counts and rates for `both_correct`, `a_only_correct`, `b_only_correct`, `both_wrong`, label disagreement, mean Jensen-Shannon divergence, A-error correction by B, B-error correction by A, per-family correction, oracle Top-1, and equal-weight Top-1/Macro-F1. Compute Jensen-Shannon with epsilon `1e-12` and natural logarithms.

- [ ] **Step 5: Run diagnostic and Phase3 regression tests**

```bash
python -m pytest tests/test_text_attribution_diagnostic_analysis.py tests/test_text_attribution_phase3_selection.py -q
```

- [ ] **Step 6: Commit run diagnostics**

```bash
git add research/text_attribution/diagnostic_analysis.py tests/test_text_attribution_diagnostic_analysis.py tests/text_attribution_test_utils.py
git commit -m "feat: validate phase3 attribution bundles"
```

---

### Task 4: Train B0 and B1 on the Exact Phase3 Views

**Files:**
- Create: `research/text_attribution/phase4_classical.py`
- Test: `tests/test_text_attribution_phase4_classical.py`

**Interfaces:**
- Consumes: loaded V5 dataset, frozen split manifest, V5 variant catalog, seed and view.
- Produces: `ClassicalBranchResult`, `train_aligned_classical_branches(request)` and formal B0/B1 run directories consumable by Task 3.

- [ ] **Step 1: Write failing exact-view and test-isolation tests**

```python
from research.text_attribution.phase4_classical import (
    ClassicalBranchRequest,
    train_aligned_classical_branches,
)


def test_classical_branches_match_phase3_validation_keys(tmp_path):
    request = make_classical_request(tmp_path, seed=42, view_id="fast-to-balanced")
    result = train_aligned_classical_branches(request)
    assert set(result.runs) == {"B0", "B1"}
    assert result.runs["B0"].request_keys == request.expected_validation_keys
    assert result.runs["B1"].request_keys == request.expected_validation_keys
    assert not set(request.test_request_keys) & set(result.runs["B0"].request_keys)


def test_classical_training_never_reads_test_text(tmp_path):
    request = make_classical_request(
        tmp_path,
        seed=42,
        view_id="closed",
        poison_test_text_property=True,
    )
    result = train_aligned_classical_branches(request)
    assert result.runs["B0"].sample_count == len(request.expected_validation_keys)
```

Add a test proving `closed` uses all train variants, `fast-to-balanced` trains only fast and validates only balanced, and the reverse view swaps roles.

- [ ] **Step 2: Run classical tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_phase4_classical.py -q
```

- [ ] **Step 3: Implement the request/result contract**

```python
@dataclass(frozen=True)
class ClassicalBranchRequest:
    data_path: Path
    catalog_path: Path
    split_manifest_path: Path
    output_root: Path
    seed: int
    view_id: str
    expected_validation_keys: tuple[str, ...]
    test_request_keys: frozenset[str]
    expected_identity: RunIdentity
    persist_model: bool


@dataclass(frozen=True)
class ClassicalBranchRun:
    branch_id: str
    run_dir: Path
    request_keys: tuple[str, ...]
    sample_count: int


@dataclass(frozen=True)
class ClassicalBranchResult:
    runs: Mapping[str, ClassicalBranchRun]
```

- [ ] **Step 4: Reuse existing B0/B1 pipelines without refitting on validation**

Load data with `load_training_dataset`, apply `load_split_manifest` and `build_phase3_data_view`, and assert validation keys exactly equal the A4 expected keys. Train:

```python
for branch_id in ("B0", "B1"):
    model = build_classical_pipeline(branch_id, seed=request.seed)
    model.fit(
        [row.text for row in data_view.train_samples],
        [row.family_id for row in data_view.train_samples],
    )
    probabilities = predict_family_probabilities(
        model,
        [row.text for row in data_view.validation_samples],
        family_labels=family_labels,
    )
```

Every branch saves `run_config.json`, `label_maps.json`, `metrics.json`, `predictions.jsonl` and `COMPLETED.json`. Save `model.joblib` only when `request.persist_model` is true; the orchestrator sets it true only for the pre-registered production seed 42 and false for the other seven seeds. Mark `test_evaluated=false`, `smoke_test=false`, `preflight_only=false`; bind dataset/split/catalog hashes and create a deterministic run fingerprint. Do not serialize raw text. This keeps 48 validation fits while avoiding 48 duplicated sparse-model artifacts.

- [ ] **Step 5: Run classical, baseline-model, and data-view tests**

```bash
python -m pytest tests/test_text_attribution_phase4_classical.py tests/test_text_attribution_classical_models_v2.py tests/test_text_attribution_phase3_data.py -q
```

- [ ] **Step 6: Commit aligned classical training**

```bash
git add research/text_attribution/phase4_classical.py tests/test_text_attribution_phase4_classical.py
git commit -m "feat: train aligned phase4 classical branches"
```

---

### Task 5: Cross-Fit Temperatures and E1 Fusion Weights

**Files:**
- Create: `research/text_attribution/probability_fusion.py`
- Test: `tests/test_text_attribution_probability_fusion.py`

**Interfaces:**
- Consumes: `AlignedBranchProbabilities`, `Phase4FastTrackConfig`.
- Produces: `FusionFitRequest`, `FusionArtifact`, `CrossFitFusionResult`, `fit_probability_fusion(request)`, `apply_probability_fusion(artifact, branches)`.

- [ ] **Step 1: Write failing deterministic-fold and calibration tests**

```python
import numpy as np

from research.text_attribution.probability_fusion import (
    assign_group_folds,
    fit_probability_fusion,
)


def test_group_fold_assignment_is_deterministic_and_has_no_group_leakage():
    groups = ("g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5")
    first = assign_group_folds(groups, true_families=("a", "b") * 5, folds=5, seed=7)
    second = assign_group_folds(groups, true_families=("a", "b") * 5, folds=5, seed=7)
    assert np.array_equal(first, second)
    for group in set(groups):
        assert len(set(first[index] for index, value in enumerate(groups) if value == group)) == 1


def test_fusion_prefers_complementary_branch_and_returns_oof_only():
    aligned = make_complementary_aligned_probabilities()
    result = fit_probability_fusion(make_fusion_request(aligned))
    assert result.oof_probabilities.shape == (len(aligned.request_keys), 2)
    assert np.allclose(result.oof_probabilities.sum(axis=1), 1.0)
    assert result.final_artifact.weights["A4"] < 1.0
    assert set(result.fold_artifacts) == {0, 1, 2, 3, 4}
```

Add tests for: every validation row predicted once; a fold missing one family fails; temperature is finite and positive; grid weights are multiples of 0.05 and sum to one; Macro-F1 drop filter; all four tie-break levels; byte-stable artifact JSON.

- [ ] **Step 2: Run fusion tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_probability_fusion.py -q
```

- [ ] **Step 3: Implement deterministic stratified semantic-group folds**

Sort groups by a SHA-256 key derived from `fold_seed`, group name and the group's family-count vector. Greedily assign each group to the fold minimizing `(missing_family_penalty, family_count_imbalance, total_rows, fold_id)`. Validate that every fold and every four-fold training complement contains all family labels; otherwise raise a data-insufficiency error.

- [ ] **Step 4: Implement positive scalar temperature fitting**

Treat probabilities as clipped pseudo-logits `log(clip(p, 1e-12, 1))`. Optimize `log_temperature` in `[-4,4]` with `scipy.optimize.minimize_scalar(method="bounded")`, minimize multiclass NLL, require optimizer success, and return `temperature = exp(log_temperature)`. Applying temperature means `softmax(log(p) / temperature)`.

- [ ] **Step 5: Implement the exact simplex search and tie-break**

Enumerate integer units summing to `round(1 / weight_step)` for the three ordered branches. For each candidate, average calibrated probabilities and compute NLL, Macro-F1 and worst-family F1. Discard candidates whose Macro-F1 is more than 0.005 below the best calibrated single branch. Sort remaining candidates by:

```python
(
    round(nll, 6),
    -worst_family_f1,
    nonzero_branch_count,
    -weights["A4"],
    tuple(weights[name] for name in ("A4", "B0", "B1")),
)
```

- [ ] **Step 6: Implement five-fold OOF and final full-validation artifacts**

```python
@dataclass(frozen=True)
class FusionArtifact:
    schema_version: int
    branch_order: tuple[str, ...]
    temperatures: Mapping[str, float]
    weights: Mapping[str, float]
    family_labels: tuple[str, ...]
    fit_request_keys_hash: str
    config_hash: str


@dataclass(frozen=True)
class CrossFitFusionResult:
    oof_probabilities: np.ndarray
    fold_ids: np.ndarray
    fold_artifacts: Mapping[int, FusionArtifact]
    final_artifact: FusionArtifact
```

For fold `k`, fit calibrators and weights on rows with `fold_id != k`, apply only to `fold_id == k`, and assert every row is filled exactly once. Fit `final_artifact` on all validation rows only after OOF probabilities are complete.

- [ ] **Step 7: Run fusion tests and metric regression tests**

```bash
python -m pytest tests/test_text_attribution_probability_fusion.py tests/test_text_attribution_phase4_metrics.py -q
```

- [ ] **Step 8: Commit E1 probability fusion**

```bash
git add research/text_attribution/probability_fusion.py tests/test_text_attribution_probability_fusion.py
git commit -m "feat: cross-fit phase4 probability fusion"
```

---

### Task 6: Fit the Selective Abstain Policy

**Files:**
- Create: `research/text_attribution/selective_policy.py`
- Test: `tests/test_text_attribution_selective_policy.py`

**Interfaces:**
- Consumes: E1 OOF probabilities, branch probabilities, fold IDs and truth.
- Produces: `SelectiveFitRequest`, `SelectivePolicyArtifact`, `SelectiveCrossFitResult`, `fit_selective_policy(request)`, `decide_attribution(artifact, evidence)`.

- [ ] **Step 1: Write failing threshold and state tests**

```python
from research.text_attribution.selective_policy import (
    AttributionEvidence,
    decide_attribution,
    fit_selective_policy,
)


def test_policy_meets_precision_then_maximizes_coverage():
    request = make_selective_request_with_feasible_threshold()
    result = fit_selective_policy(request)
    assert result.oof_metrics["accepted_precision"] >= 0.9
    assert result.oof_metrics["coverage"] >= 0.4
    assert result.final_artifact.max_probability_min > 0.0


def test_runtime_abstains_on_conflict_or_invalid_artifact():
    artifact = make_selective_artifact()
    conflict = AttributionEvidence(
        fused_probabilities=(0.51, 0.49),
        branch_probabilities={
            "A4": (0.9, 0.1),
            "B0": (0.1, 0.9),
            "B1": (0.5, 0.5),
        },
        effective_character_count=400,
    )
    assert decide_attribution(artifact, conflict).status == "abstain"
```

Add tests that Phase4 never returns `unknown`, `family_only` has at least one candidate, probabilities sum to one, invalid/missing artifacts abstain, and no accepted samples yields JSON `null` precision rather than division by zero.

- [ ] **Step 2: Run policy tests and verify RED**

```bash
python -m pytest tests/test_text_attribution_selective_policy.py -q
```

- [ ] **Step 3: Implement evidence features**

For every row compute calibrated maximum probability, Top-1/Top-2 margin, normalized entropy, three-branch label disagreement count, and mean pairwise Jensen-Shannon divergence. Reject nonfinite values, label-order mismatch or probability rows not summing to one.

- [ ] **Step 4: Implement cross-fitted deterministic policy search**

Maximum-probability thresholds use observed quantiles `[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]`. To keep the Fast Track bounded, do not construct a four-feature Cartesian product. Generate exactly the union of: 21 confidence-only rules; 21 × 5 confidence-plus-margin rules; 21 × 5 confidence-plus-entropy rules; 21 × 5 confidence-plus-JS rules, where the secondary feature uses quantiles `[0.00, 0.25, 0.50, 0.75, 1.00]`; and 21 × 3 confidence-plus-disagreement rules with disagreement maximum `0,1,2`. This yields at most 399 candidates per fold. Disabled secondary thresholds are stored as JSON `null`. For each held-out fold, select on the other four folds using this order:

1. accepted precision ≥ 0.90;
2. coverage ≥ 0.40;
3. maximize coverage;
4. maximize worst-family accepted recall;
5. minimize ECE;
6. choose the lexicographically smallest threshold tuple.

If no candidate satisfies both hard thresholds, select minimum selective risk, then maximum coverage, and mark `deployment_ready=false`. Apply the chosen fold policy only to its held-out fold. Fit the final deployment policy on all validation OOF rows after reporting is complete.

- [ ] **Step 5: Implement the runtime output contract**

```python
@dataclass(frozen=True)
class AttributionDecision:
    schema_version: int
    status: str
    top_families: tuple[tuple[str, float], ...]
    confidence: float | None
    prediction_set: tuple[str, ...]
    known_probability: float | None
    warnings: tuple[str, ...]
```

`status` is only `family_only` or `abstain`. `known_probability` remains `None`. Return calibrated Top-3 for accepted rows. Use warnings `single_text_evidence`, `unseen_version_risk`, `low_confidence`, `branch_conflict`, or `artifact_mismatch` as applicable.

- [ ] **Step 6: Run policy and fusion tests**

```bash
python -m pytest tests/test_text_attribution_selective_policy.py tests/test_text_attribution_probability_fusion.py -q
```

- [ ] **Step 7: Commit the selective policy**

```bash
git add research/text_attribution/selective_policy.py tests/test_text_attribution_selective_policy.py
git commit -m "feat: add phase4 selective attribution policy"
```

---

### Task 7: Orchestrate One-Command E0→E1→Abstain and Report the Result

**Files:**
- Create: `research/text_attribution/phase4_reporting.py`
- Create: `scripts/run_family_attribution_phase4_fast_track.py`
- Test: `tests/test_run_family_attribution_phase4_fast_track.py`

**Interfaces:**
- Consumes: config, V5 data/catalog/split, A0 root and A4 root.
- Produces: complete Fast Track directory, `phase4_fast_track_summary.json`, `phase4_fast_track_report.md`, seed 42 artifacts and process exit status.

- [ ] **Step 1: Write failing CLI contract and matrix tests**

```python
from scripts.run_family_attribution_phase4_fast_track import main


def test_dry_run_discovers_exact_matrix_without_training(tmp_path, capsys):
    roots = write_complete_a0_a4_matrix(
        tmp_path,
        seeds=(42, 52, 62, 12, 22, 32, 72, 82),
        views=("closed", "fast-to-balanced", "balanced-to-fast"),
    )
    status = main([
        "--config", "configs/text_attribution/training/family_phase4_fast_track.v1.json",
        "--input", str(roots.data),
        "--catalog", str(roots.catalog),
        "--split-manifest", str(roots.split),
        "--a0-root", str(roots.a0),
        "--a4-root", str(roots.a4),
        "--output-dir", str(tmp_path / "out"),
        "--dry-run",
    ])
    assert status == 0
    assert not (tmp_path / "out").exists()
    assert '"test_evaluated": false' in capsys.readouterr().out


def test_end_to_end_fixture_writes_model_result_and_no_test_metrics(tmp_path):
    roots = write_complete_a0_a4_matrix(
        tmp_path,
        seeds=(42, 52, 62, 12, 22, 32, 72, 82),
        views=("closed", "fast-to-balanced", "balanced-to-fast"),
    )
    output = tmp_path / "result"
    assert run_fixture_fast_track(roots, output) == 0
    summary = read_json(output / "phase4_fast_track_summary.json")
    assert summary["selection_scope"] == "validation_only"
    assert summary["test_evaluated"] is False
    assert set(summary["systems"]) == {"A0", "A4", "E1", "E1-selective"}
    assert (output / "artifacts/seed42-fusion.json").is_file()
    assert (output / "COMPLETED.json").is_file()
    for system in summary["systems"].values():
        assert "test" not in system
```

Add failures for incomplete seed/view matrix, output already exists, A0/A4 fingerprint mismatch, test rows in predictions, a failed classical child, and interrupted staging publication.

- [ ] **Step 2: Run orchestration tests and verify RED**

```bash
python -m pytest tests/test_run_family_attribution_phase4_fast_track.py -q
```

- [ ] **Step 3: Implement the CLI preflight and run matrix**

CLI arguments are exactly:

```text
--config
--input
--catalog
--split-manifest
--a0-root
--a4-root
--output-dir
--dry-run
--bootstrap-resamples
```

`--bootstrap-resamples` is allowed only to reduce local fixture/smoke cost and must be recorded; formal output requires the configured 2,000. Discover paths `A0-{view}-seed{seed}` and `A4-{view}-seed{seed}`. Validate the entire matrix before creating staging or fitting B0/B1.

- [ ] **Step 4: Implement one continuous seed/view loop**

For each configured seed and view:

1. load and validate A0/A4;
2. recompute E0 extended metrics from A4 predictions;
3. train B0/B1 on the exact view;
4. reload B0/B1 through the same strict run loader;
5. align A4/B0/B1;
6. write pairwise complementarity;
7. fit E1 five-fold OOF fusion and final artifact;
8. fit five-fold selective policy;
9. write per-run predictions and metrics;
10. release in-memory sparse matrices and models before the next run.

Set `persist_model = (seed == config.production_seed)` in step 3. The other seven seeds keep probabilities and metrics but discard fitted joblib objects after each run.

Never continue after a failed cell; write `FAILED.json` in staging and exit nonzero.

- [ ] **Step 5: Implement exact aggregate result and selection rules**

For A0, A4 and E1, summarize each view's Macro-F1, Accuracy, NLL, ECE, worst-family F1, mean, sample standard deviation, min, max and group Bootstrap CI. Report per-family P/R/F1 for E1 and A4. E1 is `promoted` only when:

```python
closed_gain >= 0.01
and fast_to_balanced_gain >= -0.01
and balanced_to_fast_gain >= -0.01
and e1_worst_family_f1 >= a4_worst_family_f1
```

Independently mark `selective_deployment_ready` only when accepted precision ≥ 0.90, coverage ≥ 0.40, ECE ≤ 0.05, and no family has zero accepted recall when it has eligible support. If E1 is not promoted, the summary must still identify A4 as the retained research model and explain the failed gates.

- [ ] **Step 6: Implement report and artifact manifest**

The JSON top level must contain:

```json
{
  "schema_version": 1,
  "selection_scope": "validation_only",
  "test_evaluated": false,
  "seeds": [42, 52, 62, 12, 22, 32, 72, 82],
  "views": ["closed", "fast-to-balanced", "balanced-to-fast"],
  "systems": {},
  "paired_gains": {},
  "per_family": {},
  "complementarity": {},
  "selective": {},
  "decision": {}
}
```

The Markdown begins with a one-line answer: E1 promoted, A4 retained, or evidence invalid. It then shows the three-view A0/A4/E1 table, selective precision/coverage, worst families, pairwise oracle headroom, failed gates, and next recommended action. The seed 42 artifact manifest binds config hash, source commit, dataset/split/catalog hashes, A4 run fingerprint, B0/B1 model hashes, fusion hash, policy hash and `test_evaluated=false`.

- [ ] **Step 7: Implement staging and atomic publication**

Create `output_dir.parent`, then `tempfile.mkdtemp(prefix=f".{output_dir.name}.run-", dir=output_dir.parent)`. Write all artifacts there, fsync JSON files, write `COMPLETED.json` last, and publish with `os.replace(staging, output_dir)`. If `output_dir` exists before execution, fail without deleting it.

- [ ] **Step 8: Run orchestration and focused regression tests**

```bash
python -m pytest tests/test_run_family_attribution_phase4_fast_track.py tests/test_text_attribution_phase4_classical.py tests/test_text_attribution_probability_fusion.py tests/test_text_attribution_selective_policy.py -q
```

- [ ] **Step 9: Commit the end-to-end Fast Track**

```bash
git add research/text_attribution/phase4_reporting.py scripts/run_family_attribution_phase4_fast_track.py tests/test_run_family_attribution_phase4_fast_track.py
git commit -m "feat: run phase4 attribution fast track"
```

---

### Task 8: Package, Run, and Return the Server Result

**Files:**
- Create: `requirements-attribution-phase4-fast-track.txt`
- Create: `scripts/package_attribution_phase4_fast_track.py`
- Create: `docs/ATTRIBUTION_PHASE4_FAST_TRACK_RUNBOOK.md`
- Test: `tests/test_attribution_phase4_fast_track_package.py`

**Interfaces:**
- Consumes: repository source files only.
- Produces: deterministic `attribution-phase4-fast-track-server.zip`, verified manifest and copy-paste server commands.

- [ ] **Step 1: Write failing package allowlist and secret-exclusion tests**

```python
from scripts.package_attribution_phase4_fast_track import (
    build_phase4_fast_track_package,
    verify_phase4_fast_track_package,
)


def test_package_contains_fast_track_code_and_no_runtime_artifacts(tmp_path):
    archive = tmp_path / "phase4.zip"
    result = build_phase4_fast_track_package(PROJECT_ROOT, archive)
    verified = verify_phase4_fast_track_package(archive)
    assert verified.file_count == result.file_count
    names = set(verified.relative_paths)
    assert "research/text_attribution/probability_fusion.py" in names
    assert "scripts/run_family_attribution_phase4_fast_track.py" in names
    assert not any(name.startswith("data/") for name in names)
    assert not any("outputs/" in name for name in names)
    assert not any(name.endswith((".joblib", ".safetensors", ".pt")) for name in names)
```

Also reject absolute paths, `..`, symlinks, `.env`, API-key patterns, logs, cache, checkpoints, models and any unmanifested ZIP member.

- [ ] **Step 2: Run package tests and verify RED**

```bash
python -m pytest tests/test_attribution_phase4_fast_track_package.py -q
```

- [ ] **Step 3: Add the bounded CPU requirements**

```text
numpy>=1.24,<2.0
scipy>=1.10,<2
pandas>=2.0,<3
scikit-learn==1.5.2
joblib>=1.3,<2
pytest>=8,<9
```

- [ ] **Step 4: Implement deterministic allowlist packaging**

Include Phase4 files, their direct Phase2/Phase3 data/metrics dependencies, config, tests, runbook, specification and this plan. Use sorted paths, fixed ZIP timestamps, SHA-256 per member and a top-level `PHASE4_FAST_TRACK_SERVER_MANIFEST.json`. Do not include A4 weights because the server already owns them.

- [ ] **Step 5: Write the exact Mac and server runbook**

Mac packaging:

```bash
python -m scripts.package_attribution_phase4_fast_track \
  --project-root "$PWD" \
  --output "$HOME/Desktop/attribution-phase4-fast-track-server.zip"
```

Server extraction and environment:

```bash
cd /home/zhongxu/projects/aigc_detector
unzip -o ~/attribution-phase4-fast-track-server.zip
python -m pip install -r requirements-attribution-phase4-fast-track.txt
```

Preflight:

```bash
python -u -m scripts.run_family_attribution_phase4_fast_track \
  --input "$P3_DATA" \
  --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json \
  --split-manifest "$P3_SPLIT" \
  --a0-root "$P3_ROOT/A0" \
  --a4-root "$P3_ROOT/A4" \
  --output-dir "$P3_ROOT/../family-phase4/fast-track-v1" \
  --dry-run
```

Formal run inside tmux:

```bash
tmux new-session -d -s attribution-phase4-fast-track \
  "cd /home/zhongxu/projects/aigc_detector && python -u -m scripts.run_family_attribution_phase4_fast_track --input '$P3_DATA' --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json --split-manifest '$P3_SPLIT' --a0-root '$P3_ROOT/A0' --a4-root '$P3_ROOT/A4' --output-dir '$P3_ROOT/../family-phase4/fast-track-v1' 2>&1 | tee '$P3_ROOT/../family-phase4-fast-track.log'"
```

Completion check:

```bash
jq '{status, test_evaluated}' "$P3_ROOT/../family-phase4/fast-track-v1/COMPLETED.json"
jq '{decision, paired_gains, selective}' "$P3_ROOT/../family-phase4/fast-track-v1/phase4_fast_track_summary.json"
```

Result archive:

```bash
tar -czf ~/attribution-phase4-fast-track-result.tar.gz \
  -C "$P3_ROOT/../family-phase4/fast-track-v1" \
  phase4_fast_track_summary.json \
  phase4_fast_track_report.md \
  diagnostics \
  E1-fusion \
  selective \
  artifacts \
  COMPLETED.json
```

- [ ] **Step 6: Run package and offline-help tests**

```bash
python -m pytest tests/test_attribution_phase4_fast_track_package.py tests/test_attribution_phase3_server_package.py -q
python -m scripts.run_family_attribution_phase4_fast_track --help
python -m scripts.package_attribution_phase4_fast_track --help
```

- [ ] **Step 7: Commit packaging and operations**

```bash
git add requirements-attribution-phase4-fast-track.txt scripts/package_attribution_phase4_fast_track.py docs/ATTRIBUTION_PHASE4_FAST_TRACK_RUNBOOK.md tests/test_attribution_phase4_fast_track_package.py
git commit -m "docs: ship phase4 fast track workflow"
```

---

### Task 9: Full Verification and Fast Result Gate

**Files:**
- Modify only if verification exposes a concrete failure in a Phase4 file.

**Interfaces:**
- Consumes: all completed Tasks 1–8.
- Produces: verified source package and an explicit server execution handoff.

- [ ] **Step 1: Run all Phase4 tests**

```bash
python -m pytest \
  tests/test_text_attribution_phase4_config.py \
  tests/test_text_attribution_phase4_metrics.py \
  tests/test_text_attribution_diagnostic_analysis.py \
  tests/test_text_attribution_phase4_classical.py \
  tests/test_text_attribution_probability_fusion.py \
  tests/test_text_attribution_selective_policy.py \
  tests/test_run_family_attribution_phase4_fast_track.py \
  tests/test_attribution_phase4_fast_track_package.py \
  -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run Phase2/Phase3 regression tests touched by shared contracts**

```bash
python -m pytest \
  tests/test_text_attribution_metrics_v2.py \
  tests/test_text_attribution_classical_models_v2.py \
  tests/test_text_attribution_phase3_data.py \
  tests/test_text_attribution_phase3_selection.py \
  tests/test_run_family_attribution_phase3.py \
  tests/test_attribution_phase3_server_package.py \
  -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 3: Build and verify the server ZIP twice**

```bash
python -m scripts.package_attribution_phase4_fast_track --project-root "$PWD" --output /tmp/phase4-fast-track-a.zip
python -m scripts.package_attribution_phase4_fast_track --project-root "$PWD" --output /tmp/phase4-fast-track-b.zip
shasum -a 256 /tmp/phase4-fast-track-a.zip /tmp/phase4-fast-track-b.zip
```

Expected: the two SHA-256 values are identical.

- [ ] **Step 4: Run a local synthetic end-to-end smoke**

```bash
python -m pytest tests/test_run_family_attribution_phase4_fast_track.py::test_end_to_end_fixture_writes_model_result_and_no_test_metrics -q
```

Expected: one test passes and the fixture summary contains A0, A4, E1 and E1-selective.

- [ ] **Step 5: Check source hygiene and staged scope**

```bash
git diff --check
git status --short
```

Confirm no raw data, outputs, model weights, checkpoint, cache or unrelated user change is staged.

- [ ] **Step 6: Commit verification-only corrections if any were required**

If verification required a Phase4 correction, stage only the corrected Phase4 source and its regression test, then commit:

```bash
git commit -m "fix: harden phase4 fast track verification"
```

If no correction was required, do not create an empty commit.

---

## Execution Boundary After This Plan

Fast Track 完成并在服务器得到结果后，按 `phase4_fast_track_summary.json` 只做一次决策：

- E1 晋级且选择性门槛通过：将 A4+B0+B1+融合+abstain 作为第一版可用溯源模型，下一计划补推理 API 和多文本聚合。
- E1 晋级但选择性门槛未通过：保留 E1 研究模型，下一计划优先增加校准/OOD 数据，不宣传单文本可用。
- E1 未晋级但互补 oracle 增益 ≥ 0.02：下一计划只增加一个严格对齐的第二神经分支 E2/E3。
- E1 未晋级且 oracle 增益 < 0.02：停止继续堆融合，直接做学习曲线和 V6 扩数，不再消耗时间调小阈值。

本计划不要求用户在每个内部任务后手工跑命令。实施时连续完成代码、测试和打包，用户第一次需要操作是在服务器执行 preflight 和 formal run；随后发送两个总结文件即可得到下一步结论。
