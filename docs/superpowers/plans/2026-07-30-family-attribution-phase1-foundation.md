# Family Attribution Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立正式的 10 家族文本溯源训练基础框架，完成冻结数据切分、B0–B3 传统基线、统一概率预测、版本诊断、校准指标和可视化报告，为后续神经模型与条件级联提供稳定接口。

**Architecture:** 保留现有 `baseline_*` 先导实验模块不动，在 `research/text_attribution/` 下建立正式训练模块。所有分类器统一接收 `AttributionSample`，输出对齐到固定家族标签顺序的概率矩阵；具体模型 ID 只作为元数据和诊断维度。固定的语义提示词组切分清单是训练入口的强制依赖，任何模型都不得自行重新随机切分。

**Tech Stack:** Python 3.11、scikit-learn、NumPy、SciPy、joblib、Matplotlib、Seaborn、pytest/unittest 兼容测试。

---

## 范围与后续计划

本计划只实现第一阶段，可独立产出可训练、可评测、可保存的家族分类系统：

1. V5 JSONL 与模型目录校验；
2. 冻结的 train/validation/test 语义组切分；
3. B0 字符 TF-IDF + LR；
4. B1 显式风格特征 + LR；
5. B2 字符与风格特征联合 + LR；
6. B3 词/字符 TF-IDF + 校准 Linear SVM；
7. 统一概率、校准、版本诊断、JSON/Markdown/PNG 输出。

以下内容分别另写实施计划，不塞入本阶段：

- Phase 2：XLM-R、mDeBERTa-v3、CANINE 全量微调及可选版本辅助头；
- Phase 3：概率融合、低置信度级联、完整集成上限与可选蒸馏。

## 文件结构

### 新建

```text
research/text_attribution/
├── training_data.py          # 正式样本、模型目录和数据审计契约
├── split_manifest.py         # 冻结切分、指纹和泄漏校验
├── attribution_types.py      # 统一概率预测记录
├── attribution_metrics.py    # 家族、版本诊断和校准指标
├── style_features.py         # 显式风格特征
├── classical_models.py       # B0–B3 管线、拟合和概率对齐
└── attribution_reporting.py  # 正式 JSON/Markdown/PNG 报告

scripts/
└── run_family_attribution_baselines.py

configs/text_attribution/training/
└── family_baselines.v1.json

tests/
├── text_attribution_test_utils.py
├── test_text_attribution_training_data.py
├── test_text_attribution_split_manifest.py
├── test_text_attribution_metrics_v2.py
├── test_text_attribution_style_features.py
├── test_text_attribution_classical_models_v2.py
├── test_text_attribution_reporting_v2.py
└── test_run_family_attribution_baselines.py
```

### 保留不改

```text
research/text_attribution/baseline_data.py
research/text_attribution/baseline_models.py
research/text_attribution/baseline_reporting.py
scripts/run_attribution_pilot_baselines.py
scripts/train_text.py
scripts/train_text_attribution.py
```

旧模块继续负责 923 条先导数据的 OOF 结果复现；新模块负责 V5 正式实验。这样不会破坏已有报告，也不会把人类/AIGC 二分类脚本混入家族归因主线。

---

### Task 1: 建立正式训练数据与模型目录契约

**Files:**
- Create: `tests/text_attribution_test_utils.py`
- Create: `tests/test_text_attribution_training_data.py`
- Create: `research/text_attribution/training_data.py`

- [ ] **Step 1: 编写通用测试数据构造器**

在 `tests/text_attribution_test_utils.py` 中写入：

```python
from __future__ import annotations

import json
from pathlib import Path


FAMILIES = ("claude", "openai")
MODELS = {
    "claude-haiku-test": "claude",
    "claude-sonnet-test": "claude",
    "gpt-fast-test": "openai",
    "gpt-balanced-test": "openai",
}


def write_model_catalog(path: Path) -> Path:
    rows = [
        {
            "family_id": family_id,
            "model_id": model_id,
            "variant_role": "fast" if index % 2 == 0 else "balanced",
            "enabled": True,
        }
        for index, (model_id, family_id) in enumerate(MODELS.items())
    ]
    path.write_text(
        json.dumps(
            {"schema_version": 5, "catalog_id": "test-v5", "models": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def raw_row(model_id: str, group_index: int, *, family_id: str | None = None) -> dict:
    expected_family = MODELS[model_id]
    language = "zh" if group_index % 2 == 0 else "en"
    prefix = "这是完整测试回答" if language == "zh" else "This is a complete test answer"
    return {
        "request_key": f"{model_id}-{group_index}",
        "family_id": family_id or expected_family,
        "request_model": model_id,
        "text": f"{prefix}，模型为 {model_id}，语义组为 {group_index}。"
        if language == "zh"
        else f"{prefix} from {model_id} for semantic group {group_index}.",
        "finish_reason": "stop",
        "route_mismatch": False,
        "repeat_index": 0,
        "prompt": {
            "prompt_id": f"{language}-task-{group_index:03d}",
            "language": language,
            "task_type": "explanation",
            "target_length": "medium",
        },
        "parameters": {"temperature": 0.0},
    }


def write_dataset(path: Path, *, groups: int = 12) -> Path:
    rows = [
        raw_row(model_id, group_index)
        for group_index in range(groups)
        for model_id in MODELS
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
```

- [ ] **Step 2: 编写训练数据失败测试**

在 `tests/test_text_attribution_training_data.py` 中写入：

```python
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import (
    MODELS,
    raw_row,
    write_dataset,
    write_model_catalog,
)


class TrainingDataTests(unittest.TestCase):
    def test_loads_family_target_and_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = load_training_dataset(
                write_dataset(root / "data.jsonl", groups=3),
                write_model_catalog(root / "catalog.json"),
            )

        self.assertEqual(len(dataset.samples), 12)
        self.assertEqual(dataset.family_labels, ("claude", "openai"))
        self.assertEqual(dataset.model_labels, tuple(sorted(MODELS)))
        self.assertEqual(dataset.samples[0].family_id, "claude")
        self.assertTrue(dataset.samples[0].model_id)
        self.assertEqual(dataset.audit["kept_records"], 12)

    def test_rejects_family_catalog_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = write_model_catalog(root / "catalog.json")
            row = raw_row("gpt-fast-test", 0, family_id="claude")
            data_path = root / "data.jsonl"
            data_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "catalog family mismatch"):
                load_training_dataset(data_path, catalog)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_text_attribution_training_data.py -q
```

Expected: collection fails with `ModuleNotFoundError: research.text_attribution.training_data`.

- [ ] **Step 4: 实现正式训练数据模块**

在 `research/text_attribution/training_data.py` 中实现：

```python
"""Canonical data contract for family-level text attribution training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from research.text_attribution.baseline_data import load_clean_samples


@dataclass(frozen=True)
class CatalogModel:
    family_id: str
    model_id: str
    variant_role: str


@dataclass(frozen=True)
class AttributionSample:
    request_key: str
    text: str
    family_id: str
    model_id: str
    prompt_id: str
    semantic_prompt_group: str
    language: str
    task_type: str
    target_length: str
    temperature: float
    repeat_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingDataset:
    samples: tuple[AttributionSample, ...]
    family_labels: tuple[str, ...]
    model_labels: tuple[str, ...]
    model_to_family: Mapping[str, str]
    audit: Mapping[str, Any]


def load_model_catalog(path: str | Path) -> tuple[CatalogModel, ...]:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = payload.get("models")
    if not isinstance(rows, list):
        raise ValueError(f"Model catalog must contain a models list: {catalog_path}")

    models: list[CatalogModel] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("enabled", True):
            continue
        model = CatalogModel(
            family_id=str(row.get("family_id", "")).strip(),
            model_id=str(row.get("model_id", "")).strip(),
            variant_role=str(row.get("variant_role", "unknown")).strip() or "unknown",
        )
        if not model.family_id or not model.model_id:
            raise ValueError(f"Invalid enabled model catalog row: {row!r}")
        if model.model_id in seen:
            raise ValueError(f"Duplicate model_id in catalog: {model.model_id}")
        seen.add(model.model_id)
        models.append(model)
    if not models:
        raise ValueError(f"No enabled models in catalog: {catalog_path}")
    return tuple(models)


def load_training_dataset(
    data_path: str | Path,
    catalog_path: str | Path,
) -> TrainingDataset:
    baseline_samples, audit = load_clean_samples(data_path)
    catalog = load_model_catalog(catalog_path)
    model_to_family = {row.model_id: row.family_id for row in catalog}

    samples: list[AttributionSample] = []
    for row in baseline_samples:
        if row.model not in model_to_family:
            raise ValueError(f"model_id missing from catalog: {row.model}")
        expected_family = model_to_family[row.model]
        if row.label != expected_family:
            raise ValueError(
                "catalog family mismatch for "
                f"{row.model}: data={row.label}, catalog={expected_family}"
            )
        samples.append(
            AttributionSample(
                request_key=row.request_key,
                text=row.text,
                family_id=row.label,
                model_id=row.model,
                prompt_id=row.prompt_id,
                semantic_prompt_group=row.semantic_prompt_group,
                language=row.language,
                task_type=row.task_type,
                target_length=row.target_length,
                temperature=row.temperature,
                repeat_index=row.repeat_index,
            )
        )

    present_families = tuple(sorted({sample.family_id for sample in samples}))
    present_models = tuple(sorted({sample.model_id for sample in samples}))
    enriched_audit = dict(audit)
    enriched_audit["counts_by_model"] = {
        model_id: sum(sample.model_id == model_id for sample in samples)
        for model_id in present_models
    }
    enriched_audit["catalog_models_absent_from_clean_data"] = sorted(
        set(model_to_family) - set(present_models)
    )
    return TrainingDataset(
        samples=tuple(samples),
        family_labels=present_families,
        model_labels=present_models,
        model_to_family=model_to_family,
        audit=enriched_audit,
    )
```

- [ ] **Step 5: 运行新旧数据测试**

Run:

```bash
python -m pytest tests/test_text_attribution_training_data.py tests/test_text_attribution_baseline_data.py -q
```

Expected: all tests pass.

- [ ] **Step 6: 提交**

```bash
git add research/text_attribution/training_data.py tests/text_attribution_test_utils.py tests/test_text_attribution_training_data.py
git commit -m "feat: add family attribution training data contract"
```

---

### Task 2: 冻结语义提示词组切分清单

**Files:**
- Create: `tests/test_text_attribution_split_manifest.py`
- Create: `research/text_attribution/split_manifest.py`

- [ ] **Step 1: 编写切分、持久化和泄漏测试**

在 `tests/test_text_attribution_split_manifest.py` 中写入：

```python
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from research.text_attribution.split_manifest import (
    apply_split_manifest,
    create_split_manifest,
    load_split_manifest,
    save_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


class SplitManifestTests(unittest.TestCase):
    def test_split_is_deterministic_complete_and_group_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = load_training_dataset(
                write_dataset(root / "data.jsonl", groups=20),
                write_model_catalog(root / "catalog.json"),
            )
            first = create_split_manifest(dataset.samples, seed=42)
            second = create_split_manifest(dataset.samples, seed=42)
            splits = apply_split_manifest(dataset.samples, first)

        self.assertEqual(first, second)
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertEqual(sum(map(len, splits.values())), len(dataset.samples))
        group_to_split: dict[str, set[str]] = {}
        for split_name, samples in splits.items():
            for sample in samples:
                group_to_split.setdefault(sample.semantic_prompt_group, set()).add(
                    split_name
                )
        self.assertTrue(all(len(names) == 1 for names in group_to_split.values()))
        for samples in splits.values():
            self.assertEqual(
                {sample.family_id for sample in samples},
                set(dataset.family_labels),
            )
            self.assertEqual(
                {sample.model_id for sample in samples},
                set(dataset.model_labels),
            )

    def test_saved_manifest_rejects_changed_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = load_training_dataset(
                write_dataset(root / "data.jsonl", groups=20),
                write_model_catalog(root / "catalog.json"),
            )
            path = root / "split.json"
            save_split_manifest(path, create_split_manifest(dataset.samples, seed=42))
            loaded = load_split_manifest(path)

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                apply_split_manifest(dataset.samples[:-1], loaded)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_text_attribution_split_manifest.py -q
```

Expected: collection fails because `split_manifest` does not exist.

- [ ] **Step 3: 实现切分清单**

在 `research/text_attribution/split_manifest.py` 中实现以下公开接口：

```python
"""Frozen semantic-prompt-group splits for attribution experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

from research.text_attribution.training_data import AttributionSample


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    seed: int
    ratios: Mapping[str, float]
    dataset_fingerprint: str
    group_assignments: Mapping[str, str]


def dataset_fingerprint(samples: Sequence[AttributionSample]) -> str:
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda item: item.request_key):
        digest.update(
            (
                f"{sample.request_key}\t{sample.family_id}\t{sample.model_id}\t"
                f"{sample.semantic_prompt_group}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def create_split_manifest(
    samples: Sequence[AttributionSample],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitManifest:
    ratios = {
        "train": train_ratio,
        "validation": validation_ratio,
        "test": test_ratio,
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(
        value <= 0.0 for value in ratios.values()
    ):
        raise ValueError(f"Split ratios must be positive and sum to 1: {ratios}")

    groups = sorted({sample.semantic_prompt_group for sample in samples})
    if len(groups) < 3:
        raise ValueError("At least three semantic prompt groups are required")
    random.Random(seed).shuffle(groups)
    train_count = max(1, int(len(groups) * train_ratio))
    validation_count = max(1, int(len(groups) * validation_ratio))
    if train_count + validation_count >= len(groups):
        raise ValueError("Split ratios leave no semantic group for test")

    assignments = {
        group: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, group in enumerate(groups)
    }
    manifest = SplitManifest(
        schema_version=1,
        seed=seed,
        ratios=ratios,
        dataset_fingerprint=dataset_fingerprint(samples),
        group_assignments=assignments,
    )
    apply_split_manifest(samples, manifest)
    return manifest


def apply_split_manifest(
    samples: Sequence[AttributionSample],
    manifest: SplitManifest,
) -> dict[str, tuple[AttributionSample, ...]]:
    actual_fingerprint = dataset_fingerprint(samples)
    if actual_fingerprint != manifest.dataset_fingerprint:
        raise ValueError(
            "Dataset fingerprint does not match frozen split manifest: "
            f"{actual_fingerprint} != {manifest.dataset_fingerprint}"
        )
    buckets: dict[str, list[AttributionSample]] = {
        name: [] for name in SPLIT_NAMES
    }
    for sample in samples:
        try:
            split_name = manifest.group_assignments[sample.semantic_prompt_group]
        except KeyError as exc:
            raise ValueError(
                "Semantic group missing from split manifest: "
                f"{sample.semantic_prompt_group}"
            ) from exc
        if split_name not in buckets:
            raise ValueError(f"Unsupported split name: {split_name}")
        buckets[split_name].append(sample)

    all_families = {sample.family_id for sample in samples}
    all_models = {sample.model_id for sample in samples}
    for split_name, split_samples in buckets.items():
        if not split_samples:
            raise ValueError(f"Split is empty: {split_name}")
        missing_families = all_families - {
            sample.family_id for sample in split_samples
        }
        missing_models = all_models - {sample.model_id for sample in split_samples}
        if missing_families or missing_models:
            raise ValueError(
                f"Split {split_name} lacks families={sorted(missing_families)} "
                f"models={sorted(missing_models)}"
            )
    return {name: tuple(rows) for name, rows in buckets.items()}


def save_split_manifest(path: str | Path, manifest: SplitManifest) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_split_manifest(path: str | Path) -> SplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitManifest(
        schema_version=int(payload["schema_version"]),
        seed=int(payload["seed"]),
        ratios={str(key): float(value) for key, value in payload["ratios"].items()},
        dataset_fingerprint=str(payload["dataset_fingerprint"]),
        group_assignments={
            str(key): str(value)
            for key, value in payload["group_assignments"].items()
        },
    )
```

- [ ] **Step 4: 运行测试**

Run:

```bash
python -m pytest tests/test_text_attribution_split_manifest.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: 提交**

```bash
git add research/text_attribution/split_manifest.py tests/test_text_attribution_split_manifest.py
git commit -m "feat: freeze leakage-safe attribution splits"
```

---

### Task 3: 统一概率预测与家族评价协议

**Files:**
- Create: `tests/test_text_attribution_metrics_v2.py`
- Create: `research/text_attribution/attribution_types.py`
- Create: `research/text_attribution/attribution_metrics.py`

- [ ] **Step 1: 编写概率对齐、校准和版本诊断测试**

在 `tests/test_text_attribution_metrics_v2.py` 中写入：

```python
from __future__ import annotations

import unittest

import numpy as np

from research.text_attribution.attribution_metrics import (
    build_prediction_records,
    evaluate_family_predictions,
)
from research.text_attribution.training_data import AttributionSample


def sample(index: int, family: str, model: str) -> AttributionSample:
    return AttributionSample(
        request_key=f"row-{index}",
        text=f"text {index}",
        family_id=family,
        model_id=model,
        prompt_id=f"en-task-{index}",
        semantic_prompt_group=f"task-{index}",
        language="en",
        task_type="explanation",
        target_length="medium",
        temperature=0.0,
        repeat_index=0,
    )


class AttributionMetricsTests(unittest.TestCase):
    def test_perfect_probabilities_produce_perfect_family_metrics(self) -> None:
        samples = (
            sample(0, "claude", "claude-fast"),
            sample(1, "openai", "gpt-fast"),
            sample(2, "claude", "claude-balanced"),
            sample(3, "openai", "gpt-balanced"),
        )
        probabilities = np.asarray(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
            dtype=np.float64,
        )
        records = build_prediction_records(
            samples,
            probabilities,
            family_labels=("claude", "openai"),
            experiment_id="B0",
            split="validation",
        )
        metrics = evaluate_family_predictions(
            records,
            family_labels=("claude", "openai"),
        )

        self.assertEqual(metrics["overall"]["accuracy"], 1.0)
        self.assertEqual(metrics["overall"]["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(metrics["per_model"]["claude-fast"]["support"], 1)
        self.assertLess(metrics["calibration"]["nll"], 0.3)
        self.assertEqual(metrics["subgroups"]["language"]["en"]["support"], 4)

    def test_rejects_probability_rows_that_do_not_sum_to_one(self) -> None:
        samples = (sample(0, "claude", "claude-fast"),)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            build_prediction_records(
                samples,
                np.asarray([[0.8, 0.8]]),
                family_labels=("claude", "openai"),
                experiment_id="B0",
                split="test",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_text_attribution_metrics_v2.py -q
```

Expected: import fails because the two new modules do not exist.

- [ ] **Step 3: 定义统一预测记录**

在 `research/text_attribution/attribution_types.py` 中写入：

```python
"""Shared prediction records for classical, neural and ensemble classifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FamilyPrediction:
    request_key: str
    experiment_id: str
    split: str
    true_family: str
    predicted_family: str
    family_probabilities: Mapping[str, float]
    confidence: float
    model_id: str
    prompt_id: str
    semantic_prompt_group: str
    language: str
    task_type: str
    target_length: str
    temperature: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 4: 实现评价函数**

在 `research/text_attribution/attribution_metrics.py` 中实现：

```python
"""Family-first attribution metrics with model-version diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from research.text_attribution.attribution_types import FamilyPrediction
from research.text_attribution.training_data import AttributionSample


def build_prediction_records(
    samples: Sequence[AttributionSample],
    probabilities: np.ndarray,
    *,
    family_labels: Sequence[str],
    experiment_id: str,
    split: str,
) -> tuple[FamilyPrediction, ...]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    labels = tuple(family_labels)
    if matrix.shape != (len(samples), len(labels)):
        raise ValueError(
            f"Probability shape {matrix.shape} does not match "
            f"({len(samples)}, {len(labels)})"
        )
    if np.any(matrix < 0.0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Every probability row must be non-negative and sum to one")

    records: list[FamilyPrediction] = []
    for sample, row in zip(samples, matrix, strict=True):
        best_index = int(np.argmax(row))
        probability_map = {
            label: float(row[index]) for index, label in enumerate(labels)
        }
        records.append(
            FamilyPrediction(
                request_key=sample.request_key,
                experiment_id=experiment_id,
                split=split,
                true_family=sample.family_id,
                predicted_family=labels[best_index],
                family_probabilities=probability_map,
                confidence=float(row[best_index]),
                model_id=sample.model_id,
                prompt_id=sample.prompt_id,
                semantic_prompt_group=sample.semantic_prompt_group,
                language=sample.language,
                task_type=sample.task_type,
                target_length=sample.target_length,
                temperature=sample.temperature,
            )
        )
    return tuple(records)


def evaluate_family_predictions(
    records: Sequence[FamilyPrediction],
    *,
    family_labels: Sequence[str],
    ece_bins: int = 10,
) -> dict:
    labels = tuple(family_labels)
    true = np.asarray([row.true_family for row in records])
    predicted = np.asarray([row.predicted_family for row in records])
    probabilities = np.asarray(
        [[row.family_probabilities[label] for label in labels] for row in records],
        dtype=np.float64,
    )
    true_indices = np.asarray([labels.index(value) for value in true], dtype=int)
    one_hot = np.eye(len(labels), dtype=np.float64)[true_indices]
    clipped = np.clip(probabilities, 1e-12, 1.0)

    normalized_confusion = confusion_matrix(
        true,
        predicted,
        labels=labels,
        normalize="true",
    )
    report = classification_report(
        true,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_model_rows: dict[str, list[FamilyPrediction]] = defaultdict(list)
    for row in records:
        per_model_rows[row.model_id].append(row)
    per_model = {
        model_id: {
            "support": len(rows),
            "family_accuracy": float(
                np.mean([row.true_family == row.predicted_family for row in rows])
            ),
            "mean_confidence": float(np.mean([row.confidence for row in rows])),
            "mean_true_family_probability": float(
                np.mean(
                    [
                        row.family_probabilities[row.true_family]
                        for row in rows
                    ]
                )
            ),
        }
        for model_id, rows in sorted(per_model_rows.items())
    }
    return {
        "overall": {
            "accuracy": float(accuracy_score(true, predicted)),
            "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro")),
            "weighted_f1": float(
                f1_score(true, predicted, labels=labels, average="weighted")
            ),
            "support": len(records),
        },
        "per_family": {label: report[label] for label in labels},
        "per_model": per_model,
        "subgroups": subgroup_metrics(records, family_labels=labels),
        "calibration": {
            "ece": expected_calibration_error(records, bins=ece_bins),
            "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "nll": float(-np.mean(np.log(clipped[np.arange(len(records)), true_indices]))),
        },
        "confusion_matrix": normalized_confusion.tolist(),
    }


def expected_calibration_error(
    records: Sequence[FamilyPrediction],
    *,
    bins: int = 10,
) -> float:
    confidences = np.asarray([row.confidence for row in records], dtype=np.float64)
    correct = np.asarray(
        [row.true_family == row.predicted_family for row in records],
        dtype=np.float64,
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (confidences >= lower) & (confidences <= upper)
            if index == bins - 1
            else (confidences >= lower) & (confidences < upper)
        )
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidences[mask]))
            )
    return error


def subgroup_metrics(
    records: Sequence[FamilyPrediction],
    *,
    family_labels: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    dimensions = {
        "language": lambda row: row.language,
        "target_length": lambda row: row.target_length,
        "task_type": lambda row: row.task_type,
        "temperature": lambda row: str(row.temperature),
    }
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for dimension, getter in dimensions.items():
        values = sorted({getter(row) for row in records})
        output[dimension] = {}
        for value in values:
            subset = [row for row in records if getter(row) == value]
            true = [row.true_family for row in subset]
            predicted = [row.predicted_family for row in subset]
            output[dimension][value] = {
                "support": len(subset),
                "accuracy": float(accuracy_score(true, predicted)),
                "macro_f1": float(
                    f1_score(
                        true,
                        predicted,
                        labels=tuple(family_labels),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
    return output
```

- [ ] **Step 5: 运行评价测试**

Run:

```bash
python -m pytest tests/test_text_attribution_metrics_v2.py -q
```

Expected: 2 tests pass.

- [ ] **Step 6: 提交**

```bash
git add research/text_attribution/attribution_types.py research/text_attribution/attribution_metrics.py tests/test_text_attribution_metrics_v2.py
git commit -m "feat: add family attribution probability metrics"
```

---

### Task 4: 实现显式风格特征

**Files:**
- Create: `tests/test_text_attribution_style_features.py`
- Create: `research/text_attribution/style_features.py`

- [ ] **Step 1: 编写特征稳定性与长度消融测试**

在 `tests/test_text_attribution_style_features.py` 中写入：

```python
from __future__ import annotations

import unittest

import numpy as np

from research.text_attribution.style_features import StylometricFeatureExtractor


class StylometricFeatureTests(unittest.TestCase):
    def test_extracts_fixed_finite_feature_matrix(self) -> None:
        texts = [
            "# 标题\n\n首先，这是第一点。\n- 项目甲\n- 项目乙\n\n总之，回答结束。",
            "However, this answer is concise. Therefore, it ends here!",
        ]
        extractor = StylometricFeatureExtractor(include_length_features=True)
        matrix = extractor.fit_transform(texts)

        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(matrix.shape[1], len(extractor.get_feature_names_out()))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertGreater(matrix[0, list(extractor.get_feature_names_out()).index("markdown_heading_ratio")], 0.0)

    def test_length_ablation_removes_only_length_features(self) -> None:
        with_length = StylometricFeatureExtractor(include_length_features=True)
        without_length = StylometricFeatureExtractor(include_length_features=False)
        with_names = set(with_length.get_feature_names_out())
        without_names = set(without_length.get_feature_names_out())

        self.assertIn("char_count_log1p", with_names)
        self.assertNotIn("char_count_log1p", without_names)
        self.assertIn("comma_per_1000", without_names)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_text_attribution_style_features.py -q
```

Expected: import fails because `style_features` does not exist.

- [ ] **Step 3: 实现 sklearn 兼容特征提取器**

在 `research/text_attribution/style_features.py` 中实现：

```python
"""Interpretable multilingual style features for generated text."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")
_TOKEN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE = re.compile(r"[^。！？!?]+[。！？!?]?")
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*(?:\d+[.)、]|[一二三四五六七八九十]+、)\s*", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_INDENTED = re.compile(r"^(?: {4}|\t)", re.MULTILINE)

_LENGTH_NAMES = (
    "char_count_log1p",
    "word_count_log1p",
    "sentence_count_log1p",
    "paragraph_count_log1p",
    "mean_sentence_chars_log1p",
    "std_sentence_chars_log1p",
    "mean_paragraph_chars_log1p",
)
_STYLE_NAMES = (
    "cjk_ratio",
    "latin_ratio",
    "digit_ratio",
    "whitespace_ratio",
    "newline_ratio",
    "comma_per_1000",
    "period_per_1000",
    "colon_per_1000",
    "semicolon_per_1000",
    "question_per_1000",
    "exclamation_per_1000",
    "quote_per_1000",
    "parenthesis_per_1000",
    "dash_per_1000",
    "markdown_heading_ratio",
    "bullet_line_ratio",
    "numbered_line_ratio",
    "indented_line_ratio",
    "bold_marker_per_1000",
    "code_fence_per_1000",
    "type_token_ratio",
    "repeated_bigram_ratio",
    "zh_de_ratio",
    "zh_shi_ratio",
    "zh_because_ratio",
    "zh_however_ratio",
    "en_the_ratio",
    "en_and_ratio",
    "en_because_ratio",
    "en_however_ratio",
    "starts_with_transition",
    "ends_with_summary",
)


class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, *, include_length_features: bool = True):
        self.include_length_features = include_length_features

    def fit(self, texts: Sequence[str], y=None):
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._row(str(text)) for text in texts], dtype=np.float64)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = (*_LENGTH_NAMES, *_STYLE_NAMES) if self.include_length_features else _STYLE_NAMES
        return np.asarray(names, dtype=object)

    def _row(self, text: str) -> list[float]:
        chars = max(len(text), 1)
        visible = max(sum(not char.isspace() for char in text), 1)
        lines = text.splitlines() or [text]
        paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()] or [text]
        sentences = [part.strip() for part in _SENTENCE.findall(text) if part.strip()] or [text]
        tokens = [token.lower() for token in _TOKEN.findall(text)]
        bigrams = list(zip(tokens, tokens[1:]))
        repeated_bigrams = sum(count - 1 for count in Counter(bigrams).values() if count > 1)
        per_1000 = 1000.0 / visible

        length_values = [
            np.log1p(len(text)),
            np.log1p(len(tokens)),
            np.log1p(len(sentences)),
            np.log1p(len(paragraphs)),
            np.log1p(np.mean([len(value) for value in sentences])),
            np.log1p(np.std([len(value) for value in sentences])),
            np.log1p(np.mean([len(value) for value in paragraphs])),
        ]
        lower = text.lower().strip()
        style_values = [
            len(_CJK.findall(text)) / visible,
            len(_LATIN.findall(text)) / visible,
            len(_DIGIT.findall(text)) / visible,
            sum(char.isspace() for char in text) / chars,
            text.count("\n") / chars,
            sum(text.count(mark) for mark in (",", "，")) * per_1000,
            sum(text.count(mark) for mark in (".", "。")) * per_1000,
            sum(text.count(mark) for mark in (":", "：")) * per_1000,
            sum(text.count(mark) for mark in (";", "；")) * per_1000,
            sum(text.count(mark) for mark in ("?", "？")) * per_1000,
            sum(text.count(mark) for mark in ("!", "！")) * per_1000,
            sum(text.count(mark) for mark in ('"', "'", "“", "”", "‘", "’")) * per_1000,
            sum(text.count(mark) for mark in ("(", ")", "（", "）")) * per_1000,
            sum(text.count(mark) for mark in ("-", "—", "–")) * per_1000,
            len(_HEADING.findall(text)) / max(len(lines), 1),
            len(_BULLET.findall(text)) / max(len(lines), 1),
            len(_NUMBERED.findall(text)) / max(len(lines), 1),
            len(_INDENTED.findall(text)) / max(len(lines), 1),
            text.count("**") * per_1000,
            text.count("```") * per_1000,
            len(set(tokens)) / max(len(tokens), 1),
            repeated_bigrams / max(len(bigrams), 1),
            text.count("的") / visible,
            text.count("是") / visible,
            text.count("因为") / visible,
            text.count("但是") / visible,
            len(re.findall(r"\bthe\b", lower)) / max(len(tokens), 1),
            len(re.findall(r"\band\b", lower)) / max(len(tokens), 1),
            len(re.findall(r"\bbecause\b", lower)) / max(len(tokens), 1),
            len(re.findall(r"\bhowever\b", lower)) / max(len(tokens), 1),
            float(lower.startswith(("首先", "以下", "first", "to begin"))),
            float(lower.endswith(("总之。", "综上。", "in summary.", "overall."))),
        ]
        return [*length_values, *style_values] if self.include_length_features else style_values
```

- [ ] **Step 4: 运行风格特征测试**

Run:

```bash
python -m pytest tests/test_text_attribution_style_features.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: 提交**

```bash
git add research/text_attribution/style_features.py tests/test_text_attribution_style_features.py
git commit -m "feat: add explicit attribution style features"
```

---

### Task 5: 实现 B0–B3 传统分类管线

**Files:**
- Create: `tests/test_text_attribution_classical_models_v2.py`
- Create: `research/text_attribution/classical_models.py`

- [ ] **Step 1: 编写四条基线的概率接口测试**

在 `tests/test_text_attribution_classical_models_v2.py` 中写入：

```python
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.text_attribution.classical_models import (
    build_classical_pipeline,
    predict_family_probabilities,
)
from research.text_attribution.split_manifest import (
    apply_split_manifest,
    create_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


class ClassicalModelTests(unittest.TestCase):
    def test_all_baselines_return_aligned_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = load_training_dataset(
                write_dataset(root / "data.jsonl", groups=20),
                write_model_catalog(root / "catalog.json"),
            )
            splits = apply_split_manifest(
                dataset.samples,
                create_split_manifest(dataset.samples, seed=42),
            )
            train_texts = [sample.text for sample in splits["train"]]
            train_labels = [sample.family_id for sample in splits["train"]]
            validation_texts = [sample.text for sample in splits["validation"]]

            for experiment_id in ("B0", "B1", "B2", "B3"):
                with self.subTest(experiment_id=experiment_id):
                    model = build_classical_pipeline(experiment_id, seed=42)
                    model.fit(train_texts, train_labels)
                    probabilities = predict_family_probabilities(
                        model,
                        validation_texts,
                        family_labels=dataset.family_labels,
                    )
                    self.assertEqual(
                        probabilities.shape,
                        (len(validation_texts), len(dataset.family_labels)),
                    )
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_unknown_experiment_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported classical experiment"):
            build_classical_pipeline("B99", seed=42)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_text_attribution_classical_models_v2.py -q
```

Expected: import fails because `classical_models` does not exist.

- [ ] **Step 3: 实现四条 sklearn 管线**

在 `research/text_attribution/classical_models.py` 中实现：

```python
"""Classical family-attribution baselines with one probability contract."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from research.text_attribution.style_features import StylometricFeatureExtractor


EXPERIMENT_IDS = ("B0", "B1", "B2", "B3")


def _logistic_regression(seed: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=250_000,
        dtype=np.float32,
    )


def build_classical_pipeline(experiment_id: str, *, seed: int) -> Pipeline:
    if experiment_id == "B0":
        return Pipeline(
            [("char_tfidf", _char_vectorizer()), ("classifier", _logistic_regression(seed))]
        )
    if experiment_id == "B1":
        return Pipeline(
            [
                ("style", StylometricFeatureExtractor(include_length_features=True)),
                ("scale", StandardScaler()),
                ("classifier", _logistic_regression(seed)),
            ]
        )
    if experiment_id == "B2":
        features = FeatureUnion(
            [
                ("char", _char_vectorizer()),
                (
                    "style",
                    Pipeline(
                        [
                            (
                                "extract",
                                StylometricFeatureExtractor(
                                    include_length_features=True
                                ),
                            ),
                            ("scale", StandardScaler()),
                        ]
                    ),
                ),
            ]
        )
        return Pipeline(
            [("features", features), ("classifier", _logistic_regression(seed))]
        )
    if experiment_id == "B3":
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.995,
                        sublinear_tf=True,
                        max_features=150_000,
                        dtype=np.float32,
                    ),
                ),
                ("char", _char_vectorizer()),
            ]
        )
        calibrated_svm = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=seed),
            method="sigmoid",
            cv=3,
        )
        return Pipeline(
            [("features", features), ("classifier", calibrated_svm)]
        )
    raise ValueError(f"Unsupported classical experiment: {experiment_id}")


def predict_family_probabilities(
    model: Pipeline,
    texts: Sequence[str],
    *,
    family_labels: Sequence[str],
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(list(texts)), dtype=np.float64)
    model_labels = tuple(str(value) for value in model.classes_)
    target_labels = tuple(family_labels)
    missing = set(target_labels) - set(model_labels)
    if missing:
        raise ValueError(f"Trained model lacks family labels: {sorted(missing)}")
    aligned = np.zeros((len(texts), len(target_labels)), dtype=np.float64)
    for target_index, label in enumerate(target_labels):
        aligned[:, target_index] = raw[:, model_labels.index(label)]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned
```

- [ ] **Step 4: 运行传统模型测试**

Run:

```bash
python -m pytest tests/test_text_attribution_classical_models_v2.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: 运行先导基线回归测试**

Run:

```bash
python -m pytest tests/test_text_attribution_baseline_models.py -q
```

Expected: existing pilot baseline tests still pass.

- [ ] **Step 6: 提交**

```bash
git add research/text_attribution/classical_models.py tests/test_text_attribution_classical_models_v2.py
git commit -m "feat: add calibrated family attribution baselines"
```

---

### Task 6: 输出正式报告与模型产物

**Files:**
- Create: `tests/test_text_attribution_reporting_v2.py`
- Create: `research/text_attribution/attribution_reporting.py`

- [ ] **Step 1: 编写报告产物测试**

在 `tests/test_text_attribution_reporting_v2.py` 中写入：

```python
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.text_attribution.attribution_metrics import (
    build_prediction_records,
    evaluate_family_predictions,
)
from research.text_attribution.attribution_reporting import write_attribution_report
from research.text_attribution.training_data import AttributionSample


class AttributionReportingTests(unittest.TestCase):
    def test_writes_machine_readable_and_visual_artifacts(self) -> None:
        samples = tuple(
            AttributionSample(
                request_key=f"row-{index}",
                text=f"text {index}",
                family_id=family,
                model_id=f"{family}-model",
                prompt_id=f"en-task-{index}",
                semantic_prompt_group=f"task-{index}",
                language="en",
                task_type="explanation",
                target_length="medium",
                temperature=0.0,
                repeat_index=0,
            )
            for index, family in enumerate(("claude", "openai", "claude", "openai"))
        )
        probabilities = np.asarray(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
        )
        records = build_prediction_records(
            samples,
            probabilities,
            family_labels=("claude", "openai"),
            experiment_id="B0",
            split="validation",
        )
        metrics = {
            "B0": {
                "validation": evaluate_family_predictions(
                    records,
                    family_labels=("claude", "openai"),
                )
            }
        }

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "report"
            artifacts = write_attribution_report(
                output_dir=output_dir,
                run_config={"seed": 42, "experiments": ["B0"]},
                dataset_audit={"kept_records": 4},
                split_summary={"train": 0, "validation": 4, "test": 0},
                metrics=metrics,
                predictions=records,
            )
            for name in (
                "run_config",
                "dataset_audit",
                "metrics",
                "predictions",
                "report",
                "overall_metrics_png",
                "confusion_matrices_png",
                "per_model_diagnostics_png",
                "calibration_png",
                "subgroup_metrics_png",
            ):
                self.assertTrue(Path(artifacts[name]).is_file(), name)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_text_attribution_reporting_v2.py -q
```

Expected: import fails because `attribution_reporting` does not exist.

- [ ] **Step 3: 实现正式报告入口**

在 `research/text_attribution/attribution_reporting.py` 中定义以下稳定入口：

```python
"""Durable report artifacts for family-attribution experiments."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from research.text_attribution.attribution_types import FamilyPrediction


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_attribution_report(
    *,
    output_dir: str | Path,
    run_config: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    split_summary: Mapping[str, int],
    metrics: Mapping[str, Any],
    predictions: Sequence[FamilyPrediction],
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_config": root / "run_config.json",
        "dataset_audit": root / "dataset_audit.json",
        "metrics": root / "metrics.json",
        "predictions": root / "predictions.jsonl",
        "report": root / "report.md",
        "overall_metrics_png": root / "overall_metrics.png",
        "confusion_matrices_png": root / "confusion_matrices.png",
        "per_model_diagnostics_png": root / "per_model_diagnostics.png",
        "calibration_png": root / "calibration.png",
        "subgroup_metrics_png": root / "subgroup_metrics.png",
    }
    _write_json(paths["run_config"], dict(run_config))
    _write_json(paths["dataset_audit"], dict(dataset_audit))
    _write_json(paths["metrics"], dict(metrics))
    paths["predictions"].write_text(
        "".join(
            json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for row in predictions
        ),
        encoding="utf-8",
    )
    paths["report"].write_text(
        _markdown_report(run_config, dataset_audit, split_summary, metrics),
        encoding="utf-8",
    )
    _plot_overall(metrics, paths["overall_metrics_png"])
    _plot_confusions(metrics, paths["confusion_matrices_png"])
    _plot_per_model(metrics, paths["per_model_diagnostics_png"])
    _plot_calibration(metrics, paths["calibration_png"])
    _plot_subgroups(metrics, paths["subgroup_metrics_png"])
    return {name: str(path) for name, path in paths.items()}
```

同一文件继续实现四个私有绘图函数，必须遵循以下数据选择规则：

```python
def _metric_rows(metrics: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    return [
        (experiment_id, split_name, split_metrics)
        for experiment_id, experiment_metrics in metrics.items()
        if experiment_id != "selection"
        for split_name, split_metrics in experiment_metrics.items()
    ]


def _plot_overall(metrics: Mapping[str, Any], path: Path) -> None:
    rows = _metric_rows(metrics)
    labels = [f"{experiment}\n{split}" for experiment, split, _ in rows]
    values = [row["overall"]["macro_f1"] for _, _, row in rows]
    fig, axis = plt.subplots(figsize=(max(7, len(rows) * 1.2), 4.5))
    axis.bar(labels, values)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Family Macro-F1")
    axis.set_title("Family attribution performance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confusions(metrics: Mapping[str, Any], path: Path) -> None:
    rows = _metric_rows(metrics)
    columns = max(1, len(rows))
    fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 4), squeeze=False)
    for axis, (experiment, split, row) in zip(axes[0], rows, strict=True):
        labels = list(row["per_family"])
        sns.heatmap(
            np.asarray(row["confusion_matrix"]),
            annot=True,
            fmt=".2f",
            vmin=0.0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=axis,
        )
        axis.set_title(f"{experiment} / {split}")
        axis.set_xlabel("Predicted family")
        axis.set_ylabel("True family")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_per_model(metrics: Mapping[str, Any], path: Path) -> None:
    rows = _metric_rows(metrics)
    experiment, split, selected = rows[0]
    models = list(selected["per_model"])
    values = [selected["per_model"][model]["family_accuracy"] for model in models]
    fig, axis = plt.subplots(figsize=(max(7, len(models) * 0.7), 4.5))
    axis.bar(models, values)
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(axis="x", rotation=60)
    axis.set_ylabel("Family accuracy")
    axis.set_title(f"Per-model family diagnostics: {experiment} / {split}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_calibration(metrics: Mapping[str, Any], path: Path) -> None:
    rows = _metric_rows(metrics)
    labels = [f"{experiment}\n{split}" for experiment, split, _ in rows]
    ece = [row["calibration"]["ece"] for _, _, row in rows]
    fig, axis = plt.subplots(figsize=(max(7, len(rows) * 1.2), 4.5))
    axis.bar(labels, ece)
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("Expected Calibration Error")
    axis.set_title("Probability calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subgroups(metrics: Mapping[str, Any], path: Path) -> None:
    rows = _metric_rows(metrics)
    experiment, split, selected = rows[0]
    dimensions = list(selected["subgroups"])
    fig, axes = plt.subplots(
        1,
        len(dimensions),
        figsize=(5 * len(dimensions), 4),
        squeeze=False,
    )
    for axis, dimension in zip(axes[0], dimensions, strict=True):
        groups = selected["subgroups"][dimension]
        names = list(groups)
        values = [groups[name]["macro_f1"] for name in names]
        axis.bar(names, values)
        axis.set_ylim(0.0, 1.0)
        axis.tick_params(axis="x", rotation=45)
        axis.set_title(dimension)
        axis.set_ylabel("Family Macro-F1")
    fig.suptitle(f"Subgroup diagnostics: {experiment} / {split}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_report(
    run_config: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    split_summary: Mapping[str, int],
    metrics: Mapping[str, Any],
) -> str:
    lines = [
        "# Family Attribution Baseline Report",
        "",
        "## Run",
        "",
        f"- Seed: `{run_config.get('seed')}`",
        f"- Experiments: `{', '.join(run_config.get('experiments', []))}`",
        f"- Clean samples: `{dataset_audit.get('kept_records', 0)}`",
        "",
        "## Frozen split",
        "",
        "| Split | Samples |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {split_name} | {count} |"
        for split_name, count in split_summary.items()
    )
    lines.extend(
        [
            "",
            "## Family-level metrics",
            "",
            "| Experiment | Split | Accuracy | Macro-F1 | Weighted-F1 | ECE | Brier | NLL |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for experiment, split, row in _metric_rows(metrics):
        overall = row["overall"]
        calibration = row["calibration"]
        lines.append(
            f"| {experiment} | {split} | {overall['accuracy']:.4f} | "
            f"{overall['macro_f1']:.4f} | {overall['weighted_f1']:.4f} | "
            f"{calibration['ece']:.4f} | {calibration['brier']:.4f} | "
            f"{calibration['nll']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-version family diagnostics",
            "",
            "The rows below measure family prediction performance after grouping by the "
            "true generating model ID. They are not claims of exact-version attribution.",
            "",
            "| Experiment | Split | True model ID | Support | Family accuracy | Mean true-family probability |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for experiment, split, row in _metric_rows(metrics):
        for model_id, diagnostic in row["per_model"].items():
            lines.append(
                f"| {experiment} | {split} | {model_id} | "
                f"{diagnostic['support']} | {diagnostic['family_accuracy']:.4f} | "
                f"{diagnostic['mean_true_family_probability']:.4f} |"
            )
    return "\n".join(lines) + "\n"
```

Markdown 报告中的逐版本表只描述“按真实版本分组后的家族识别效果”，不得把它描述成最终具体版本识别能力。

- [ ] **Step 4: 运行报告测试**

Run:

```bash
python -m pytest tests/test_text_attribution_reporting_v2.py -q
```

Expected: 1 test passes and all listed artifacts exist.

- [ ] **Step 5: 提交**

```bash
git add research/text_attribution/attribution_reporting.py tests/test_text_attribution_reporting_v2.py
git commit -m "feat: report family attribution baseline results"
```

---

### Task 7: 实现正式命令行与实验配置

**Files:**
- Create: `configs/text_attribution/training/family_baselines.v1.json`
- Create: `tests/test_run_family_attribution_baselines.py`
- Create: `scripts/run_family_attribution_baselines.py`

- [ ] **Step 1: 写入可版本化配置**

创建 `configs/text_attribution/training/family_baselines.v1.json`：

```json
{
  "schema_version": 1,
  "run_id": "family-attribution-baselines-v1",
  "experiments": ["B0", "B1", "B2", "B3"],
  "seed": 42,
  "split": {
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    "test_ratio": 0.15
  },
  "model_catalog": "../zhizengzeng_model_catalog.v5.json"
}
```

- [ ] **Step 2: 编写 dry-run 与端到端测试**

在 `tests/test_run_family_attribution_baselines.py` 中写入：

```python
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_family_attribution_baselines import main
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


class FamilyAttributionBaselineCliTests(unittest.TestCase):
    def test_dry_run_validates_without_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = write_dataset(root / "data.jsonl", groups=20)
            catalog_path = write_model_catalog(root / "catalog.json")
            output_dir = root / "output"
            exit_code = main(
                [
                    "--input",
                    str(data_path),
                    "--catalog",
                    str(catalog_path),
                    "--split-manifest",
                    str(root / "split.json"),
                    "--create-split",
                    "--experiments",
                    "B0",
                    "--evaluate-selected-test",
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertFalse(output_dir.exists())

    def test_runs_b0_and_writes_reusable_model_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            split_path = root / "split.json"
            exit_code = main(
                [
                    "--input",
                    str(write_dataset(root / "data.jsonl", groups=20)),
                    "--catalog",
                    str(write_model_catalog(root / "catalog.json")),
                    "--split-manifest",
                    str(split_path),
                    "--create-split",
                    "--experiments",
                    "B0",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(split_path.is_file())
            self.assertTrue((output_dir / "models" / "B0.joblib").is_file())
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(set(metrics["B0"]), {"validation", "test"})
            self.assertEqual(metrics["selection"]["selected_experiment"], "B0")
            self.assertTrue((output_dir / "predictions.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_run_family_attribution_baselines.py -q
```

Expected: import fails because the new script does not exist.

- [ ] **Step 4: 实现命令行入口**

在 `scripts/run_family_attribution_baselines.py` 中实现以下流程：

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib

from research.text_attribution.attribution_metrics import (
    build_prediction_records,
    evaluate_family_predictions,
)
from research.text_attribution.attribution_reporting import write_attribution_report
from research.text_attribution.classical_models import (
    EXPERIMENT_IDS,
    build_classical_pipeline,
    predict_family_probabilities,
)
from research.text_attribution.split_manifest import (
    apply_split_manifest,
    create_split_manifest,
    load_split_manifest,
    save_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe family attribution baselines."
    )
    parser.add_argument(
        "--config",
        default="configs/text_attribution/training/family_baselines.v1.json",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENT_IDS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate-selected-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    catalog_path = (
        Path(args.catalog)
        if args.catalog
        else (config_path.parent / str(config["model_catalog"])).resolve()
    )
    experiments = tuple(args.experiments or config["experiments"])
    seed = args.seed if args.seed is not None else int(config["seed"])
    split_config = config["split"]
    dataset = load_training_dataset(args.input, catalog_path)
    split_path = Path(args.split_manifest)
    if args.create_split:
        if split_path.exists():
            manifest = load_split_manifest(split_path)
        else:
            manifest = create_split_manifest(
                dataset.samples,
                seed=seed,
                train_ratio=float(split_config["train_ratio"]),
                validation_ratio=float(split_config["validation_ratio"]),
                test_ratio=float(split_config["test_ratio"]),
            )
            if not args.dry_run:
                save_split_manifest(split_path, manifest)
    else:
        if not split_path.is_file():
            raise FileNotFoundError(
                "Frozen split manifest is required; pass --create-split once "
                f"to create it: {split_path}"
            )
        manifest = load_split_manifest(split_path)
    splits = apply_split_manifest(dataset.samples, manifest)

    summary = {
        "input": str(args.input),
        "clean_samples": len(dataset.samples),
        "families": list(dataset.family_labels),
        "models": list(dataset.model_labels),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "experiments": list(experiments),
        "seed": seed,
        "config": str(config_path),
        "catalog": str(catalog_path),
        "evaluate_selected_test": args.evaluate_selected_test,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_samples = splits["train"]
    train_texts = [sample.text for sample in train_samples]
    train_labels = [sample.family_id for sample in train_samples]
    all_records = []
    all_metrics = {}
    trained_models = {}

    for experiment_id in experiments:
        model = build_classical_pipeline(experiment_id, seed=seed)
        model.fit(train_texts, train_labels)
        joblib.dump(model, model_dir / f"{experiment_id}.joblib")
        trained_models[experiment_id] = model
        validation_samples = splits["validation"]
        probabilities = predict_family_probabilities(
            model,
            [sample.text for sample in validation_samples],
            family_labels=dataset.family_labels,
        )
        records = build_prediction_records(
            validation_samples,
            probabilities,
            family_labels=dataset.family_labels,
            experiment_id=experiment_id,
            split="validation",
        )
        all_records.extend(records)
        all_metrics[experiment_id] = {
            "validation": evaluate_family_predictions(
                records,
                family_labels=dataset.family_labels,
            )
        }

    def selection_key(experiment_id: str) -> tuple[float, float, str]:
        validation = all_metrics[experiment_id]["validation"]
        worst_family_f1 = min(
            float(row["f1-score"])
            for row in validation["per_family"].values()
        )
        return (
            float(validation["overall"]["macro_f1"]),
            worst_family_f1,
            experiment_id,
        )

    selected_experiment = max(experiments, key=selection_key)
    all_metrics["selection"] = {
        "selected_experiment": selected_experiment,
        "criterion": "validation_macro_f1_then_worst_family_f1",
    }
    summary["selected_experiment"] = selected_experiment

    if args.evaluate_selected_test:
        test_samples = splits["test"]
        test_probabilities = predict_family_probabilities(
            trained_models[selected_experiment],
            [sample.text for sample in test_samples],
            family_labels=dataset.family_labels,
        )
        test_records = build_prediction_records(
            test_samples,
            test_probabilities,
            family_labels=dataset.family_labels,
            experiment_id=selected_experiment,
            split="test",
        )
        all_records.extend(test_records)
        all_metrics[selected_experiment]["test"] = evaluate_family_predictions(
            test_records,
            family_labels=dataset.family_labels,
        )

    write_attribution_report(
        output_dir=output_dir,
        run_config=summary,
        dataset_audit=dataset.audit,
        split_summary={name: len(rows) for name, rows in splits.items()},
        metrics=all_metrics,
        predictions=all_records,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行 CLI 测试**

Run:

```bash
python -m pytest tests/test_run_family_attribution_baselines.py -q
```

Expected: 2 tests pass.

- [ ] **Step 6: 运行第一阶段完整测试**

Run:

```bash
python -m pytest \
  tests/test_text_attribution_training_data.py \
  tests/test_text_attribution_split_manifest.py \
  tests/test_text_attribution_metrics_v2.py \
  tests/test_text_attribution_style_features.py \
  tests/test_text_attribution_classical_models_v2.py \
  tests/test_text_attribution_reporting_v2.py \
  tests/test_run_family_attribution_baselines.py \
  tests/test_text_attribution_baseline_data.py \
  tests/test_text_attribution_baseline_models.py \
  tests/test_text_attribution_baseline_reporting.py \
  -q
```

Expected: all selected tests pass with zero failures.

- [ ] **Step 7: 提交**

```bash
git add configs/text_attribution/training/family_baselines.v1.json scripts/run_family_attribution_baselines.py tests/test_run_family_attribution_baselines.py
git commit -m "feat: run family attribution baseline experiments"
```

---

### Task 8: 在真实 V5 数据上做服务器 dry-run、验证集选型与一次性测试

**Files:**
- Modify only if needed after dry-run evidence: `configs/text_attribution/training/family_baselines.v1.json`
- Produce: `data/attribution/splits/zhizengzeng-attribution-v5-seed42.json`
- Produce: `outputs/attribution/family-baselines-v1/`

- [ ] **Step 1: 首次只做数据和切分验证**

在服务器项目根目录运行：

```bash
python -u -m scripts.run_family_attribution_baselines \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json \
  --split-manifest data/attribution/splits/zhizengzeng-attribution-v5-seed42.json \
  --create-split \
  --experiments B0 B1 B2 B3 \
  --output-dir outputs/attribution/family-baselines-v1 \
  --seed 42 \
  --dry-run
```

Expected:

- 输出干净样本数、10 个家族、实际存在的具体模型和三份切分数量；
- 不创建输出目录；
- 如某一切分缺少家族或具体模型，命令明确失败，不静默继续。

- [ ] **Step 2: 创建一次并冻结切分清单**

移除 `--dry-run`，先只运行 B0：

```bash
python -u -m scripts.run_family_attribution_baselines \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json \
  --split-manifest data/attribution/splits/zhizengzeng-attribution-v5-seed42.json \
  --create-split \
  --experiments B0 \
  --output-dir outputs/attribution/family-baselines-v1-smoke \
  --seed 42
```

Expected:

- 写出冻结切分 JSON；
- 写出 `models/B0.joblib`；
- `metrics.json` 只包含 validation，不提前读取真实测试集；
- `predictions.jsonl` 每条记录包含 10 家族概率与真实 `model_id`；
- 五张 PNG 均能打开。

- [ ] **Step 3: 检查冻结清单后运行 B0–B3**

正式运行时不再传 `--create-split`：

```bash
python -u -m scripts.run_family_attribution_baselines \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json \
  --split-manifest data/attribution/splits/zhizengzeng-attribution-v5-seed42.json \
  --experiments B0 B1 B2 B3 \
  --output-dir outputs/attribution/family-baselines-v1 \
  --seed 42
```

Expected output directory:

```text
outputs/attribution/family-baselines-v1/
├── models/
│   ├── B0.joblib
│   ├── B1.joblib
│   ├── B2.joblib
│   └── B3.joblib
├── run_config.json
├── dataset_audit.json
├── metrics.json
├── predictions.jsonl
├── report.md
├── overall_metrics.png
├── confusion_matrices.png
├── per_model_diagnostics.png
├── calibration.png
└── subgroup_metrics.png
```

- [ ] **Step 4: 只根据 validation 冻结胜出实验**

运行：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/attribution/family-baselines-v1")
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
assert set(metrics) == {"B0", "B1", "B2", "B3", "selection"}
selection = metrics["selection"]
assert selection["selected_experiment"] in {"B0", "B1", "B2", "B3"}
assert all("test" not in metrics[key] for key in ("B0", "B1", "B2", "B3"))
print(selection)
PY
```

Expected: 打印验证集选出的唯一实验，且四个基线都没有 test 指标。此时冻结选择，不再依据测试集调整特征、超参数或选择规则。

- [ ] **Step 5: 最终只对验证集胜出模型开启一次测试**

使用相同配置、相同切分和相同随机种子运行：

```bash
python -u -m scripts.run_family_attribution_baselines \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --catalog configs/text_attribution/zhizengzeng_model_catalog.v5.json \
  --split-manifest data/attribution/splits/zhizengzeng-attribution-v5-seed42.json \
  --experiments B0 B1 B2 B3 \
  --output-dir outputs/attribution/family-baselines-v1-final \
  --seed 42 \
  --evaluate-selected-test
```

Expected: 四个实验都有 validation，只有自动选中的胜出实验包含 test。

- [ ] **Step 6: 验证最终结果契约**

运行：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/attribution/family-baselines-v1-final")
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
assert set(metrics) == {"B0", "B1", "B2", "B3", "selection"}
winner = metrics["selection"]["selected_experiment"]
assert set(metrics[winner]) == {"validation", "test"}
assert all(
    "test" not in metrics[key]
    for key in ("B0", "B1", "B2", "B3")
    if key != winner
)
rows = [
    json.loads(line)
    for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
]
assert rows
assert all(len(row["family_probabilities"]) == 10 for row in rows)
assert all(row["model_id"] for row in rows)
assert all(
    abs(sum(row["family_probabilities"].values()) - 1.0) < 1e-6
    for row in rows
)
print({"winner": winner, "predictions": len(rows)})
PY
```

Expected: assertions pass and print validation-selected winner plus prediction row count。

- [ ] **Step 7: 保存运行证据**

```bash
git status --short
```

Expected: 代码和配置保持干净；数据切分与输出目录按项目数据产物策略保留，不提交大型模型和结果文件。

---

## 第一阶段完成标准

只有同时满足以下条件，才能进入 XLM-R/mDeBERTa/CANINE 计划：

- 冻结切分清单能够复用，并在数据变化时因指纹不一致而失败；
- B0–B3 使用完全相同的 train/validation/test；
- 每条 prediction 包含固定顺序的 10 家族概率；
- 具体版本只用于诊断，正式主指标始终是 Family Macro-F1；
- validation 和 test 指标严格分开；
- B0–B3 默认只查看 validation，最终 test 只对冻结的胜出基线开启一次；
- 输出 Accuracy、Macro-F1、Weighted-F1、ECE、Brier、NLL；
- 输出逐家族、逐具体版本、混淆矩阵和图表；
- 输出语言、长度、任务类型和温度子组指标；
- B0–B3 模型文件能够重新加载并执行 `predict_proba`；
- 新增测试和旧先导基线回归测试全部通过。
