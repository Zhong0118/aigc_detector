# Multilingual Text Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Chinese/English, multi-domain false-positive calibration with held-out human FPR and AI recall evaluation while continuing to use the existing Yuchuan detector weights.

**Architecture:** Dataset-specific adapters normalize legacy corpora, HC3, TNEWS, M4/SemEval, and RAID into immutable `TextSample` records. The bootstrap partitions and deterministically samples those records, fits conformal thresholds only on human scores, evaluates held-out human and AI scores, atomically publishes artifact v2, and the runtime controller selects exact language/domain/length groups.

**Tech Stack:** Python 3.10, Hugging Face `datasets`, PyArrow, requests, Transformers/PyTorch, scikit-learn metrics, FastAPI, Streamlit, unittest.

---

## File Map

- Create `detection/text_resource_adapters.py`: canonical sample type and source-specific row adapters.
- Create `detection/text_domain.py`: canonical domain mapping and bilingual runtime routing.
- Modify `detection/text_resources.py`: resource registry, resumable preparation, manifests, normalized sample iteration, status and stats.
- Modify `scripts/bootstrap_text_resources.py`: deterministic partitioning, score collection, artifact v2 metrics, progress and atomic writes.
- Modify `detection/false_positive_controller.py`: multilingual lookup and validation diagnostics.
- Modify `config.yaml`: resource profile, per-role sample limits, reliability thresholds and source switches.
- Modify `requirements.txt`: dataset, parquet and progress dependencies.
- Modify `api/dashboard.py` and `reports/generator.py`: display FPR, recall, sample evidence and warnings.
- Modify `tests/test_text_resources.py`: adapters, partitions, sampling, metrics, errors and atomic publication.
- Modify `tests/test_false_positive_controller.py`: English and mixed-language routing and artifact v2.
- Modify report/provider tests for new diagnostics.
- Modify `docs/LINUX部署与运行_完整.md`: installation, disk requirements, bootstrap and monitoring.

### Task 1: Canonical Samples and Domain Routing

**Files:**
- Create: `detection/text_resource_adapters.py`
- Create: `detection/text_domain.py`
- Test: `tests/test_text_resources.py`
- Test: `tests/test_false_positive_controller.py`

- [ ] **Step 1: Write failing adapter and routing tests**

Add fixture tests covering:

```python
sample = TextSample(
    text="A human answer.",
    label="human",
    language="en",
    domain="general_qa",
    source="hc3_en",
    source_id="question-1",
    generator=None,
    split="train",
)
self.assertEqual(sample.group_key("short"), "en:general_qa:short")

router = TextDomainRouter()
self.assertEqual(router.classify("The study evaluates a neural algorithm.").domain, "academic_technical")
self.assertEqual(router.classify("The government announced the new policy.").domain, "official_news")
```

Add HC3, TNEWS, M4 and RAID row fixtures that assert human/AI labels, language,
domain, generator and shared pair IDs.

- [ ] **Step 2: Run tests and confirm missing symbols**

Run:

```bash
python -m unittest tests.test_text_resources tests.test_false_positive_controller -v
```

Expected: failures importing `TextSample`, adapters and `TextDomainRouter`.

- [ ] **Step 3: Implement canonical records and adapters**

Implement:

```python
@dataclass(frozen=True)
class TextSample:
    text: str
    label: Literal["human", "ai"]
    language: Literal["zh", "en"]
    domain: str
    source: str
    source_id: str
    generator: str | None = None
    split: str = "train"

    def group_key(self, length_bucket: str) -> str:
        return f"{self.language}:{self.domain}:{length_bucket}"
```

Adapters must expose `iter_*_rows(rows, source_name)` generators, normalize
source domains through `normalize_source_domain`, flatten HC3 answer lists, map
M4 labels `0/1`, map RAID `model == "human"`, and reject blank or unsupported
records.

- [ ] **Step 4: Implement bilingual domain router**

Return `DomainDecision(domain, confidence, evidence)` using explicit Chinese
and English marker tables. Preserve the four canonical domains and default to
`general_qa` with low confidence.

- [ ] **Step 5: Run focused tests**

Expected: all new adapter and router fixture tests pass.

### Task 2: Resource Registry and Project-local Normalization

**Files:**
- Modify: `detection/text_resources.py`
- Modify: `requirements.txt`
- Test: `tests/test_text_resources.py`

- [ ] **Step 1: Write failing resource profile tests**

Assert that `TextResourceManager(dataset_profile="multidomain")` exposes legacy
sources plus `hc3_en`, `hc3_zh`, `tnews`, `m4_multilingual`, and
`raid_no_adversarial`, while `dataset_profile="core"` exposes only legacy
sources.

Test that a completion marker with a different specification fingerprint is
not considered ready.

- [ ] **Step 2: Run resource tests and confirm failure**

Run:

```bash
python -m unittest tests.test_text_resources -v
```

Expected: profile and fingerprint assertions fail.

- [ ] **Step 3: Extend `DatasetSpec` and the resource manager**

Add explicit fields:

```python
name: str
kind: Literal["legacy_zip", "legacy_file", "hf_dataset", "raid_csv"]
license: str
source: str
required: bool
adapter: str
dataset_id: str | None
dataset_config: str | None
splits: tuple[str, ...]
url: str | None
```

Store raw/cache state below `data/text_false_positive/corpora/<source>` and
normalized JSONL below `data/text_false_positive/normalized/<source>.jsonl`.
Write marker JSON only after normalized output is atomically renamed.

- [ ] **Step 4: Add official distribution definitions**

Use:

```text
Hello-SimpleAI/HC3
Hello-SimpleAI/HC3-Chinese
clue / tnews
d0rj/SemEval2024-task8 / subtaskA_multilingual
https://dataset.raid-bench.xyz/train_none.csv
```

Record M4's official project URL separately from its Apache-2.0 distribution
mirror. RAID reads the official non-adversarial CSV sequentially. Physical
sources with label/domain ordering are scanned fully; only selected normalized
records and detector scores are capped.

- [ ] **Step 5: Add dependencies**

Add compatible constraints:

```text
datasets>=3.0,<5
pyarrow>=16,<24
tqdm>=4.66,<5
```

- [ ] **Step 6: Run resource tests**

Expected: fixture-only tests pass without network access.

### Task 3: Deterministic Sampling and Artifact v2

**Files:**
- Modify: `scripts/bootstrap_text_resources.py`
- Test: `tests/test_text_resources.py`

- [ ] **Step 1: Write failing calibration tests**

Create fake human and AI samples across `zh/en`, all domains and both length
buckets. Assert:

```python
self.assertEqual(saved["artifact_schema_version"], 2)
self.assertEqual(group["human_calibration_samples"], 100)
self.assertEqual(group["human_validation_samples"], 40)
self.assertEqual(group["ai_validation_samples"], 40)
self.assertIn("validation_fpr_interval_95", group)
self.assertIn("ai_recall", group)
self.assertIn("ai_fnr", group)
self.assertIn("ai_recall_interval_95", group)
self.assertIn("validation_auroc", group)
```

Add tests proving samples below `min_effective_chars` are excluded, pair IDs
stay in one partition, same seed yields identical IDs, and detector failures
report their exception class.

- [ ] **Step 2: Run focused calibration tests**

Expected: artifact-v2 assertions fail against the existing human-only builder.

- [ ] **Step 3: Implement deterministic partitions**

Assign partitions from a stable SHA-256 digest of
`f"{seed}:{sample.partition_key}"`. Keep HC3 and parallel benchmark pairs
together through their shared `partition_key`. Use `sample.identity_key` for
individual deduplication and deterministic reservoir identity. Use per-group
deterministic reservoirs for human calibration, human validation, and AI
validation.

- [ ] **Step 4: Implement score collection and metrics**

Fit `high_threshold` only from human calibration scores. Compute held-out FPR,
recall, FNR and Wilson intervals, plus `roc_auc_score` when both classes are
present. Add warnings:

```text
calibrated_threshold_near_score_ceiling
holdout_fpr_above_target
insufficient_human_validation
insufficient_ai_validation
ai_recall_below_minimum
auroc_unavailable
```

- [ ] **Step 5: Make artifact publication atomic**

Write JSON to `thresholds.json.tmp`, flush and replace
`thresholds.json` only after required groups pass. Preserve an existing valid
artifact on any exception.

- [ ] **Step 6: Add progress, status and stats profiles**

Print source preparation, selected bucket counts, model scoring progress and
failure summaries. `--profile stats` must operate without constructing the
detector.

- [ ] **Step 7: Run calibration tests**

Expected: all text-resource tests pass.

### Task 4: Runtime Multilingual Threshold Consumption

**Files:**
- Modify: `detection/false_positive_controller.py`
- Test: `tests/test_false_positive_controller.py`
- Test: `tests/test_text_provider_decision.py`

- [ ] **Step 1: Write failing runtime tests**

Test exact English lookup, Chinese news lookup, uncalibrated fallback,
mixed-language strict-threshold fallback, and propagation of:

```text
threshold_ai_recall
threshold_ai_validation_samples
threshold_validation_auroc
threshold_reliability_status
threshold_warnings
```

- [ ] **Step 2: Run runtime tests and confirm failure**

Run:

```bash
python -m unittest tests.test_false_positive_controller tests.test_text_provider_decision -v
```

- [ ] **Step 3: Replace embedded domain markers with `TextDomainRouter`**

Keep language detection local or move it into the router without changing the
public `FalsePositiveDecision` contract.

- [ ] **Step 4: Implement artifact-v2 lookup and diagnostics**

Lookup exact language/domain/length, then language/domain, language, and
built-in fallback. For mixed text without a calibrated mixed group, choose the
higher applicable high threshold and add `mixed_language_threshold_fallback`.

- [ ] **Step 5: Run runtime tests**

Expected: controller and provider tests pass.

### Task 5: Configuration and User-facing Evidence

**Files:**
- Modify: `config.yaml`
- Modify: `api/dashboard.py`
- Modify: `reports/generator.py`
- Test: `tests/test_false_positive_report.py`
- Test: `tests/test_report_and_providers.py`

- [ ] **Step 1: Add failing report tests**

Assert generated output contains human FPR evidence, AI recall evidence,
validation counts and reliability warnings when present, and remains readable
for built-in fallback decisions.

- [ ] **Step 2: Add configuration**

Add:

```yaml
dataset_profile: multidomain
bootstrap_human_calibration_limit: 5000
bootstrap_human_validation_limit: 1000
bootstrap_ai_validation_limit: 1000
bootstrap_max_source_items: 500000
min_reliable_human_validation: 200
min_reliable_ai_validation: 200
min_acceptable_ai_recall: 0.70
resource_sources:
  raid_adversarial:
    enabled: false
```

Keep existing keys as compatibility aliases for one release.

- [ ] **Step 3: Update Dashboard and report rendering**

Show the selected language/domain/length group, threshold, human FPR with
sample count, AI recall with sample count, AUROC and warnings. Do not describe
raw detector scores as calibrated probabilities.

- [ ] **Step 4: Run report tests**

Expected: report/provider tests pass.

### Task 6: Deployment Documentation and Regression

**Files:**
- Modify: `docs/LINUX部署与运行_完整.md`
- Modify: `docs/LINUX_DEPLOYMENT.md`

- [ ] **Step 1: Document installation and disk/network behavior**

Document `datasets`, PyArrow, approximately 1-2GB additional non-adversarial
source data, project-local paths, proxy inheritance, resumability and the fact
that no new detector weights are trained.

- [ ] **Step 2: Document commands**

Use:

```bash
python -m scripts.bootstrap_text_resources --profile stats --dataset-profile multidomain
python -m scripts.bootstrap_text_resources --profile all --dataset-profile multidomain --force-calibration
python -m scripts.bootstrap_text_resources --profile status --dataset-profile multidomain
```

Document that Hugging Face offline variables are enabled only after successful
bootstrap.

- [ ] **Step 3: Run focused suite**

Run:

```bash
python -m unittest \
  tests.test_text_resources \
  tests.test_false_positive_controller \
  tests.test_text_provider_decision \
  tests.test_false_positive_report \
  tests.test_report_and_providers -v
```

Expected: all focused tests pass.

- [ ] **Step 4: Run full suite**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass without downloading external datasets.

- [ ] **Step 5: Verify syntax and dependency metadata**

Run:

```bash
python -m compileall detection scripts api reports
python -m pip check
```

Expected: compilation succeeds and `pip check` reports no broken requirements.

- [ ] **Step 6: Review final diff**

Confirm no downloaded corpora, model weights, API keys, caches or unrelated
user changes are staged or modified by this implementation.
