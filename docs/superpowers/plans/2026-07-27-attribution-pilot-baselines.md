# Attribution Pilot Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a server-ready experiment runner that evaluates character n-gram Logistic Regression, mixed TF-IDF LinearSVC, and a frozen multilingual E5 linear probe on the current attribution pilot JSONL.

**Architecture:** Keep data validation/splitting, model evaluation, and reporting in separate modules. Every method consumes the same cleaned rows and the same deterministic semantic-prompt folds, returns out-of-fold predictions, and is summarized by one reporting layer. Frozen E5 embeddings are extracted once, fingerprinted, cached, and reused across folds without training encoder parameters.

**Tech Stack:** Python 3.10+, scikit-learn, NumPy, pandas, matplotlib, seaborn, PyTorch, Hugging Face Transformers.

---

## File structure

- Create `research/text_attribution/baseline_data.py`: JSONL loading, quality filtering, deduplication, semantic prompt groups, deterministic grouped folds.
- Create `research/text_attribution/baseline_models.py`: sparse pipelines, embedding probe, frozen E5 extraction and cache validation.
- Create `research/text_attribution/baseline_reporting.py`: metrics aggregation, Markdown/JSON/CSV/prediction artifacts, PNG/PDF figures.
- Create `scripts/run_attribution_pilot_baselines.py`: CLI orchestration and failure-isolated method execution.
- Create `tests/test_text_attribution_baseline_data.py`: filtering and leakage regression tests.
- Create `tests/test_text_attribution_baseline_models.py`: sparse CV and linear probe tests using real small matrices.
- Create `tests/test_text_attribution_baseline_reporting.py`: artifact and figure integration tests.
- Modify `requirements.txt`: declare matplotlib and seaborn used by the report runner.
- Modify `docs/ZHIZENGZENG_DATA_PRODUCTION.md`: document the server command and outputs.

### Task 1: Clean data and create leakage-safe folds

**Files:**
- Create: `research/text_attribution/baseline_data.py`
- Test: `tests/test_text_attribution_baseline_data.py`

- [ ] **Step 1: Write failing filtering and grouping tests**

```python
rows, audit = load_clean_samples(fixture)
assert audit["raw_records"] == 7
assert audit["kept_records"] == 2
assert audit["excluded_by_reason"]["truncated"] == 1
assert audit["excluded_by_reason"]["content_filtered"] == 1
assert audit["excluded_by_reason"]["route_mismatch"] == 1
assert audit["excluded_by_reason"]["language_mismatch"] == 1
assert audit["excluded_by_reason"]["exact_duplicate"] == 1
assert semantic_prompt_group("zh-science-explain-001") == "science-explain-001"
assert semantic_prompt_group("en-science-explain-001") == "science-explain-001"
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_data -v
```

Expected: import failure for `research.text_attribution.baseline_data`.

- [ ] **Step 3: Implement immutable sample records and cleaning**

```python
@dataclass(frozen=True)
class BaselineSample:
    request_key: str
    text: str
    label: str
    prompt_id: str
    semantic_prompt_group: str
    language: str
    temperature: float
    target_length: str

def semantic_prompt_group(prompt_id: str) -> str:
    return re.sub(r"^(zh|en)-", "", prompt_id)
```

Reject non-stop rows, route mismatches, empty text, language mismatches, and duplicate `(label, prompt_id, temperature, normalized_text)` keys. Return both clean samples and an audit dictionary.

- [ ] **Step 4: Implement six deterministic semantic-group folds**

```python
def assign_group_folds(samples, n_splits=6, seed=42):
    groups = sorted({sample.semantic_prompt_group for sample in samples})
    shuffled = random.Random(seed).sample(groups, len(groups))
    group_to_fold = {group: index % n_splits for index, group in enumerate(shuffled)}
    return [group_to_fold[sample.semantic_prompt_group] for sample in samples]
```

Validate that every semantic group maps to one fold, all labels have training samples in every fold, and `2 <= n_splits <= number_of_groups`.

- [ ] **Step 5: Run data tests**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_data -v
```

Expected: all tests pass.

### Task 2: Implement the two sparse baselines

**Files:**
- Create: `research/text_attribution/baseline_models.py`
- Test: `tests/test_text_attribution_baseline_models.py`

- [ ] **Step 1: Write a failing real cross-validation test**

```python
result = evaluate_sparse_method(samples, folds, method="char_lr", seed=42)
assert result.method == "char_lr"
assert len(result.predictions) == len(samples)
assert sorted({row.fold for row in result.predictions}) == [0, 1]
assert 0.0 <= result.oof_metrics["macro_f1"] <= 1.0
```

Repeat for `tfidf_svm`, and assert unknown method names fail explicitly.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_models -v
```

Expected: import failure for `baseline_models`.

- [ ] **Step 3: Implement sparse pipelines**

```python
def build_sparse_pipeline(method: str, seed: int):
    if method == "char_lr":
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5),
                min_df=2, max_features=120_000, sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                max_iter=3000, class_weight="balanced",
                random_state=seed,
            )),
        ])
    if method == "tfidf_svm":
        return Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2),
                    min_df=2, max_features=80_000, sublinear_tf=True,
                )),
                ("char", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5),
                    min_df=2, max_features=120_000, sublinear_tf=True,
                )),
            ])),
            ("classifier", LinearSVC(class_weight="balanced", random_state=seed)),
        ])
    raise ValueError(f"Unsupported sparse method: {method}")
```

- [ ] **Step 4: Implement shared out-of-fold evaluation**

For every fold, fit only on other folds, predict the held-out fold, calculate fold Accuracy/Macro-F1/Weighted-F1, and preserve request metadata in `PredictionRow`. Calculate aggregated OOF metrics only after every sample receives exactly one prediction.

- [ ] **Step 5: Run model tests**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_models -v
```

Expected: all sparse baseline tests pass.

### Task 3: Implement frozen E5 embeddings and linear probe

**Files:**
- Modify: `research/text_attribution/baseline_models.py`
- Test: `tests/test_text_attribution_baseline_models.py`

- [ ] **Step 1: Write failing cache and linear-probe tests**

```python
fingerprint = embedding_cache_fingerprint(
    samples, model_id="intfloat/multilingual-e5-base", max_length=512
)
assert fingerprint == embedding_cache_fingerprint(
    samples, model_id="intfloat/multilingual-e5-base", max_length=512
)
result = evaluate_embedding_probe(samples, folds, embeddings, seed=42)
assert result.method == "frozen_e5_probe"
assert len(result.predictions) == len(samples)
```

Use deterministic NumPy embeddings so the test exercises the real Logistic Regression probe without downloading a Hub model.

- [ ] **Step 2: Verify new tests fail**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_models -v
```

Expected: missing embedding functions.

- [ ] **Step 3: Implement cache fingerprint and validation**

Hash ordered request keys, normalized texts, model ID, max length, and pooling version. Store `embeddings`, `request_keys`, and JSON metadata in compressed NPZ. Reject a cache whose fingerprint or row order differs.

- [ ] **Step 4: Implement frozen E5 extraction**

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModel.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
).to(device).eval()

with torch.inference_mode():
    encoded = tokenizer(
        [f"passage: {text}" for text in batch],
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    hidden = model(**encoded).last_hidden_state
    vectors = mean_pool(hidden, encoded["attention_mask"])
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
```

Never call `backward()`, never create an optimizer, and release model/GPU memory after extraction.

- [ ] **Step 5: Implement the fold-local linear probe**

Use `LogisticRegression(max_iter=3000, class_weight="balanced")` on cached embeddings with the same folds and prediction schema as sparse methods.

- [ ] **Step 6: Run embedding/probe tests**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_models -v
```

Expected: all tests pass without network access.

### Task 4: Generate metrics, text reports, and figures

**Files:**
- Create: `research/text_attribution/baseline_reporting.py`
- Test: `tests/test_text_attribution_baseline_reporting.py`

- [ ] **Step 1: Write failing artifact integration test**

```python
artifacts = write_baseline_report(
    samples=samples,
    audit=audit,
    results=[char_result, svm_result, probe_result],
    output_dir=output_dir,
    run_config={"seed": 42},
)
assert Path(artifacts["metrics_json"]).is_file()
assert Path(artifacts["metrics_csv"]).is_file()
assert Path(artifacts["predictions_jsonl"]).is_file()
assert Path(artifacts["report_markdown"]).is_file()
assert len(artifacts["figures"]) == 4
```

- [ ] **Step 2: Verify the report test fails**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_reporting -v
```

Expected: missing reporting module.

- [ ] **Step 3: Implement metrics and subgroup summaries**

Write JSON-serializable overall, per-fold, per-class, language, temperature, and target-length metrics. Use a fixed global label order for every confusion matrix and `zero_division=0`.

- [ ] **Step 4: Implement output tables and predictions**

Write `metrics.json`, tidy `metrics.csv`, one `predictions.jsonl` row per method/sample, `dataset_audit.json`, `run_config.json`, and a Markdown report explaining the one-model-per-family limitation.

- [ ] **Step 5: Implement publication-quality figures**

Use a colorblind-safe palette and save each figure as 300-DPI PNG and vector PDF:

```python
save_figure(fig, output_dir / "figures" / "overall_metrics")
save_figure(fig, output_dir / "figures" / "fold_metrics")
save_figure(fig, output_dir / "figures" / "confusion_matrices")
save_figure(fig, output_dir / "figures" / "subgroup_metrics")
```

- [ ] **Step 6: Run reporting tests**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_reporting -v
```

Expected: all tests pass and all PNG files are non-empty.

### Task 5: Add the server CLI and documentation

**Files:**
- Create: `scripts/run_attribution_pilot_baselines.py`
- Modify: `requirements.txt`
- Modify: `docs/ZHIZENGZENG_DATA_PRODUCTION.md`
- Test: `tests/test_text_attribution_baseline_models.py`

- [ ] **Step 1: Write a failing CLI dry-run test**

```python
exit_code = main([
    "--input", str(fixture),
    "--output-dir", str(output_dir),
    "--methods", "char_lr", "tfidf_svm",
    "--dry-run",
])
assert exit_code == 0
assert not output_dir.exists()
```

- [ ] **Step 2: Verify the CLI test fails**

Run:

```bash
python -m unittest tests.test_text_attribution_baseline_models -v
```

Expected: missing CLI entry point.

- [ ] **Step 3: Implement CLI orchestration**

Support:

```text
--input
--output-dir
--methods char_lr tfidf_svm frozen_e5
--encoder-model intfloat/multilingual-e5-base
--device cuda:0
--batch-size 16
--max-length 512
--hf-cache models/huggingface
--embedding-cache
--folds 6
--seed 42
--dry-run
```

Run sparse methods first and persist their results before loading E5. If E5 fails, write `method_failures.json` and retain sparse artifacts.

- [ ] **Step 4: Declare visualization dependencies**

Add:

```text
matplotlib>=3.8,<4
seaborn>=0.13,<1
```

- [ ] **Step 5: Document the server command**

```bash
python -u -m scripts.run_attribution_pilot_baselines \
  --input data/attribution/raw/zhizengzeng-local-pilot-v1.jsonl \
  --output-dir outputs/attribution/zhizengzeng-local-pilot-v1-baselines \
  --methods char_lr tfidf_svm frozen_e5 \
  --encoder-model intfloat/multilingual-e5-base \
  --device cuda:0 \
  --batch-size 16 \
  --folds 6
```

- [ ] **Step 6: Run CLI dry-run and traditional-method smoke test**

Run dry-run on the real 923-row JSONL, then run `char_lr tfidf_svm` locally. Expected: clean sample counts printed, six folds resolved, metrics/report/figures written without E5 network access.

### Task 6: Full verification

**Files:**
- Verify all files above.

- [ ] **Step 1: Run focused tests**

```bash
python -m unittest \
  tests.test_text_attribution_baseline_data \
  tests.test_text_attribution_baseline_models \
  tests.test_text_attribution_baseline_reporting -v
```

- [ ] **Step 2: Run the full suite**

```bash
python -m unittest discover -s tests -v
```

- [ ] **Step 3: Compile changed Python modules**

```bash
python -m compileall -q \
  research/text_attribution \
  scripts/run_attribution_pilot_baselines.py
```

- [ ] **Step 4: Inspect generated figures and final server file list**

Verify all four PNGs render, all four PDFs are non-empty, no API/HF secrets appear in artifacts, and document exactly which source files must be uploaded.
