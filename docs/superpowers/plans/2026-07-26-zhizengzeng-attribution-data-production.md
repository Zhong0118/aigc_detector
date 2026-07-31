# Zhizengzeng Attribution Data Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configuration-driven, resumable Zhizengzeng data collector for ten LLM families, with auditable prompt/model manifests, repetitions, bounded retries, quality reports, and local/server run profiles.

**Architecture:** Keep the existing gateway client and prompt bank as protocol and prompt sources. Add a validated configuration layer that resolves model catalogs and run profiles into deterministic `PlannedRequest` objects, then use a dedicated production runner to append success and error JSONL records safely. A separate audit command checks coverage and output quality without mutating raw data.

**Tech Stack:** Python 3.12, `requests`, JSON/JSONL, `unittest`, PowerShell server launcher.

---

### Task 1: Freeze model, prompt, and run manifests

**Files:**
- Create: `configs/text_attribution/zhizengzeng_model_catalog.json`
- Create: `configs/text_attribution/prompt_manifest.pilot-v1.json`
- Create: `configs/text_attribution/runs/smoke.json`
- Create: `configs/text_attribution/runs/local_pilot.json`
- Create: `configs/text_attribution/runs/server_10k.json`
- Test: `tests/test_text_attribution_run_config.py`

- [ ] **Step 1: Write a failing manifest contract test**

The test must require ten unique `family_id` values, the approved exact model IDs, 24 prompt records matching `PILOT_PROMPTS`, and request totals of 20 and 960 for smoke and local pilot.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_text_attribution_run_config -v
```

Expected: import or missing-file failure because the configuration layer and manifests do not exist.

- [ ] **Step 3: Add the five JSON manifests**

Use this run-profile shape:

```json
{
  "run_id": "local-pilot-v1",
  "gateway": "zhizengzeng",
  "model_catalog": "../zhizengzeng_model_catalog.json",
  "model_roles": ["pilot"],
  "prompt_manifest": "../prompt_manifest.pilot-v1.json",
  "prompt_ids": ["*"],
  "sampling": [
    {"temperature": 0.0, "top_p": null, "max_tokens": 1536},
    {"temperature": 0.8, "top_p": null, "max_tokens": 1536}
  ],
  "repetitions": 2,
  "max_requests": 960,
  "enabled": true
}
```

`server_10k.json` must set `enabled` to false and require 100 prompts, so it cannot run while only 24 prompts exist.

- [ ] **Step 4: Run the manifest test and keep failures limited to missing loader code**

### Task 2: Add validated configuration loading and repetitions

**Files:**
- Create: `research/text_attribution/run_config.py`
- Modify: `research/text_attribution/collection.py`
- Test: `tests/test_text_attribution_run_config.py`
- Test: `tests/test_text_attribution_collection.py`

- [ ] **Step 1: Write failing tests for configuration resolution**

Tests must assert:

```python
resolved = load_run_config(Path("configs/text_attribution/runs/local_pilot.json"))
self.assertEqual(len(resolved.models), 10)
self.assertEqual(len(resolved.prompts), 24)
self.assertEqual(resolved.request_count, 960)
```

Also require rejection of disabled profiles, duplicate families, missing prompt IDs, a mismatched hard cap, and a server profile whose required prompt count is unmet.

- [ ] **Step 2: Write a failing repetition test**

Require `build_request_plan(..., repetitions=2)` to return distinct requests with `repeat_index` values 0 and 1.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_text_attribution_run_config tests.test_text_attribution_collection -v
```

- [ ] **Step 4: Implement the minimal loader and repetition field**

Define immutable `ModelSelection`, `RunSampling`, and `ResolvedRunConfig` types. Resolve manifest paths relative to the run-profile file, validate exact Cartesian size, and add `repeat_index: int = 0` to `PlannedRequest`.

- [ ] **Step 5: Run focused tests and verify GREEN**

### Task 3: Build the production runner

**Files:**
- Create: `research/text_attribution/production.py`
- Create: `scripts/collect_attribution_run.py`
- Test: `tests/test_text_attribution_production.py`

- [ ] **Step 1: Write failing tests for request identity, resume, and failures**

Require the request key to include `repeat_index`, completed successes to be skipped, failures to be written without secrets, 401 to stop the run, and 429/5xx to receive bounded retries.

- [ ] **Step 2: Run the production tests and verify RED**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_text_attribution_production -v
```

- [ ] **Step 3: Implement `run_collection` and the CLI**

The CLI interface must be:

```powershell
python -m scripts.collect_attribution_run `
  --config configs/text_attribution/runs/smoke.json `
  --dry-run
```

and:

```powershell
python -m scripts.collect_attribution_run `
  --config configs/text_attribution/runs/smoke.json `
  --confirm
```

Without `--confirm`, real network generation must be rejected. Output paths default to
`data/attribution/raw/<run_id>.jsonl` and
`data/attribution/errors/<run_id>.errors.jsonl`.

- [ ] **Step 4: Run production tests and verify GREEN**

### Task 4: Add deterministic quality audit

**Files:**
- Create: `research/text_attribution/audit.py`
- Create: `scripts/audit_attribution_data.py`
- Test: `tests/test_text_attribution_audit.py`

- [ ] **Step 1: Write failing audit tests**

Given fixture JSONL, require counts by model, family, language, task, sampling and repeat; success coverage; empty-text count; truncation rate; response-route mismatch count; duplicate-text rate; and missing-cell detection against the run profile.

- [ ] **Step 2: Run audit tests and verify RED**

- [ ] **Step 3: Implement audit report generation**

The command must be:

```powershell
python -m scripts.audit_attribution_data `
  --config configs/text_attribution/runs/smoke.json `
  --input data/attribution/raw/zhizengzeng-smoke-v3.jsonl `
  --output data/attribution/reports/zhizengzeng-smoke-v3.audit.json
```

- [ ] **Step 4: Run audit tests and verify GREEN**

### Task 5: Add server handoff and operating documentation

**Files:**
- Create: `scripts/run_zhizengzeng_server.ps1`
- Create: `docs/ZHIZENGZENG_DATA_PRODUCTION.md`
- Modify: `docs/TEXT_ATTRIBUTION_LOCAL_PILOT.md`

- [ ] **Step 1: Add a launcher that checks `.env.local` and requires a run config**

The launcher must invoke the same Python module used locally and write logs under `logs/`.
It must not contain a key or copy `.env.local`.

- [ ] **Step 2: Document the exact 20, 960, and future 10,000 request formulas**

State explicitly that 960 is the total across ten models and that each model contributes 96 samples.

- [ ] **Step 3: Document safe upload and resume commands**

Include dry-run, real run, audit, interruption/resume, and output-directory instructions.

### Task 6: Run the real smoke collection and verify the project

**Files:**
- Produce: `data/attribution/raw/zhizengzeng-smoke-v3.jsonl`
- Produce: `data/attribution/errors/zhizengzeng-smoke-v3.errors.jsonl` only if failures occur
- Produce: `data/attribution/reports/zhizengzeng-smoke-v3.audit.json`

- [ ] **Step 1: Verify the dry-run has exactly 20 requests**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_run `
  --config configs/text_attribution/runs/smoke.json `
  --dry-run
```

- [ ] **Step 2: Execute the 20-request smoke**

Run the same command with `--confirm`. Do not proceed to 960 unless all ten families generate
two non-truncated, non-empty outputs.

- [ ] **Step 3: Generate and inspect the quality report**

- [ ] **Step 4: Run the complete regression suite**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Compile the new Python modules**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m compileall research\text_attribution scripts
```

- [ ] **Step 6: Report actual smoke results**

Report per-family success, any route mismatch, truncation, exact files produced, and whether the
960-request profile is ready. Do not start the 960-request billable run without a separate explicit
confirmation after the audit result.
