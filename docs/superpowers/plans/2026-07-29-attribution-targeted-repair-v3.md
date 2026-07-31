# Attribution Targeted Repair V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the useful V2 collection, replace three unreliable concrete models, regenerate four bad cells, and assemble an exactly balanced and auditable 600-record V3 dataset.

**Architecture:** Use the existing production collector for a twelve-request candidate probe and a ninety-request replacement-model batch. Add a small explicit-repair manifest loader that converts four named repair cells into `PlannedRequest` objects, then add a deterministic assembler that excludes retired/bad V2 rows, inserts replacement and repair rows, preserves source provenance, and validates the final records against the V3 request plan.

**Tech Stack:** Python 3.10, dataclasses, JSON/JSONL, existing OpenAI-compatible gateway, `unittest`, existing attribution audit and EDA modules.

---

### Task 1: Candidate probe configuration

**Files:**
- Create: `configs/text_attribution/zhizengzeng_replacement_candidates.v3.json`
- Create: `configs/text_attribution/runs/replacement_candidates_probe_v3.json`
- Modify: `tests/test_text_attribution_v2_collection_config.py`

- [ ] **Step 1: Write a failing repository-configuration test**

Add a test that loads `replacement_candidates_probe_v3.json` and asserts:

```python
resolved = load_run_config(
    CONFIG_ROOT / "runs" / "replacement_candidates_probe_v3.json"
)
self.assertEqual(resolved.request_count, 12)
self.assertEqual(resolved.requests_per_model, 2)
self.assertEqual(
    {model.family_id for model in resolved.models},
    {"deepseek", "glm"},
)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_text_attribution_v2_collection_config -v
```

Expected: failure because the candidate configuration does not exist.

- [ ] **Step 3: Create the six-candidate catalog and twelve-request run**

The catalog contains:

```text
deepseek-chat
deepseek-v3.2
deepseek-v3.1-terminus
glm-4.5-flash
glm-4-air-250414
glm-4-flashx-250414
```

Each family uses unique candidate variant roles so catalog validation remains
strict. The run reuses `prompt_manifest.smoke-v2.json`, with the Chinese
deterministic and English diverse profiles already used by the production
smoke.

- [ ] **Step 4: Run the focused test and verify it passes**

Expected: twelve requests, two per candidate.

### Task 2: Execute and select replacement candidates

**Files:**
- Generate: `data/attribution/raw/zhizengzeng-attribution-replacement-candidates-v3.jsonl`
- Generate: `data/attribution/errors/zhizengzeng-attribution-replacement-candidates-v3.errors.jsonl`
- Generate: `outputs/attribution/replacement-candidates-v3-selection.md`

- [ ] **Step 1: Dry-run the candidate plan**

```bash
python -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/replacement_candidates_probe_v3.json \
  --dry-run
```

Expected: `request_count: 12`.

- [ ] **Step 2: Execute the bounded paid probe**

```bash
python -u -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/replacement_candidates_probe_v3.json \
  --env-file .env.local \
  --confirm \
  --max-workers 4 \
  --timeout-seconds 180 \
  --max-attempts 2 \
  --empty-max-attempts 2 \
  --empty-circuit-threshold 2 \
  --retry-base-seconds 3 \
  --sleep-seconds 0.5
```

- [ ] **Step 3: Select one model per affected family**

Require both prompts to be non-empty, route-matched, language-correct, and
`finish_reason=stop`. Reject reasoning-budget distortion. Rank remaining
candidates by length compliance and latency, then record every candidate's
result and the selected DeepSeek/GLM model in the selection report.

### Task 3: Explicit four-cell repair collection

**Files:**
- Create: `research/text_attribution/repair.py`
- Create: `scripts/collect_attribution_repairs.py`
- Create: `configs/text_attribution/repairs/server_600_v3_cells.json`
- Create: `tests/test_text_attribution_repair.py`

- [ ] **Step 1: Write failing manifest-loader tests**

Test that the loader:

```python
manifest = load_repair_manifest(path)
self.assertEqual(manifest.request_count, 4)
self.assertEqual(len(manifest.plan), 4)
self.assertEqual(
    {cell.supersedes_request_key for cell in manifest.cells},
    {
        "dffc229da12a8f06663cad34a64ea67013a48a03b8c31c6d965f3a2ca30dc7c8",
        "66c2494c633f61e61c10bdc203a2cd5b7b22297b55e3d157512a7a7762fe7cd2",
        "875b51035c5b89f6c44b1faa509ef631af05d2acc02c4044cd8a60985701f874",
        "f8b39a7d74ec9e4c80ca926e04efd5d89ee6fbddc549a0c8005fe92952998dbe",
    },
)
```

Also test rejection of duplicate repair IDs, duplicate superseded keys,
unknown models/prompts, non-positive output budgets, and blank reasons.

- [ ] **Step 2: Run tests and verify they fail**

Expected: import failure because `repair.py` does not exist.

- [ ] **Step 3: Implement the minimal repair loader**

Define immutable `RepairCell` and `ResolvedRepairManifest` dataclasses.
Resolve model and prompt identities through the existing V3 catalog and prompt
manifest. Convert each cell to one `PlannedRequest`, using sampling ID
`repair-v3` so its physical collection key cannot collide with the original
record.

- [ ] **Step 4: Implement the collector CLI**

The CLI supports `--dry-run`, `--confirm`, the same bounded retry/concurrency
options as `collect_attribution_run`, and prints the four repair IDs plus their
superseded keys. Real collection delegates to `run_collection` with the
explicit plan.

- [ ] **Step 5: Verify focused tests pass**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_text_attribution_repair -v
```

### Task 4: Deterministic 600-record assembler

**Files:**
- Create: `research/text_attribution/assembly.py`
- Create: `scripts/assemble_attribution_v3.py`
- Create: `tests/test_text_attribution_assembly.py`

- [ ] **Step 1: Write failing assembly tests**

Create fixtures representing:

```text
base rows: unchanged, retired-model, and superseded rows
replacement rows: new model rows
repair rows: physical repair request keys
```

Assert that assembly:

- removes every retired-model row;
- removes all four superseded V2 rows;
- requires exactly one repair row per repair cell;
- rewrites the repair row's logical `request_key` to the superseded key;
- stores its physical key under `repair.collection_request_key`;
- stores `source_run_id` and changes `run_id` to the V3 run ID;
- rejects duplicate logical keys, missing repairs, unexpected replacement
  models, truncated repair rows, route mismatches, and a total other than the
  V3 request count.

- [ ] **Step 2: Run tests and verify they fail**

Expected: import failure because `assembly.py` does not exist.

- [ ] **Step 3: Implement assembly and validation**

Expose:

```python
assemble_v3_dataset(
    *,
    v3_config,
    repair_manifest,
    base_rows,
    replacement_rows,
    repair_rows,
) -> tuple[list[dict], dict]
```

Build the expected request-key set from `build_plan_from_config(v3_config)`.
The final logical keys must equal that set exactly. Sort output by the V3 plan
order and emit a machine-readable report containing source counts, excluded
counts, inserted counts, final model/family counts, and validation failures.

- [ ] **Step 4: Implement the CLI and atomic publication**

Write the completed JSONL and report to temporary sibling files, then replace
the requested outputs only after all validation succeeds.

- [ ] **Step 5: Verify focused tests pass**

Run:

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_text_attribution_assembly -v
```

### Task 5: Freeze V3 catalog and collection profiles

**Files:**
- Create: `configs/text_attribution/zhizengzeng_model_catalog.v3.json`
- Create: `configs/text_attribution/zhizengzeng_replacement_batch.v3.json`
- Create: `configs/text_attribution/runs/replacement_batch_90_v3.json`
- Create: `configs/text_attribution/runs/server_600_v3.json`
- Modify: `tests/test_text_attribution_v2_collection_config.py`

- [ ] **Step 1: Add failing count and identity tests**

Assert:

```text
V3 catalog: 20 models, 10 families, two roles per family
replacement batch: 3 models
replacement run: 90 requests, 30 per model
server V3 run: 600 requests, 30 per model
```

Also assert that the three retired model IDs are absent and the three selected
replacement IDs occupy the intended family/role slots.

- [ ] **Step 2: Run tests and verify they fail**

Expected: missing V3 configuration files.

- [ ] **Step 3: Create the V3 files using probe winners**

Copy unchanged V2 entries byte-for-byte at the field level. Replace only:

```text
OpenAI fast
DeepSeek balanced
GLM fast
```

The replacement batch uses the same ten prompts and deterministic/diverse
sampling profiles as `server_600_v2`, producing exactly ninety physical
requests whose keys match their V3 plan cells.

- [ ] **Step 4: Run configuration tests and dry-runs**

Expected: 90 and 600 exact request counts.

### Task 6: Documentation, bundle, and full verification

**Files:**
- Modify: `docs/ATTRIBUTION_DATASET_V2_RUNBOOK.md`
- Generate: `outputs/attribution/attribution_server_targeted_repair_v3_20260729.zip`

- [ ] **Step 1: Document the three server commands**

Document candidate probe, replacement batch, four-cell repair, and assembly.
State that V2 raw files remain immutable and the assembled output is the only
600-record training candidate.

- [ ] **Step 2: Run the complete test suite**

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run all dry-runs**

Expected:

```text
candidate probe: 12
replacement batch: 90
explicit repairs: 4
assembled V3 target: 600
```

- [ ] **Step 4: Build and inspect the server archive**

Include source, configs, scripts, tests, prompt manifests, and runbook.
Exclude `.env.local`, API keys, JSONL data, logs, `__pycache__`, and `.pyc`.
Record archive size and SHA256.
