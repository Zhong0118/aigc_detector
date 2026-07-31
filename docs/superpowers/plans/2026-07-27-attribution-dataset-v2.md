# Attribution Dataset V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a validated, resumable 20-model API collector and configuration foundation for the 6000-response attribution dataset.

**Architecture:** Extend the existing manifest-driven collector without breaking pilot-v1 profiles. V2 sampling profiles carry their own repetition counts, prompt metadata carries output budgets, and the production planner resolves the effective request parameters before creating stable request keys.

**Tech Stack:** Python dataclasses, JSON manifests, requests, pytest/unittest.

---

### Task 1: Add V2 manifest and planning semantics

**Files:**
- Modify: `research/text_attribution/prompts.py`
- Modify: `research/text_attribution/collection.py`
- Modify: `research/text_attribution/run_config.py`
- Modify: `research/text_attribution/production.py`
- Test: `tests/test_text_attribution_run_config.py`
- Test: `tests/test_text_attribution_production.py`

- [ ] Write failing tests proving duplicate families are allowed, variant roles are unique, per-profile repetitions produce `1+2` requests, and prompt budgets override legacy sampling budgets.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Add backward-compatible V2 fields and planning logic.
- [ ] Run focused tests and verify they pass.

### Task 2: Support safe per-model request overrides

**Files:**
- Modify: `research/text_attribution/gateway.py`
- Modify: `research/text_attribution/production.py`
- Test: `tests/test_attribution_adapters.py`
- Test: `tests/test_text_attribution_production.py`

- [ ] Write failing tests for safe override merging and protected core fields.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement override propagation and validation.
- [ ] Run focused tests and verify they pass.

### Task 3: Add the 20-model V2 catalog and 40-request smoke profile

**Files:**
- Create: `configs/text_attribution/zhizengzeng_model_catalog.v2.json`
- Create: `configs/text_attribution/prompt_manifest.v2.json`
- Create: `configs/text_attribution/runs/smoke_v2.json`
- Create: `configs/text_attribution/runs/server_6k_v2.json`
- Test: `tests/test_text_attribution_run_config.py`

- [ ] Write tests for 20 unique models, two variants per family, 100 prompts, 40 smoke requests, and 6000 production requests.
- [ ] Run the tests and verify the expected failures.
- [ ] Add the catalogs and profiles.
- [ ] Run the tests and verify they pass.

### Task 4: Update the CLI and operator documentation

**Files:**
- Modify: `scripts/collect_attribution_run.py`
- Create: `docs/ATTRIBUTION_DATASET_V2_RUNBOOK.md`
- Test: `tests/test_text_attribution_production.py`

- [ ] Write a failing preview test for sampling-profile and per-length counts.
- [ ] Implement the preview output and document Windows/Linux commands.
- [ ] Verify dry-run output for 40 and 6000 planned requests.

### Task 5: Regression verification

- [ ] Run all text-attribution collection tests.
- [ ] Run the full test suite.
- [ ] Inspect the diff for secrets and unrelated changes.

