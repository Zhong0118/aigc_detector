# Concurrent Attribution Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe bounded concurrency, per-model circuit breaking, and tested same-family model replacement probes to the attribution data collector.

**Architecture:** API requests run in a `ThreadPoolExecutor`; a per-model lock enforces one in-flight request per concrete model, while the caller thread performs all JSONL writes. A shared circuit state stops deterministic route failures and repeated empty completions. Candidate model probing remains a separate command so `/models` listings cannot silently enter paid production runs.

**Tech Stack:** Python standard library (`concurrent.futures`, `threading`), `requests`, existing unittest suite and OpenAI-compatible gateway.

---

### Task 1: Concurrent collection contract

**Files:**
- Modify: `tests/test_text_attribution_production.py`
- Modify: `research/text_attribution/production.py`

- [ ] Add a test gateway that tracks total and per-model active requests.
- [ ] Add a failing test proving two different models overlap when `max_workers=2`.
- [ ] Add a failing test proving two requests for the same model remain serialized.
- [ ] Run `python -m unittest tests.test_text_attribution_production -v` and confirm the new `max_workers` calls fail before implementation.
- [ ] Implement worker execution with `ThreadPoolExecutor`, per-model locks, and main-thread JSONL writes.
- [ ] Re-run the production tests and confirm both concurrency tests pass.

### Task 2: Retry limits and circuit breaker

**Files:**
- Modify: `tests/test_text_attribution_production.py`
- Modify: `research/text_attribution/production.py`

- [ ] Add a failing test where the first request returns 404 and the second request for that model is never sent.
- [ ] Add a failing test where two requests each exhaust two empty-text attempts and the third is circuit-skipped.
- [ ] Add a failing assertion that a 400 error record contains a redacted, bounded response body excerpt.
- [ ] Implement deterministic HTTP status and consecutive-empty circuit state.
- [ ] Add `circuit_skipped` to `RunSummary` and progress events.
- [ ] Run the production tests and verify the new tests pass without regressing 401/429/resume behavior.

### Task 3: CLI controls

**Files:**
- Modify: `scripts/collect_attribution_run.py`
- Modify: `tests/test_text_attribution_production.py`

- [ ] Add parser assertions for `--max-workers`, `--empty-max-attempts`, and `--empty-circuit-threshold`.
- [ ] Verify the parser test fails before changing the script.
- [ ] Pass the three controls to `run_collection` and include `circuit_skipped` in the final summary.
- [ ] Construct one gateway session per worker instead of sharing a `requests.Session`.
- [ ] Run the parser and production tests.

### Task 4: Candidate model probing

**Files:**
- Create: `research/text_attribution/model_probe.py`
- Create: `scripts/probe_attribution_models.py`
- Create: `tests/test_text_attribution_model_probe.py`
- Create: `configs/text_attribution/zhizengzeng_model_candidates.v2.json`

- [ ] Write failing tests for bilingual eligibility, partial failure, and secret-redacted error output.
- [ ] Implement candidate manifest loading and bilingual probing.
- [ ] Implement the CLI with `--dry-run`, `--confirm`, `--max-workers`, and explicit output paths.
- [ ] Add Kimi and MiMo same-family candidate versions.
- [ ] Run the model-probe test module.

### Task 5: Server handoff and verification

**Files:**
- Modify: `docs/ATTRIBUTION_DATASET_V2_RUNBOOK.md`
- Modify: `docs/ZHIZENGZENG_DATA_PRODUCTION.md`
- Rebuild: `outputs/attribution/attribution_server_v2_bundle.zip`

- [ ] Document the candidate probe command and the rule that listed models are not automatically eligible.
- [ ] Document the default 4-worker collection command and serial fallback.
- [ ] Run the focused tests for production, run configuration, gateway, and model probing.
- [ ] Run the full text-attribution unittest suite.
- [ ] Build the server archive from the verified source state and record its SHA-256.
