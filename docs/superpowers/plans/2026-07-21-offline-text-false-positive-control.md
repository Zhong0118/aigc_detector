# Offline Text False-Positive Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace online source search and binary averaging with offline segmented text detection, conservative three-way decisions, project-local resource bootstrap, and an explanatory dashboard.

**Architecture:** Existing text detector models remain the base scorers. A sentence-aware segmenter produces stable windows, and a false-positive controller aggregates provider and segment evidence into `low_risk`, `uncertain`, `high_risk_ai`, or `insufficient_evidence` using artifact-backed per-domain thresholds. A bootstrap command downloads approved human calibration corpora and Hugging Face models into project-local folders; runtime loads the resulting thresholds and never searches the public web.

**Tech Stack:** Python 3.10, unittest, PyTorch/Transformers, Hugging Face Hub, FastAPI, Streamlit, YAML.

---

### Task 1: Segmentation and decision policy

**Files:**
- Create: `detection/text_segmentation.py`
- Create: `detection/false_positive_controller.py`
- Create: `tests/test_false_positive_controller.py`

- [ ] Write failing tests for sentence-boundary windows, weighted aggregation, uncertainty, missing evidence, and calibrated threshold lookup.
- [ ] Run `python -m unittest tests.test_false_positive_controller -v` and confirm failures are caused by missing modules.
- [ ] Implement the minimal segmenter and controller.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Local detector and provider integration

**Files:**
- Modify: `detection/text_detector.py`
- Modify: `detection/providers.py`
- Modify: `api/analysis.py`
- Modify: `tests/test_text_detector_routing.py`
- Modify: `tests/test_report_and_providers.py`

- [ ] Add failing tests for segment score details and text decisions.
- [ ] Remove the online public-source branch from the engine.
- [ ] Run local text models by stable segments and aggregate without maximum-only document decisions.
- [ ] Add controller details to `DetectionPackage` and API JSON while retaining old fields.
- [ ] Run focused tests.

### Task 3: Project-local automatic resources

**Files:**
- Create: `detection/text_resources.py`
- Create: `scripts/bootstrap_text_resources.py`
- Create: `tests/test_text_resources.py`
- Modify: `scripts/run_api.ps1`
- Modify: `config.yaml`

- [ ] Write failing tests for manifest paths, idempotent downloads, dataset parsing, and threshold artifact generation.
- [ ] Implement project-local model/data directories and approved corpus manifests.
- [ ] Make `run_api.ps1` invoke bootstrap only when required artifacts are absent; subsequent runs skip completed work.
- [ ] Run focused tests and a no-network status check.

### Task 4: Dashboard and reports

**Files:**
- Modify: `api/dashboard.py`
- Modify: `reports/generator.py`
- Modify: `tests/test_report_and_providers.py`

- [ ] Add failing report tests for uncertain and insufficient-evidence results.
- [ ] Display calibrated AI risk, low/high thresholds, uncertainty reasons, analyzed coverage, and segment scores.
- [ ] Remove public-source wording and tests.
- [ ] Run focused tests.

### Task 5: Removal and complete verification

**Files:**
- Delete: `detection/public_source_matcher.py`
- Delete: `tests/test_public_source_matcher.py`
- Modify: `docs/AIGC_FALSE_POSITIVE_RESEARCH_2026-07-21.md`

- [ ] Confirm no runtime import or config key refers to public source matching.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run bootstrap status/dry-run and compile checks.
- [ ] Start the API, call `/api/v1/providers` and a text analysis endpoint, then stop the process.
