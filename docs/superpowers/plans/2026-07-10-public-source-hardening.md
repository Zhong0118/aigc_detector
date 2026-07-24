# Public Source Matcher Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make public-source matching resilient, semantically correct, safe for external pages, and capable of sending only unmatched text to downstream AI detectors.

**Architecture:** Keep `PublicSourceMatcher` as the source-evidence boundary and keep AI scoring in `ApiFirstDetectionEngine`. The matcher returns structured evidence and offsets; the engine chooses full text, unmatched text, or no AI model based on matcher status without mixing source and AI scores.

**Tech Stack:** Python 3, dataclasses, requests, difflib, unittest, YAML.

---

### Task 1: Short-Text Guard And Original-Text Mapping

**Files:**
- Modify: `detection/public_source_matcher.py`
- Test: `tests/test_public_source_matcher.py`

- [ ] Add a failing test asserting that normalized text shorter than `min_input_chars` returns `too_short` without calling the search client.
- [ ] Add a failing test asserting that matched normalized spans can remove the corresponding original text, including punctuation between normalized characters.
- [ ] Run `python -m unittest tests.test_public_source_matcher` and verify both tests fail for the missing behavior.
- [ ] Add `min_input_chars`, normalized-to-original index mapping, and `unmatched_text()` extraction.
- [ ] Re-run the matcher test module and verify it passes.

Expected interface:

```python
matcher = PublicSourceMatcher(min_input_chars=80)
result = matcher.match(text)
remaining = matcher.unmatched_text(text, result.matched_spans)
```

### Task 2: Per-Page Failure Isolation And Fetch Safety

**Files:**
- Modify: `detection/public_source_matcher.py`
- Test: `tests/test_public_source_matcher.py`

- [ ] Add a failing test where the first candidate fetch raises an error and a second candidate contains a strong match; expect `matched` plus one recorded issue.
- [ ] Add a failing test where all candidate fetches fail; expect `error` and no exception escaping.
- [ ] Add failing URL validation tests for localhost/private hosts and non-HTTP schemes.
- [ ] Run the matcher tests and confirm the expected failures.
- [ ] Separate candidate discovery from page fetching, catch each failure independently, and add result counters/issues.
- [ ] Implement safe redirect validation, text content-type filtering, and bounded streaming reads.
- [ ] Re-run the matcher tests.

Expected issue shape:

```python
{"stage": "fetch", "url": "https://example.test/page", "error": "403 ..."}
```

### Task 3: Public-Source Label And Partial-Text Routing

**Files:**
- Modify: `detection/providers.py`
- Modify: `tests/test_report_and_providers.py`

- [ ] Change the existing strong-match test expectation from `human/0.05` to `public_source/0.0` and verify it fails.
- [ ] Add a failing integration test proving that a `partial` match sends only unmatched text to the local text detector.
- [ ] Add a failing Hive text test proving `text_override` is used instead of file content.
- [ ] Run the provider tests and confirm failures are due to missing routing behavior.
- [ ] Add a text analysis scope object in `detect`, thread `text_override` through Hive V2/V3, demo, and local provider calls, and annotate provider details.
- [ ] If unmatched normalized text is below the configured minimum, return a structured skipped result rather than a fabricated score.
- [ ] Re-run provider tests.

Expected strong result:

```python
DetectionPackage(score=0.0, label="public_source", ...)
```

### Task 4: Configuration And Reports

**Files:**
- Modify: `config.yaml`
- Modify: `reports/generator.py`
- Modify: `tests/test_report_and_providers.py`

- [ ] Add failing report tests for `public_source`, `partial`, `weak`, `too_short`, and `error` states.
- [ ] Run report tests and confirm missing state explanations.
- [ ] Add `min_input_chars`, `min_unmatched_chars`, `max_page_bytes`, and redirect/fetch limits to configuration.
- [ ] Update report summary, label mapping, evidence and recommendations without exposing raw unmatched text.
- [ ] Re-run report tests.

### Task 5: Verification And Process Record

**Files:**
- Create: `docs/PUBLIC_SOURCE_MATCHER_VALIDATION.md`

- [ ] Run `python -m unittest tests.test_public_source_matcher`.
- [ ] Run `python -m unittest tests.test_report_and_providers`.
- [ ] Run `python -m unittest discover`.
- [ ] Run `git diff --check`.
- [ ] Perform a best-effort live test with a known public-domain passage and record whether search, fetch, and matching each succeeded.
- [ ] Document reproduced failures, root causes, chosen fixes, commands, outcomes, known limits, privacy implications, and deployment guidance.

No task changes unrelated files such as `requirements.txt`, launch scripts, generated outputs, or existing user documentation.
