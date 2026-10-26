# IAB Balanced v2 200-Per-Cell Execution Plan

> **For agentic workers:** Execute this plan in the isolated `attribution-pilot-baseline` worktree. Keep each change testable and preserve the existing v1 data and production checkout.

**Goal:** Acquire the largest verifiable IAB V3.2 subset up to 200 generated images per source/semantic cell, then run D0, A0, and OFA V1 once with auditable development-only results.

**Architecture:** Use the frozen v1 selection and hash-verified files as the prefix of an availability-capped v2 selection. Read only selected ZIP members through the public Dataverse Range API, record transfer and training evidence, then let a lock-protected monitor launch the dataset pipeline and the three independent runs exactly once.

**Tech Stack:** Python 3.10, requests, PyTorch, torchvision, Pillow, scikit-learn, pytest, Bash, Codex heartbeat automation.

**Spec:** User-provided 2026-10-20 research execution request; repository protocol: `docs/IAB_BALANCED_PROTOCOL.md`.

## Global Constraints

- Target ImageAttributionBench V3.2, Dataverse dataset 11127188; never replace or delete `iab-balanced-v1`.
- Select up to 200 images for each of 22 generation sources × 10 semantic categories; use only actually available verified members and never duplicate images to meet a target.
- Select up to 4,400 real images per semantic category when available; report real-image shortages and class imbalance.
- Run D0 ResNet50 truth detection, A0 ResNet50 closed-set attribution on generated images only, and OFA V1 reference verification as separate evaluations.
- Keep the official holdout unopened; record per-epoch train/development metrics, sample predictions, configuration, hashes, logs, and exit state.
- Do not call paid APIs, disable TLS verification, print or commit temporary credentials, modify `sync`, or change the production service.

---

### Task 1: Make acquisition resumable, bounded, and measurable

**Files:**
- Modify: `scripts/download_iab_subset.py`
- Modify: `research/image_attribution/iab_selection.py`
- Test: `tests/test_iab_delivery.py`
- Test: `tests/test_iab_selection.py`

- [ ] Add failing tests for stable public API URLs when no temporary link file is supplied, HTTP Range success/failure/timeout accounting, and selection byte totals.
- [ ] Run those tests and confirm they fail for the missing behavior.
- [ ] Use the stable public Dataverse datafile endpoint by default; keep TLS verification enabled and do not require expiring signed URLs.
- [ ] Record attempted/successful/failed requests, timeouts, transferred bytes, request duration, and effective throughput in cell receipts and download output.
- [ ] Extract real images with bounded Range reads, avoiding full real-category ZIP downloads; retain verified prior samples through hash-checked hard links.
- [ ] Add selected raw/compressed-byte totals to the frozen selection audit.
- [ ] Run focused acquisition and selection tests.

### Task 2: Gate and observe the one-time pipeline

**Files:**
- Modify: `scripts/monitor_iab_v2.py`
- Modify: `scripts/run_iab_m2.sh`
- Test: `tests/test_iab_monitor.py`
- Test: `tests/test_iab_epoch_metrics.py`
- Test: `tests/test_iab_v2_manifest.py`

- [ ] Add failing tests for direct/proxy public Range probes, verified-proxy-only download environment, required model assets, busy-GPU waits, and duplicate-launch prevention.
- [ ] Run those tests and confirm the intended missing behavior.
- [ ] Probe public direct and configured-forward-proxy requests with ordinary HTTPS Range GETs; pass proxy variables only to acquisition after a successful probe.
- [ ] Require all selected-cell receipts and a frozen manifest before launching training; keep launch intent, PID identity, lock, logs, and terminal status durable.
- [ ] Use the existing server ResNet50 and OFA checkpoints as read-only inputs; create private research-worktree paths for copied assets and outputs.
- [ ] Run focused monitor, manifest, and training-metric tests.

### Task 3: Verify and start the experiment

**Files:**
- Create: `docs/superpowers/plans/2026-10-20-iab-balanced-v2-200-execution.md`
- Create/retain: `outputs/iab-v2-monitor/`
- Create: `outputs/image_experiments/iab-balanced-v2-200/`

- [ ] Verify metadata version and all 230 source/semantic cells against the cached v1 archive catalogs.
- [ ] Freeze one deterministic selection, preserving the v1 selected member lists and capping each cell at its true available maximum.
- [ ] Confirm projected output fits available disk and report selected counts and byte estimates before acquisition.
- [ ] Start one detached download worker at 16 concurrent cell workers; leave the Codex monitor to check every 10 minutes and gate D0/A0/OFA until receipt and manifest validation succeeds.
- [ ] Run the full image-experiment test suite, review changed/untracked files for secrets and large binaries, then report branch, data, execution state, and follow-up status.
