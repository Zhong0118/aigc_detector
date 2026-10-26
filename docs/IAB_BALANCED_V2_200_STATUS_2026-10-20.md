# IAB Balanced v2 200-per-cell run status

**Date:** 2026-10-20  
**Server branch:** `attribution-pilot-baseline`  
**Worktree:** `/home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline`  
**Dataset target:** ImageAttributionBench V3.2, Dataverse dataset 11127188

## Frozen selection

The independent `iab-balanced-v2-200` selection was generated with seed 42 from the cached V1 archive catalogs. It preserves V1's selected members as the prefix and deterministically fills each generated-source/semantic cell up to 200 without replacement.

- Generated images: 22 sources × 10 semantic categories × 200 = **44,000**; all 220 generated cells reached 200 in the available catalog.
- Real images: **2,000 per semantic category**, 20,000 total. The catalogs contain 2,000 real members per category, below the requested 4,400; no images were duplicated.
- Frozen selection: **64,000** rows total across 230 source/semantic cells.
- Selected member bytes: **55,045,246,179 bytes** (about 55.0 GB decimal / 51.3 GiB). This is a selection estimate, not a claim that these files were downloaded.
- V1 remains at its original path and was not overwritten.

## Acquisition attempt

The monitor started one 16-worker acquisition process. It reused all 230 cached catalogs and began fetching only selected members through the public Dataverse Range endpoint. The first direct 1 MiB probe returned HTTP 206 but took 77.8 seconds; an earlier probe took 3.9 seconds, so observed direct throughput varied substantially. The configured forward-proxy probe failed with `ProxyError` and was not passed to the downloader.

Selected-file Range requests then returned HTTP 403. A follow-up ordinary HTTPS Range check on three affected public file IDs also returned 403 for both a one-byte range and the ZIP-tail range. The downloader recorded 230 failed source/semantic cells, zero successful cell receipts, and zero completed request bytes in those final failures. The worker exited with code 1.

There are currently 8,064 image files and 8,064 sidecar files from partial cell work, but no complete cell receipts, `download_status.json`, frozen sample manifest, or training run. These partial files are not a completed dataset and were retained for inspection. No credentials were copied, TLS verification remained enabled, and no access control was bypassed.

## Experiment and monitor status

D0 (ResNet50 truth detection), A0 (ResNet50 generated-source classification), and V1/OFA (reference-based source verification) have **not run**. No metrics or model claims are available for this dataset.

The branch contains the selection, acquisition, manifest, per-epoch metrics, monitor, and test changes. The server worktree has a private Python environment and copies of the existing ResNet50/OFA weights at `outputs/image_experiments/assets/`; these environment/model files are ignored by Git. The v1 dataset and prior runs remain untouched.

The monitor records transport probes, disk/GPU state, worker PID identities, locks, and logs. It will not automatically restart a failed download or training worker. The 10-minute Codex heartbeat automation is active and will report the failed state without relaunching it; it is configured to stop after a validated experiment summary exists.

## Verification and blocker

The relevant image tests pass: **57 passed**. Python compilation, `bash -n scripts/run_iab_m2.sh`, and `git diff --check` also pass.

The remaining blocker is HTTP 403 from the data service for selected file Range requests, with the configured proxy path unavailable. Acquisition and training remain gated until an authorized public route returns valid Range responses.
