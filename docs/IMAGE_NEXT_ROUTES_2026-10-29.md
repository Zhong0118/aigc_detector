# IAB next visual routes — 2026-10-29
User explicitly approved implementation, official model downloads, experiments, Git results and30-minute follow-up on2026-10-29. This document registers a new phase; cdd1b9530694b78117b4c9d1799ebd03d48162e2 and all prior experiments remain immutable.

## Scope and success criteria
Execute in order: FFHQ false-positive audit → official frozen OmniAID CLIP v2 → frozen OpenCLIP detection and attribution controls → unknown rejection comparisons. Main new head training budget30epochs, seed42, batch32, Adam .001, StepLR(1,.9). Frozen official model, KNN and threshold calibration have epoch=null. Finish only after actual inference/training, saved-model checks, same-ID comparison and Git aggregate results; preparation/tests are not model results.

## Fixed inputs and evaluation
Use outputs/image_experiments/iab-balanced-v7-200-q179 (63821=43821AI+20000real; nominal200/cell, actual187–200 after approved179 exclusions).
Manifest SHA9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3; selection SHA1c8dee2e17cc128867562186cee7bc4f8cb82042b26ddb6d5569d703ab77876b.
All old42000 IDs/splits/groups/labels/hashes fixed. Original train plus existing seed42/group development selection/calibration/evaluation roles, no resampling. Holdout never inferred/scored/tuned.
Predeclared OpenCLIP jobs: D-clean, D-held-FFHQ (same held-semantic protocol as D2-semantic), A22-linear, A22-cosine-KNN, A20-open. Additional held-FFHQ detector is necessary to answer whether the new representation improves the diagnosed generalization failure; it is not a new dataset.
KNN uses only training exemplars, k1..20 selected on selection macroF1 with smallest-k tie; no gradient epochs. Open20 removes kling/grok3 from train/selection; unknown calibration only kling, evaluation only grok3. Compare MSP acceptance and cosine similarity to train-only known centroids, both calibrated to <=5% acceptance on unknown calibration. Report known coverage, known correct-and-accepted, false acceptance/rejection, AUROC, Wilson and paired group intervals; no transfer to the22-way deployed interface. Standard frozen CLIP features+new heads are adaptations, not faithful reproduction of UnivFD training.

## Implementation and preservation
New scripts/configs/modules only, reuse existing partition_rows/evaluate_scores/fit_head if compatible, no unrequested core refactor. FFHQ audit consumes saved scores and nonholdout image metadata, checks dimensions/format/crop and existing preprocessing identities; associations do not prove semantic shortcuts. No evaluation-driven threshold revision.
OmniAID uses previously pinned official CLIP v2 source/config/checkpoint/base assets and source hash review. Safe weights_only loading, offline safetensors tensor equality, strict keys, source/license evidence; no unsafe fallback. Actual2selection images verify complete reload and single/batch consistency<=1e-5. New ready config must bind asset/manifest/receipt hashes; old disabled config unchanged.
OpenCLIP uses official ViT-L-14 OpenAI checkpoint, correct QuickGELU and standard RGB resize/center-crop224 pipeline. Pin implementation/dependencies separately from existing .venv-image, bind exact assets and preprocessing. Freeze encoder, extract once only nonholdout, bind cache to all ordered IDs and input hashes. Head selection metric AUROC for detection, macroF1 for attribution. Full reload/image-vs-cache checks required before completed status.
Unique new outputs under outputs/image-next-routes-20261029 and model-specific new experiment roots; create exclusively. Existing failed runs remain failed; no blind retry or overwrite.

## Resources and launch
Only ssh school, research worktree/branch attribution-pilot-baseline. No production/text/service changes, no process termination, no GPU1 appropriation. PhysicalGPU0 idle<=1024MiB plus shared outputs/image_experiments/gpu0-research.lock required. Sequential pipeline, boot/start/cmdline identity recorded, no duplicate workers. Current free disk77GiB; check >=20GiB after asset budget before launching. Normal TLS/known host keys, no paid APIs or access-control bypass. Official weights downloaded once with hashes, preserve partials on failure. If a gate fails, preserve diagnostics and stop dependent stages.

## Delivery and limitations
Small source/config/tests/docs plus completed aggregate metrics/history/confusion/per-source/paired results and SHA provenance in timestamped docs/image_research_results snapshots. No models/images/dataset/features/per-image big predictions/credentials/signed URLs in Git. Sensitive/size/JSON/CSV/SHA/diff checks before normal push to same branch; never sync/force-push.
Single seed, already-viewed developer evaluation, prior parent-dev model-selection bias and unknown official pretraining overlap mean exploratory comparison, not independent external performance or production-readiness. Any future independent test is separate authorization/protocol.

## Execution ledger
-2026-10-29: human approved sequence and autonomous implementation; no repeat approval required for routine design choices.
-Existing tree clean at cdd1b95, both GPUs idle,77GiB free.
-New heartbeat ffhq active every30minutes; previous completed automations stay deleted.
-FFHQ audit / OmniAID assets / OpenCLIP implementation delegated to separate file owners under subagent-driven-development. CPU preparation may overlap; GPU experiments remain ordered.

-FFHQ audit completed: authoritative outputs/image-next-routes-20261029/ffhq-audit/verified/. Reproduced56/93 vs same-ID1/93; all93FFHQ and100celebaHQ RGB8 PNG256 with matching measuredheaders. No confirmed pixelcause. D1-control/D3 actualreal-eval probabilitiesidentical, not independentreplications. 39combinedtests passed. OmniAID first download timeoutpreserved; boundednetworkdiagnosis ongoing; no modelrunyet.

## Serial execution and follow-up
Entry: `CUDA_VISIBLE_DEVICES=0 .venv-image/bin/python scripts/run_image_next_routes.py --execute`.
This is a one-shot queue, not a restart command. Its unique state directory is `outputs/image-next-routes-20261029/queue`; inspect `launch.json`, `registration.json`, `status.json` and each stage's launch/log before acting. The queue stops on any failed child or changed registered source/config. It preserves every child identity and exit code. `awaiting_git_archive` means calculations finished, not delivery completed.

Order: verified official asset receipts → CPU safe conversion → two actual selection-image OmniAID reload checks → frozen OmniAID evaluation → all five OpenCLIP jobs → six closed comparisons → open-set comparison. The final OpenCLIP open20 job trains its head for 30 epochs and evaluates both MSP and train-prototype cosine rejection; KNN and the frozen detector have no training epoch. GPU stages independently acquire the existing shared research lock and recheck physical GPU 0 availability. No original holdout scoring is permitted.

Outputs:
- Frozen detector: `outputs/image_experiments/omniaid-v2-q179-20261029`.
- OpenCLIP: `outputs/image_experiments/openclip-q179-20261029`, one subdirectory per registered job.
- Paired reports: `outputs/image-next-routes-20261029/{closed-comparison,rejection-comparison}`.
- Asset transport receipts: `outputs/image_experiments/assets/omniaid_v2`. Original school direct timeout and subsequent interrupted SSH stream evidence remain archived. A single explicit official HTTPS Range recovery reuses the SHA-checked archived prefix; final complete source SHA remains mandatory. The downloader never performs conversion or model inference; those belong to the serial queue.

Code/guard tests and the OpenCLIP CPU strict weight-load receipt are preparation evidence only. No new model score is established until the corresponding completed runtime status and saved predictions pass validation. The enabled OmniAID config legitimately transitions after conversion and actual-image verification; each transition is copied and hashed in queue evidence. It should be committed with subsequent results after review, without changing the historical initial config copy.

- New-route preflight: 103 tests passed (queue, OpenCLIP, role-matched comparisons, rejection, OmniAID and FFHQ audit). The independent review found no remaining Critical/Important issue after argument-lock, asset-reference and child-identity fixes. OpenCLIP official checkpoint passed complete SHA/size and CPU strict-load checks; no actual new GPU model result yet.
