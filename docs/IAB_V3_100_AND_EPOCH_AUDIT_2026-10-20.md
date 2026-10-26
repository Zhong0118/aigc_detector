# IAB 100-per-cell decision and epoch audit — 2026-10-20

User explicitly selected 100 generated images per source × semantic cell. The independent iab-balanced-v3-100 selection takes fixed prefixes of v2, without resampling or removing old files. Target: 22,000 generated + 20,000 real = 42,000. Real data is capped by actual archive availability (2,000 per semantic category). Network errors are not availability shortages. Selected original image bytes: 28,530,713,675. 10,591 old images were hash-verified and hardlinked.

Parent selection SHA256: ca0d44617c25375bbc498b320369bf9eb3c0f9e96d204c93d1edc01552481235.
New selection SHA256: 2ff3d56207b214f72924ed8a08a14e62aaf369562c4f13cd5555b14d61d6821e.
Configuration: configs/image_experiments/iab_v3_100_run.json; active copy outputs/iab-v2-monitor/config.json. Four workers, direct route. Production sync and text services were not changed.

## Current failure, not a completed recovery
Eight-worker formal download failed with 403. Four-worker direct validation passed 32 complete images with CRC/decode, but subsequent formal download also failed. At 17:02 +08, v3 contains 11,075 image/sidecar pairs: 3,156 generated and 7,919 real, zero completed cell receipts of 230. Latest attempt: 716 requests, 484 successes, 232 failures, zero timeouts. Image increase over reuse baseline: 484. Failed worker exited. Failed state/log preserved, not reset. Subsequent single public GET requests returned normal 303 redirects to S3 and 206 responses. This does not establish sustained recovery or a confirmed root cause. See outputs/iab-v2-monitor/v3-latest-audit.json and v3-failure-diagnosis.json. Historical log contains older attempts; the audit isolates the latest attempt after its last catalog event.

The existing ten-minute automation remains active and was updated with the new target and failure evidence. Do not blindly restart on one successful small probe. No training has run on v3 yet.

## Text epoch evidence
Read-only existing logs under /home/zhongxu/projects/aigc_detector/outputs/attribution/family-phase3/A0/A0-closed-seed{42,52,62}/trainer_history.json:

| Seed | Epoch 3 dev macro-F1 | Best dev macro-F1 | Best epoch |
|---|---:|---:|---:|
|42|48.78%|56.50%|4|
|52|54.13%|60.66%|5|
|62|50.32%|55.69%|6|

Mean improvement from epoch 3 to the best recorded epoch: 6.54 percentage points. These are within-run development curves under a six-epoch schedule, not a separately controlled three-versus-six-epoch experiment or test results. See outputs/iab-v2-monitor/text-epoch-audit.json. Six is the configured limit, with patience two; three is not universally sufficient.

## Image epoch references and our protocol
ImageAttributionBench paper appendix: ResNet50 standard protocol uses 20 epochs. Default other methods use 10 standard / 15 semantic epochs, with exceptions. Paper: https://arxiv.org/html/2605.12967v1 . Its pinned resnet50.yaml instead lists nEpochs:30, while README examples use 10; document this discrepancy rather than conflating them. Config: https://raw.githubusercontent.com/mttry/ImageAttributionBench/19bb7d805142d30e67689dae009ece170866c5f3/training/config/model/resnet50.yaml .

OFA paper https://ojs.aaai.org/index.php/AAAI/article/download/40851/44812 does not specify a fixed epoch count found in this audit. Author training code defaults to 100000 optimizer steps, not epochs: https://raw.githubusercontent.com/jumpycat/OFAAttribution/3780cbffeeac48c0bf5e72a025f4803e408b9dfa/train_noiser_fft_bank.py . Our V1 uses frozen author weights and records epoch:null; no training epochs.

User subsequently reduced D0 and A0 to fixed 12-epoch, batch-32 configurations, selecting by development metrics without holdout tuning. This is a user-requested engineering budget change from the paper’s 20-epoch ResNet50 protocol, not an exact reproduction of that training budget. A0 uses the ResNet50 source-classification method with the reduced training budget; D0 is an independent binary adaptation. Download workers 8→4 does not change training epochs. GPU0 runs the jobs sequentially; GPU1's existing Ollama service remains untouched. Prior short GPU0 batch-32 FP32 measurement peaked around 3.64 GiB total GPU memory and is not a completed training run.

## Validation
42 tests passed across prefix selection, selection, monitor, delivery, manifest, epoch metrics, protocol and summary tests. git diff --check and bash -n scripts/run_iab_m2.sh passed. Tests cover fixed-prefix selection and exact worker output/manifest matching, so a reuse-from parent or another worktree cannot cause false duplicate-worker detection. These checks validate code behavior, not network stability or experimental performance.

## 17:28 +08 follow-up: sustained failure reproduced
Four-worker full-image diagnostic verified 484 images in 152 seconds, then received HTTP 403 at dataverse.harvard.edu (awselb/2.0, 118-byte HTML), not an observed S3 signature error. The diagnostic stopped on the first denial. Evidence: outputs/iab-v2-monitor/v3-sustained-range-diagnosis.json. This is consistent with an ingress request-rate limit, whose precise policy is not known.
Added a shared global public-request interval to the existing downloader (CLI --request-interval, default 1 second); retained four connections. 36 relevant tests passed, plus shell syntax and diff checks. A separate 600-image, four-worker, globally paced validation is running under download.lock, tracked by paced-validation-process.json; it writes no dataset files and does not reset the failed formal download state. Results will be v3-paced-range-validation.json. Only all 600 passing plus process exit authorizes the existing locked monitor recovery flow. Current dataset count remains 11,075; diagnostic samples are not additional dataset samples. This is a candidate repair, not a claimed sustained recovery.


## User scope change: v4-20 (2026-10-20)

The current run now uses iab-balanced-v4-20: 22 generator sources × 10 semantics × 20 images = 4400 AI, with 440 real per semantic = 4400 real. Every AI source, including 4o, has exactly 200 selected images. Old v1/v2/v3 files and selections remain; old 4o surplus is excluded from the new selection. Selection is a fixed prefix of v3-100, SHA256 c822f03e5c691f4a86a01af62b9fd85116bbdbe7c4ad25cdbcb9ea882eb7e562. At transition, 1718 AI and 3650 real were SHA256-checked and hard-linked. Evidence: outputs/iab-v2-monitor/v4-20-switch.json and v4-20-latest-audit.json. Runtime config and heartbeat now target v4-20. Acquisition starts only through monitor; 4 direct workers continue. D0/A0 remain independent ResNet50 binary/22-source runs, 12 epochs each; OFA remains frozen source verification. This is a small engineering pilot, not SOTA or a full paper reproduction. D0 uses the same IAB subset relabeled real=0/AI=1; no separate GenImage training or cross-generator holdout has been added.
