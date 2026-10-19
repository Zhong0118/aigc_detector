# ImageAttributionBench V3.2 balanced server baseline

This run uses the publisher's images from DOI [10.7910/DVN/O4S4IV](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O4S4IV&version=3.2). No generation API is called. This is a development baseline, not a full-data reproduction or a deployment qualification.

## Acquisition and coverage

The public inventory has 350 files / 528,670,084,577 bytes, including one unlabeled `_test_imgs` image that is excluded. The 230 source/semantic archive catalogs contain 457,565 addressable image members, including 20,000 real images. This is a directory-level count, not a claim that every published image was downloaded or decoded.

The 22 generator folder labels are preserved verbatim: `4o`, `CogView3_PLUS`, `FLUX`, `KANDINSKY`, `PIXART`, `PLAYGROUND_2_5`, `SD1_5`, `SD2_1`, `SD3`, `SD3_5`, `SDXL`, `dalle3`, `gemini`, `grok3`, `hidream`, `hunyuan`, `ideogram`, `infinity`, `janus-pro`, `kling`, `mid-5.2`, `mid-6.0`. Semantic categories: COCO, FFHQ, ImageNet, bedroom, cat, celebahq, church, classroom, dog, wild. Only the spelling `ImageNet-1k` is normalized to `ImageNet` for semantic grouping; source labels are not substituted.

Target: 50 images per generator/semantic cell = 11,000 AI images; 1,100 real per semantic category = 11,000 real images. Each generator is 1/22 of AI samples; AI and real each occupy 50% overall. Images are selected from verified common publisher prompt IDs in each semantic category with seed 42, one image per prompt/source. This aligns prompt composition across sources and prevents sibling images from crossing partitions. Prompt text is not included in the archives, so cross-source prompt-ID alignment remains a publisher-naming assumption.

**Release gap:** `SDXL/FFHQ/SDXL_FFHQ.zip` (file ID 11193787) declares ZIP disk 1, but the V3.2 inventory omits its `.z01` disk 0. Of its 2,000 catalog entries, 1,606 start in the missing volume and 394 are addressable in the last volume. Only complete last-volume members may be sampled. This leaves 93 common FFHQ prompt IDs across all 22 generators. The 50 FFHQ prompts are sampled from those 93; this is not uniform sampling of the full original FFHQ prompt population. Other publisher cells also contain fewer than their nominal 2,000 images.

HTTP Range downloads avoid fetching all 529 GB. Python `zipfile` reads central directories; stored/deflated members are bounded, decoded and checked against their ZIP CRC32. Per-image SHA256, decoded-pixel hash, pHash, dimensions, source directory, official file IDs and member paths are retained. **Whole-volume publisher MD5 cannot be verified for AI partial range downloads**, and is explicitly recorded as unverified. Real-image selection takes 55% of each archive, so those ten small archives are downloaded/reused in full and their publisher MD5 is verified; the already verified real/dog archive is hard-linked and reused. The selected real members are then decoded locally on the server. Real full-archive caches add about 2.08 GB of disk usage. Public download links expire after one hour; an atomically refreshed private link file is reloaded by active workers without changing image selection. Temporary signed public download URLs stay in ignored, permission-restricted acquisition files and are excluded from Git.

## Fixed development protocol

The target subset is 22,000 images (about 14.33 GB of original image bytes). Actual completed counts and failures must come from `download_status.json`, receipts and `audit.json`, not this target.

Group together identical publisher semantic/prompt IDs across generators, identical bytes/pixels and pHash distance <=4 (transitive union). Five disjoint pHash bands provide an exact candidate filter for that distance threshold. Split groups 70/15/15 with seed 42, stratifying by semantic category and source coverage. Grouping may change exact partition sizes; record actual counts per source, semantic category and partition. No official test set is used, and the local holdout is not used for selection or evaluation.

50 images per cell gives 500 images per source before splitting, roughly 350 training and 70 development images per source. This is enough to exercise acquisition, training, reload, inference and a first comparison; it is too small for a strong 22-way full-network generalization claim. Expanding training later must preserve partition/group boundaries and requires a new manifest revision.

## Separate tasks

- **D0 truth detection:** a separate two-output ResNet50 with ImageNet V1 initialization; real/AI cross-entropy, Adam 0.001, StepLR(1, 0.9), batch 32, 20 epochs. Select by development AUROC, earliest tie. Report AUROC, accuracy, macro-F1, real FPR and AI TPR at 0.5; any development-calibrated low-FPR threshold is exploratory.
- **A0 attribution:** the ImageAttributionBench ResNet50 end-to-end architecture/training recipe, with a separate 22-output linear head and AI-only samples. Same optimizer, scheduler, batch and epochs; select by development macro-F1, earliest tie. Report top-1, top-3, macro-F1, per-source and per-semantic results. Real is not an attribution class.
- **OFA attribution:** complete frozen author model: DnCNN residual, FFT magnitude/log1p, ResNet50 fingerprint encoder, learned asymmetric reference/query pair head. No CLIP substitution, cosine prototype substitution or training of the original reconstruction simulator. Select up to 100 train references per source, round-robin over semantics with seed 42. Evaluate on exactly the A0 development AI query IDs. Report source-verification AUROC separately from the closed-set argmax top-1/top-3/macro-F1. Pair match scores are not calibrated multiclass probabilities.

ResNet uses RGB, short-side resize 256, center crop 224 and ImageNet normalization. OFA uses its author's 256 center crop and normalization/denoiser/FFT pipeline. The IAB author loader additionally crops Grok images to `(0,0,width-100,height-50)` conditioned on the known true source label. This baseline deliberately uses source-independent preprocessing for single-image inference and **does not reproduce that source-conditioned crop**. Together with the custom grouped subset/split and AI-only output classes, this prevents claiming exact reproduction of the paper's headline results. Raw watermarks and dataset artifacts may remain predictive; no semantic-free or artifact-free conclusion is warranted.

## CUDA numerical verification

The initial CUDA regression failed its single/batch consistency check. Layer-wise diagnosis found default cuDNN TF32 caused a 0.00391 denoiser-residual difference and 0.48382 fingerprint-feature difference on the checked image. Disabling TF32 reduced but did not eliminate batch-dependent residual cancellation. Fixed denoiser microbatch 1 with FP32 reduced feature error to 9.54e-7 and pair score error to 2.68e-7 in the diagnostic. The full saved-bank/reloaded-author-weights regression then passed (score difference 5.96e-8). Failed runs/logs remain preserved as failed and are not main experiment results.

Both TF32 paths are disabled in the experiment runner. OFA's denoiser always uses microbatch 1; its encoder uses the requested outer batch. No weight or mathematical model change is made. Model/code hashes, environment, manifest identity, reference bank, predictions and exit status are retained for each run.

## Locations and commands

All paths below are relative to the isolated server worktree `/home/zhongxu/projects/aigc_detector/.worktrees/image-m0-m2-paper`.

- Raw subset, archive catalogs, receipts and frozen manifest: `outputs/image_experiments/iab-balanced-v1/`
- Acquisition/diagnostic logs: `outputs/migration-20261019/`
- Author/ImageNet pretrained weights: `outputs/image_experiments/assets/`
- Planned completed experiment directories: `d0-resnet50-iab-v1`, `a0-resnet50-iab-v1`, `ofa-iab-v1` under `outputs/image_experiments/`.

Use the isolated `.venv-image/bin/python`, `LD_LIBRARY_PATH=$PWD/.venv-image/lib`, and GPU 0 only. The production worktree and existing text service are unchanged. A directory existing does not prove an experiment finished: require `status.json` with `status=completed`, saved/reloaded inference checks and metrics from that exact manifest.

The earlier Tiny full-download/training queue was cancelled to prioritize this requested IAB comparison. Verified Tiny shards and resumable partials are retained but do not contribute to IAB results. No Tiny full-data training result is claimed.

`scripts/run_iab_m2.sh` runs the three tasks after completed acquisition. `scripts/summarize_iab_m2.py` verifies matching development query IDs and manifest identities, then reports the fixed 0.5-threshold detection→attribution cascade separately from the independent tasks. Missed AI images remain in the cascade denominator; real false alarms are counted separately. This does not establish unknown-source rejection.
