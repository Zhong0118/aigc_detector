# IAB next routes implementation plan
> Execute approved tasks in this research worktree, with subagent-driven-development and independent review.
Goal: finish diagnostic and preregistered200-q179 comparisons and archive verified results.
Architecture: independent wrappers around unchanged split/evaluation primitives, pinned local assets, frozen encoders, exclusive outputs and serialGPU0 queue.
Tech: Python/PyTorch/torchvision/Pillow/scikit-learn, existing .venv-image plus isolated OpenCLIP environment.
Spec: docs/IMAGE_NEXT_ROUTES_2026-10-29.md

- [x] Task1 scripts/audit_ffhq_errors.py + tests/test_ffhq_audit.py: first fail tests for holdout/role/ID mismatches; implement bounded metadata audit; run actual saved-score audit; independently verify56/93 andmatchedcontrol counts; archive aggregate report.
- [x] Task2 newOmniAID preparation wrapper/config/tests: download exactly pinned assets, verify SHA/size, weights_only conversion/equality, review source; execute strict fullload under sharedlock only aftertask1; actualselection-image consistency; explicit ready config and frozen evaluation. Never modifyold disabled config.
- [x] Task3 isolatedOpenCLIP adapter/runner/config/tests: pinnedofficial library andweights/preprocess, cache provenance and leakage guards tested before extraction; use same30epochheadrecipe; D-clean/D-heldFFHQ/A22-linear/A22-KNN/A20-open, selection-only model/kchoice, calibration-only MSP/prototype thresholds.
- [x] Task4 serialqueue: GPU0idle+lock+PIDidentity, exclusive output, failurepreservation, no retry/resume of failed or existing queue state; no duplicate inference. Launch only reviewed code.
- [x] Task5 completionaudit: recompute metricsfromsaved evalscores, compareoriginalsameIDs/classorders/roles/groups,1000pairedgroupbootstrap; qualifyknowncoverage/FFHQ/generalization/pretrainingoverlap. NewtimestampedGitresult snapshot, validate copiedSHA/JSON/CSV/no secrets/no binary/size/diff, normalcommit+push.
- [ ] Task6 maintainffhqheartbeatwithrealstate/paths/hashes/remainingwork; deleteonlyafterallregisterednewjobsandfinalGitdeliverycomplete.

Completion evidence: docs/image_research_results/2026-10-29T072357Z-next-routes-final/README.md and acceptance/verification.json. Models and reports are complete; Task6's final deletion follows the ordinary Git delivery verification. The original queue failure was preserved and the explicit Namespace recovery used a separate queue and verification paths.
