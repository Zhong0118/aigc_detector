# Dataverse link reuse recovery — 2026-10-20

## Scope and authorization
User explicitly requested reuse of download links or batched ranges, preserving frozen selections and old data, and recovery only after sustained validation. Changes are in the attribution-pilot-baseline research worktree only. D0/A0 remain 12 epochs. No model or production text-service changes.

## Evidence and chosen change
Prior four-worker diagnostic succeeded on 484 full images, then received 403 at dataverse.harvard.edu (awselb/2.0, HTML 118 bytes). Exact rate policy is unknown. Per-image public requests unnecessarily repeated the entry redirect. Dataverse officially supports direct downloads through temporary presigned S3 URLs: https://guides.dataverse.org/en/latest/installation/big-data-support.html .

The existing PublicLinks helper now remembers successful HTTPS redirects from Dataverse to the observed official S3 host, per file ID, in memory only. It derives expiry from X-Amz-Date and X-Amz-Expires and falls back to the normal public GET 60 seconds before expiry. URLs are not persisted, printed or rewritten. No credentials, WAF bypass, disabled TLS or alternate egress retries are used. Unknown/failed redirects are not cached. Only public-entry requests retain a shared one-second interval; cached storage reads use the four workers without per-image pacing. 401/403/429 stops subsequent requests through the shared link object. Existing range length, Content-Range, ZIP member CRC, decoded image and file hash checks remain.

Added per-cell dataverse_requests, storage_requests and link_reuses metrics. Requests_attempted remains logical range requests, so redirects can produce both an entry request and a storage request. New unit tests cover reuse, expiry, untrusted redirects, URL-free statistics and shared fail-fast behavior. 43 relevant tests and git diff --check passed.

## Batching assessment
Offline estimate using max 64 KiB gaps and max 4 MiB merged spans reduces 42,000 member ranges to approximately 21,654, with 3,343 bytes of positive gap overhead. This is an approximation, not an implemented transfer plan; it does not handle split-volume boundaries or measure actual traffic. Link reuse is the selected minimal change for this recovery; archive member ordering and selection are unchanged. Evidence: outputs/iab-v2-monitor/range-coalescing-assessment.json.

## Validation and recovery gate
Superseded the earlier paced-only diagnostic, recording its identity and termination separately. New validation uses four workers and 1,000 full missing images: four real cells ×150 plus four generated cells ×100. It writes no dataset files and holds download.lock. Inputs remain the frozen v3 selection. Progress: outputs/iab-v2-monitor/link-reuse-validation.log; results: v3-link-reuse-validation.json; process identities: link-reuse-validation-process.json.

The initial diagnostic target was 1000, an agent-chosen target rather than a user requirement. The first segment reached its 15-minute budget after 945 full images passed (600 real +345 generated across all eight cells), 10 Dataverse entry requests, and 3 transient transfer failures recovered by retries. No access denial or CRC/decode failure occurred. After the user explicitly prioritized productive downloading, the agent accepted this completed evidence beyond the former 484-image failure scale and stopped the remaining standalone diagnostic rather than wait for a round number. This is 945 passed, not 1000. Acceptance and hashes: link-reuse-acceptance.json. The former failure state/log was archived under both supervisor.lock and download.lock; monitor --advance --once launched the formal downloader exactly once. All subsequent verification is part of acquisition with retained selected images.

Parent SHA256: ca0d44617c25375bbc498b320369bf9eb3c0f9e96d204c93d1edc01552481235.
V3 SHA256: 2ff3d56207b214f72924ed8a08a14e62aaf369562c4f13cd5555b14d61d6821e.
Formal download restored via monitor at approximately 17:59 +08. Latest measured progress is in outputs/iab-v2-monitor/link-reuse-live-progress.json. Dataset completeness and model-training results are still pending. Old failure evidence is preserved under recovery/link-reuse-resume-20261020T095851Z. The ten-minute automation has been updated to this policy.

## 18:12 +08 mixed acquisition scheduling
User required generated images to download alongside real images. The former explicit real-first queue occupied all four workers with 2000-image real cells. Replaced only acquisition scheduling with one shared four-worker pool, initially reserving two slots per side and reallocating idle slots when a queue ends. The frozen selection and sample identity are unchanged. Regression test simulates fast AI jobs while real jobs remain blocked, verifies AI continues progressing and every cell executes once. 41 related tests passed; shell syntax and diff checks passed. No extra network-only diagnostic was run.
Stopped only the identity-verified old downloader child; waited for its supervisor exit; archived state/log under supervisor.lock and download.lock; monitor alone relaunched. Evidence: outputs/iab-v2-monitor/mixed-queue-recovery.json. Switching baseline: 3156 AI +10573 real =13729 images.
At 18:09, productive byte throughput since recovery was about 0.408 MB/s. Remaining selected image bytes approximately 24.1 GB implied 16.4 hours if that rate persisted, excluding training. This was a real-image-only interval, not a reliable mixed-image forecast; update using subsequent retained mixed-image bytes. No count-based extrapolation from small real images to larger generated images. Download-complete validation still directly gates D0/A0 12 epochs and frozen OFA.

## 18:24 +08 productive direct/Clash comparison
User explicitly authorized testing the local Clash route using only retained production downloads. The actual child HTTP_PROXY and HTTPS_PROXY were verified as http://127.0.0.1:17898, tunneled to Mac 127.0.0.1:12321. Four workers and the two-real/two-generated schedule stayed fixed. Both windows excluded 30-second startup and measured 294.053 seconds of full image bytes retained in the unchanged selection. Samples were sequential and different, so this is operational throughput evidence, not a same-image controlled benchmark.

|Route|AI images/bytes|Real images/bytes|Total bytes/s|
|---|---|---|---:|
|Direct|99 / 155672519|343 / 45664152|684694|
|Clash|73 / 102148204|279 / 36117135|470205|

Direct was 1.456 times faster in this comparison and faster in each type separately. Both had no cell-level failures; proxy emitted at least six recovered request failures in partial counters, which are not a complete matched-window failure-rate comparison. All proxy-acquired images remain in the dataset. Chose direct, stopped only the verified proxy child, archived state/log under both locks, and relaunched only through monitor. Evidence: route-direct-production.json, route-proxy-production-live.json, route-comparison-result.json, post-route-comparison-progress.json. Failure/history backup: recovery/keep-faster-direct-20261020T102439Z.

Added explicit proxy mode to prevent silent direct fallback during measurements. Connectivity probe reduced to one byte, never used for throughput comparisons. Acquisition logs now emit cumulative per-cell progress each 25 verified rows when network reads occurred. 42 relevant tests and diff/shell syntax checks passed. Automation follows the selected direct route and productive byte growth; D0/A0 12-epoch and frozen OFA completion gates are unchanged.


## 18:36 +08 用户授权的8路直连正式落盘比较

保留原清单与所有图片，只将并发4改8，随后因吞吐下降退回4。两个窗口均排除启动30秒，统计240秒内完成的图片实际字节：4路100张AI/432张真实，209,370,256字节，872,376 B/s；8路46张AI/361张真实，100,632,597字节，419,302 B/s。8路约为4路48.1%。这不是相同图片的严格A/B，不能断言4路在所有网络时段总是最佳。两轮最新已输出的累计请求分别646/646、421/421成功，均零记录失败/超时，统计可能滞后且不精确对应测量窗口。没有图片校验/单格失败。8路进程已核验参数且没有代理变量；真实/生成各4路。

结果在outputs/iab-v2-monitor/concurrency-comparison-result.json；进程证据concurrency-8-process.json。两次切换均核验身份、停止原child并确认退出、持有supervisor/download双锁归档，然后仅通过monitor启动；恢复为4路直连，生成/真实各2路。旧日志在recovery/concurrency-8-*与concurrency-4-*，数据和selection SHA256不变。调度/缓存/监控24项测试通过。未修改生产或训练协议。
