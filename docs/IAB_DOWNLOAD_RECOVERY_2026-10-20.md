# IAB v2 download recovery — 2026-10-20

Research worktree: `/home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline`. Branch: `attribution-pilot-baseline`.

- Verified local Clash configuration and HTTP CONNECT: mixed proxy port 12321. Port 49621 responds 404 to CONNECT and is not the proxy.
- Recreated dedicated loopback reverse tunnel: server 17898 → local 12321, with SSH keepalive and forwarding failure detection. No unknown listeners were terminated.
- Direct and proxy each passed 6 public Range probes with concurrency 2.
- Proxy then passed 24 full selected-member downloads, ZIP CRC and image decoding at concurrency 2: 24/24 requests, 0 failures; 62.96 seconds. This is a small validation sample, not a full-dataset throughput guarantee.
- Changed only monitor download concurrency from 16 to 2. No training protocol or selection changes. Existing uncommitted code retained.
- Backed up pre-change scripts, previous failed state and log at `/home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline/outputs/iab-v2-monitor/recovery/20261020T074016Z`. Archived old failure file so it is not counted as a new-attempt failure.
- Under supervisor/download locks, confirmed previous worker stopped, archived its failed marker, then launched through monitor --advance --once. Did not start a competing downloader.
- Original selection SHA256: `ca0d44617c25375bbc498b320369bf9eb3c0f9e96d204c93d1edc01552481235`; unchanged after resume. V1 untouched.
- Verification at 2026-10-20T15:45:06.571624+08:00: 8298 images, 234 new sidecars rehashed, 0 new cell failures, process alive. See `outputs/iab-v2-monitor/recovery-verified.json` for exact evidence.
- 33 acquisition/selection/manifest/monitor tests passed; git diff --check and bash syntax check passed.
- Existing ten-minute iab-v2 heartbeat updated with tunnel repair and evidence-gated recovery instructions. Download completion still gates manifest validation and one D0/A0/OFA pipeline. No experiment results are claimed.

The SSH tunnel and proxy depend on this Mac remaining online. Prior failure logs cannot establish that every failure was HTTP 403; current tests demonstrate successful HTTP 206 but do not prove high-concurrency rate limiting was the sole cause.
