# 多模型溯源数据集 V4 服务器运行说明

## 目标

V4 使用用户确认的 20 个模型，每个模型 30 条，最终严格组装为 600 条：

`118 条 V2 可复用样本 + 16×30=480 条新采样本 + 2 条截断修复样本 = 600 条`

V4 使用独立的 run ID 和输出路径，不覆盖 V2/V3。采集文件按
`request_key` 断点续跑；命令中断后原命令重跑即可，已经成功的请求不会再次调用 API。

## 第一步：40 条烟测

先进入项目和 Python 环境：

```bash
cd ~/projects/aigc_detector
conda activate zx
```

确认 `.env.local` 中存在：

```text
ZHIZENGZENG_API_KEY=你的密钥
ZHIZENGZENG_BASE_URL=https://api.zhizengzeng.com/v1
```

运行 20 个模型各 2 条的烟测：

```bash
bash scripts/run_attribution_selected_v4_smoke.sh .env.local
```

输出：

```text
data/attribution/raw/zhizengzeng-attribution-user-selected-smoke-v4.jsonl
data/attribution/raw/zhizengzeng-attribution-user-selected-smoke-v4.errors.jsonl
outputs/attribution/user-selected-smoke-v4/audit.json
outputs/attribution/user-selected-smoke-v4/eda/
```

脚本末尾包含严格烟测门禁。只有在成功数为 40，且无空文本、截断、
`route_mismatch=true` 或完全重复文本时才输出 `Smoke gate passed.`。否则脚本以
非零状态停止，不应进入 480 条采集。某个模型连续发生两次终态网络或 5xx
错误时会熔断该模型的剩余任务，避免慢接口拖住整轮；400/403/404 会立即熔断。

## 第二步：480 条增量采集、2 条修复和组装

假设原始 558 条 V2 数据位于：

```text
data/attribution/raw/zhizengzeng-attribution-server-600-v2.jsonl
```

运行：

```bash
tmux new -s attribution-v4
conda activate zx
cd ~/projects/aigc_detector
bash scripts/run_attribution_selected_v4.sh \
  data/attribution/raw/zhizengzeng-attribution-server-600-v2.jsonl \
  .env.local
```

在 tmux 中按 `Ctrl+b`，再按 `d` 可退出但保持任务运行；重新进入：

```bash
tmux attach -t attribution-v4
```

脚本依次完成：

1. 对 480 条配置和 2 条修复配置做本地 dry-run。
2. 并发采集 16 个模型，每个模型内部串行，默认并发 8。
3. 定向重采 Claude 和 MiMo 的两条截断样本。
4. 严格校验并组装 600 条；缺任何一条、出现重复或截断都会停止。
5. 生成数据审计、统计表和可视化。

主要结果：

```text
data/attribution/raw/zhizengzeng-attribution-incremental-batch-480-v4.jsonl
data/attribution/raw/zhizengzeng-attribution-incremental-batch-480-v4.errors.jsonl
data/attribution/raw/zhizengzeng-attribution-quality-repairs-v4.jsonl
data/attribution/raw/zhizengzeng-attribution-quality-repairs-v4.errors.jsonl
data/attribution/curated/zhizengzeng-attribution-server-600-v4.jsonl
data/attribution/curated/zhizengzeng-attribution-server-600-v4.assembly.json
outputs/attribution/server-600-v4-eda/
```

组装报告必须包含：

```json
{
  "expected_records": 600,
  "assembled_records": 600,
  "kept_from_base": 118,
  "added_replacements": 480,
  "added_repairs": 2,
  "status": "valid"
}
```

## 速度和故障参数

默认参数针对“尽快失败、可恢复重跑”设置：

- 全局并发数 8，但同一模型同时最多一个请求。
- 单请求超时 300 秒。
- 普通错误最多尝试 2 次。
- 400/403/404 立即停止该模型后续请求。
- 同一模型连续两次空文本、网络异常或 5xx 后停止该模型后续请求。
- 429 仍按退避重试处理，不触发网络熔断。

可按网关稳定程度临时调整，而无需改代码：

```bash
MAX_WORKERS=6 TIMEOUT_SECONDS=420 \
bash scripts/run_attribution_selected_v4.sh \
  data/attribution/raw/zhizengzeng-attribution-server-600-v2.jsonl \
  .env.local
```

如果脚本在严格组装处停止，不要删除 JSONL，也不要从头重采。检查
`.errors.jsonl`，修复路由或网络后原命令重跑，采集器会只补缺失的
`request_key`。
