# 多模型溯源数据集 V3 定向替换运行说明

本轮不重跑原有 600 个请求。原始 V2 成功数据保持不变，只新增：

- 3 个替代模型各 30 条，共 90 条；
- 2 条截断样本和 2 条重复样本的定向修复，共 4 条。

合并器从原 558 条中保留 506 条，加入 90 条替代数据和 4 条修复数据，
最终严格得到 600 条。任何缺失、意外模型、重复请求键、空文本、路由漂移
或截断都会使合并失败，不会写出不完整的最终数据集。

## 已冻结的替代模型

| 家族与角色 | 退役模型 | V3 模型 | 依据 |
|---|---|---|---|
| OpenAI fast | `gpt-4.1-nano-2025-04-14` | `gpt-4o-mini-2024-07-18` | 先前探测已通过 |
| DeepSeek balanced | `deepseek-v4-pro` | `deepseek-chat` | 本轮 2/2 成功且候选中最快 |
| GLM fast | `glm-4.5-air` | `glm-4-flashx-250414` | 本轮 2/2 成功且候选中最快 |

候选探测总计 12/12 成功，空文本、截断和路由漂移均为 0。

## 服务器准备

上传项目代码，并把原始 558 条数据放到：

```text
data/attribution/raw/zhizengzeng-attribution-server-600-v2.jsonl
```

在项目根目录创建 `.env.local`，不要把密钥提交到 Git：

```dotenv
ZZZ_API_KEY=你的密钥
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
```

进入可写的 Conda 环境，然后先检查两个采集计划：

```bash
python -m scripts.collect_attribution_run \
  --config configs/text_attribution/runs/replacement_batch_90_v3.json \
  --dry-run

python -m scripts.collect_attribution_repairs \
  --manifest configs/text_attribution/repairs/server_600_v2_quality_repairs.v3.json \
  --dry-run
```

第一条必须显示 `request_count: 90` 和 `requests_per_model: 30`；第二条必须
显示 `request_count: 4`。

## 在 tmux 中运行

```bash
tmux new -s attribution-v3
conda activate zx
cd ~/projects/aigc_detector
bash scripts/run_attribution_targeted_repair_v3.sh \
  data/attribution/raw/zhizengzeng-attribution-server-600-v2.jsonl \
  .env.local 2>&1 | tee outputs/attribution/targeted-repair-v3.log
```

按 `Ctrl+B`，再按 `D` 可退出 tmux；使用
`tmux attach -t attribution-v3` 返回。

采集器支持断点续跑。网络故障或个别失败时，直接重复同一条脚本命令；已经
成功写入的请求键会自动跳过，不会从头重新采集。

## 最终结果

脚本成功结束后会生成：

```text
data/attribution/raw/zhizengzeng-attribution-replacement-batch-90-v3.jsonl
data/attribution/raw/zhizengzeng-attribution-quality-repairs-v3.jsonl
data/attribution/curated/zhizengzeng-attribution-server-600-v3.jsonl
data/attribution/curated/zhizengzeng-attribution-server-600-v3.assembly.json
outputs/attribution/server-600-v3-eda/
```

合并报告中的 `status` 必须为 `valid`，且应满足：

```text
assembled_records = 600
kept_from_base = 506
added_replacements = 90
added_repairs = 4
```

最终 JSONL 中，修复样本同时保留逻辑 `request_key`、原采集
`collection_request_key`、`supersedes_request_key` 和 `source_run_id`，
可以完整追溯其替换来源。
