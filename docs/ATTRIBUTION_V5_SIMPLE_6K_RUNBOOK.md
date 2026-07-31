# V5 多模型溯源数据采集：简化版 6000 条

## 目标

V5 只服务于多模型溯源分类训练：

```text
20 个具体模型 × 100 个提示词 × 3 次生成 = 6000 条
```

每个模型计划 300 条。训练准备标准不是一条不缺，而是每个模型至少有
95% 可用记录，即至少 285 条。

V5 不复用 V2/V3/V4 的旧样本，旧数据继续保留作流程验证或外部测试。

## 简化规则

- `kimi-k2` 替换为 `kimi-k2.5`。
- `mimo-v2-flash` 替换为 `mimo-v2.5-pro`。
- 短文本和中等文本使用 `max_tokens=4096`。
- 长文本使用 `max_tokens=8192`。
- 空文本、路由错配和 `finish_reason=length/max_tokens` 不计入可用样本。
- 再次运行时，成功样本自动跳过；截断等不可用样本会重新请求。
- 不在采集阶段强制要求文本严格满足目标字数，长度问题留到训练消融处理。

## 第一步：只测试两个替代模型

```bash
cd ~/projects/aigc_detector
conda activate zx
bash scripts/run_attribution_v5_probe.sh .env.local
```

只有看到下面这行才运行正式数据：

```text
V5 replacement probe passed: kimi-k2.5 and mimo-v2.5-pro
```

## 第二步：先运行第一个 600 条分片

建议在 tmux 中执行：

```bash
tmux new -s attribution-v5
conda activate zx
cd ~/projects/aigc_detector
bash scripts/run_attribution_v5_6k.sh .env.local 1 1
```

第一个分片包含10个领域、中文和英文、不同长度的提示词，是最终6000条的一部分。

## 第三步：继续剩余九个分片

```bash
bash scripts/run_attribution_v5_6k.sh .env.local 2 10
```

也可以一次运行全部：

```bash
bash scripts/run_attribution_v5_6k.sh .env.local 1 10
```

如果某个分片低于95%，脚本会保存该分片审计并显示警告，但不会停止后续分片。
鉴权失败仍会立即停止。已经成功的样本会自动跳过，不需要删除或重写 JSONL。

## 输出

```text
data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl
data/attribution/errors/zhizengzeng-attribution-server-6k-v5.errors.jsonl
outputs/attribution/server-6k-v5/shard_01.audit.json
...
outputs/attribution/server-6k-v5/shard_10.audit.json
outputs/attribution/server-6k-v5/final_audit.json
outputs/attribution/server-6k-v5/eda/
```

最终重点查看 `final_audit.json`：

```text
usable_successes
usable_coverage_rate
usable_by_model
model_usable_coverage
minimum_model_usable_coverage
```

95%现在是最终质量目标而不是采集阻断条件。即使低于该值，脚本也会生成完整
`final_audit.json` 和 EDA。完成全部分片后再根据全局结果建立定向修复清单。

## 速度调整

默认并发8、超时420秒。普通请求和空文本最多尝试3次；同一模型连续3条
空文本或瞬时网络/5xx失败后，才熔断该模型在当前分片内的剩余请求。
网络不稳定时可以降低并发：

```bash
MAX_WORKERS=6 TIMEOUT_SECONDS=600 \
bash scripts/run_attribution_v5_6k.sh .env.local 2 10
```
