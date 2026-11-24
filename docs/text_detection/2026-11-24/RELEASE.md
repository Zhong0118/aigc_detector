# 2026-11-24 文本检测发布记录

验收时间 2026-11-24T11:34:25.150222+00:00。API 8611 的 PID 为 2842921，工作台 8612 的 PID 为 2842923。
切换到两个服务健康约 3.5 秒。实测期间 API 与工作台均在 CPU 运行。

本地仓库同步两项防护。

- `ec95bb0` 在未校准分支的报告和看板隐藏不可比风险分。
- `b58569a` 在证据不足或阈值为空时跳过文本归因，覆盖 HTTP 500 回归。

## 英文 en-v2 统一校准正式切流

英文长短路由共用 en-v2 权重和 `en` 校准组。温度为 1.272868，严格线 93.9149%，筛查线 90.7749%。中文权重、窗口、温度、严格线、筛查线及归因配置保持原值。合并后的路由签名覆盖中英文实际权重和温度。

| 正式实测 | 字符数 | 分数 | 严格结论 | 筛查 | 归因状态 | 报告 ID | 用时 |
|---|---:|---:|---|---|---|---:|---:|
| 英文真人论文正文 | 3737 | 14.53% | `low_risk` | 未命中 | `skipped` | 145 | 0.670 s |
| 英文 AI 论文正文 | 3597 | 98.03% | `high_risk_ai` | 命中 | `ok` | 146 | 0.704 s |
| 英文真人学术摘要 | 736 | 18.91% | `low_risk` | 未命中 | `skipped` | 147 | 0.159 s |
| 英文筛查样本 | 1113 | 91.71% | `uncertain` | 命中 | `ok` | 148 | 0.311 s |
| 英文短 AI 样本 | 368 | 97.00% | `high_risk_ai` | 命中 | `ok` | 149 | 0.572 s |
| 中文 3,531 字原报告 | 3518 | 95.64% | `high_risk_ai` | 命中 | `ok` | 142 | 10.620 s |
| 中文真人长新闻 | 1527 | 75.94% | `uncertain` | 未命中 | `ok` | 143 | 0.988 s |
| 中文短 AI 样本 | 186 | 96.18% | `high_risk_ai` | 命中 | `ok` | 144 | 0.539 s |

中文三条请求在切流前后分数、阈值组、严格结论、筛查标记及归因一致。工作台 AppTest 验证了校准分数及复核标签。正式历史库按报告 ID 验证追加成功，已有记录哈希一致。临时端口 18611/18612 已释放。

英文短句及部分前沿生成器的召回偏低。离线口径 A 在原英文短文上筛查漏检 28/48，在 RAID+ 冻结样本上筛查漏检 12/80。低风险结果需结合原文及写作过程复核。英文归因沿用现有配置，其候选家族仅供参考。

## 配置与回退

- 发布目录 `/home/zhongxu/projects/aigc_detector/deployments/text-binary-20261124-en-v2a`。
- 合并评分协议 `token-logit-v1:2e084c9491123921`。
- 正式配置 SHA-256 `401a997f26578394f2be4f5d36047761b7e3eb47d9369d154b72077ccf4d4016`。
- 英文权重 SHA-256 `11e172d0550e884f3ebfe34a7fe6db08c4c66b758736714943e305dbc382789b`。
- 合并阈值 SHA-256 `2b56f5f5baa98462ce95c7143cbdd536064d5d3c4ac758fe148145355e864d03`，合并温度 SHA-256 `2fbefb99be6a1895e352ccf3bc817a6158197ce126455fbc80a85df1b0c59a77`。
- 原部署 `/home/zhongxu/projects/aigc_detector/deployments/text-binary-20261123-v2` 保留，启动快照位于新目录下的 `evidence/rollback-live.json`。
- 回退命令在 school 的新部署目录运行。

```bash
cd /home/zhongxu/projects/aigc_detector/deployments/text-binary-20261124-en-v2a
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 /home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline/.venv-image-next/bin/python release_en_v2a.py rollback
```
