# 文本溯源恢复记录

2026-11-17 已恢复 school 正式服务的文本来源家族归因。用户原文经本机转发生成的报告 111 返回 `status=ok`、`family_status=family_only`，AI 风险分数仍为 0.878，筛查提示仍为“疑似 AI，需复核”。

## 修复内容

旧 `A0-closed-seed42` 产物标记为 `A0-window-split`，却保留了基础 A0 的 recipe hash `78051efc4353f36c4ca55ed8f1619268a80a6065fc460bb134343a6dc09c0a8f`，且缺少完整窗口验证协议。严格加载器拒绝产物后，融合分支返回错误，页面却将缺失的 `family_status` 显示为 `None`。

使用仓库现有已修正的 `scripts.run_family_attribution_window_split` 重训 A0，保留严格产物校验。没有手改旧 hash、跳过校验、硬编码来源或降低来源阈值。

- 训练集沿用服务器已有 v5 标注数据及 seed42 划分，没有新下载数据集，没有把用户原文加入训练。
- 底座为缓存的 `FacebookAI/xlm-roberta-base`，10 个来源家族，6 轮，单张 GPU，最大长度 512，正文窗口/步长 510。
- 3,981 篇训练文档展开为 7,009 个窗口；912 篇验证文档展开为 1,573 个窗口，按文档平均窗口概率选择检查点。
- 规范 recipe hash 为 `d2cfe938219e71ab28cf3101065434ca2be566fbb9c2fa4b02a8c4d4637d8ef2`。
- 新 A0 验证准确率 65.13%、macro-F1 0.63956；这是 A0 单分支的验证集结果，未在本轮评估完整融合的独立测试准确率。

新产物位于 `/home/zhongxu/projects/aigc_detector/outputs/attribution/family-window-split/A0-canonical-seed42-20261117`。正式配置仅更新 A0 的模型及 tokenizer 两个路径，B0/B1/B3、四路各 0.25 权重、来源判断阈值以及二分类检测模型均保持原配置。

前端新增归因加载/推理失败的显式提示，缺失状态不再显示 `None`。本地模块为 `api/dashboard.py`，正式服务较新基础版本对应模块为 `api/text_view.py`，部署时仅应用同一小补丁。

## 真实请求验收

通过本机 `127.0.0.1:8001` 提交原文，报告 111 的来源候选如下：

| 家族 | 融合得分 |
| --- | ---: |
| Kimi | 28.14% |
| 豆包 | 15.91% |
| GLM | 11.91% |
| DeepSeek | 11.29% |
| Qwen | 9.34% |

这些是 10 类闭集家族候选的模型得分。原文实际生成模型未知，首位得分不能当成已证明的真实来源，也不能定位具体模型版本。

正式 API 和页面健康检查通过，本机访问地址仍为 `http://localhost:8501`。实际部署模块使用真实返回运行 Streamlit AppTest，确认 Top-K 可见；另外验证错误状态能显示失败提示。源码测试 `python -m pytest -q tests`：749 项及 214 个子测试通过，8 条第三方弃用警告。新增回归已在改动前失败、改动后通过。

## 运行与回退

服务目录仍为 `/home/zhongxu/projects/aigc_detector/deployments/text-binary-20261117-v1`，当前 `AIGC_CONFIG` 指向该目录的 `config-attribution-canonical.yaml`。API/页面 PID 为 2539885/2541535，端口仍为 8611/8612。原数据库和历史报告保留，历史错误报告不回写；重新提交会生成新结果。

当前部署清单见 `attribution-deployment.json`，产物哈希及请求摘要见 `attribution-repair.json`。服务器保存旧配置、旧产物、`deployment-before-attribution-repair.json`、`pids-before-attribution-repair.json` 和 `api/text_view.before-attribution-repair.py`；需要回退时使用清单中的原配置及启动参数，不覆盖原权重和历史记录。
