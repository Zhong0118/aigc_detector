# 图片 M0—M2 执行状态

日期：2026-10-19。以用户本轮消息为准：CLIP/逻辑回归不再是必选主模型，优先核验完整 OFA 和 ImageAttributionBench 端到端方法。仅做小规模开发实验，不运行正式测试，不部署。

## M0 对齐证据

- GitHub `sync`: `2b37c811f7787beba8e570cf4d5bac9b1ef27b30`，已通过 `git ls-remote` 核实。
- 原本地分支 `codex/attribution-pilot-baselines` HEAD `209935734f145179813452984fe07e43c650a3fb`，相对 sync 为 0 ahead / 9 behind；原目录有文本产物、AGENT.md、演示文件等未提交修改，未迁移或覆盖。
- school `/home/zhongxu/projects/aigc_detector` HEAD `b8adb8bc55a1a1e6896e4ba548a9982431cef354`，相对 sync 落后 1 个提交，存在 16 个已跟踪文件的未提交修改及其他未跟踪文件。服务器拥有较新的运行资产不等于代码提交最新。
- 服务器修改涉及 dashboard、config、文本阈值、provenance、reports、依赖及文本研究模块；归属未确认，未迁移。新分支基于已发布 sync，不声称包含服务器全部运行变化。
- 独立本地 worktree `/Users/zhongxu/.codex/worktrees/image-m0-m2-paper/aigc_detector`；分支 `codex/image-m0-m2-paper-20261019`。
- 原生产工作树、服务、模型、环境保持原状。未调用任何付费检测/报告/生成 API。

## 实际图片链路

`api/analysis.py` → `ingestion/loader.py` 图片 RGB 读取、metadata、fingerprint → `detection/providers.py` 配置驱动 API/demo/local → `provenance/pipeline.py` 的 C2PA、历史指纹、水印和来源分支 → `reports/generator.py` 模板/可选 LLM → 存储。

可复用：图片读取/指纹、来源 CLI JSON 边界、已有训练脚本的类标签解码与本地文件约定、独立研究目录和 pytest。`scripts/train_image_attribution.py` 已有统计或 CLIP 特征 + LogisticRegression，但随机划分、无固定 manifest、无独立真假训练，不能直接充当本次协议。

待完成/风险：`detection/image_detector.py` 是 weights=None scaffold；OFA provider 只是命令适配接口，并非论文实现；`clip_prototype` 实际使用 pHash，不是 CLIP 表示。非文本检测在无可用分数时仍回退 0/human，是现有工程风险，本轮离线实验不经过此链路，也不顺手修改共享生产模块。

服务器现有 `models/attribution/image_source_classifier.joblib` 34,313 bytes；无可追溯的新协议训练记录，保留只读，不当作新基线。服务器 data 1.9 GiB、models 6.8 GiB 主要为文本资产，未发现可直接复用的已核验图片数据集。

## 资源与环境

school 两张 RTX 4090 D；初查 GPU0 15 MiB、0%，GPU1 10,655 MiB；无进程被终止。磁盘 1.8 TiB、可用约247 GiB、86%已用，无其他数据盘，低于任务书20%保留线。已向用户请求本轮最多5 GiB且保留200 GiB的例外，在答复前不新增服务器实验资产。

生产 Python 为 3.10.20，torch 2.6.0+cu124、torchvision 0.21.0+cu124、numpy 2.2.6、sklearn 1.7.2、pyarrow 24.0.0，与 sync 约束部分不符，不修改生产环境。

本地磁盘约139 GiB可用、68%已用；独立 `.venv-image`，Python3.12、torch2.6、torchvision0.21、numpy1.26.4、sklearn1.5.2、pyarrow19.0.1；实际 MPS GPU 可用。Python3.12相对README推荐3.10/3.11属于记录在案的本地环境差异。

## M1 / M2

执行中，所有指标须来自真实 run 的产物；论文数字不填写在本项目指标栏。方法核验见 IMAGE_METHOD_AUDIT.md，最终运行记录将更新本文件。
