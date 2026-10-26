# IAB v4-20 后续实验代码与冻结协议

2026-10-20。研究工作树：`/home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline`。

**状态：代码准备完成，正式 D1–D3 / A1–A3 未启动。** 自动流水线仍只有 D0、A0 各 12 epochs 和冻结 OFA。本文不是实验结果报告。

## 数据与共同协议

- 使用已冻结的 `outputs/image_experiments/iab-balanced-v4-20`：22 来源 × 10 语义 × 20 张生成图，加每语义 440 张真实图，共 8,800 张。多下载的旧图不加入。
- selection SHA256：`c822f03e5c691f4a86a01af62b9fd85116bbdbe7c4ad25cdbcb9ea882eb7e562`。未下载完整、没有正式 manifest 或身份不符时拒绝执行。
- 不重新抽样，不重划原始 train/dev/holdout。原 holdout 不参与选模、校准、推理或指标；程序只读取其 manifest 元数据检查身份隔离，不打开 holdout 图片。
- 原 dev 按 `SHA256("42:" + group_id) % 3` 分配 selection / calibration / evaluation；同组不拆开。额外检查字节哈希和像素哈希，发现跨角色重复直接报错。
- train 保留原训练组；selection 选择 epoch；calibration 拟合阈值；evaluation 报告结果。各角色按任务过滤，不足类别时失败并保留诊断，不偷偷降低门槛或改划分。
- 新训练固定 ResNet50/ImageNet V1、本地 `resnet50-0676ba61.pth`，全骨干训练；12 epochs，batch 32，seed 42，Adam lr 0.001，StepLR(step_size=1,gamma=0.9)。D0 类二分类使用原逆频率 CE 权重，来源任务使用无权重 CE。
- 无增强时预处理保持 Resize(256) → CenterCrop(224) → ImageNet normalize。最佳 epoch 仍按真假 AUROC／来源 macro-F1，平局取最早；只用 selection。
- 仅 GPU0，复用 `gpu0-research.lock`，显存已用超过 1 GiB 拒绝运行；不停止别人的进程、不访问 GPU1 服务。

## 固定实验矩阵

主配置：`configs/image_experiments/iab_followups_v4_20.json`。下表中的来源和语义是在查看基线结果前固定的工程先导折，不代表穷尽所有来源／语义泛化。

| ID | 实现 | 是否重训 | 评估与边界 |
|---|---|---:|---|
| D1 | 训练侧 JPEG 与降采样增强，各自概率 0.5；JPEG quality 从 50/75/95 选，缩放从 0.5/0.75 选，恢复原尺寸后统一预处理 | 是 | 同一干净 evaluation 上 accuracy、macro-F1、AUROC、FPR/TPR；训练指标也用干净图片 |
| D1-control | 同一划分、同一训练协议，无增强 | 是 | 与 D1 严格匹配的对照；不能把划分也变了的 D0 对比称为纯增强消融 |
| D2 | 预先留出 `4o`；其图片不进 train/selection/calibration，evaluation 只用留出来源 AI 与真实对照 | 是 | 来源留出真假检测；其他来源可训练，但不能声称所有未知生成器泛化 |
| D2-semantic | 留出 `FFHQ` 语义 | 是 | 同来源集合的跨语义真假检测 |
| D3 | 已有 D0 评分；在 calibration 真实图上选 FPR 目标 0.01/0.02/0.05 的阈值，evaluation 独立报告；200 次校准负例组 bootstrap | 否 | 报实际 FPR/TPR、负例数、1/n 分辨率、Wilson 区间、阈值分位数；小样本不保证低 FPR |
| A1 | 训练排除 `kling` 和 `grok3`，其余 20 来源闭集训练；kling 只用于未知校准，grok3 只用于未知评估 | 是 | 以最大 softmax 分数接受；校准未知误接收目标 5%；未知 AUROC、已知接受率、已知正确且接受率、未知拒识率、混淆矩阵、每来源／语义结果 |
| A2 | 已有 A0，在完全相同 evaluation 图片上分别测 clean、JPEG95/75/50、resize0.75/0.5 | 否 | 每次只改变一个后处理因素，报告 Top1/Top3、macro-F1、混淆矩阵、相对 clean 差值 |
| A2-semantic | 留出 `FFHQ` 语义，来源词表保持一致 | 是 | 跨语义闭集来源分类 |
| A3 | A0 各来源概率按已核验 family 求和；同时报告来源和家族层级指标 | 否 | 已实现汇总器，映射未核验则阻止运行；不是新训练的层级网络 |

所有配置 `automatic_execution=false`，不会加进监控脚本或 `run_iab_m2.sh`。

### 结果解释限制

1. D3/A2/A3 复用的 D0/A0 已用整个原 dev 选过最佳 epoch。因此即使本次 calibration/evaluation 不重叠，仍继承选模偏差，只能标注探索性开发评估。独立确认需要另行批准新协议／新数据；不能把当前 holdout 拿来反复调参数。
2. A1 必须用排除两个未知来源后重新训练的 checkpoint，不能拿已经见过全部 22 来源的 A0 伪装开放集。`1-max_softmax` 是拒识分数，不是校准后的未知概率。来源未知不等于模型家族未知。
3. 小样本分组可能导致某角色缺类；程序会明确报错。不得静默退化为少于声明来源数的实验。后续决定扩数据／调整协议时另存版本，保留当前配置和失败现场。
4. 每次评估的 Wilson 区间按图片独立假设计算，存在组相关时有效样本数更低。阈值 bootstrap 按校准组重采样，不使用 evaluation 样本。
5. A3 的家族／版本映射文件 `configs/image_experiments/iab_family_mapping_review.json` 当前刻意设为 `reviewed=false`，不能把同供应商自动当成同模型家族，也不能把目录别名当成已核验后端版本。需要为每个来源填写 `family` 与一手 `evidence` 后再审核。
6. A2 的“跨版本”暂未配置为可运行实验：新版本可能没有训练类别，必须先确定是版本闭集、家族迁移还是未知拒识，并有可靠版本映射。这是明确的协议依赖，不用普通 22 类结果替代。

## 手动入口

**现在仅允许查看方案。** 下列 plan 不加载图片或模型，不写输出目录，也不启动下载。

```bash
cd /home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline
.venv-image/bin/python scripts/run_image_followup.py --experiment D1
.venv-image/bin/python scripts/run_image_followup.py --experiment A1
```

未来基线完成、用户决定先做哪些之后，才使用 train / evaluate 加 `--execute`。两者都先检查 D0、A0、OFA 完成状态；train 还要求该配置确实需要重训。示例仅说明接口，本轮没有执行：

```bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$PWD/.venv-image/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv-image/bin/python scripts/run_image_followup.py --experiment D1-control --action train --execute
.venv-image/bin/python scripts/run_image_followup.py --experiment D1-control --action evaluate --execute
```

D1/D2/A1/A2-semantic 先 train，再 evaluate；D3/A2/A3 直接 evaluate 已有基线。输出固定到 `outputs/image_experiments/followups-v4-20/<ID>/<train|evaluate>/`，目录已存在则拒绝覆盖，不自动重试失败运行。

每个输出包含 resolved config、配置/manifest/角色/代码/权重哈希、运行日志、环境、真实退出状态、指标及资产哈希。训练保存 effective rows、每 epoch 指标、checkpoint、重载与单图/批量一致性结果；评估保存逐样本角色、来源、语义、条件和完整概率。原 baseline checkpoint 的 manifest 哈希表示父 manifest；新增 `followup_identity` 同时绑定实际角色清单，防止跨配置复用 checkpoint。

## 实现与验证边界

- 新模块 `research/image_attribution/followups.py`：划分、变换、阈值及层级指标。
- 新入口 `scripts/run_image_followup.py`：只读 plan、显式 train/evaluate、依赖/资源/身份检查、输出记录。
- 复用 `scripts/run_image_experiment.py`；只新增可选训练 transform，默认路径等价，dev 和训练评估始终干净。修改前副本保存在 `outputs/iab-v2-monitor/recovery/followups-code-preparation-20261020/`。
- `tests/test_image_followups.py` 使用合成数据检查角色隔离、重复图片、留出来源、拒识、阈值 ties、校准独立性、家族概率求和、变换边界、默认不执行以及微型网络单轮保存/重载。合成测试不属于任何正式 D/A 实验，不能当成本项目指标。
- 正式 D1–D3/A1–A3 尚无训练结果或模型资产。本轮不扩展 V2/C0，不修改下载清单、现有自动化执行集合或生产工作树。
