# 文本家族溯源 Phase4 可靠性、融合与开放集设计

_面向“闭集研究指标 + 可拒判实用系统”的后续路线，版本 1.0，2026-08-03_

## 文档状态

| 项目 | 内容 |
| --- | --- |
| 状态 | 设计已获方向批准，书面规格自审完成，等待用户确认 |
| 当前数据 | V5 简化版 6K，冻结语义提示词组切分 |
| 当前神经骨干 | `FacebookAI/xlm-roberta-base` |
| 当前研发基线 | A4：首部、中央、尾部三窗口家族分类 |
| 目标形态 | 高置信度输出家族 Top-K；证据不足时 `unknown` 或 `abstain` |
| 当前范围 | Phase3.5 诊断、Phase4 融合/级联、选择性输出、开放集前置设计 |
| 后置范围 | V6 扩数、LoRA 对照、最终 locked test |
| 测试集状态 | 必须继续冻结，当前设计与实现阶段均不得读取 |

本文档承接：

- [`2026-07-30-hierarchical-llm-attribution-training-design.md`](2026-07-30-hierarchical-llm-attribution-training-design.md)
- [`2026-08-02-family-attribution-phase3-enhancement-design.md`](2026-08-02-family-attribution-phase3-enhancement-design.md)
- [`2026-07-26-dual-mode-llm-source-attribution-research-design.md`](2026-07-26-dual-mode-llm-source-attribution-research-design.md)

它把原总方案中的“阶段四：融合与条件级联”细化为实施级设计，并规定何时进入数据扩充、开放集和 LoRA。本文档不改写 Phase3 的历史选择记录，也不把后确认结果伪装成原始晋级。

## 1. 已知证据与正式解释

### 1.1 Phase3 结果

Phase3 初始 `42/52/62` 三个 seed 的预设晋级规则没有选择 A4，正式 `stage_decision.json` 因而保留 A0。随后按预先规定的 `12/22/32/72/82` 五个确认 seed 形成 8-seed 汇总：

| 验证视图 | A0 Macro-F1 | A4 Macro-F1 | A4 绝对增益 |
| --- | ---: | ---: | ---: |
| closed | 0.563941 | 0.586917 | +0.022976 |
| fast-to-balanced | 0.371458 | 0.389063 | +0.017605 |
| balanced-to-fast | 0.361519 | 0.375155 | +0.013636 |

A4 在 closed 的 8 个 seed 中赢得 6 个，并同时改善 closed NLL、ECE 和平均最弱家族 F1。因此：

- Phase3 的**历史选择**仍是 A0，不覆盖原 decision；
- Phase3 的**确认后研发提名**是 A4；
- Phase4 的 E0 使用 A4，但必须在 `phase4_parent_decision.json` 中记录这是 `post_confirmation_nominee`；
- 最终论文或报告同时陈述三 seed 选择与八 seed 确认，不使用“Phase3 原本晋级 A4”的错误表述。

### 1.2 当前可用性边界

A4 明显高于十分类随机水平，但 closed Macro-F1 约 0.59，两个版本留出视图约 0.38。当前模型可以作为研究基线、Top-K 辅助信号和多文本聚合分支，不能用于单篇文本的确定性责任归属。

尤其需要保留以下限制：

- 当前分类器只处理十个已知家族，未知模型会被强制分类；
- 家族表现差异很大，平均分不能代表每个家族都可靠；
- 当前版本留出每个家族主要依赖两种版本，无法替代跨时间真实漂移评估；
- F1 高不等于预测为该家族时 Precision 必然高；
- 来源归因概率不是平台日志、水印、签名或密码学证明。

## 2. 目标与非目标

### 2.1 目标

Phase4 需要回答五个问题：

1. A4 与字符、显式风格和第二神经模型的错误是否互补；
2. 简单概率融合能否稳定提高 closed、版本留出和最弱家族表现；
3. 条件级联能否接近完整集成上限，同时控制第二模型调用率；
4. 在允许拒判后，系统能否在可接受覆盖率下达到高 Precision；
5. 当前瓶颈主要来自模型表示、数据规模、版本覆盖，还是单文本证据上限。

### 2.2 非目标

Phase4 不做：

- 读取或评估 frozen test；
- 精确具体模型版本的产品承诺；
- 直接训练 7B QLoRA；
- 在没有 OOD 校准数据时声称支持 `unknown`；
- 深层 stacking、自动机器学习或大规模权重网格搜索；
- 把人类/AIGC 二分类和十家族归因重新训练成一个标签空间；
- 根据 Phase4 validation 结果无限追加模型、seed 或阈值试验。

## 3. 用户输出契约

### 3.1 三种状态

```text
family_only  已知家族证据达到冻结门槛，输出校准 Top-K
unknown      独立 OOD 校准支持“不属于已知十家族”
abstain      文本过短、置信不足、分支冲突或证据质量不足
```

Phase4 第一版只承诺 `family_only` 和 `abstain`。`unknown` 必须等 Phase5 OOD 数据与校准完成后启用。未启用 Unknown 时，分布外风险只能作为 `abstain` 原因，不能伪装成 Unknown 判定。

### 3.2 输出字段

```json
{
  "schema_version": 1,
  "status": "family_only | unknown | abstain",
  "top_families": [
    {"family": "qwen", "probability": 0.61},
    {"family": "glm", "probability": 0.18},
    {"family": "deepseek", "probability": 0.09}
  ],
  "confidence": 0.61,
  "prediction_set": ["qwen", "glm"],
  "known_probability": null,
  "second_model_called": false,
  "calibration_revision": "family-phase4-calibration-v1",
  "artifact_fingerprint": "sha256:...",
  "warnings": ["single_text_evidence", "unseen_version_risk"]
}
```

约束：

- `confidence` 必须来自冻结校准流程，不能直接重命名原 Softmax 最大值；
- `family_only` 必须至少包含一个候选；
- `unknown` 和 `abstain` 的 `confidence` 含义必须与家族概率分开；
- 默认不输出 `model_id`；
- 缺少、损坏或版本不匹配的 artifact 必须返回 `abstain`，不能回退为无校准十选一。

## 4. 总体数据流

```mermaid
flowchart TD
    input[文本] --> detector[现有 AIGC 检测器]
    detector -->|human 或低 AI 证据| stop[不运行家族溯源]
    detector -->|uncertain| abstain0[abstain]
    detector -->|AI 证据充分| primary[A4 三窗口]
    input --> char[字符 TF-IDF LR]
    input --> style[显式风格 LR]
    primary --> light[校准与 E1 轻量融合]
    char --> light
    style --> light
    light --> gate{证据充分?}
    gate -->|是| policy[选择性输出策略]
    gate -->|否| second[第二神经模型]
    second --> full[E2/E3 重新校准融合]
    full --> policy
    policy -->|已知且可靠| family[family_only Top-K]
    policy -->|OOD 已启用且成立| unknown[unknown]
    policy -->|其余| abstain[abstain]
```

检测与归因保持两个独立任务。检测结果只决定是否进入归因流程，不作为家族分类训练标签。归因结果也不能反向证明输入一定由 AI 生成。

## 5. Phase3 证据包与不可变校验

### 5.1 不需要完整服务器目录

Phase3 已经在每个正式运行的 `metrics.json` 中保存：

- Accuracy、Macro-F1、Weighted-F1；
- 每家族 Precision、Recall、F1、Support；
- 归一化混淆矩阵；
- 具体模型诊断；
- NLL、ECE、Brier；
- 语言、长度、任务和 temperature 子组。

`predictions.jsonl` 保存逐样本真值、预测、十家族概率和必要元数据。因此缺少 `logs/attribution-phase3` 不阻塞 Phase3.5；`trainer_history.json` 与 `loss_components.json` 已覆盖训练历史。Accuracy、Precision 和 Recall 没有完整显示在终端属于可观察性问题，不需要重跑训练。

### 5.2 诊断包允许文件

诊断包只允许包含：

```text
metrics.json
predictions.jsonl
run_config.json
resolved_recipe.json
label_maps.json
view_manifest.json
dataset_audit.json
training_result.json
trainer_history.json
loss_components.json
COMPLETED.json
stage_decision.json
selection_report.md
promoted_recipe.json
eight-seed-summary.json
```

明确排除：

- `model/`
- `tokenizer/`
- `checkpoints/`
- Hugging Face 缓存
- 原始 V5 文本
- API Key、`.env` 和请求凭据
- `validation_outputs.npz`，因为逐样本概率已在 `predictions.jsonl` 中保存

### 5.3 对齐硬约束

任何诊断或融合前必须验证：

- 所有分支使用相同 dataset fingerprint、split manifest hash 和 catalog hash；
- 家族标签集合与顺序完全一致；
- `test_evaluated` 严格为 `false`；
- 正式运行不是 smoke 或 preflight；
- 每个目标 `request_key` 在每个参与分支中恰好出现一次；
- 同一 `request_key` 的真值、语义组和子组元数据完全一致；
- 概率有限、位于 `[0,1]` 且每行和为 1；
- 运行矩阵满足预设 seed 与 view 完整性。

任一条件不满足时整体失败，不做交集截断、不补零、不静默重排标签。

## 6. Phase3.5：诊断与互补性分析

### 6.1 指标

统一输出：

- Accuracy、Macro-Precision、Macro-Recall、Macro-F1、Weighted-F1；
- 每家族 Precision、Recall、F1、Support；
- Top-1、Top-2、Top-3 Accuracy；
- NLL、ECE、Brier；
- 最弱家族 F1 与最弱家族 Recall；
- 按具体版本、语言、长度、任务、temperature 的分组指标；
- 按 seed 和 view 的均值、样本标准差、最小值和最大值；
- 以 `semantic_prompt_group` 为重采样单位的 95% Bootstrap CI；
- 混淆矩阵及最主要错误对。

Bootstrap 固定 2,000 次重采样和独立分析 seed。禁止按单条文本重采样，因为同一语义提示词组的回答相关。

选择性指标使用以下固定定义，避免与多分类 Macro-Precision 混淆：

- `accepted_precision = accepted_correct_top1 / accepted_count`，即所有未拒判样本中的 Top-1 正确率；
- `coverage = accepted_count / eligible_count`，其中 `eligible_count` 不含进入归因前已被检测器拦截的样本；
- `selective_risk = 1 - accepted_precision`；
- 逐家族 accepted Precision、Recall 和 coverage 仍需单独报告；某家族没有 accepted 样本时记为 `undefined` 并触发覆盖告警，不记作 0，也不从汇总中静默删除；
- Macro-Precision、Macro-Recall、Macro-F1 始终按完整十家族分类定义计算，不使用 accepted Precision 替代。

### 6.2 互补性指标

对任意两个分支 A、B 计算：

- `both_correct`、`a_only_correct`、`b_only_correct`、`both_wrong`；
- 预测标签分歧率；
- 概率 Jensen-Shannon divergence；
- 错误样本上的纠正率；
- 每家族纠正率；
- oracle top-1：任一分支正确即视为正确的理论上限；
- 简单等权平均的无拟合基线。

Oracle 只用于判断是否存在互补空间，不能作为可部署结果或晋级指标。

### 6.3 Phase4 进入门槛

满足任一条件即可进入 E1/E2：

1. 对齐分支的 oracle closed Macro-F1 比 A4 高至少 0.02；
2. 等权融合已提高 closed Macro-F1 至少 0.01；
3. 即使整体 F1 增益不足 0.01，高置信度选择性 Precision 或最弱家族有明确改善。

若全部不满足，则跳过复杂融合，直接进入学习曲线和 V6 数据决策。

### 6.4 输出

```text
outputs/attribution/family-phase4/diagnostics/
├── diagnostic_manifest.json
├── phase3_diagnostics.json
├── phase3_diagnostics.md
├── model_complementarity.json
├── risk_coverage.json
├── paired_seed_results.json
├── confusion_matrices/
└── phase4_recommendation.json
```

## 7. 防泄漏的交叉拟合协议

Phase4 的校准器、融合权重和选择性阈值不能在完整 validation 上拟合后再使用同一预测报告性能。

对现有 validation 预测按 `semantic_prompt_group` 做固定 5 折交叉拟合：

1. 使用冻结哈希把语义组分配到折；
2. 每轮用 4 折拟合温度、融合权重或门控阈值；
3. 只在剩余 1 折生成该轮 OOF 预测；
4. 合并五折 OOF 预测用于 E1–E3 选择和报告；
5. 方案完全冻结后，才在全部 validation 上拟合最终部署 artifact；
6. frozen test 始终不参与任何一步。

每折必须包含全部十个家族。若某折缺类，使用确定性分层分组分配重新构造；仍无法满足时停止并报告数据不足，不能退化为按行随机拆分。

## 8. Phase4 实验矩阵

### 8.1 E0：A4 研发基线

E0 固定为：

- recipe：`A0+A4`
- window mode：`head-center-tail`
- recipe hash：`1d4b21b8e2bb1477d4872049805cf6d1ca8e037ec52e2a538f851eb335dd71f2`
- 家族单头；不继承 A1/A2/A3；
- 研究报告使用 8-seed 统计；
- 最终产品 artifact 的 production seed 必须在 test 前预先登记，不能按 test 选择。

### 8.2 E1：轻量融合

```text
E1 = A4 + 字符 TF-IDF Logistic Regression + 显式风格 Logistic Regression
```

若现有 B0/B1/B2 预测与 V5 指纹、切分或请求键不一致，必须使用冻结 V5 split 重跑相应传统分支。不能混用旧 pilot OOF、不同数据版本或不同标签顺序。

每个分支先拟合一个正温度参数。融合权重满足非负单纯形约束。第一版使用步长 `0.05` 的确定性单纯形搜索，以 OOF NLL 为主要目标，并丢弃相对最佳单分支 Macro-F1 下降超过 `0.005` 的权重。NLL 相同到 `1e-6` 时依次选择：

1. 最弱家族 F1 更高；
2. 非零分支更少；
3. A4 权重更高；
4. 权重元组字典序更小。

不使用深层元分类器。

### 8.3 E2：完整集成上限

```text
E2 = E1 + 第二神经模型，所有分支始终运行
```

第二神经模型只在 Phase3.5 证明互补后进入。候选优先从已经完成的 mDeBERTa/CANINE 结果中选择；缺少完全对齐预测时可以按冻结协议重跑，但不得根据 test 选择。

第二模型进入 E2 至少满足：

- 在 A4 错误样本上有可观纠正率；
- 至少一个弱家族改善；
- 两个版本留出方向没有同时明显劣化；
- 不是仅凭最高单 seed 分数晋级。

E2 只测准确率上限，不是默认部署方案。

### 8.4 E3：条件级联与 abstain

E3 默认先运行 E1。门控特征限定为：

- 校准后的最大家族概率；
- Top-1/Top-2 margin；
- 预测熵；
- 各轻量分支的标签分歧；
- 各轻量分支概率的平均 Jensen-Shannon divergence。

门控采用确定性阈值网格，不训练深层门控网络。选择顺序为：

1. E3 OOF Macro-F1 与 E2 差距不超过 `0.005`；
2. 两个版本留出相对 E0 均不下降超过 `0.01`；
3. 最弱家族 F1 不低于 E0；
4. 满足以上约束后最小化第二模型调用率；
5. 调用率相同时选择 NLL 更低的规则。

第二模型调用后重新校准融合。若最终证据仍不满足选择性阈值，返回 `abstain`。

### 8.5 E4：可选蒸馏

只有 E2/E3 相对 E0 有稳定增益且平均推理成本不可接受时才启动。蒸馏教师使用冻结 OOF/validation 软标签，不读取 test。若蒸馏模型相对 E3：

- closed Macro-F1 下降超过 `0.01`，或
- 任一版本留出下降超过 `0.015`，或
- 选择性 Precision/coverage 明显恶化，

则放弃 E4。

## 9. 选择性输出策略

### 9.1 第一阶段只做 abstain

在没有独立 Unknown 数据前，策略只决定“输出已知家族”还是“拒判”。第一版使用全局阈值，不使用每家族阈值；只有某家族在 calibration 中至少有 100 个独立语义组时，才允许预注册家族阈值消融。

候选阈值由 OOF 风险—覆盖率曲线选择。临时研发门槛为：

- accepted Precision ≥ 0.90；
- coverage ≥ 0.40；
- closed ECE ≤ 0.05；
- 无家族在 accepted 子集中 Precision 或 Recall 直接坍塌为 0；
- 低于最小有效长度或概率异常时强制 `abstain`。

若无法同时达到 Precision 与 coverage 门槛，系统不得标记为“可用”，但仍可保留研究结果。

### 9.2 Top-K 与 prediction set

默认展示 Top-3，但 `prediction_set` 由冻结校准规则产生：

- 集合大小 1：可输出单一 `family_only` 候选；
- 多个候选：输出集合并明确不确定性；
- 空集或多分支严重冲突：`abstain`。

Top-K 排名本身不等于置信保证。

## 10. Phase5：Unknown 与 OOD 校准

### 10.1 OOD 数据类型

至少覆盖：

- 有来源证明的人类文本；
- 不属于已知十家族的模型；
- 已知家族的新版本或新部署快照；
- 翻译、润色、摘要和截断；
- 人工编辑后的 AIGC；
- 一个模型重写另一个模型的输出；
- 跨时间和跨网关复采。

### 10.2 切分

```text
ood_train_exposure
ood_validation
ood_calibration
ood_locked_test
```

同一来源模型、版本、时间快照、原始文档和变换链不得跨集合。用于阈值拟合的未知来源不能同时作为 locked unknown 测试来源。

### 10.3 Knownness 证据

第一版联合：

- 最大校准概率；
- Energy 或等价 logit 分数；
- Top-1/Top-2 margin；
- 预测熵；
- 分支一致性；
- 到已知家族/模型原型的距离（只有可复现嵌入时启用）。

`known_probability` 使用独立 OOD calibration 拟合。原型距离、余弦相似度或 Energy 不能直接命名为概率。

### 10.4 指标

- Unknown AUROC、AUPR；
- FPR@95TPR；
- OSCR；
- Known Accuracy；
- Unknown Recall；
- Risk-Coverage；
- conformal prediction-set coverage 与平均集合大小；
- 人类文本被误输出家族的比例。

正式产品门槛需依据业务风险另行冻结。研发阶段使用 AUROC ≥ 0.85、accepted Precision ≥ 0.90 且 coverage ≥ 0.40 作为继续投入门槛，不将其表述为通用行业标准。

## 11. 鲁棒性与多文本聚合

### 11.1 必做消融

- 数据规模：10%、25%、50%、75%、100%；
- 字符长度：256、512、1024；
- Markdown：原始与删除；
- 标点：原始与规范化；
- 空白：原始与统一；
- 语言与任务；
- 快版本到强版本及反向；
- 单窗口与三窗口；
- 翻译、润色、人工编辑和跨模型重写。

学习曲线解释：

- 100% 仍持续上升：优先扩数据；
- 很早平台：优先检查表示能力或标签可辨识性；
- 训练提高而组外验证停滞：提示词/版本过拟合；
- closed 提高而版本留出不提高：模型主要记忆具体版本。

### 11.2 多文本聚合

对同一待测来源的独立文本比较 `K=1/3/5/10`。聚合必须跨不同语义组，不能用同一提示词的重复生成虚增证据。第一版使用校准概率均值和长度/质量权重，不使用最大分。

报告：

- K 对 Macro-F1、Top-K 和 risk-coverage 的影响；
- 达到 0.90 accepted Precision 所需最小 K；
- 短文本组合与单篇长文本的差异；
- 证据冲突时的 `abstain` 比例。

多文本聚合可能比更换大模型更符合“端点指纹”的实际条件，应在 LoRA 前完成。

## 12. V6 数据扩充门控

仅当学习曲线或错误诊断证明数据与版本覆盖是主要瓶颈时，从 6K 扩到 20K–40K。新增数据优先增加变化维度，而不是重复现有提示词：

- 每家族尽量覆盖三个以上版本或时间快照；
- 同一 prompt group 跨全部候选家族配对；
- 中英文、不同领域、任务和长度；
- 多种采样设置；
- 跨时间与跨网关复采；
- 变换链和人工编辑；
- 人类与未知模型 OOD。

V6 冻结切分至少包括：

```text
train
selection_validation
calibration
locked_test
version_held_out
time_held_out
transformation_held_out
unknown_family_held_out
```

切分单位仍是语义提示词组、原始文档和生成链。原始 JSONL 不可变，派生视图用 manifest 引用。

## 13. Phase6：LoRA 与新骨干

LoRA 是参数更新策略，不是自动提高识别能力的算法。它只在 Phase4/Phase5 和学习曲线完成后启动。

### 13.1 候选顺序

1. `Qwen3-Embedding-0.6B + LoRA`：嵌入、原型、开放集和多文本聚合主候选；
2. `Qwen3-1.7B-Base + LoRA`：十家族闭集强对照；
3. XLM-R Large 或 BGE-M3：长文本编码器对照；
4. 7B QLoRA：仅在数万级数据且小模型仍受表示能力限制时考虑。

正式实施前必须重新核对模型许可证、模型卡、revision、上下文长度和显存需求；本文档不把 2026-08-03 之后可能变化的模型信息视为冻结事实。

### 13.2 选择顺序

```text
冻结特征 + 线性探针
        ↓ 前两名
全量微调或 LoRA 对照
        ↓ 前一至两名
开放集、版本留出和风险覆盖率
```

不同时改变骨干、数据、损失和窗口。≤600M 候选比较全量微调与 LoRA；1B–4B 优先 LoRA；7B 仅使用 QLoRA 强基线。

### 13.3 晋级门槛

新骨干至少满足一项，并且其余关键指标不明显退化：

- closed Macro-F1 比 Phase4 最佳方案提高至少 0.02；
- 两个版本留出平均提高至少 0.03；
- 相同 coverage 下选择性风险明显降低；
- Unknown/abstain 指标显著改善；
- 相同性能下推理成本或存储明显降低。

若没有满足，不因参数更多或训练更新而保留。

## 14. 深模块与接口

后续代码放在 `research/text_attribution/`，不把逻辑堆入现有 Phase3 CLI。

```text
research/text_attribution/
├── diagnostic_analysis.py
├── probability_fusion.py
├── selective_policy.py
├── open_set_calibration.py
└── phase4_reporting.py
```

### 14.1 诊断模块

```python
def analyze_prediction_bundle(request: DiagnosticRequest) -> DiagnosticReport:
    ...
```

该接口隐藏文件发现、指纹校验、请求键对齐、指标重算、Bootstrap、互补性和报告模型。调用者只提供显式 roots、期望 recipes、seed/view 矩阵和输出目录。

### 14.2 融合模块

```python
def fit_probability_fusion(request: FusionFitRequest) -> FusionArtifact:
    ...

def apply_probability_fusion(
    artifact: FusionArtifact,
    branches: AlignedBranchProbabilities,
) -> FusedProbabilities:
    ...
```

接口隐藏交叉拟合、温度校准、单纯形搜索、tie-break、指标与最终重拟合。artifact 使用显式 JSON 数值和哈希，简单权重不使用不透明 pickle。

### 14.3 选择性策略模块

```python
def fit_selective_policy(request: SelectiveFitRequest) -> SelectivePolicyArtifact:
    ...

def decide_attribution(
    artifact: SelectivePolicyArtifact,
    evidence: AttributionEvidence,
) -> AttributionDecision:
    ...
```

接口隐藏阈值网格、风险—覆盖率、级联门控、prediction set 和状态决策。调用者不直接实现阈值判断。

### 14.4 开放集校准模块

```python
def fit_open_set_calibration(request: OpenSetFitRequest) -> OpenSetArtifact:
    ...

def score_knownness(
    artifact: OpenSetArtifact,
    evidence: AttributionEvidence,
) -> KnownnessResult:
    ...
```

只有拥有至少两类可替换 knownness adapter（例如纯概率证据与嵌入原型证据）后才建立 adapter seam；第一版不为单一实现预设抽象层。

## 15. Artifact 与目录契约

```text
outputs/attribution/family-phase4/
├── diagnostics/
├── E0/
├── E1/
├── E2/
├── E3/
├── E4/
├── selection/
└── artifacts/
    ├── phase4_parent_decision.json
    ├── fusion.json
    ├── calibrators.json
    ├── cascade_policy.json
    ├── selective_policy.json
    └── artifact_manifest.json
```

每个 artifact 至少绑定：

- schema version；
- 训练代码 commit；
- 数据、split、catalog hash；
- 标签顺序；
- 输入 branch ID、recipe hash 和 run fingerprint；
- cross-fitting 规则与 seed；
- 校准样本身份 hash；
- 指标、门槛和选择原因；
- `test_evaluated=false`；
- 创建时间与运行环境版本。

已存在的正式输出目录不得覆盖。使用 staging 目录写完并校验后原子发布。失败目录保留最小错误证据，但不能保存半写入的正式 decision。

## 16. 错误处理与恢复

以下情况必须快速失败：

- 输入目录缺失或矩阵不完整；
- recipe/hash/标签/请求键不一致；
- 发现 test 指标；
- smoke/preflight 混入正式选择；
- 概率异常、重复请求键或缺类折；
- 校准拟合产生非有限参数；
- 输出目录已存在；
- Unknown 未校准却请求 `unknown` 状态。

诊断和融合属于轻量确定性任务，失败后从头重跑，不设计 checkpoint。神经模型重跑继续使用 Phase3 的原子目录和 checkpoint 合同。磁盘清理只能删除已完成训练的 checkpoint 或已淘汰模型权重，保留 JSON、预测、fingerprint 和选择记录。

## 17. 测试策略

### 17.1 单元测试

- 概率、标签和请求键校验；
- Macro-Precision/Recall/F1、Top-K、ECE、NLL、Brier；
- 组级 Bootstrap 可复现；
- 两分支互补性和 oracle 上限；
- 温度校准和单纯形权重 tie-break；
- 风险—覆盖率和 prediction set；
- `family_only/unknown/abstain` 状态不变量；
- artifact JSON round-trip 与 hash 校验。

### 17.2 集成测试

- 使用微型 10 家族 fixture 完成 5 折交叉拟合；
- E0→E1→E2→E3 完整选择；
- 不一致 split、标签顺序和缺失请求键被拒绝；
- test 指标、smoke、preflight 被拒绝；
- staging 成功原子发布，失败不占用正式目录；
- 无 OOD artifact 时运行时绝不返回 `unknown`。

### 17.3 回归测试

- Phase2/Phase3 现有行为和文件契约保持不变；
- Phase4 不要求重新训练才能读取 Phase3 Accuracy/Recall；
- 相同输入 bundle 重跑得到字节稳定的 JSON（时间字段除外或固定）；
- 诊断包不包含权重、缓存、原始数据或密钥模式。

## 18. 阶段门槛与执行顺序

| 阶段 | 必要输入 | 完成门槛 | 下一步 |
| --- | --- | --- | --- |
| 3.5 诊断 | Phase3 精简结果包 | 完整指标、CI、互补性、oracle | 融合或扩数据 |
| E1 | A4 + 对齐字符/风格概率 | OOF 增益或选择性收益 | E2 |
| E2 | 互补第二神经模型 | 明确完整集成上限 | E3 |
| E3 | 冻结门控与拒判约束 | 接近 E2，调用率受控 | 鲁棒性/OOD |
| E4 | E3 有收益但成本过高 | 蒸馏退化在门槛内 | 可选保留 |
| Phase5 | 独立 OOD 数据 | Unknown 与 risk-coverage 达门槛 | V6/LoRA 决策 |
| V6 | 学习曲线证明数据不足 | 20K–40K 冻结数据与新 split | 新骨干 |
| Phase6 | 冻结数据和候选骨干 | LoRA/全调受控对照 | 最终冻结 |
| Final | 全部 artifact 与阈值冻结 | 一次 locked test | 报告/部署 |

执行纪律：

1. 先完成诊断包和 Phase3.5，不训练新模型；
2. 只有互补性成立才实现 E1/E2；
3. 先实现 `abstain`，再收集数据实现 `unknown`；
4. 先做学习曲线与多文本聚合，再决定扩数；
5. LoRA 只在数据/表示诊断后启动；
6. locked test 最后一次性使用；
7. 任何阶段失败都允许停止，不要求为了流程完整强行进入下一阶段。

## 19. 最终冻结与报告

测试前必须冻结：

- 最终骨干、窗口和 production seed；
- 字符/风格分支版本；
- 校准器和融合权重；
- 第二模型与级联规则；
- 最小文本长度；
- family/unknown/abstain 阈值；
- Top-K 与 prediction-set 规则；
- 数据、split、catalog、代码和 artifact hash；
- 测试比较对象及报告指标。

最终一次性报告：

- closed 十家族 Accuracy、Macro-P/R/F1、Weighted-F1 和逐家族 P/R/F1；
- 版本、时间、语言、长度、任务和变换留出；
- NLL、ECE、Brier 和可靠性；
- Top-K、risk-coverage、调用率和 P95 时延；
- Unknown/OOD 指标；
- 多文本聚合；
- 端到端“检测→归因”错误；
- 组级 Bootstrap 95% CI；
- 已知失败模式和不支持的声明。

测试结果出来后，任何针对结果的模型、阈值或数据修改都必须使用新的独立 locked test 才能形成新版本结论。

## 20. 第一实施切片

本设计批准后的第一实施切片严格限定为：

> 不依赖模型权重的 Phase3 诊断、指标补全和模型互补性分析管线。

它读取精简结果包，重算 Accuracy/Precision/Recall/F1/Top-K/校准/子组/CI，验证 8-seed × 3-view 证据，输出 A4 与传统/第二神经分支的互补上限和 Phase4 建议。它不训练模型、不拟合最终融合、不读取 test，也不改变 Phase2/Phase3 文件。

只有该切片的结果证明融合值得继续，后续实施计划才进入 E1–E3。
