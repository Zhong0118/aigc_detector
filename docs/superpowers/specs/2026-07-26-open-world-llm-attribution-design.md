# 开放世界 LLM 文本来源归因技术设计

**状态：** 初版工程草案，已被 `2026-07-26-dual-mode-llm-source-attribution-research-design.md` 取代。保留用于记录第一次方案，不再作为模型选型和训练实施依据。  
**日期：** 2026-07-26  
**范围：** 黑盒条件下，根据最终文本推断候选模型家族和已知模型版本，并在证据不足时输出 `unknown` 或 `abstain`。  
**不包含：** 仅凭被多次改写的最终文本，保证恢复最初生成模型；把分类概率表述为密码学或平台级来源证明。

## 1. 目标与术语

现有系统已经能够判断文本是否疑似 AI 生成。本阶段不重复解决二分类，而是替换老旧的字符 TF-IDF 闭集来源分类器，建立分层、可校准、能够拒绝未知模型的来源归因系统。

系统区分以下概念：

- `original_generator`：最初生成主要内容的模型。
- `last_transformer`：最后一次生成、润色、翻译或重写文本的模型。
- `generation_chain`：按时间排列的模型与人工处理链。
- `model_family`：Qwen、Llama、GPT、Claude、GLM 等家族。
- `model_id`：中转站或模型提供方返回的精确模型标识。
- `deployment_snapshot`：同一 `model_id` 在某一采集时间段内的部署快照。

V1 默认识别 `last_transformer`。只有 API 日志、水印、C2PA 或可信生成链记录能够支持更强的 `original_generator` 结论。

## 2. 输出契约

文本归因结果采用以下状态：

```text
exact        已知模型精确归因，家族头与精确模型头一致
family_only  能支持家族结论，但不足以支持精确版本
unknown      输入不属于任何已知模型分布
abstain      证据冲突、文本过短或不确定性过高
```

统一输出至少包含：

```json
{
  "status": "exact | family_only | unknown | abstain",
  "model_family": "qwen",
  "model_id": "provider/model-version-or-null",
  "confidence": 0.91,
  "known_probability": 0.96,
  "prediction_set": ["provider/model-a"],
  "prototype_distance": 0.18,
  "calibration_revision": "cal-2026-08-01",
  "dataset_revision": "attribution-corpus-v1",
  "evidence": [],
  "warnings": []
}
```

`confidence` 必须来自独立校准集，未经校准的余弦相似度、Softmax 或多个分支分数不能称为概率。

## 3. 推荐模型路线

### 3.1 主线：Qwen3-Embedding-0.6B + LoRA

首选骨干为 `Qwen/Qwen3-Embedding-0.6B`：

- 约 0.6B 参数，单张 24GB 显卡可快速迭代。
- 支持 100+ 语言，适合中英文及后续多语言扩展。
- 最长 32K 上下文；V1 训练先使用 1024/2048 token，长文采用分块聚合。
- 输出 1024 维向量，适合监督对比学习、模型原型和开放集距离。

训练方式：

- BF16 LoRA；0.6B 模型默认不使用 4-bit 量化，以避免不必要的量化噪声。
- 初始 `r=32`、`alpha=16`、`dropout=0.05`。
- 先适配注意力投影，再通过消融决定是否加入 `gate/up/down` FFN 层。
- 加入家族分类头、精确模型分类头和归一化嵌入头。

主线不是因为 Qwen 名称与候选生成模型相同，而是因为它在规模、中文、多语言、长上下文和可训练性之间平衡较好。必须用家族留出实验检查骨干自身是否对 Qwen 来源产生偏置。

### 3.2 强闭集基线：Qwen2.5-7B-Instruct + QLoRA

该路线复现 FDLLM 的关键设定：

- 4-bit NF4 QLoRA。
- 多分类交叉熵。
- 最大长度 1024。
- 同一数据切分、相同训练预算。

它用于回答“复杂分层算法是否真的优于 FDLLM 式闭集分类”，不作为默认部署模型。7B 模型只有在 OOD、未知拒识和风险覆盖率显著优于 0.6B 主线时才晋级。

### 3.3 编码器对照：BAAI/bge-m3

`BAAI/bge-m3` 支持 100+ 语言和 8192 token，作为多语长文本编码器对照。它的原始训练目标偏检索，可能主动消除文本表面风格，因此只有在模型归因指标上实证优于 Qwen3-Embedding 时才替换主线。

### 3.4 传统基线

保留当前字符 TF-IDF + LogisticRegression，作用仅限于：

- 发现明显的模板、标点和供应商免责声明泄漏。
- 衡量深度模型相对于低成本基线的真实增益。
- 在深度模型不可用时返回低等级研究提示，不输出精确来源结论。

## 4. 学习目标

设编码器输出为 `z = E(x)`，总损失为：

```text
L = L_exact
  + λ_family L_family
  + λ_contrast L_supervised_contrastive
  + λ_domain L_domain_invariance
```

- `L_exact`：已知 `model_id` 分类。
- `L_family`：模型家族分类。
- `L_supervised_contrastive`：同模型跨提示、领域、温度样本靠近；不同模型在相同提示上的回答远离。
- `L_domain_invariance`：降低主题、任务和模板对来源预测的贡献。

批次构造：

- 正样本：同模型、不同 `prompt_group_id`、领域或采样设置。
- 硬负样本：不同模型回答相同 `prompt_group_id`。
- 超硬负样本：同家族不同版本回答相同提示。

V1 先实现交叉熵与原型距离；监督对比和领域不变损失在数据工厂稳定后加入，防止在标签尚不可信时增加训练复杂度。

## 5. 开放集决策与校准

每个已知模型保存校准集嵌入原型和协方差统计。推理阶段联合使用：

- 精确模型校准概率。
- 家族校准概率。
- Energy score。
- 到最近模型原型的距离。
- Top-1/Top-2 margin。
- 家族头与精确模型头一致性。

决策顺序：

1. 文本低于最小有效长度或没有足够片段时返回 `abstain`。
2. 已知性分数低或原型距离超阈值时返回 `unknown`。
3. 精确模型预测集只有一个候选、margin 足够且家族一致时返回 `exact`。
4. 多个候选属于同一家族时返回 `family_only`。
5. 其余冲突返回 `abstain`。

阈值只在以下校准数据上拟合：

- 已知模型的独立 prompt/time 校准集。
- 完整留出的模型版本。
- 完整留出的模型家族。

优先使用 temperature scaling 加 conformal prediction set。模型输出还需报告风险—覆盖率曲线，不能只报告准确率。

## 6. 数据工厂

### 6.1 API 中转站前置核验

接入真实 API 前必须确认：

- 是否返回真实 `provider`、`model_id` 和请求 ID。
- 模型别名是否动态路由或无通知升级。
- 是否注入统一 system prompt、敏感词处理或输出后处理。
- 是否支持 temperature、top_p、max_tokens 和 seed。
- 是否可以取得原始请求与原始响应。

不能验证精确版本时，数据只标注到模型别名或家族，不伪造精确型号。

API Key 仅从环境变量读取。对象字符串、异常、日志、JSONL 和模型产物均不得包含 Key。

### 6.2 样本字段

每条样本至少保存：

```text
sample_id
prompt_group_id
prompt_template_id
prompt_hash
text
language
domain
task_type
length_bucket
provider
model_family
model_id
deployment_snapshot
collection_time
system_prompt_hash
temperature
top_p
seed
request_id
original_generator
last_transformer
generation_chain
transformation_type
dataset_revision
split
```

API Key、Authorization header 和未脱敏的用户隐私字段不进入样本。

### 6.3 配对生成

一个 `prompt_group_id` 对应一个语义任务，并发送给所有候选模型。所有来自同一 `prompt_group_id` 的回答只能进入同一个数据切分。

提示集覆盖：

- 长短问答、解释、论证、摘要、报告、通知、学术技术说明。
- 改写、翻译、创意写作和对话回答。
- 中英文及不同输出长度。
- 真实业务提示和合成受控提示。

拒答和免责声明保留在 `natural_traffic` 分层中，同时建立移除捷径特征的 `clean_content` 分层。两者分别评估。

### 6.4 生成规模

按门控逐步扩展：

1. API 冒烟：2,000–3,000 条。
2. 试验集：20,000–40,000 条。
3. 正式集：150,000–300,000 条。

只有当分组 OOD 评估证明有效后才进入下一规模，避免昂贵但无效的数据生产。

## 7. 数据切分

禁止按行随机切分。切分单位为 `prompt_group_id`，并额外构造：

- `prompt_held_out`
- `domain_held_out`
- `time_held_out`
- `version_held_out`
- `family_held_out_unknown`
- `transformation_held_out`
- `natural_traffic`

切分 manifest 在生成前固定，并由稳定哈希分配。训练脚本只能消费 manifest，不能自行重新随机切分。

## 8. 评估与晋级

闭集指标：

- Accuracy、Macro-F1、Family Macro-F1、Top-K。

校准指标：

- NLL、Brier、ECE、可靠性图。
- Risk-Coverage、在指定覆盖率下的准确率。

开放集指标：

- Unknown AUROC/AUPR。
- FPR@95TPR。
- OSCR。
- 已知模型准确率与未知拒识率联合报告。

鲁棒性指标：

- 润色、翻译、同义词、截断、人类编辑和跨模型重写后的性能变化。

模型晋级条件：

- 不以同分布 Macro-F1 单独决策。
- 主指标为 `domain/time held-out Family Macro-F1`、`OSCR` 和 `risk@coverage`。
- 新模型必须在相同数据、相同切分、相近训练预算下击败当前基线。

## 9. 现有代码集成

保留 `ProvenanceAttributionEngine.attribute(path, "text")` 公共入口。

改造边界：

- `provenance/text_attribution_schema.py`：数据与结果契约。
- `provenance/text_attribution_policy.py`：开放集分层决策。
- `provenance/relay_client.py`：中转站边界和密钥安全。
- `scripts/collect_text_attribution_data.py`：数据采集。
- `scripts/train_text_attribution_v2.py`：分组切分、基线和深度模型训练。
- `provenance/attribution.py`：加载 v2 产物并返回分层状态。
- `provenance/pipeline.py`：显式来源分析不再被 AI 二分类分数硬拦截。

旧 joblib 产物继续可读，但标记为 `legacy_closed_set`。v2 产物必须包含：

```text
artifact_schema_version
backbone_name
adapter_config
dataset_revision
split_manifest_hash
label_ontology
model_to_family
calibration_revision
thresholds
metrics
training_code_revision
```

## 10. 风险与边界

- API 别名漂移会制造错误标签，优先级高于模型结构优化。
- 同家族模型可能共享训练数据、tokenizer 和后训练方式，精确版本可能不可辨。
- 翻译和强改写可能抹去最初模型指纹。
- 分类骨干可能偏向自身家族，必须做整家族留出实验。
- 短文本证据不足时必须拒绝输出。
- 文本分类结论是统计证据，不替代平台日志、水印或签名。

## 11. 参考依据

- FDLLM, arXiv 2501.16029
- OpenTuringBench, EMNLP 2025
- StyleDistance, NAACL 2025
- BEDAA, Findings of ACL 2025
- Authorship Attribution in Multilingual Machine-Generated Texts, 2025
- Qwen3-Embedding-0.6B、Qwen2.5-7B-Instruct、BGE-M3 官方模型卡
