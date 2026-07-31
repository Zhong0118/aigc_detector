# 开放世界 LLM 文本来源归因阶段实施计划（初版，已停止）

> 本计划过早预设了 Qwen3-Embedding 主线，已被
> `2026-07-26-dual-mode-llm-source-attribution-stages.md` 取代。
> 已完成的样本 schema 和四态决策代码保留为通用基础设施，但后续研究不受现有代码结构约束。

> **执行要求：** 每个生产行为先写失败测试并确认按预期失败，再写最小实现。真实网络请求只在用户提供中转站协议后执行；测试默认使用本地受控响应。

**目标：** 建立可生成新鲜训练数据、按模型家族与精确版本分层归因、能够拒绝未知模型并输出校准置信度的文本来源归因系统。

**主线模型：** `Qwen/Qwen3-Embedding-0.6B + LoRA`  
**强基线：** `Qwen/Qwen2.5-7B-Instruct + 4-bit QLoRA + CE`  
**编码器对照：** `BAAI/bge-m3`  
**传统基线：** 字符 TF-IDF + LogisticRegression

---

## 阶段 0：契约、切分与开放集决策骨架

**文件：**

- 新建 `provenance/text_attribution_schema.py`
- 新建 `provenance/text_attribution_policy.py`
- 新建 `tests/test_text_attribution_v2.py`

- [ ] **任务 0.1：样本契约**

先写测试，覆盖：

- 合法直接生成样本能够序列化且不包含 API Key。
- 空文本、空 `prompt_group_id`、缺少家族或模型 ID 时明确失败。
- 改写样本同时保存 `original_generator`、`last_transformer` 和生成链。
- 样本 ID 与 prompt hash 稳定。

运行：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_text_attribution_v2 -v
```

预期 RED：模块不存在。

- [ ] **任务 0.2：分组切分**

先写测试，证明：

- 同一 `prompt_group_id` 永远进入同一 split。
- 相同 seed 产生相同结果。
- 不同 seed 能改变部分分配。
- 非法比例被拒绝。

实现稳定 SHA-256 分组分配，不读取全局随机状态。

- [ ] **任务 0.3：分层开放集决策**

先写测试，覆盖：

- 高已知性、高精确概率、家族一致时返回 `exact`。
- 精确候选不唯一但同属一家族时返回 `family_only`。
- 已知性不足或原型距离过远时返回 `unknown`。
- 精确头和家族头冲突时返回 `abstain`。
- 文本过短时返回 `abstain`。

实现纯策略模块，不加载模型，以便单元测试和后续替换校准器。

**阶段出口：**

- 数据和推理结果拥有稳定 schema。
- Unknown/Abstain 不再依赖具体深度模型才能测试。

---

## 阶段 1：API 中转站安全适配与冒烟采集

**前置输入：**

- 中转站 base URL。
- API 协议文档或一个脱敏请求示例。
- 可用模型别名列表。
- Key 对应的环境变量名；Key 本身不写入仓库。

**文件：**

- 新建 `provenance/relay_client.py`
- 新建 `scripts/collect_text_attribution_data.py`
- 扩展 `tests/test_text_attribution_v2.py`
- 修改 `.gitignore`

- [ ] **任务 1.1：安全配置**

测试：

- 缺失环境变量时快速失败。
- `repr`、异常和日志不出现 Key。
- base URL 只接受 HTTP(S)。
- Authorization header 仅在请求边界创建。

- [ ] **任务 1.2：协议适配**

测试本地受控响应，覆盖：

- OpenAI-compatible `/v1/chat/completions` 成功响应。
- 中转站自定义错误、429、5xx、超时和无效 JSON。
- 响应中的 `model` 与请求别名分别保存。
- 重试不会重复写入已完成 `sample_id`。

- [ ] **任务 1.3：不可变原始记录与归一化样本**

原始响应写入项目本地私有目录，训练 JSONL/Parquet 只包含白名单字段。写入采用临时文件加原子替换。

- [ ] **任务 1.4：2,000–3,000 条冒烟数据**

先以 10–12 个模型、100 个共享 prompt、两个采样设置采集。执行：

- 模型 ID 和别名漂移检查。
- 拒答、空响应、语言、长度、重复和提示回显统计。
- 统一 system prompt 或免责声明捷径检查。

**阶段出口：**

- 只有当精确标签可信时才允许进入精确型号训练。
- 标签不可信时自动降级为家族级任务。

---

## 阶段 2：数据集构造和泄漏审计

**文件：**

- 新建 `scripts/build_text_attribution_dataset.py`
- 新建 `tests/test_text_attribution_dataset.py`
- 新建 `data/attribution/manifests/` 下的版本化 manifest

- [ ] **任务 2.1：提示与 split manifest**

在任何批量生成前固定 `prompt_group_id`、任务、领域、语言、长度和 split。

- [ ] **任务 2.2：质量控制**

实现并测试：

- 精确重复和 MinHash 近重复。
- 提示回显。
- 供应商免责声明。
- 拒答、错误页和截断。
- 语言、长度、任务和领域覆盖。

- [ ] **任务 2.3：标签与生成链**

改写、翻译和人工编辑数据保存完整 `generation_chain`。训练目标分别生成 `last_transformer` 与 `original_generator` 视图，V1 只部署前者。

- [ ] **任务 2.4：数据卡**

输出版本、模型、时间、任务、语言、领域、采样参数、过滤原因、已知局限和 license/使用约束。

**阶段出口：**

- 20,000–40,000 条试验数据。
- prompt/domain/time/version/family-held-out 切分可复现。

---

## 阶段 3：传统基线与 FDLLM 强基线

**文件：**

- 修改 `scripts/train_text_attribution.py`
- 新建 `scripts/evaluate_text_attribution.py`
- 新建 `tests/test_text_attribution_training.py`

- [ ] **任务 3.1：修复传统基线切分**

将随机按行切分替换为读取 split manifest。产物保存完整数据版本、分布、指标和校准信息。

- [ ] **任务 3.2：TF-IDF 泄漏基线**

分别评估：

- 原始文本。
- 移除免责声明和固定模板。
- 长度、标点等纯风格特征。
- grouped/OOD 测试。

- [ ] **任务 3.3：FDLLM 式 Qwen2.5-7B QLoRA**

实现 CE 多分类强基线。训练参数、最大长度、LoRA 模块、GPU、种子和模型 revision 全部写入产物。

**阶段出口：**

- 获得可复现的下限和 FDLLM 风格上限。
- 不再用随机切分高分决定模型是否有效。

---

## 阶段 4：Qwen3-Embedding 分层主模型

**文件：**

- 新建 `training/text_attribution/model.py`
- 新建 `training/text_attribution/data.py`
- 新建 `training/text_attribution/losses.py`
- 新建 `scripts/train_text_attribution_v2.py`
- 扩展训练测试

- [ ] **任务 4.1：家族与精确模型双头**

实现共享嵌入、家族头、精确型号头。先使用 CE 验证数据与接口。

- [ ] **任务 4.2：LoRA 消融**

比较：

- 冻结骨干只训练头。
- 注意力 LoRA。
- 注意力 + FFN LoRA。

同一切分、相同 seed、相近训练 token 预算。

- [ ] **任务 4.3：监督对比学习**

按共享 prompt 构造硬负样本，加入 SupCon。检查同模型跨领域距离和同家族近邻分离。

- [ ] **任务 4.4：BGE-M3 对照**

在同一数据和评估脚本上训练 BGE-M3。只依据 OOD、OSCR、校准和风险覆盖率选择骨干。

**阶段出口：**

- 主模型选择报告。
- 不能仅凭同分布 Macro-F1 晋级。

---

## 阶段 5：开放集、校准与增量模型

**文件：**

- 新建 `training/text_attribution/calibration.py`
- 新建 `training/text_attribution/prototypes.py`
- 修改 `provenance/text_attribution_policy.py`

- [ ] **任务 5.1：模型原型与距离**

从独立校准集计算模型和家族原型、类内距离统计。

- [ ] **任务 5.2：概率校准**

实现 temperature scaling，输出 NLL、Brier 和 ECE。

- [ ] **任务 5.3：Unknown 阈值与 conformal set**

用完整留出的模型和家族拟合 known/unknown 阈值，输出 prediction set。

- [ ] **任务 5.4：增量新模型**

比较：

- 只增加原型。
- 冻结骨干训练新分类头。
- LoRA + 旧类 replay + 蒸馏。

检查旧类遗忘，不接受仅报告新类准确率。

**阶段出口：**

- `exact/family_only/unknown/abstain` 四状态可校准运行。

---

## 阶段 6：鲁棒性、主动探针与现有系统集成

**文件：**

- 修改 `provenance/attribution.py`
- 修改 `provenance/pipeline.py`
- 修改 `api/dashboard.py`
- 修改 `reports/generator.py`
- 扩展现有 provenance/report 测试

- [ ] **任务 6.1：鲁棒性集**

评估翻译、润色、同义替换、截断、人工编辑、跨模型重写和混合文本。

- [ ] **任务 6.2：主动黑盒验证**

如果推理时可以查询待识别端点，从候选 prompt 中选择模型间距离大、模型内方差小的 10–30 个探针，聚合多响应后验。

- [ ] **任务 6.3：v2 产物加载**

`ProvenanceAttributionEngine` 兼容旧产物，并优先读取 v2 的校准器、原型、标签本体和阈值。

- [ ] **任务 6.4：解除不合理硬门控**

显式来源分析请求不受 AI 二分类阈值硬拦截；普通自动流程仍可采用成本门控。

- [ ] **任务 6.5：用户界面与报告**

显示模型家族、精确型号、Unknown/Abstain、校准 revision、证据和警告；不把统计归因写成确定来源证明。

---

## 每阶段验证命令

聚焦测试：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_text_attribution_v2 -v
```

现有归因回归：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest tests.test_attribution_adapters tests.test_provenance_pipeline -v
```

完整测试：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m unittest discover -s tests -v
```

语法检查：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m compileall provenance scripts training api reports
```

依赖检查：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m pip check
```

## 真实 API 测试所需信息

用户提供中转站时，只需要：

- base URL 和协议页面。
- 一个不含真实 Key 的请求/响应示例。
- 模型别名列表。
- Key 将写入哪个环境变量，例如 `AIGC_RELAY_API_KEY`。

不要把 Key 写入 `config.yaml`、`.env` 示例、测试 fixture、命令历史或聊天中可复制的代码块。真实请求完成后必须检查日志与生成数据中不存在 Key。
