# 双模式 LLM 来源溯源阶段文档

**目标：** 先证明任务、数据和算法成立，再训练和部署；现有代码不作为研究约束。  
**资源：** 本地开发环境、CloseAI/智增增双网关、2 × RTX 4090D。  
**当前状态：** 暂停继续扩展旧归因代码，进入协议、数据与算法验证。

## 阶段 R0：安全和任务冻结

### 工作

- 轮换已经暴露的 CloseAI Key。
- 将两个站点使用不同环境变量和独立预算。
- 冻结两类任务：
  - Passive-AA：单篇文本被动归因。
  - Active-FP：可查询端点主动指纹。
- 冻结来源语义：V1 被动模式默认识别 `last_transformer`。

### 交付

- threat model。
- 标签本体。
- 预算与停止条件。

### 出口条件

- Key 不出现在 Git、日志和聊天复制代码中。
- 研究目标不再混淆被动归因与主动指纹。

## 阶段 R1：双网关协议核验

### 工作

- CloseAI：
  - `https://api.openai-proxy.org/v1`
  - 老师说明中的 Claude/Gemini 地址分别带 `/anthropic/v1`、`/google/v1`
  - CloseAI 当前公开 SDK 文档给出的 base URL 分别是
    `https://api.openai-proxy.org/anthropic`、`https://api.openai-proxy.org/google`
  - 必须根据实际 SDK 拼接规则做最小请求，防止出现重复 `/v1/v1/messages`
- 智增增：
  - `https://api.zhizengzeng.com/v1`
  - `https://api.zhizengzeng.com/anthropic`
  - `https://api.zhizengzeng.com/google`
- 用轮换后的 Key 测试 `/models` 或平台模型列表。
- 每站选择 2–3 个低成本模型，每模型做 5–10 个最小请求。
- 核对请求别名、响应 `model`、request ID、速率限制和错误格式。
- 比较 OpenAI 兼容格式与 Claude/Gemini 原生格式是否引入不同文本。

### 出口条件

- 能确认哪些标签是精确模型、哪些只是网关 alias。
- 明确两个站点的并发、重试和计费边界。

## 阶段 R2：3,200 条数据冒烟

### 设计

```text
8 模型 × 2 网关 × 100 prompt × 2 采样设置
```

prompt 覆盖中英文、任务、领域和长度；同一个 `prompt_group_id` 发给所有模型和网关。

### 审计

- 重复、拒答、提示回显、固定免责声明。
- 网关可预测性。
- 模型可预测性。
- 同模型跨网关距离。
- 同 prompt 跨模型距离。
- 模型标签和模型版本漂移。

### 出口条件

- 标签错误率和缺失率可接受。
- 不存在一个简单免责声明即可区分模型的明显捷径。
- 至少有若干同家族近邻模型可用于验证精确归因难度。

## 阶段 R3：第一次使用 4090D——骨干筛选

### 数据

3,200–10,000 条经过审计的数据，不等待正式大数据。

### 实验

- TF-IDF/字符基线。
- byte-level 基线。
- 4–5 个候选骨干冻结特征。
- 线性探针。
- 同 prompt grouped split。
- 网关 A 训练、网关 B 测试及反向实验。
- 留出完整模型和完整家族。

### GPU 分配

- GPU 0：候选骨干 A/C。
- GPU 1：候选骨干 B/D。
- 暂不做双卡 DDP。

### 出口条件

- 依据 OOD Family-F1、OSCR、ECE 和风险覆盖率选择前两名。
- 不能依据同分布 Accuracy 选择。

## 阶段 R4：72,000 条试验集

### 数据

```text
12 模型 × 2 网关 × 500 prompt × 3 设置 × 2 重复
```

另外采集：

- 跨模型润色。
- 翻译。
- 截断。
- 人工轻/中/重度编辑。
- 时间复采探针。

### 出口条件

- 生成链标签完整。
- prompt/domain/gateway/time/model-family 切分固定。
- 数据卡和 manifest 可复现。

## 阶段 R5：第二次使用 4090D——算法消融

### 实验矩阵

1. 闭集 CE。
2. 家族 + 精确模型双头。
3. 双头 + paired triplet。
4. 双头 + SupCon。
5. 加 nuisance suppression。
6. 加 prototype/open-set。
7. 语义 + byte。
8. 语义 + byte + observer profile。

### 训练策略矩阵

- 冻结骨干。
- LoRA。
- 全量微调。
- 7B/8B QLoRA 强基线。

### 双卡使用

- 前期两卡并行跑消融。
- 最优大模型再使用 DDP/FSDP。
- 所有实验固定数据 manifest、seed 和 Token 预算。

### 出口条件

- 至少一个方法在多种 OOD 切分稳定超过传统与 FDLLM 式基线。
- Unknown FPR、ECE 和 risk-coverage 达到预设门槛。

## 阶段 R6：主动探针

### 工作

- 从 500 个候选 prompt 计算区分度 `J(p)`。
- 建立 8/16/32 个探针集合。
- 比较随机、人工、自动探针。
- 测试隐藏 system prompt、温度变化、RAG/CoT 和网关变化。

### 出口条件

- 报告查询成本—准确率曲线。
- 主动结果与被动结果分别报告。

## 阶段 R7：时间漂移和增量模型

### 工作

- 固定探针隔周或隔月重采。
- 监控模型原型漂移。
- 新模型依次比较：
  - 只加原型。
  - 新增分类头。
  - LoRA + replay + 蒸馏。
- 检查旧类遗忘。

### 出口条件

- 形成模型更新 SOP 和回滚机制。

## 阶段 R8：部署

### 何时部署

满足以下条件后才部署：

- frozen test 从未参与调参。
- time-held-out 已通过。
- unknown 和 calibration 指标达到门槛。
- 推理显存、吞吐和延迟已测量。
- 所有产物记录数据、代码、骨干和校准 revision。

### 形态

- 独立推理服务，不直接嵌入旧脚本。
- 输出 `exact/family_only/unknown/abstain`。
- 现有项目仅通过薄适配器调用服务。
- 强证据、被动统计归因和主动指纹分层展示。

## 下一步唯一应做的事

1. 轮换 CloseAI Key。
2. 将两个新 Key 分别放入 `.env.local`。
3. 提供智增增 Key 是否也已分配、两个站点可用余额/预算上限。
4. 进行阶段 R1 的极小协议测试，不批量生成。
