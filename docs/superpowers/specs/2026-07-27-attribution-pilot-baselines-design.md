# 多模型文本溯源先导基线设计

**日期：** 2026-07-27  
**输入：** `zhizengzeng-local-pilot-v1.jsonl`  
**目标：** 使用当前先导数据判断十条模型路线是否存在可泛化的文本差异，不将结果解释为完整模型家族泛化能力。

## 1. 数据口径

脚本直接读取采集 JSONL，标签使用 `family_id`。由于当前每个家族只有一个具体模型，本轮十分类同时混合了“家族”和“具体 API 路线”含义，报告必须显式注明这一限制。

进入基线的数据需要满足：

- `text` 非空；
- `finish_reason == "stop"`，排除 `length`、`max_tokens` 和 `content_filter`；
- `route_mismatch == false`；
- 输出语言与提示词语言一致；
- 对同一标签、提示词、温度下的完全相同文本只保留一条。

清洗前后数量、各类剔除原因和每个标签的剩余样本数写入 `dataset_audit.json`，不覆盖原始 JSONL。

## 2. 防泄漏切分

不能随机拆分单条文本。中英文提示词是语义配对的，例如
`zh-science-explain-001` 与 `en-science-explain-001` 必须进入同一折。

脚本从 `prompt_id` 去掉 `zh-` 或 `en-` 前缀，得到 `semantic_prompt_group`。使用固定随机种子和 6 折分组交叉验证，每次完整留出两个语义提示组。所有方法共享完全相同的折分配，输出逐样本 OOF 预测。

## 3. 三类基线

### 3.1 字符 n-gram + Logistic Regression

- 字符 TF-IDF，3–5 gram；
- `min_df=2`，限制最大特征数；
- Logistic Regression，`class_weight="balanced"`；
- 用于检验标点、格式、局部字符模式和稳定措辞是否可分。

### 3.2 混合 TF-IDF + Linear SVM

- 英文/空格词 1–2 gram；
- 字符边界 3–5 gram；
- FeatureUnion 合并两类稀疏特征；
- LinearSVC，`class_weight="balanced"`；
- 用于建立低成本强闭集基线。

### 3.3 冻结多语言编码器 + 线性探针

- 编码器：`intfloat/multilingual-e5-base`；
- 编码器全程 `eval()` 和 inference mode，不更新任何参数；
- 输入添加 `passage: ` 前缀，最大 512 tokens；
- 对最后隐层执行 attention-mask mean pooling 和 L2 归一化；
- 在 GPU 上批量提取一次嵌入并缓存为 NPZ；
- 每一折仅训练 Logistic Regression 线性分类头。

这里“训练”只指线性分类头训练，不是 LoRA，也不是编码器微调。

## 4. 指标与图表

每种方法输出：

- 6 折 Accuracy、Macro-F1、Weighted-F1 的均值和标准差；
- 聚合 OOF Accuracy、Macro-F1 和逐类别 Precision/Recall/F1；
- 10×10 归一化混淆矩阵；
- 按中文/英文、temperature、目标长度分组的 Accuracy 与 Macro-F1；
- 每条样本的真实标签、预测标签、折号和关键元数据。

生成以下可视化，并同时保存 PNG（300 DPI）和 PDF：

1. 三种方法总体指标对比；
2. 六折性能分布；
3. 三种方法的归一化混淆矩阵；
4. 按语言、温度和目标长度的分组性能。

## 5. 输出结构

默认输出到：

```text
outputs/attribution/zhizengzeng-local-pilot-v1-baselines/
├── run_config.json
├── dataset_audit.json
├── metrics.json
├── metrics.csv
├── predictions.jsonl
├── report.md
├── embeddings/
│   └── multilingual-e5-base.npz
└── figures/
    ├── overall_metrics.png/.pdf
    ├── fold_metrics.png/.pdf
    ├── confusion_matrices.png/.pdf
    └── subgroup_metrics.png/.pdf
```

传统模型和最终线性探针可另外保存为实验产物，但标记为 `pilot_only`，不得直接作为正式部署模型。

## 6. 服务器执行与错误处理

入口命令接收输入路径、输出目录、方法列表、编码器、设备、批大小和缓存目录。传统方法可以单独运行；冻结编码器下载失败或显存不足时给出明确错误，不影响已完成的传统基线结果。嵌入缓存包含输入文件指纹、模型 ID 和最大长度，参数不匹配时拒绝错误复用。

默认使用 `cuda:0` 和混合精度提取嵌入，两张 4090D 不做 DDP；第二张卡保留给后续并行骨干对比。

## 7. 结果解释边界

本轮结果只用于回答：

- 当前数据是否包含可学习的模型路线差异；
- 哪些模型最容易互相混淆；
- 传统风格特征和冻结语义特征谁更有效；
- 下一轮应扩充哪些提示词、模型版本和采样参数。

本轮不能证明：

- 对同一家族未见版本的泛化；
- 对未知模型或未知家族的识别；
- 在不同网关和不同时间上的稳定性；
- 最终开放集溯源能力。

完成本轮后，根据混淆矩阵选择每个家族的新增具体模型，扩充提示词并进入 3,200–10,000 条阶段数据。
