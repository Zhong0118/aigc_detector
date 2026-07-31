# 双模式开放世界 LLM 来源溯源研究设计

**状态：** 待方案确认后实施  
**日期：** 2026-07-26  
**计算资源：** 2 × RTX 4090D  
**数据来源：** CloseAI、智增增双中转站及公开数据集  
**原则：** 不沿用现有代码作为算法边界，不预设 Qwen、LoRA 或任何单一骨干必胜。

## 1. 研究问题重新定义

“LLM 来源溯源”包含两个不同问题，不能用一个分类器混在一起。

### 1.1 被动文本归因

输入是一篇已经存在的文本 `x`，无法再次访问生成端点。目标是：

```text
P(模型家族 | x)
P(已知具体模型 | x, 模型家族)
P(未知模型 | x)
```

这是 FDLLM、OpenTuringBench 和当前项目主要处理的问题。它适合内容审核和历史文本分析，但强改写后存在不可辨识上限。

### 1.2 主动端点指纹识别

可以向待识别黑盒端点发送一组诊断问题 `P={p1,...,pk}`，观察响应集合 `Y`：

```text
P(模型版本 | Y, P)
```

这属于 LLMmap 一类的主动指纹。它往往只需少量高信息量查询，精确版本识别能力通常强于单篇被动文本。

### 1.3 推荐目标

建立双模式系统：

- **Passive-AA：** 任意文本、分层开放集归因。
- **Active-FP：** 可查询端点、主动探针和多响应聚合。

二者共享数据工厂和模型本体，但训练目标、评估指标和可声明的结论分开。

## 2. 变量与因果视角

观测文本不是只由模型决定：

```text
x ~ P(x | m, p, d, l, s, g, t, r)
```

其中：

- `m`：生成模型及版本。
- `p`：prompt 语义与模板。
- `d`：领域和任务。
- `l`：语言。
- `s`：temperature、top_p、seed 等采样参数。
- `g`：API 网关及其 system prompt、路由和后处理。
- `t`：采集时间和部署快照。
- `r`：润色、翻译、人工修改等变换链。

真正希望保留的是与 `m` 有关的信号，抑制 `p/d/g` 等捷径，并明确测量 `l/s/t/r` 对模型指纹的影响。

双中转站特别有价值，因为 `g` 可以成为可控变量。对同一模型、prompt 和采样参数分别通过两个网关生成，能够检测分类器是在识别上游模型，还是在识别网关。

## 3. 三条候选算法路线

### 路线 A：单骨干闭集分类

```text
文本 → Transformer → Softmax(model_id)
```

优点是实现快，可复现 FDLLM；缺点是主题泄漏、未知模型强制分类和时间漂移。只作为基线。

### 路线 B：模型无关的配对度量学习

```text
文本 → 指纹编码器 → 嵌入空间 → 模型/家族原型 → 开放集决策
```

利用同 prompt 跨模型和同模型跨 prompt 的配对数据，把“内容相同而模型不同”的回答作为硬负样本。适合增量模型和 Unknown，但可能丢失互补的字符、tokenizer 与似然信息。

### 路线 C：多视图分层指纹网络

把语义残差、字节风格和观察者似然画像融合，并同时支持被动与主动模式。训练和数据要求最高，但最符合真实溯源。

**推荐：** 路线 C 作为研究主线，A/B 为消融和退化版本；不预先指定其中任何一个 Transformer 骨干。

## 4. 多视图指纹表示

### 4.1 语义—风格编码分支

编码器 `f_sem(x)` 学习高层生成习惯、结构和推理表达。候选骨干通过实验选择：

- 多语编码器：XLM-R-large、BGE-M3 或同等级长文本编码器。
- 小型生成骨干：0.5B–2B 多语 causal LM。
- 其他当前可用的中英双语编码器。

先比较冻结特征和线性探针，再对前两名比较全量微调、LoRA 和 QLoRA。

### 4.2 字节与 tokenizer 痕迹分支

`f_byte(x)` 直接处理 UTF-8 byte、字符 n-gram 或 byte-level Transformer，捕获：

- 标点和空白模式。
- tokenizer 边界痕迹。
- 中英文符号、数字和列表格式。
- 局部重复与短语模板。

该分支不依赖某一家族 tokenizer，可使用字符 CNN、轻量 byte Transformer 或 ByT5 小模型。传统 TF-IDF 是它的最低成本版本。

### 4.3 观察者语言模型画像

选择若干不同架构的小型开放权重观察者 `R1,...,RJ`，对文本计算：

- token log-likelihood 的均值、方差、偏度和分位数。
- rank/entropy 直方图。
- 不同 tokenizer 下的压缩率。
- 句段级似然变化和重复模式。

得到：

```text
z_obs = [stats(R1, x), ..., stats(RJ, x)]
```

它不是试图让观察者“猜来源”，而是用多个坐标系描述文本在不同模型分布下的位置。

### 4.4 多视图融合

```text
z_sem  = f_sem(x)
z_byte = f_byte(x)
z_obs  = f_obs(x)

α = softmax(G([z_sem; z_byte; z_obs; quality(x)]))
z = α1 z_sem + α2 z_byte + α3 z_obs
```

训练时使用 branch dropout，防止模型只依赖免责声明、文本长度或某一个观察者。短文本、长文本和不同语言可以学习不同融合权重。

## 5. 分层配对训练目标

### 5.1 家族与精确模型分层

使用条件分解：

```text
P(m | x) = P(family(m) | x) · P(m | family(m), x)
```

这使系统可以区分：

- 新家族：`unknown_family`。
- 已知家族的新版本：`family_only`。
- 已知精确版本：`exact`。

### 5.2 总损失

```text
L = λm L_model
  + λf L_family
  + λp L_paired_triplet
  + λc L_supervised_contrastive
  + λh L_hierarchy
  + λn L_nuisance
  + λo L_outlier_exposure
```

其中：

```text
L_model  = CE(y_model, h_model(z))
L_family = CE(y_family, h_family(z))
```

配对三元组损失：

```text
L_paired_triplet =
max(0,
    d(z[m,p1], z[m,p2])
  - d(z[m,p],  z[n,p])
  + margin)
```

- 正对：同一模型、不同 prompt/领域/温度/网关。
- 硬负对：不同模型、同一 prompt。
- 超硬负对：同一家族不同版本、同一 prompt。

层次一致性损失：

```text
L_hierarchy =
KL(sum_model_in_family P(model|x) || P(family|x))
```

干扰变量抑制使用 gradient reversal 或互信息最小化：

```text
L_nuisance =
- CE(prompt_group, h_prompt(GRL(z)))
- CE(gateway,      h_gateway(GRL(z)))
- CE(domain,       h_domain(GRL(z)))
```

目标不是让语言完全不可预测，而是通过消融决定语言是否作为独立专家路由，避免错误地删除真实的跨语言指纹。

Outlier exposure 每个训练 fold 留出完整模型或家族，将它们只作为 unknown 校准样本，禁止进入已知类训练。

## 6. 开放集评分

对每个已知模型维护原型 `μm` 和协方差 `Σm`：

```text
d_m(x) = (z-μm)^T Σm^-1 (z-μm)
```

联合已知性分数：

```text
S_known(x) =
a · [-Energy(x)]
+ b · [-min_m d_m(x)]
+ c · Top1Margin(x)
+ d · FamilyModelAgreement(x)
+ e · BranchAgreement(x)
```

系数只在独立 known/unknown 校准集上拟合。然后用 conformal prediction 产生候选集合：

- 候选集为空：`unknown`。
- 一个精确模型：`exact`。
- 多个模型但同一族：`family_only`。
- 多家族冲突：`abstain`。

必须报告 ECE、Brier、NLL、OSCR、FPR@95TPR 和风险—覆盖率，不能只报告 Macro-F1。

## 7. 主动指纹与探针优化

训练数据中每个 prompt 都可以估计：

```text
J(p) =
InterModelDistance(p)
/
(IntraModelVariance(p) + ε)
```

选择探针集合时同时考虑互补性：

```text
P* = argmax_P [
sum_{p in P} J(p)
- β sum_{pi != pj} Similarity(pi, pj)
]
```

主动端点推理不是逐条投票，而是聚合似然：

```text
log P(m | Y, P)
∝ log P(m) + sum_i w(pi) log P(yi | m, pi)
```

初期比较：

- 随机 prompt。
- 人工设计 prompt。
- 基于 `J(p)` 的自动选择。
- 8、16、32 次查询的成本—准确率曲线。

## 8. 双网关数据设计

### 8.1 因子实验

```text
模型 m × 网关 g × prompt p × 采样 s × 重复 r × 时间 t
```

建议规模：

- 协议冒烟：每站 2–3 个模型，每模型 5–10 个请求。
- 数据冒烟：8 模型 × 2 网关 × 100 prompt × 2 设置 = 3,200 条。
- 试验集：12 模型 × 2 网关 × 500 prompt × 3 设置 × 2 重复 = 72,000 条。
- 时间复采：隔周或隔月对固定探针子集重采，测量模型和路由漂移。

### 8.2 网关泄漏测试

训练一个只预测网关的分类器：

- 若同模型条件下网关准确率接近随机，说明网关痕迹较弱。
- 若准确率很高，必须定位 system prompt、免责声明、JSON 转换或后处理，并建立去捷径数据视图。
- 来源模型训练按网关留出：在网关 A 训练、网关 B 测试，反向再测。

### 8.3 模型标签可信性

每次采集保存：

- 请求中的模型别名。
- 响应中的 `model` 字段。
- 网关 request ID。
- 请求时间。
- OpenAI/Claude/Gemini 协议类型。

同一别名若返回不同上游 `model`，只标注为 alias，不当作精确版本。模型列表和实际可调用性通过轮换后的 Key 单独验证。

## 9. 模型选择不是拍脑袋

第一轮候选不超过 5 个骨干，每个只做冻结特征：

1. 一个经典多语 encoder。
2. 一个长文本多语 embedding encoder。
3. 一个 0.5B–1B 多语 causal LM。
4. 一个 1B–3B 多语 causal LM。
5. 一个 byte-level 模型。

晋级顺序：

```text
冻结特征 + 线性探针
        ↓ 前 2 名
全量微调 vs LoRA
        ↓ 前 1–2 名
加入配对/对比/开放集目标
        ↓
多视图融合
```

LoRA 只是参数更新策略：

- ≤600M encoder：比较全量微调与 LoRA。
- 1B–4B：优先 LoRA，再视显存比较全量微调。
- 7B–8B：QLoRA 作为强基线，只有显著增益才保留。

不因 FDLLM 使用 Qwen 就默认继续使用 Qwen，也不因 embedding benchmark 高就假设它能保留生成风格。

## 10. 2 × 4090D 使用时机

### 现在：不需要服务器

完成：

- API Key 轮换。
- 两网关协议和模型列表核验。
- prompt 因子设计。
- 3,200 条数据冒烟与标签/泄漏审计。

数据生成是网络和 Token 成本约束，不是 GPU 约束。

### 第一次上服务器：冒烟数据通过后

使用 3,200–10,000 条数据做：

- 候选骨干冻结特征提取。
- 线性探针。
- TF-IDF/byte 基线。
- 网关预测和 prompt 泄漏审计。

两张卡分别跑不同骨干，比一开始做 DDP 更高效。

### 第二次上服务器：20,000–72,000 条试验集后

运行：

- 前两名骨干的全量微调/LoRA 消融。
- 分层 CE、配对三元组和 SupCon。
- 家族/版本留出开放集实验。
- 概率校准和 risk-coverage。

### 最终训练

只有算法和切分冻结后才扩大到 10 万级以上，并使用：

- 单模型 DDP/FSDP，或
- 两卡分别训练主模型和独立复现实验。

模型训练完成不等于部署。只有 frozen test、时间留出和 Unknown 指标达到门槛后，才打包推理服务。

## 11. 与现有代码的关系

现有 `provenance` 代码仅作为：

- 旧 TF-IDF 基线。
- 最终 API 兼容层。
- 测试过的通用 sample schema 与四态决策接口。

新研究实现放入独立目录，例如：

```text
research/text_attribution/
training/text_attribution/
data/attribution/
```

它可以独立训练、评估和导出，不调用旧融合逻辑。确定获胜算法后再写一个薄适配器接入现有系统。

## 12. 安全

- 已经贴入聊天的 CloseAI Key 视为泄露，必须轮换。
- 新 Key 只放 `.env.local` 或服务器 secret manager。
- 测试、日志、异常、JSONL、模型产物和命令行均不包含 Key。
- 先用极小请求验证费用和速率，不直接批量生成。
- 每站配置独立预算、并发和硬停止上限。

