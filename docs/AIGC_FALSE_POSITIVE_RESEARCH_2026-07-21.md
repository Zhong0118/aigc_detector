# AIGC 文本误判控制技术设计

日期：2026-07-21  
范围：仅解决“人类文本被误判为 AI”的问题，重点覆盖中文经典文学、现代文学、论文/技术文档、公文/新闻和普通问答。  
非目标：本阶段不做生成模型溯源，不以固定 20,000 条数据为目标，不把互联网检索结果当作作者身份判据。  
状态：可直接指导后续编码；本文档本身不修改运行代码。

## 1. 结论

现有文本检测模型可以继续沿用，第一阶段不需要重新训练一套大型检测模型。需要新增的是一个独立的 `FalsePositiveController`：它接收现有各检测分支的原始分数、文本长度、文体、模型分歧和分布外风险，输出经过校准的风险分数、三段式结论和拒判原因。这个控制器可先采用逻辑回归或小型 LightGBM，训练成本远低于重新训练大模型；最终高风险阈值不手工写死为 0.75 或 0.8，而是由独立的人类校准集按文体和长度计算。

只有当“冻结现有模型 + 误判控制器 + 分组阈值”仍无法达到困难人类文本的误报目标时，第二阶段才微调现有本地 encoder。微调的重点不是继续堆普通 AI 样本，而是加入被当前系统打高分的真实人类文本，即 hard negatives。云端 API 模型无法微调，只能通过控制器校准、降低权重或在某些文体中禁用。

推荐最终输出：`low_risk`、`uncertain`、`high_risk_ai`、`insufficient_evidence`。任何输出都表示风险等级而非作者身份的证明。联网或本地来源匹配另行输出 `source_evidence`，只有“可信、已知为人类且有时间/版本依据的本地语料”才能作为降低风险的证据；普通网页命中只能说明文字已公开，不能直接说明不是 AI。

## 2. 第一性原理

AI 文本检测是在高度重叠的“人类文本分布”和“机器文本分布”之间进行统计检验。专业文档、古典文学、公文和非母语写作经常具有低困惑度、固定结构和强连贯性，这些特征也可能被检测器学成“AI 特征”。因此，误判不是简单调高一个统一阈值就能彻底解决，而是训练分布、目标文体、文本长度和模型置信度共同失配的结果。

本项目应把优化目标写成约束问题：在总体及关键人类文体的高风险误报率 `FPR <= alpha` 的前提下，尽可能提高 AI 召回率和自动判定覆盖率。建议第一版取 `alpha=0.02`。这与“最大化 accuracy”不同，因为真实业务中 AI 文本比例可能较低，即使总体准确率很高，少量误报也会使高风险结果中混入大量真人文本。

校准器不会凭空创造识别能力。它的作用是把多个不可直接比较的原始分数映射到当前业务分布，并选择满足误报约束的决策阈值。真正降低误判的三个来源是：加入目标文体的人类困难样本、对分布外文本拒判、在文章级使用经过校准的组合证据。若基础模型对某一文体完全没有区分能力，最终仍需要微调基础模型。

## 3. 可直接复用的数据

不存在一个数据集可以单独解决中文经典文学、论文、公文和技术文档的全部误判。可复用数据适合提供普通样本、跨域测试和校准方法，但项目仍需要一份规模不必很大、来源可信的目标困难人类集。数据必须保留来源和许可证，不能因为文件能下载就默认允许产品训练。

| 数据 | 可用内容 | 本项目用途 | 可直接程度与注意事项 |
|---|---|---|---|
| [MAGE 数据集](https://huggingface.co/datasets/yaful/MAGE) | 约 44.8 万条人类/机器文本，10 类人类来源、27 个生成模型，含 OOD 与复述测试；字段为 `text/label/src`，其中 `0=machine, 1=human` | 预训练误判控制器、验证跨领域退化、复现 OOD 流程 | Hugging Face 数据卡标注 Apache-2.0，可直接下载；以英文为主，不能用于确定中文最终阈值 |
| [HC3-Chinese](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) | 中文人类回答与 ChatGPT 回答 | 补充中文问答域、训练基础融合器 | 数据内部许可证混合：部分为 MIT、CC-BY、CC0，另有 CC-BY-NC、未知许可及百度百科来源。只导入许可证白名单子集，并保存 `source` |
| [DetectRL-ZH / NLPCC 2025 Task 1](https://github.com/NLP2CT/NLPCC-2025-Task1) | 32,400 条训练数据、2,800 条开发数据，覆盖中文学术、摘要等领域和多种模型；测试含攻击、混合和不同长度 | 当前最贴近中文场景的研究基准，适合外部评估和复现实验 | 官方明确开发集只能调参、不能训练，且比赛数据有额外限制；仓库未给出足够明确的生产再分发许可，因此默认作为研究/评估集，正式交付前需获得许可确认 |
| [MULTITuDE](https://zenodo.org/records/10013755) | 74,081 条、11 种语言（含中文）、8 个生成模型 | 多语言与跨生成器外部测试 | 数据来自多个上游语料，许可证随来源变化；先做评估，逐源核验后再决定是否进入训练 |
| [RealDet](https://github.com/Xiaoweizhu57/RealDet) | 已发布校准集、人类/机器测试集和 MCP 保形校准实现 | 复现固定 FPR 的阈值计算，不作为中文主训练集 | 仓库规模较小且未展示清晰许可证，默认只参考方法；需要使用其数据时先确认授权 |
| [中文维基百科 dump](https://dumps.wikimedia.org/zhwiki/latest/) | 百科、专业概念和说明性真人文本 | 构建专业/知识型 hard negatives | Wikimedia 文本通常为 CC BY-SA 4.0/GFDL，必须保留归属、版本和许可证，并遵守相同方式共享要求 |
| [中文维基文库 dump](https://dumps.wikimedia.org/zhwikisource/latest/) | 古典文学和公共领域文献 | 构建经典文学 hard negatives 与可信本地来源库 | 作品本身可能属于公有领域，但站点整理文本仍需按 Wikimedia 条款处理；保存作品、版本、来源 URL 和许可证 |

建议数据使用顺序为：MAGE 用于快速建立通用融合基线；HC3 的许可白名单数据补充中文问答；DetectRL-ZH、MULTITuDE 和 RealDet 保留为外部审计；中文维基百科/文库及项目内部有授权的论文、公文、技术文档用于构建真正决定误报表现的困难人类校准集。不要使用由其他检测器自动打标签、但没有人工来源证明的数据作为“人类真值”。

## 4. 数据规模与划分

本阶段不追求某个好看的固定总数。决定误报测量可信度的是独立人类样本数、文体覆盖和是否存在泄漏。建议先建设 4,000-8,000 条高质量样本完成第一版，其中至少 50% 为真实人类文本，并优先覆盖当前误判严重的文体。公开数据可以提供 AI 样本和普通人类样本，项目自建部分主要补困难人类样本。

建议把文体先合并为四个稳定路由：`literature`、`academic_technical`、`official_news`、`general_qa`；长度分为 `<200`、`200-800`、`>800` 个有效汉字。某一“文体 × 长度”单元的人类校准样本少于 300 时，不单独学习阈值，而是回退到文体阈值、语言阈值，最后回退到全局阈值。每个关键文体的人类锁定测试样本最低 500 条，推荐 1,000 条；这比随意扩充大量同分布 AI 文本更有价值。

数据固定划分为 `train`、`validation`、`calibration`、`locked_test` 四部分。`train` 只训练融合器、路由器或可选的基础模型；`validation` 选择特征、超参数、Platt 概率映射和低风险阈值；`calibration` 在所有模型均已冻结后只计算高风险阈值和 OOD 阈值；`locked_test` 只做最终验收。若必须在 calibration 上拟合概率映射，则应预先把它再固定拆成 `probability_calibration` 和 `threshold_calibration`，二者不能复用。同一原始文档、同一章节的相邻切片、同一提示词对应的人机版本、复述和改写必须通过 `root_document_id` 放在同一 split。划分前执行精确哈希和字符 n-gram MinHash 去重，禁止先切片再随机划分。

统一记录格式：

```json
{
  "sample_id": "uuid",
  "root_document_id": "stable-root-id",
  "text": "...",
  "label": 0,
  "language": "zh",
  "domain": "literature",
  "is_hard_negative": true,
  "source_dataset": "zhwikisource",
  "source_uri": "...",
  "source_version": "2026-07 dump",
  "license": "CC-BY-SA-4.0",
  "generator": null,
  "transform": "original",
  "content_sha256": "..."
}
```

其中统一规定 `label=0` 为人类、`label=1` 为机器；导入 MAGE 时必须显式反转其标签，避免静默训练错误。无法确认标签或许可证的数据进入隔离区，不进入训练、校准和验收。

## 5. 目标架构

```text
输入文本
  -> 文本规范化、语言/长度检查
  -> 文体路由 TextRiskRouter
  -> 可选来源证据 SourceEvidence（不参与普通网页“洗白”）
  -> 上下文分段 TextSegmenter
  -> 冻结的现有 DetectorAdapter[]
  -> 片段级特征与文章级聚合
  -> OODDetector
  -> ScoreFuser（小型可训练控制器）
  -> GroupCalibrator（分文体/长度阈值）
  -> DecisionPolicy
  -> low_risk / uncertain / high_risk_ai / insufficient_evidence
```

### 5.1 复用现有模型

`detection/providers.py` 仍负责调用现有本地模型和 API provider，但不再直接对 `status=ok` 的分数做简单平均。每个 provider 必须返回原始 AI 分数、状态、实际检测字符范围和模型版本。无分数、请求失败、文本过短或模型不支持该语言时必须保留缺失标记，不能用 `0.0` 代替，否则“没有证据”会被错误解释成“确定为人类”。

现有 `detection.threshold: 0.8` 仅作为控制器未加载时的兼容回退值，不再作为文本最终判定阈值。图像、音频和视频仍可保留原逻辑，本设计只替换文本融合路径。

### 5.2 文体路由

路由器不判断 AI，只选择参考分布。MVP 可以使用句法/关键词规则加文本 embedding 最近质心；有足够标注后再训练小型中文 encoder。输出应为 `domain`、`confidence` 和候选概率。路由置信度过低时使用更保守的全局阈值并提高拒判概率，不能强行塞入某个低阈值文体。

### 5.3 上下文分段和聚合

联网检索使用的 80-150 字切片不适合作为 AI 检测窗口。AI 检测建议按句界切成 256-512 token 窗口，重叠约 25%，并保留前后 2 句上下文。短于最小可靠长度的文本直接返回 `insufficient_evidence` 或 `uncertain`。

文章聚合禁止使用片段最大分。重叠区域只归属于一个中心窗口，先计算长度加权均值、片段分数标准差、高风险覆盖率和相邻高风险窗口数，再把这些统计量交给融合器。这样可以定位局部风险，同时避免一篇长文因一个异常片段被整体判为 AI。

### 5.4 OOD 拒判

使用冻结中文 encoder 的文档 embedding，在人类 train 集上按文体建立参考向量。MVP 可先用 PCA 后的 kNN 余弦距离，再使用人类 calibration 距离确定 OOD 高分位阈值；计算 train 内部距离时必须排除样本自身。距离超过阈值时标记 `ood=true`。OOD 不是 AI 证据，它只表示当前阈值不可靠，因此应把结果推向 `uncertain`。古文、少数体裁和新型技术格式会主要受益于这一规则。

### 5.5 误判控制器

第一版使用带 L2 正则的逻辑回归，输入特征建议限定为：

```text
各 provider 原始分数
各 provider 缺失掩码
片段长度加权均值、P90、标准差
高分片段覆盖率、相邻高分片段数
有效字符数和窗口数
文体路由概率与置信度
OOD 距离和 OOD 标记
可信人类来源覆盖率（若有）
```

不要把网页数量、URL 数量或普通网页匹配度直接当作“人类分数”。不要加入来源数据集名称、生成器名称等会造成训练集捷径的字段。若本地基础检测器也在本项目数据上微调，控制器训练必须使用该检测器的 out-of-fold 预测，不能使用模型见过同一文本后产生的训练分数。

控制器输出 `raw_fused_score`。若 validation 上逻辑回归不足，再比较 LightGBM，但只能依据 validation 选择，不能查看 locked test 后反复换模型。概率校准可在 validation 上使用 Platt scaling；数据足够时再比较 isotonic regression。所有映射冻结后才能读取 calibration 计算高风险阈值。最终风险阈值由下一层决定，不使用 `0.5` 作为默认分类点。

### 5.6 分组阈值与三段决策

对每个样本分组 `g=(language, domain, length_bin)`，从该组的人类 calibration 分数中计算高风险阈值。设升序人类分数为 `s[0:n]`：

```text
alpha = 0.02
k = ceil((n + 1) * (1 - alpha))
k > n 时 T_high(g) = +inf，否则 T_high(g) = s[k - 1]
仅当 score > T_high(g) 时进入高风险候选
```

这里使用严格大于号是为了保守处理离散分数和并列值，不能直接改成 `>=`。阈值文件同时保存样本量、分位点、分数函数版本和生成时间。该方法在分数函数预先冻结、calibration 与线上样本可交换时，可把高风险人类误报控制在目标附近；发生分布漂移时保证不再成立，所以 OOD 和线上监控不可省略。

`T_low(g)` 在 validation 集上选择，目标是保证低风险区有足够的人类精确率；中间全部为不确定。初始决策规则：

```text
无任何有效 detector 分数             -> insufficient_evidence
文本过短、OOD 过高或路由置信度过低    -> uncertain
score > T_high(g) 且证据覆盖充分       -> high_risk_ai
score <= T_low(g)                      -> low_risk
其他                                  -> uncertain
```

`high_risk_ai` 还应要求有效检测覆盖率不低于 70%，并至少有两个相邻高风险窗口或两个独立 provider 支持。准确数值由 validation 选择，再由 calibration 固定阈值。控制器、阈值和基础模型版本必须作为一套 artifact 发布，不能只替换其中一项。

## 6. 代码设计

建议新增以下文件，保持现有 provider 调用边界：

```text
detection/text_segmentation.py       # 句界切分、窗口范围、重叠去重
detection/text_routing.py            # 文体、长度和路由置信度
detection/ood_detector.py            # embedding 参考库和 OOD 距离
detection/score_fusion.py            # 特征构造与逻辑回归推理
detection/false_positive_controller.py # 总编排和三段决策
detection/calibration_store.py       # 阈值分组、回退层级、版本校验
training/prepare_fp_dataset.py       # 导入、许可白名单、标签映射、去重、分组划分
training/train_fp_controller.py      # OOF 特征、融合器训练、概率校准
evaluation/evaluate_false_positives.py # FPR、TPR、置信区间和分桶报告
artifacts/text_fp_controller/        # 模型、thresholds.json、manifest.json
```

核心数据结构：

```python
@dataclass
class SegmentEvidence:
    start: int
    end: int
    provider_scores: dict[str, float | None]
    provider_status: dict[str, str]
    aggregated_features: dict[str, float]

@dataclass
class FalsePositiveDecision:
    calibrated_score: float | None
    decision: str
    threshold_group: str | None
    low_threshold: float | None
    high_threshold: float | None
    uncertainty_reasons: list[str]
    domain: str
    domain_confidence: float
    ood_score: float | None
    analyzed_coverage: float
    segments: list[SegmentEvidence]
    artifact_version: str
```

`ApiFirstDetectionEngine.detect()` 的文本分支应改为收集 `ProviderResult` 后调用：

```python
decision = fp_controller.evaluate(
    text=effective_text,
    provider_results=results,
    source_evidence=public_source_result,
)
```

对外 API 在兼容原 `score/label/threshold/provider_results` 的同时新增 `decision`、`uncertainty_reasons`、`threshold_group`、`score_kind="calibrated_ai_risk"`、`analyzed_coverage` 和 `segments`。旧前端若仍只认识 `human/ai`，可暂时把 `low_risk` 映射为 `human`、`high_risk_ai` 映射为 `ai`，但 `uncertain` 与 `insufficient_evidence` 不得映射成 `human`。

建议配置：

```yaml
detection:
  text_false_positive_control:
    enabled: true
    artifact_dir: "artifacts/text_fp_controller"
    target_fpr: 0.02
    min_effective_chars: 120
    segment_tokens: 384
    segment_overlap: 0.25
    context_sentences: 2
    min_analyzed_coverage: 0.70
    min_group_calibration_humans: 300
    ood_quantile: 0.99
    route_fallback_order:
      - "language_domain_length"
      - "language_domain"
      - "language"
      - "global"
    fail_closed_to_uncertain: true
```

## 7. 训练与发布流程

1. 导入公开数据并生成 `dataset_manifest.json`，记录下载地址、版本、SHA256、许可证和标签映射；许可证不在白名单的数据直接隔离。
2. 规范化文本，只清理编码和不可见字符，不删除标点、换行或古文字符；计算原文哈希和规范化哈希。
3. 按 `root_document_id` 去重和分组划分，生成不可变的 train、validation、calibration、locked_test 清单。
4. 冻结现有 provider，在四个 split 上批量生成原始分数、状态和版本。API 失败保留缺失值，不补零。
5. 在 train 上训练融合器、路由器和 OOD 参考索引；在 validation 上选择特征、窗口、概率映射和低风险阈值，不接触 locked_test。
6. 冻结全部分数函数后，在 calibration 上计算 OOD 阈值和各组 `T_high`，写入 `thresholds.json`；不得再根据 calibration 结果修改融合器。
7. 只运行一次 locked_test 形成验收报告；未达标时记录新版本并重新建立 calibration/test，禁止继续在同一锁定测试集上调参。
8. artifact manifest 绑定基础模型名称及哈希、融合器哈希、数据 manifest、代码 commit、阈值版本和生成日期。版本不匹配时线上返回 `insufficient_evidence`。

第二阶段微调本地 encoder 时，先运行当前系统从有授权的人类池中挖掘高分误判样本，人工确认后加入 hard-negative train。每轮保留一部分新来源文档只进入 locked_test。建议先做 LoRA 或最后若干层微调，并以 `FPR` 约束选 checkpoint；不要按 accuracy 最高选择模型。

## 8. 测试与验收

主要指标为高风险区总体人类 FPR、各困难文体 FPR、`TPR@FPR=1%`、`TPR@FPR=2%`，均报告 Wilson 95% 置信区间。补充报告 AUROC、AUPRC、Brier Score、ECE、不确定率、自动判定覆盖率、无有效 provider 率和平均延迟。不能只报告 accuracy 或 Macro-F1。

第一版建议验收门槛：

| 指标 | 目标 |
|---|---|
| locked test 总体高风险 FPR | 点估计 `<=2%`，同时报告 95% CI |
| 经典文学、论文/技术、公文困难集 FPR | 每类点估计 `<=3%` |
| 同分布 `TPR@FPR=2%` | `>=70%`，不足时优先增加不确定而非放宽 FPR |
| 未见领域/生成器 `TPR@FPR=2%` | 只报告不承诺；作为后续提升基线 |
| ECE | `<=0.05` |
| 自动判定覆盖率 | 初期目标 `60%-80%`，其余允许不确定 |
| 无有效分数、过短或高 OOD 输入 | 100% 不输出确定的人类/AI 结论 |

消融实验至少包含：现有简单平均、仅全局校准、分文体/长度校准、加入 OOD、加入 hard negatives、加入上下文分段。需要同时比较 FPR、TPR 和拒判率，避免通过把所有内容都判为不确定来虚假降低误报。

必须建立以下回归用例：整段《红楼梦》或《西游记》；古文加现代评论；论文摘要和公式附近文本；规范化公文；技术 API 文档；短文本；一个 provider 高分而其他分支缺失；所有 provider 失败；部分来源命中后只检测剩余文本；长文单个片段异常；同一文档不同切片不得跨 split。

## 9. 预期效果与边界

分组保形阈值可以在与 calibration 同分布的样本上把高风险 FPR 控制在约 1%-2% 目标附近，但会牺牲召回率并增加不确定结果；分布变化后不再有保证。根据 MAGE、MCP/RealDet 和文体条件阈值研究，少量目标域校准数据通常比继续增加同分布生成文本更有效。对本项目，合理的工程目标是：在 AI 召回下降不超过 10 个百分点的条件下，使困难人类集 FPR 相对当前基线下降 30%-60%。这是阶段目标，不是上线前承诺，最终数值必须由 locked test 给出。

若只增加一个二分类“复判模型”，却继续使用随机切片、统一阈值和简单平均，误判可能只是转移而不会消失。真正可交付的最小闭环是：可信困难人类数据、冻结模型原始分数、小型融合器、独立 calibration、分组阈值、OOD 拒判和带置信区间的锁定测试。

## 10. 检索留痕与依据

检索时间为 2026-07-21，优先使用论文原文、ACL Anthology、PMLR、官方数据卡和作者仓库。主要检索词包括：

```text
AI generated text detection false positives distribution shift unseen domains
Can AI-Generated Text be Reliably Detected paraphrasing AUROC
GPT detectors biased against non-native English writers false positive
MAGE machine generated text detection dataset license Hugging Face
M4 multi-generator multi-domain multilingual detection GitHub
conformal calibration false positive rate AI generated text RealDet
conditional thresholds style router AI-generated text detection MoSEs
Chinese AI generated text detection dataset NLPCC DetectRL-ZH
HC3 Chinese dataset license source
MULTITuDE dataset Chinese license Zenodo
sentence-level mixed human AI text detection context
```

关键研究依据：

| 来源 | 可复核结论 | 本设计采用方式 |
|---|---|---|
| [Sadasivan et al.](https://arxiv.org/abs/2303.11156) | 复述可使 DetectGPT AUROC 从 96.5% 降至 59.8%，通用检测存在理论限制 | 输出风险和拒判，不声称作者身份证明 |
| [Liang et al.](https://arxiv.org/abs/2304.02819) | 7 个检测器对 91 篇 TOEFL 真人作文平均误报率 61.22%，低困惑度是关键因素 | 专业、规范和古典文本必须进入困难人类集 |
| [MAGE](https://aclanthology.org/2024.acl-long.3/) | 最佳模型在未见领域可误判 61.95% 的人类文本；少量域内校准能显著改善 OOD 表现 | 目标域校准优先于盲目扩数据 |
| [M4](https://aclanthology.org/2024.eacl-long.83/) | 未见领域和生成器显著降低泛化 | 按领域、生成器和根文档划分测试 |
| [Practical Examination](https://arxiv.org/abs/2412.05139) | 部分检测器在未见/攻击场景的 `TPR@FPR=1%` 可降至 0 | 固定 FPR 评估，不能只看 F1/accuracy |
| [MCP/RealDet](https://aclanthology.org/2025.acl-long.601/) | 按长度保形校准可在可交换性条件下控制 FPR | 使用分组有限样本分位数确定高风险阈值 |
| [MoSEs](https://aclanthology.org/2025.emnlp-main.294/) | 文体路由与条件阈值优于统一阈值，论文报告平均提升 11.34% | 建立文体路由和阈值回退层级 |
| [SenDetEX](https://aclanthology.org/2025.emnlp-main.268/) | 上下文感知句级检测较基线提升约 4.7%-6.8% F1，2-3 句上下文已提供主要收益 | 分段检测保留相邻上下文，不孤立判单句 |

最终实施顺序为：先准备许可清晰的困难人类 calibration/locked test；再把 `providers.py` 的简单平均替换为控制器；随后加入分组阈值和 OOD；完成消融和锁定测试后，才决定是否微调现有本地检测模型。模型溯源作为独立后续任务，不与本阶段的数据、标签或验收指标混合。
