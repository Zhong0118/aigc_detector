# AIGC 检测与溯源月度汇报设计及论文技术摘录

检索日期：2026-07-02  
参考上次文件：`C:/Users/zx/Desktop/0530.pptx`

## 1. 上次汇报内容承接

上次 PPT 的主线是“系统架构模块化”：内容接入与预处理、多个识别引擎、来源分析、Top-K 溯源分类器、存储索引、API 和可视化面板。核心已经从“单一检测器”扩展为“检测 + 溯源 + 指纹 + 报告”的完整框架。

这次建议把主线推进为：

> 从可运行框架到可部署、可扩展、可验证的 AIGC 检测与溯源系统。

本次老师交代的三件事可以对应成三条汇报线：

1. **工程交付**：完成 Linux 服务器部署文档，明确依赖、模型缓存、API Key、C2PA、水印、训练产物和启动流程。
2. **样本扩展**：准备采集/生成其他模型的文本、图片、音频、视频样本，用于检测评测和来源归因训练。
3. **前沿调研**：梳理 AIGC 内容识别、模型来源归因、内容凭证和水印技术，并反推系统下一步实现优先级。

## 2. 建议 PPT 结构

建议 15-16 页，不要再用上次“连续架构模块说明”的形式，而是做成“进展 + 调研 + 落地路线”。

| 页码 | 标题 | 本页要讲什么 | 视觉建议 |
|---:|---|---|---|
| 1 | AIGC 检测与溯源系统阶段汇报 | 标题、副标题：服务器部署、样本扩展、前沿技术调研 | 深色封面，右侧放“检测-溯源-部署”流程图 |
| 2 | 本月工作总览 | 三条线：部署文档、样本扩展、技术调研 | 三泳道时间线 |
| 3 | 上次到这次的变化 | 上次是架构设计，这次转向可部署和可扩展 | Before/After 对比 |
| 4 | Linux 部署文档交付 | 展示文档覆盖：系统依赖、Python 环境、模型缓存、API、C2PA、水印、训练产物 | 部署清单 |
| 5 | 当前系统落地状态 | FastAPI/Streamlit/SQLite/ingestion/report 已可跑；真实 Hive/Sightengine、C2PA、水印仍需接入 | 模块状态矩阵 |
| 6 | 样本扩展目标 | 文本、图片、音频、视频都要构建“真实/AI/来源标签”样本 | 数据闭环图 |
| 7 | 计划采集的生成模型 | 文本：ChatGPT/DeepSeek/Kimi/Claude/Gemini/Qwen；图片：GPT Image/Midjourney/Flux/SDXL/Firefly；音频：OpenAI TTS/ElevenLabs/Suno/Udio/CosyVoice 等 | 模态 x 模型表 |
| 8 | AIGC 检测技术趋势 | 被动检测不再可靠，必须组合概率特征、视觉特征、跨域基准和主动标识 | 技术雷达图 |
| 9 | 文本检测论文可用结论 | DetectGPT/Fast-DetectGPT/Binoculars/RAID/OpenTuringBench | 论文对比表 |
| 10 | 图像检测论文可用结论 | UniversalFakeDetect/DIRE/AI-GenBench/2026 out-of-box benchmark | 模型泛化风险图 |
| 11 | 音频与视频检测趋势 | 音频/多模态 deepfake 检测开始强调 benchmark、局部篡改定位、跨域泛化 | 音频波形 + 视频帧流程 |
| 12 | 溯源技术一：C2PA 内容凭证 | 可读取模型、时间、处理历史等 signed metadata；但平台转码会丢失，不能单独依赖 | “强证据但易丢失”卡片 |
| 13 | 溯源技术二：水印 | SynthID、AudioSeal、VideoSeal、Stable Signature 代表主动标识路线 | 文本 token / 音频波形 / 视频帧三联图 |
| 14 | 来源归因路线 | 文本用 OpenTuringBench + 分类器；图像用 CLIP embedding + Top-K；音视频复用帧/片段聚合 | Top-K 条形图 |
| 15 | 对现有系统的改进映射 | 调研结果如何落到 detection、provenance、storage、dashboard | 系统架构增量图 |
| 16 | 下阶段计划 | 真实 API 接入、C2PA wrapper、水印 adapter、样本库、评测面板 | 路线图 |

## 3. 可直接放进 PPT 的核心判断

### 本次汇报一句话

本月工作从“架构能跑通”推进到“服务器可部署、样本可扩展、检测与溯源路线有前沿依据”：系统下一步不应依赖单一检测器，而应采用 **被动检测 + 主动水印 + C2PA 内容凭证 + Top-K 来源归因 + 历史指纹库** 的多证据融合方案。

### 技术趋势总结页

- **被动检测**：DetectGPT、Binoculars、CLIP 图像检测等方法可以作为检测分支，但跨模型、跨压缩、跨平台泛化仍是核心问题。
- **主动标识**：SynthID、AudioSeal、VideoSeal 等水印方案更适合作为“强证据”，但只对采用该水印体系的生成模型有效。
- **内容凭证**：C2PA 可以保存生成模型、编辑历史、签名信息，是标准化溯源方向；但社交平台转码、截图、重新编码会导致元数据丢失。
- **模型来源归因**：当前更现实的做法是基于开放数据集和本地样本库训练 Top-K 分类器，把结果作为辅助证据而不是绝对结论。

## 4. 样本扩展设计

### 文本样本

建议类别：

- 人写文本：新闻、论文摘要、学生作业、技术文档、问答。
- 闭源模型：ChatGPT、Claude、Gemini、Kimi、文心、通义。
- 开源/国产模型：DeepSeek、Qwen、GLM、Llama、Mistral。

每条 JSONL 建议字段：

```json
{"text": "...", "label": "ai", "source_model": "DeepSeek", "prompt_type": "summary", "language": "zh", "date": "2026-07-02"}
```

PPT 可用句：

> 文本样本不只标“AI/人类”，还要标注来源模型、任务类型和语言，这样后续可以同时训练二分类检测器和 Top-K 来源归因器。

### 图片样本

建议类别：

- 真实图：公开摄影数据集/自采集图片。
- 闭源生成：GPT Image、Midjourney、Adobe Firefly、Google Imagen/Gemini Image。
- 开源生成：Stable Diffusion/SDXL、FLUX、Hunyuan、Qwen-Image、Z-Image。

目录建议：

```text
data/attribution/image_samples/
  Real/
  GPT_Image/
  Midjourney/
  Firefly/
  FLUX/
  SDXL/
```

PPT 可用句：

> 图像来源归因建议先复用现有 CLIP embedding + LogisticRegression 路线，后续再引入 UniversalAttribution 的开放集思想。

### 音频/视频样本

建议类别：

- 音频：真人录音、TTS、voice cloning、音乐生成。
- 视频：真实视频、AI 生成视频、换脸/口型驱动/局部篡改视频。

PPT 可用句：

> 音视频检测不能只做整段二分类，需要保存片段级或帧级证据，后续 dashboard 展示“哪些片段更可疑”会更有说服力。

## 5. 论文和技术摘录

### 文本检测与文本来源归因

| 论文/技术 | 核心结论 | PPT 可用点 | 对本项目的用处 |
|---|---|---|---|
| DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature (2023) | 机器生成文本倾向落在模型 log probability 的负曲率区域，不需要额外训练分类器 | 可作为“零样本检测”代表方法 | 解释 PPL/概率曲率为何能作为文本检测特征 |
| Fast-DetectGPT (2023/2024) | 用条件概率曲率替代大量扰动，声称比 DetectGPT 快很多 | 可说明工程部署需要考虑推理成本 | 适合放在“本地轻量检测增强路线” |
| Binoculars (2024) | 使用一对相近语言模型的 perplexity ratio，零样本检测 ChatGPT 等文本，论文报告在很低 FPR 下仍有较高召回 | 可做“两个模型交叉看文本”的直观图 | 后续可作为文本 detector adapter 的候选 |
| RAID benchmark (2024) | 覆盖 6M+ 生成文本、11 个模型、8 个领域、11 种攻击；发现检测器容易被攻击、采样策略和未见模型影响 | 强调不能只报单一准确率，要测跨域和攻击鲁棒性 | 建议系统评测页加入 TPR@FPR、攻击前后对比 |
| AI-generated Text Forensic Systems Survey (2024) | 将文本取证分为 detection、attribution、characterization 三层 | 可作为你系统“三层输出”的理论依据 | 对应检测、来源模型、内容类型/意图说明 |
| OpenTuringBench (2025) | 面向开放大模型生成文本，支持检测和作者/模型归因任务 | 直接支撑上次 PPT 中的文本来源分类器 | 继续作为训练 `text_source_classifier.joblib` 的主数据源 |
| Practical Examination of AI-Generated Text Detectors (2024) | 多个检测器在实用场景和攻击下 TPR@0.01 FPR 可能很低 | 可提醒老师“检测结果必须带置信度和限制说明” | 报告生成中加入“不作为唯一证据”的提示 |

### 图像检测与图像来源归因

| 论文/技术 | 核心结论 | PPT 可用点 | 对本项目的用处 |
|---|---|---|---|
| Towards Universal Fake Image Detectors (2023) | 使用大规模预训练视觉语言模型特征，最近邻/线性探针比专门训练的 fake detector 泛化更好 | 支撑“CLIP embedding + 轻量分类器”的合理性 | 与当前 image attribution 路线高度一致 |
| DIRE (2023) | 扩散生成图像更容易被扩散模型重建，重建误差可作为检测特征 | 可讲“扩散模型有自己的重建痕迹” | 后续可作为图像检测增强分支 |
| Fake or JPEG? (2024) | 很多生成图像检测数据集带有 JPEG 压缩、尺寸等偏差，检测器可能学到伪线索 | 可放在“为什么要做标准化样本预处理” | 样本库要统一压缩、尺寸、来源分布 |
| Universal Synthetic Image Origin Attribution (2024) | CLIP 特征可用于开放集 synthetic image origin attribution，对未见生成器也有一定泛化 | 可直接解释“来源归因不是只做 AI/Real，而是 Top-K 源模型” | 项目已有 `third_party/UniversalAttribution` 和 CLIP 路线 |
| AI-GenBench (2025) | 提出按时间顺序训练/测试的持续 benchmark，模拟新生成器不断出现 | 可做“生成模型迭代导致检测器老化”的图 | 建议评测按生成模型发布时间切分 |
| 2026 out-of-box image detector benchmark | 16 种检测方法、23 个预训练变体、12 个数据集、291 个生成器；现代商业模型可能击败多数检测器 | 可作为“单一检测器不可依赖”的强论据 | 系统要保留 provider ensemble 和模型更新机制 |
| GPT-Image-2 Twitter Dataset (2026) | 社交平台上的 AI 图像会丢失 C2PA 内容凭证，导致加密溯源不可用 | 可说明“平台传播后要靠多证据融合” | storage 中要保存上传时检测到的 metadata 和 hash |

### 音频、视频和多模态检测

| 论文/技术 | 核心结论 | PPT 可用点 | 对本项目的用处 |
|---|---|---|---|
| AudioSeal (2024) | 面向 AI 语音的局部水印检测，支持片段级定位，检测速度适合大规模场景 | 可放在音频水印接入页 | 对应 `provenance/metaseal_adapter.py` 中的 audio watermark |
| Video Seal (2024) | 开源神经视频水印框架，考虑视频编码、几何变换等失真，支持高效视频水印 | 可放在视频水印路线页 | 对应 image/video watermark adapter |
| AUDDT (2025) | 系统整理 28 个音频 deepfake 数据集并提供统一评测工具 | 可说明音频检测重点转向跨数据集评测 | 后续训练音频 detector 时可作为评测参考 |
| FakeSound2 (2025) | 音频 deepfake 不只要二分类，还要 localization、traceability、generalization | 可放在“音频下一步增强” | 支持 dashboard 展示局部可疑片段 |
| DeepfakeBench-MM (2025) | 大规模多模态 deepfake benchmark，覆盖音频、视觉、音频驱动人脸等多种伪造管线 | 可作为多模态检测趋势 | 支撑视频检测从抽帧转向音画联合 |

### 主动水印、内容凭证和标准化溯源

| 技术 | 核心结论 | PPT 可用点 | 对本项目的用处 |
|---|---|---|---|
| C2PA / Content Credentials | 用签名 metadata 记录媒体来源和编辑历史；`c2patool` 可读取、验证、写入 manifest | “强证据：有则很有价值，无则不能说明不是 AI” | Linux 部署中准备 `tools/c2patool/c2patool`，后续替换 placeholder parser |
| C2PA security analysis (2026) | 独立安全分析认为 C2PA 仍不应单独用于高风险场景 | 可避免把 C2PA 讲成万能方案 | 系统报告中 C2PA 只作为证据之一 |
| SynthID Text/Video (Google DeepMind, 2024-2025) | 文本通过调制 token 概率嵌入水印，视频给每帧嵌入不可见水印；Google 也推出 SynthID Detector | 可作为“行业主动标识代表” | 可以在调研页提出：未来若有 API，可接入 watermark verification |
| Stable Signature (2023) | 在 latent diffusion decoder 中植入图像水印，生成图像天然带签名 | 可说明开源扩散模型也可以主动水印 | 可作为图片水印技术背景 |
| Stable Signature is Unstable (2024) | 通过微调扩散模型可移除 Stable Signature，说明模型内水印也可被攻击 | 可放“水印不是绝对可靠” | 报告需输出 watermark confidence 和风险说明 |
| Watermarking for AI Content Detection review (2025) | 系统综述文本、视觉、音频水印，指出鲁棒性、标准化和隐私仍是挑战 | 可作为水印调研总引 | 帮助组织“主动标识技术栈” |
| Vaporizer (2026) | 多种文本水印在翻译、改写、同义替换等攻击下可能被削弱或移除 | 可说明文本水印要和被动检测结合 | 不应只依赖 TextSeal/SynthID 类结果 |
| 中国《人工智能生成合成内容标识办法》/EU AI Act Article 50 | 监管方向要求机器可读标识、显著标识、水印/元数据等多层机制 | 可作为项目现实意义 | PPT 结尾强调系统和监管趋势一致 |

## 6. 这次 PPT 中可以直接使用的三张图

### 图 1：系统证据融合架构

```text
上传内容
  -> 预处理与指纹
  -> 被动检测：文本/图像/音频/视频 detector
  -> 主动证据：C2PA / SynthID / AudioSeal / VideoSeal
  -> 来源归因：Top-K model attribution
  -> 历史索引：SHA256 / pHash / embedding 检索
  -> 综合报告：AI 概率、来源候选、证据链、限制说明
```

### 图 2：检测技术的可靠性分层

```text
强证据：C2PA 签名、官方水印命中
中证据：来源归因 Top-K、历史指纹匹配
弱证据：PPL、stylometry、视觉 artifact、频谱异常
最终输出：融合置信度 + 证据解释 + 局限性
```

### 图 3：下一阶段路线图

```text
第 1 周：真实 Hive/Sightengine provider 接入
第 2 周：c2patool wrapper 替换 placeholder
第 3 周：构建多模型文本/图片样本库并训练 Top-K 分类器
第 4 周：AudioSeal/VideoSeal adapter + dashboard 证据展示
```

## 7. 建议下阶段任务优先级

1. **先接一个真实检测 provider**：Hive 或 Sightengine，替换当前 demo provider 的核心演示链路。
2. **实现 c2patool wrapper**：Linux 部署文档已经明确路径，下一步让后端真正调用官方工具。
3. **构建样本库**：文本 JSONL + 图片分目录，先覆盖 5-8 个来源模型，每类至少几百条/张。
4. **重训来源归因模型**：文本用 OpenTuringBench + 自建 JSONL，图片用 CLIP embedding + 自建图片目录。
5. **补充水印 adapter**：优先 AudioSeal，因为音频水印论文和开源工具较完整；图片/视频再接 VideoSeal。
6. **把限制写进报告**：检测结果必须包含“证据来源、置信度、可能失败场景”，避免老师质疑误判问题。

## 8. 参考来源

- DetectGPT: https://arxiv.org/abs/2301.11305
- Fast-DetectGPT: https://arxiv.org/abs/2310.05130
- Binoculars: https://arxiv.org/abs/2401.12070
- RAID benchmark: https://arxiv.org/abs/2405.07940
- AI-generated text forensic survey: https://arxiv.org/abs/2403.01152
- OpenTuringBench: https://arxiv.org/abs/2504.11369
- Practical examination of text detectors: https://arxiv.org/abs/2412.05139
- Towards Universal Fake Image Detectors: https://arxiv.org/abs/2302.10174
- DIRE: https://arxiv.org/abs/2303.09295
- Fake or JPEG?: https://arxiv.org/abs/2403.17608
- Universal Synthetic Image Origin Attribution: https://arxiv.org/abs/2408.09153
- AI-GenBench: https://arxiv.org/abs/2504.20865
- Out-of-box AI image detector benchmark: https://arxiv.org/abs/2602.07814
- GPT-Image-2 Twitter Dataset: https://arxiv.org/abs/2604.25370
- AudioSeal: https://arxiv.org/abs/2401.17264
- Video Seal: https://arxiv.org/abs/2412.09492
- AUDDT: https://arxiv.org/abs/2509.21597
- FakeSound2: https://arxiv.org/abs/2509.17162
- DeepfakeBench-MM: https://arxiv.org/abs/2510.22622
- C2PA specifications: https://spec.c2pa.org/specifications/specifications/2.2/index.html
- c2patool docs: https://opensource.contentauthenticity.org/docs/c2patool/
- C2PA security analysis: https://arxiv.org/abs/2604.24890
- Google DeepMind SynthID text/video: https://deepmind.google/blog/watermarking-ai-generated-text-and-video-with-synthid/
- Stable Signature: https://arxiv.org/abs/2303.15435
- Stable Signature is Unstable: https://arxiv.org/abs/2405.07145
- Watermarking for AI Content Detection review: https://arxiv.org/abs/2504.03765
- Vaporizer watermark attack: https://arxiv.org/abs/2605.07481
