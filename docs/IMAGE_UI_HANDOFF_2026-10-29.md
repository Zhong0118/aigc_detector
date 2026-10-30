# 文本与图片检测工作台 — 页面实现交接

> 2026-10-30交接入口已更新：[页面需求](IMAGE_UI_REQUIREMENTS_2026-10-30.md)、[模型接入](IMAGE_MODEL_INTEGRATION_2026-10-30.md)、[资产清单](IMAGE_MODEL_INVENTORY_2026-10-30.json)。本文件保留历史上下文，新版决策与契约优先。

日期：2026-10-29。状态：需求与实现建议，尚未实现新页面或图片在线API。

用户已选择保留 Streamlit + FastAPI；本轮改为把页面交给另一个模型实施，本线程继续研究实验。首版不等待所有论文路线跑完：接入已有验收证据的能力，新实验完成后再登记为可选模型。不能把“所有已登记实验完成”解释为“所有算法都已跑过”。

## 1. 交付目标与范围

四个页面：**文本检测、图片检测、实验对照、历史报告**。真实推理可用，结果语义正确，失败不伪装成功，模型与阈值版本可追溯，页面往返不丢结果。复用已有依赖和接口，不另造前端框架或任务队列。

先在独立预览进程、独立API配置、独立数据库与上传临时目录验收。不得重启或修改线上文本服务、sync生产树、生产数据库、线上模型；不得启动训练、下载模型、更新实验划分或运行holdout。预览端口须实查空闲后登记，不能直接占用原服务端口。部署替换线上服务不属于本交接范围。

服务端研究工作树：`/home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline`，分支 `attribution-pilot-baseline`；只能通过 `ssh school` 访问。页面实现者先检查工作树归属和未提交修改。正在开展的新实验源码、已登记配置及旧输出均为保护区，页面通过新适配层调用加载能力，不改研究训练代码以迁就UI。

## 2. 已存在能力与缺口

| 已核对代码 | 当前行为 | 首版处理 |
|---|---|---|
| `api/dashboard.py` | Text/File/Report三个tab；`render_analysis()`偏文本；`request_with_progress()`固定18/42/78/100% | 重整四页，图片独立renderer；没有后端进度就只显示等待/耗时 |
| `api/main.py`, `api/routes.py` | `/api/v1/providers`、`/analyze/text`、`/analyze/file`、`/detect`、`/provenance`、`/report/{content_id}` | 保持文本兼容，扩展图片响应与能力发现 |
| `api/analysis.py` | ingestion→detector→provenance→report→storage；临时上传finally清理 | 图片新增明确的真实模型分支，不借用文本规则 |
| `detection/providers.py` | 本地模型仅执行文本；无有效分数的通用分支可能落为0/human | 图片无模型/推理失败必须unavailable/error，不能0分判真人 |
| `detection/image_detector.py` | `weights=None`未训练scaffold | 禁止作为产品图片模型，禁止demo/hash分数兜底 |
| `provenance/attribution.py` | 多种旧研究provider，部分为phash样本匹配 | 不把phash匹配叫生成器指纹，不混合不同分数语义 |
| `storage/database.py` | 默认读取根config数据库；当前记录不等于不可变完整报告 | 预览显式配置独立DB；保存版本化结果快照 |
| `tests/test_dashboard_integration.py` | Streamlit AppTest覆盖部分文本行为 | 保留回归并增加图片/失败/导航测试 |

`requirements.txt`历史上允许较旧Streamlit；使用导航新API前核验实际版本。优先原生控件、requests、现有绘图依赖；不用仅为样式引入另一套前端或鉴权系统。

## 3. 默认模型与证据边界

### 默认图片检测

采用 **D-OpenCLIP**：冻结官方OpenAI ViT-L/14 QuickGELU视觉编码器，768维L2特征，本项目30轮二分类线性头。使用真实产物中的类别顺序 `[real, ai]`、224 RGB短边缩放/中心裁剪、bicubic及CLIP mean/std。不得用名字接近但权重/激活/预处理不同的CLIP替换。

默认显示原登记阈值0.5下的结论和“模型检测分数”。不要写成经过校准的“真实世界AI生成概率”。目标1%/5%FPR校准阈值可在研究模式展示，并同时说明对应开发cal/eval实测FPR；跨域不保证目标误报率。

### 默认来源候选

采用 **A0 ResNet50 22来源**。A2-clean报告是对该模型在相同evaluation子集的评估，不是另一套默认训练权重。展示Top3（可展开TopK）及条件说明：

> 在图片属于AI生成、且来源属于已知22类的条件下，以下标签最接近模型已学习的来源。

不是生成器身份鉴定，不等于架构家族。来源名称必须从checkpoint class map读取，保持精确顺序，不猜商业服务实际版本。检测未达阈值时主结论不强行归因；若允许查看闭集参考排序，必须明确其条件。

主22类模型 `unknown.score=null`、`unknown.status=unavailable`，不能用1−最大softmax、TopK剩余质量或20类拒识阈值构造未知概率。

### 研究模式

- OFA：冻结原模型、每来源100训练参考的配对验证分数，和来源softmax分开。必须绑定原参考ID及其哈希，不现场从上传图增补参考库。
- 来源融合：固定50/50与calibration拟合权重各为独立已登记结果，不能根据页面输入或evaluation重新选权重。A-Fusion特征融合与ResNet/A-Fusion分数融合是两个层级。
- 未知拒识：原20已知类、kling校准、grok3评估是专用研究模型。展示已知保留率、正确且保留率和未知误接收率，不只展示“拒绝未知93.65%”；不能将阈值套到22类主模型。
- OmniAID及其他完成模型可作为登记的研究对照。OmniDFA、DNA-Det、ConvNeXt、OFA全预训练、T+1以及A3的运行状态独立显示；未经完成验收不可提供上传推理。

### 已实查的默认产物

路径相对上述研究工作树；这些文件在服务器，不应上传Git。

| 用途 | 路径 | SHA256 |
|---|---|---|
| OpenCLIP视觉原始权重 | `outputs/image_experiments/assets/openclip-vit-l-14-openai/open_clip_model.safetensors` | `9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e` |
| 检测头 | `outputs/image_experiments/openclip-q179-20261029/D-OpenCLIP/linear/head.pth` | `0432c050a2c9afcb1845e31651586032e5672f0e4096713fdd4a9f0d64ea4bbb` |
| 检测provenance | `outputs/image_experiments/openclip-q179-20261029/D-OpenCLIP/provenance.json` | `1be0094babb0fb93d44aeed361045b02bac4432764c0c78cb18e31832cf848b8` |
| 来源模型 | `outputs/image_experiments/a0-resnet50-iab-v7-200-q179-e30/model.pth` | `c3756dd56fbd3d151ebda8db69b683fd8c78cd773a9967af6e70a3565891562b` |
| 来源资产清单 | `outputs/image_experiments/a0-resnet50-iab-v7-200-q179-e30/asset_manifest.json` | `16101f1d80a7b74e4fcac0d6883d3db745e7feaf8936a0c30a275dcf1291f4c2` |

加载参考：`research/image_attribution/openclip_route.py::load_backbone`、`scripts/run_image_openclip.py::load_head`；ResNet参考`research/image_attribution/models.py`及`a0.../resolved_config.json`。使用weights_only/安全格式和strict键校验，完整class map、预处理与选模记录一起绑定；不要直接执行训练脚本main。

当前研究环境`.venv-image-next`有OpenCLIP；页面服务环境须独立核验兼容性。默认不要自动抢GPU：先明确登记CPU或受控GPU运行策略，验收该实际设备的结果与耗时；GPU0研究任务运行期间预览不得抢占。单个常驻推理worker复用加载后的模型，有界并发，不为每次Streamlit rerun重新加载权重。

## 4. 四页信息与交互

### 文本检测

保留文本粘贴和已支持文档上传、分块证据、阈值、来源与报告。保留原文本模型逻辑和测试；图片不继承文本分块、文本家族阈值或LLM假分析。

### 图片检测

桌面双列：左侧上传/预览/输入信息，右侧就绪状态/操作/结果；小屏纵向排列。流程为上传→校验→真实分析→结果，允许清空、更换、导出。结果按检测结论、来源候选、证据与限制组织。技术SHA和配置放展开区，不占主流程。

按钮有禁用与忙碌反馈，重复点击不重复提交；导航往返保留已上传内容和结果。前端超时不可自动触发后端重跑。只在用户明确提交时分析，不因上传控件或筛选rerun推理。

### 实验对照

只读登记的完成实验；分开检测、来源分类、未知拒识三类视图。显示样本数、数据角色、阈值、训练方式/epoch、模型版本、指标与局限。相同ID配对比较直接引用保存报告；不同分母不得显示“提升”箭头。实验页不能启动GPU训练或选择新阈值。

### 历史报告

分页列表，按模态/时间/状态筛选，查看详情并导出JSON/Markdown。历史记录是当时的模型、阈值和预处理快照，不能按当前默认模型重新解释。默认保存结果及必要输入元数据，不持久化原图；如产品确需保存原图，再显式说明存储与清理策略。

## 5. 最小API建议（以下是拟新增能力，不是现有API）

优先扩展现有`POST /api/v1/analyze/file`，图片返回新增的`image_result`独立对象，并保留旧字段兼容；只有确有隔离契约需要时再建专用图片端点。首版无需统一迁移v2。

建议补`GET /api/v1/capabilities`，返回支持格式/字节与像素上限、模型bundle、ready/loading/unavailable/error/busy状态、支持模式及外传策略。历史列表与实验目录补最小只读接口；其字段和路径由实现前的OpenAPI契约确定，不能只做前端假列表。

`image_result`最小结构：

```text
schema_version, analysis_id, request_id, created_at, status
input: sha256, media_type, width, height, decode_policy
model_bundle: id, code_commit, encoder/head/class_map/preprocessing hashes
detection: status, score, score_kind, label, threshold, threshold_provenance
attribution: status, label_space_id, conditional_on, top_k
unknown: status, score                 # 主22类 unavailable/null
research: []                           # 独立任务/标签空间/分数语义
provenance, limitations, timing
```

这是建议响应契约，不是要求按字面照造所有字段名。先固定契约并写测试，再接适配层。允许检测成功/归因失败返回partial并说明；缺失值null，不得NaN或0兜底。检测失败不产生“真人”结论。客户端不能指定任意文件路径/checkpoint。

统一错误至少区分invalid_input、too_large、decode_failed、model_unavailable、artifact_mismatch、busy、inference_failed、report_not_found；携带request_id和是否可重试。服务端必须验证真实格式、非空、损坏、字节数、宽高/总像素、多帧，前端限制不能替代后端。

EXIF方向、透明通道、RGB、裁剪/插值策略必须与模型原预处理一致或明确拒绝不支持输入；不能静默加ImageOps.exif_transpose等操作后仍标原协议。记录原文件字节SHA与预处理策略。预览不冒充模型实际裁剪输入。

默认上传只送到本项目预览后端，不外传图床、第三方检测或LLM；关闭demo provider、未经核验的外部provider和依赖其生成的图片“解释”。C2PA缺失不是伪造证据；metadata读取不等于签名已验证。没有已验证定位方法时，不绘热力图、框或“AI区域”。

## 6. 必须正确展示的实验口径

数值以结构化结果复核，下面仅为当前验收截面：

| 能力 | 已验收结果与局限 |
|---|---|
| D-OpenCLIP | 同2310 evaluation，AUROC99.8908%，FPR@.5 1.9408%，TPR98.9482% |
| A0 ResNet22来源 | 同1331 AI evaluation，Top1 77.1600%，Top3 92.9376% |
| held-FFHQ检测 | OpenCLIP误报15/93=16.1290%，虽优于ResNet56/93，仍有明显跨域误报 |
| ResNet open20 | grok3未知拒绝59/63=93.6508%，但已知保留576/1212=47.5248%，正确且保留539/1212=44.4719% |

来源：`docs/image_research_results/2026-10-29T072357Z-next-routes-final/{README.md,snapshot.json,acceptance/verification.json}`及各run/provenance。OFA和两级融合另见`docs/IMAGE_200_Q179_RESULTS_2026-10-28.md`。

全部属于单seed、已看过的开发角色；部分父dev参与过选模。数据为63821张、每生成小格187–200张、179新增冲突图隔离。未推理19294张holdout，预训练数据重叠未知。不能宣称独立测试、未知生成器通用保证或生产可用。

## 7. 视觉与动效

白底为主、黑色正文、蓝色强调，红色仅用于错误等特殊状态；使用开源图标，不使用emoji。信息层次先任务/结果，再证据/技术细节。按钮有hover、按下、焦点、禁用状态；卡片/导航/展开收起提供出现、过渡和退出效果，遵守系统reduced-motion，避免闪烁与布局跳动。

Streamlit rerun不能保证任意DOM之间真正Morph。首版采用稳定布局、会话状态、原生控件与轻量CSS微动效；无法保证的跨页退出/形变场景明确记录限制，不为动画重写前端、不使用脆弱DOM注入模拟复杂产品交互。

## 8. 实施顺序和验收

1. 核对版本、模型清单、现有调用链、未提交修改；登记隔离预览配置和API契约。
2. 新图片适配层+真实默认模型；先通过模型/类别/预处理校验，缺失能力明确unavailable。
3. 四页导航及图片独立renderer；保留文本回归与状态，加入历史和导出。
4. 视觉细节、可访问性、响应式；运行AppTest及实际预览浏览器验收。
5. Git只交代码/配置模板/文档/脱敏验收记录；不提交上传图片、模型、数据、缓存、逐图预测、密钥或签名URL。

验收必须有：
- [ ] 原线上文本服务/数据库无修改；预览进程、端口、DB、配置独立。
- [ ] 明确使用本文登记的检测编码器+头和来源权重，类别顺序/预处理/哈希都验证。
- [ ] 使用原selection的至少2张已核验图做实际适配层全重载、单张/批量、离线结果一致性检查，沿用1e-5门槛；测试环境不得访问holdout。
- [ ] 缺权重、损坏/超限输入、错误class map、推理异常、忙碌均不会伪造成功分数；不默认外传图片。
- [ ] 图片结果无文本分块/家族推断，无假热力图；22类未知仍null，研究分数不混入主结果。
- [ ] 页面往返保持输入/结果，重复点击、超时与刷新不会意外重复推理；会话之间不泄露上传内容。
- [ ] 历史/导出可还原当时模型、阈值、预处理、输入哈希，敏感服务器路径不出现在普通用户报告。
- [ ] 关键指标有分母与限制，冻结/KNN的epoch为null；新未完成算法不能标已可用。
- [ ] 提供运行命令、端口、环境依赖、验证结果和已知限制；不自行切换线上流量。

## 9. 给实现模型的简短任务说明

> 请根据本文件实现独立预览的Streamlit+FastAPI四页工作台。先核验实际代码、依赖、模型产物和哈希，采用最简单的兼容扩展，保留文本流程。图片默认D-OpenCLIP检测与A0 ResNet22来源候选；研究模式保持不同分数语义，主22类未知不可用。不得启用未训练scaffold/demo、虚构进度/定位/解释，不训练模型、不改已登记研究源码和配置、不访问holdout、不重启线上服务。先实现真实后端与错误语义，再做页面、历史、导出和动效。独立验证后提交代码与脱敏证据，模型数据不进Git；交付预览入口与启动方法。


## 2026-10-30实验状态更新
本节覆盖前文的实验运行状态截面，接口方案和预览隔离约定不变。已完成的全部研究结果、模型路径/SHA和局限见[阶段总览](image_research_results/2026-10-30T0444Z-final/README.md)。新增ConvNeXt三头、DNA来源线性头、OmniDFA来源线性/K10原型及OFA20+1均完成验收；这表示离线研究工件可复核，不表示服务已经接入。

默认检测仍优先考虑OpenCLIP，22来源候选排序仍可用原A0 ResNet（披露parent-dev选模偏差）；DNA是可展示的冻结来源对照。FFHQ固定.5下OpenCLIP15/93误报，ResNet和ConvNeXt均56/93；cal目标1%后OpenCLIP在FFHQ实际仍7/93误报，不写成1%跨域保证。OFA20+1拒绝63/63未知但仅保留58/1212已知、正确且保留28/1212，应标示覆盖损失，只置研究模式。20类阈值不能套用22类主接口，主接口未知拒识仍为null/未校准。页面重构与模型接入尚由接手模型实施，本实验任务没有部署页面或修改线上服务。
