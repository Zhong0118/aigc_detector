# 图片检测与来源溯源：执行路线与版本记录

更新：2026-10-26。本文区分已运行、已准备和复现候选；论文指标不是本项目结果。

本轮最新运行状态：100张版42,000张和230/230收据已完成，父22,000成员完整性检查通过，100张20轮D0已实际开始；A0与冻结OFA按原队列继续。新增四组等待这批基线完成。

## 1. 项目与代码边界

| 位置 | 已核验状态 | 处理 |
|---|---|---|
| GitHub `sync` | `2b37c811f7787beba8e570cf4d5bac9b1ef27b30` | 协作基线，不合入、不强推 |
| 服务器研究工作树 `attribution-pilot-baseline` | 原HEAD `4bcbf7f` 比sync领先10提交、落后0；已另存历史图片改动为`708600c` | 本轮唯一代码修改与训练位置；普通推送同名单数功能分支 |
| 服务器生产工作树 `sync` | `b8adb8b`，存在文本检测、溯源、配置等未提交修改 | 保留现场，不拉取覆盖、不改服务 |
| Mac本地旧项目 | `codex/attribution-pilot-baselines` / `2099357`，大量删除和未跟踪文件 | 不作为最新训练代码，不恢复或覆盖这些修改 |

研究工作树：`/home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline`。
历史44个图片源文件已打包并保存diff、SHA和安全扫描，位置`outputs/image-research-expansion-20261026/`。146项已有回归通过。数据、权重、完整日志不进入Git。

现有应用链路：`api/analysis.py` → ingestion → detection → provenance → reports → storage。生产的文本服务继续使用原配置；研究入口不修改该调用链。旧`detection/image_detector.py`含未训练骨干/启发式逻辑，不作为已验证检测器。现有外部provider名称不代表相应论文已复现。

## 2. 两项独立任务

| 项目 | Detection：真假检测 | Attribution：生成来源 |
|---|---|---|
| 输入假设 | 不知道图片真假 | 对AI生成图做条件来源分析 |
| 训练标签 | real=0、AI=1 | 作者发布目录的22个生成来源；不加入real类 |
| 第一轮新模型 | 冻结DINOv2-S/14 + 独立2类线性头 | DINOv2 RGB、OFA指纹、两者融合各自的22类线性头 |
| 选择指标 | selection子集AUROC | selection子集Macro-F1 |
| 校准 | calibration真实图选择1%、5%目标FPR阈值 | 闭集阶段保留calibration；不能冒充已实现未知拒识 |
| 最终评估 | evaluation的AUROC、accuracy、固定0.5与校准阈值的FPR/TPR和Wilson区间 | evaluation的Top-1/Top-3、Macro-F1、混淆、来源/语义分组指标 |
| 输出语义 | 独立检测softmax分数；不把它自动称为校准概率 | 独立来源softmax；OFA配对分数另列，不能混为检测概率 |

同一个冻结DINO特征缓存可以复用，两个任务的分类头、损失、选模和输出相互独立。这不是联合多任务训练。

## 3. 按顺序执行的代表性实验

先等待现有100张版D0/A0各20轮及冻结OFA完成，再运行下表4项。原50张/12轮、50张/20轮、100张/20轮先做同旧dev比较，保留所有模型。

| 编号 | 输入/预处理 | 模型与权重 | 训练 | 解释边界 |
|---|---|---|---|---|
| D0/A0 原对照 | RGB Resize256、CenterCrop224、ImageNet归一化 | ImageNet预训练ResNet50，分别真假2类/来源22类 | 已有端到端训练，20轮 | 工程对照，不宣称SOTA |
| V1 原OFA | DnCNN残差→FFT幅度→log1p→指纹编码 | 作者冻结128维编码器及配对头，每来源100训练参考 | 不训练，epoch=null | 来源验证宏AUC，不是22分类准确率或真假检测 |
| D-DINO | 官方RGB bicubic Resize256、CenterCrop224、ImageNet归一化；CLS384后L2 | 官方DINOv2-S/14冻结 + Linear(384,2) | 独立二分类CE，20轮 | 本项目线性探测器，不是OmniAID复现 |
| A-RGB | 同D-DINO | 同一冻结DINO + Linear(384,22) | AI-only CE，20轮 | RGB基准 |
| A-FP | 原OFA256裁剪与归一化；每图DnCNN微批、/256、log1p(abs(fft2)) | 冻结作者OFA128 + 新Linear(128,22) | AI-only CE，20轮 | 论文指纹编码复用，分类头为项目适配，不取代原配对验证 |
| A-Fusion | 两分支分别L2→拼接→整体L2 | 冻结384+128特征 + Linear(512,22) | AI-only CE，20轮 | 不加权混合原OFA匹配分数；更大头是容量混杂因素 |

四项均固定seed42、batch32、Adam lr0.001/weight_decay0、StepLR(step1,gamma0.9)、无图像增强、骨干冻结。各自selection最优epoch，平分取最早。真实图和所有AI图用于D-DINO；三组来源分类使用完全相同的AI角色IDs。来源结果以原group_id做1000次配对bootstrap，比较RGB/指纹/融合，不能仅凭总体分数判定融合胜出。一个seed不足以证明稳定收益，融合后仍需要额外种子或固定鲁棒性条件的确认。

独立配置：`configs/image_experiments/iab_research_v6_100.json`。服务器唯一`--watch`等待器每1800秒检查，应用关闭不影响服务器队列；检查`watch.json`与`launch.json`身份，禁止重复启动。
入口：`scripts/run_image_research_queue.py`，默认只读plan；`--advance`仅在两个已完成基线证书一致后启动一次。100张manifest已实际固定为`e8152a198db5b39481482cab18be9a56d173ba6a25dc330bc40cf1fe7cd4b06e`，配置明确绑定；基线完成证书一致后再写独立`bound-protocol.json`，不重新划分数据。失败或未知退出保留状态，禁止自动覆盖重训。

## 4. 论文路线的可复现性与优先级

| 方法 | 真正任务/公开实现 | 当前可用性及差异 | 本轮顺序 |
|---|---|---|---|
| OmniAID | 真假检测，CLIP ViT-L/14@336与SVD残差专家 | 官方v2 checkpoint约3.24GB，另需CLIP基础权重；推理预处理与早期论文参数有差异。必须按v2配置匹配，核对checkpoint缺失键；不能用DINO分类头替代后称为复现 | 预训练冻结评估候选，先取得完整资产并单独注册协议；本次4项队列不假装包含它 |
| OFAAttribution | 来源指纹提取与参考匹配 | 作者指纹/DnCNN/配对权重已可用，现有冻结链路已跑；原模拟器训练与辅助real/reconstruction头不等于通用真假检测器 | 保留原配对验证；新增指纹线性头消融 |
| UniversalAttribution | OpenCLIP ViT-B/16中间层CLS + 正则逻辑回归/近邻 | 需LAION2B对应骨干；作者脚本默认权重与配置不同，须核对。LBFGS迭代不是epoch；代码未见明确许可证，不能直接批量复制发布 | 下一阶段低成本对照；固定层/正则或只用selection选择 |
| OmniDFA | 两个ConvNeXt-Small分支、SupCon、原型少样本来源分析 | 公开checkpoint约1.2GB/个；原完整训练8×A100，当前不承诺重现。须检查预训练来源重叠、随机裁剪及5/10-shot协议差异；代码许可待核实 | 优先考虑冻结权重原型评估，不从头复刻大训练 |
| DNA-Det（AAAI2022） | 局部patch、图像变换预训练、CE+对比学习的GAN架构归因 | 16个64×64patch/图；原平衡采样可显著放大batch。旧架构标签与现代模型精确版本不是同一任务；checkpoint/代码许可仍需确认 | 次级候选，不在首轮混入未知拒识结论 |
| ConvNeXt / OpenCLIP / DINOv2 | 通用视觉骨干 | 需要我们独立训练任务头；不能把通用预训练性能称为本任务SOTA | DINOv2已真正取得权重；其他作为后续同协议骨干替换 |

OmniAID另已准备`scripts/evaluate_omniaid.py`和禁用配置`configs/image_experiments/omniaid_v2_evaluation.json`：严格离线作者源码/权重哈希、固定336预处理、GPU共享锁、校准/评估分离、失败状态保留。作者基础CLIP固定revision仅提供约1.71GB bin，须用安全weights-only读取转换到safetensors并登记哈希，另需3.24GB检测权重。完整重载凭据和真实资产尚未齐，当前不运行、不宣称复现成功。

逐项论文、代码、权重、预处理、损失和训练协议证据见本目录的`IMAGE_DETECTION_METHOD_EVIDENCE_2026-10-26.md`、`IMAGE_ATTRIBUTION_METHOD_EVIDENCE_2026-10-26.md`。没有使用论文数值充当本项目指标。

## 5. 数据与来源真实性

当前100张版固定22来源×10语义×100=22,000 AI；真实每语义最多2,000，共20,000；总42,000。已保留50张版所有22,000旧成员的split、group_id、标签和hash。新数据若桥接旧分区则停止并报告，不移动holdout。

作者官方链接指向的HF镜像现有24个AI来源，新增FLUX2_KLEIN和Z_IMAGE，对应作者代码的FLUX.2-klein-4B及Z-Image-Turbo。250个Parquet文件页脚元数据已核验，总559,101行；所有240个AI小格至少1,880行，容量上支持200/500，但不等于已下载或可直接无缝替换现有ZIP样本ID。

| 来源请求 | 核验结果 | 本轮标签处理 |
|---|---|---|
| GPT Image新版本 | `4o`是作者GPT-4o目录；尚未见公开GPT Image 1/1.5数据文件 | 保留4o，不能改名成GPT Image 1/1.5 |
| Seedream / Nano Banana及Pro | 论文/生成脚本提及，当前公开文件树未见对应目录 | 标为数据缺口，不调用付费API补造 |
| FLUX | 当前FLUX是FLUX.1-schnell；镜像新增FLUX.2-klein-4B | 原22类不变；新类需另建增量版本 |
| Gemini | 现有gemini对应作者Gemini 2.0实验图像模型 | 不改称Nano Banana |
| Midjourney / 可灵 | 已有mid-5.2、mid-6.0、kling | MJ版本按目录保留；kling后端具体版本不猜测 |
| GenImage | 适合作为真假检测跨来源公开基准，CC-BY-NC-SA | 外部评测另登记来源重叠与划分；不能用其旧来源填补新模型缺口 |

数据公开入口与逐格容量见`IMAGE_DATASET_EVIDENCE_2026-10-26.md`及`image_dataset_cells_20261026.csv`。Dataverse登记CC-BY-4.0，HF卡标Apache-2.0，许可证来源差异需要保留，不能自行统一。

## 6. D/A后续策略与顺序

| 策略 | 改变的因素 | 评估规则/何时运行 |
|---|---|---|
| D1 | 训练JPEG/缩放增强 | 必须匹配同架构无增强对照、同seed/roles/20轮；旧20张版D1与对照已有结果 |
| D2 | 跨生成来源或跨语义泛化 | 训练排除预登记来源/语义；未知来源不能参与选模或阈值校准 |
| D3 | 只改检测阈值 | 冻结权重、calibration真实图定阈值；同时报告FPR与TPR。降误报可能损失召回，不称全面提升 |
| A1 | 未知来源拒识 | 排除未知来源后重训；未知校准和未知评估来源分离；MSP分数不代表可靠概率；原50张协议有实现与结果 |
| A2 | JPEG/缩放/裁剪、跨语义 | 同一冻结checkpoint在固定evaluation IDs上比较；裁剪或跨语义需分别注册，不把后处理混成一次调参 |
| A3 | 家族/版本层级 | 先核验每个目录的家族和具体版本证据；不能从模型名称猜映射 |
| 融合 | 特征拼接或分数校准后融合 | 当前仅注册特征拼接消融；加权分数方案须仅在calibration拟合，独立evaluation报告 |

本轮不一次性启动所有论文训练和后续实验。先回答代表模型是否有收益，再根据误差、置信区间、算力和数据决定微调/鲁棒性/未知拒识的顺序。未启动项目不能写成已完成。

## 7. 扩至200、500的执行条件

用户已授权必要时扩容；当前100张清单运行中，保持不变。扩容必须另建版本，优先取旧固定清单前缀并复用文件；现有100张所有ID、split、group、hash保持。先评估训练与selection差距、来源混淆、低FPR召回及同dev规模曲线；若明显是标签错误或预处理问题先修复，不能用加数据掩盖问题。

**本轮发现的分配偏差：** 实际100版train25,446/dev4,960/holdout11,594。原实现按“语义集合＋来源集合”的小分层取整数比例；1,775个仅有一个新组的分层全部进入holdout，涉及4,820张新增AI。AI训练样本实际从7,612增至11,456，不是翻倍。数据分区仍隔离，旧成员未移动，但该取整偏差会削弱扩容收益，必须在结果解释中报告。审计证据：`outputs/image-research-expansion-20261026/split-allocation-audit.json`。

已独立准备`research/image_attribution/extension_split.py::group_and_split_extension_v2`，未来仅对无父锚点的新组按域分离SHA256固定种子分配70/15/15期望比例，避免小分层全入holdout；旧成员、旧holdout、锚点组及近重复保护完全保留。它**未接入当前100流水线**，未来启用须另登记版本和实际计数。

200格对应44,000 AI；500格对应110,000 AI。当前Dataverse真实子集仍只有20,000，不虚构平衡数。若使用HF更多真实图，必须核对与旧资产的逐图身份/去重，并独立登记新增版本。类别先验变化和训练步数变化均需报告；若引入类权重，先补100张同权重对照，不能把两个因素算作单纯数据收益。

仅在逐格实际可得性、完整性、预估存储和父分组保护通过后扩200，再根据200结果决定500。网络失败不是来源数量不足；没有指定业务验收指标，不能事后发明‘达到某准确率即成功’门槛。当前单seed的开发集结果只提供工程研究证据。

## 8. 产物、接口及监控

- 研究输出：`outputs/image_experiments/independent-image-tasks-v6-100/{D-DINO,A-RGB,A-FP,A-Fusion}/`。
- 每项保存resolved config、role IDs、逐epoch、最佳头、重载/单图批量校验、逐样本预测、metrics和provenance；特征缓存记录权重、代码、清单及ordered IDs。
- 配对结果：`paired-comparison.json`。DINO真假和ResNet D0在相同evaluation IDs上比较；来源三组与ResNet A0亦同IDs。旧基线用整个父dev选模，比较仍有探索性偏差。
- 旧三条件汇总：`scripts/compare_image_scale_epochs.py`，只读已存分数；50/12→50/20观察轮数，50/20→100/20观察规模；同3140条真假、1496条AI dev；新全dev另列。
- 独立研究推理：`scripts/predict_image_research.py --detection-run ... --attribution-run ... --images ...`。分别返回detection和attribution字段，默认CPU；可选择经calibration得到的检测阈值。未知来源未校准时输出null，不伪造未知概率。暂不改变线上API配置。
- 当前30分钟自动化继续原下载/20轮基线，并在前置证据核验后推进这4项代表实验。共享研究GPU锁，使用物理GPU1，保留GPU0 Ollama。失败保留现场，不自动重复训练。
