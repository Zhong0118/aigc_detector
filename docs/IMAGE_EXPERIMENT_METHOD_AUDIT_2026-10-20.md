# 图片实验方法、实现与运行审计（2026-10-20）

本文件描述服务器 `attribution-pilot-baseline` 的实际实现和已完成记录。历史文档中的 Tiny 数据、20 epochs 和未取得 IAB 状态不是当前结果。所有指标均为项目开发评估；未对 holdout 推理或调参，不能称为论文全量复现或 SOTA。

## 方案与状态

| 编号 | 当前实际方案 | 是否训练新模型 | 实际状态 | 与论文关系 |
| --- | --- | --- | --- | --- |
| D0 | ImageNet V1 ResNet50，全骨干微调，2048→2，real=0/AI=1，按训练类数加权 CE；Adam .001、StepLR(1,.9)、batch32、seed42、12轮，dev AUROC选模 | 是 | v4-20已完成12轮，最佳第10轮 | 借用IAB ResNet骨干/优化器的项目独立真假检测基线；不是论文验证的SOTA检测器 |
| D1-control | D0无增强训练，同D1的train/selection/calibration/evaluation IDs和优化参数 | 是 | 完成12轮，最佳第10轮 | 项目匹配对照 |
| D1 | 在对照上仅加入训练侧JPEG质量50/75/95与缩放.5/.75，两种增强分别p=.5；指标输入保持干净 | 是 | 完成12轮，最佳第12轮 | 项目增强消融；不等于论文原样配置 |
| D2 | 排除4o训练和校准，评估该未知生成来源的真假泛化；另预登记FFHQ跨语义版本 | 是 | 代码/配置已准备，未运行 | 项目泛化协议；并非论文的cat/FFHQ/bedroom单语义训练协议 |
| D3 | 复用D0，calibration负例拟合目标1/2/5% FPR阈值，独立角色evaluation报告FPR/召回、Wilson区间及分组bootstrap | 否 | 已完成 | 项目阈值分析；D0原选模见过整个父dev，必须标注探索性 |
| A0 | 同一ResNet50骨干，2048→22，仅AI生成来源，无真实类；不加权CE，12轮，dev macro-F1选模 | 是 | 完成12轮，最佳第12轮 | IAB公开ResNet实现的子集适配，非严格全论文复现 |
| A1 | 20个已知来源训练；kling仅用于未知校准，grok3仅用于未知评估；max-softmax拒识阈值，未知误接收目标5% | 是 | 已有代码/固定配置，未运行 | 项目开放集基线；不是OFA论文额外训练的T+1开放集分类头 |
| A2 | 复用A0做clean/JPEG95/75/50、resize.75/.5的配对鲁棒性评估；另预登记FFHQ跨语义重训版本 | 主版本否，语义版本是 | 未运行；跨版本协议未定 | 项目鲁棒性协议；不是论文DS .5/.25、JPEG65/30、blur sigma3/5的完整原样设置 |
| A3 | 依据有证据的来源→家族映射，对A0来源概率求和，报告家族和具体来源指标 | 否 | 代码已有，映射reviewed=false，禁止执行 | 项目输出聚合，不是HiFi-Net分层网络训练，不能把家族准确率当22类准确率 |
| V1/OFA | 冻结作者DnCNN→残差FFT/log1p→ResNet50 128维指纹→非对称学习配对头；每来源100训练参考，dev查询，平均配对分数 | 本轮否 | v4-20已完成 | 使用作者训练权重的完整指纹推理；未重训模拟器/提取器，未训练论文T+1拒识头 |
| C0 | 历史CLIP+轻量分类器兼容入口 | 可选 | 本轮v4-20未作为对照运行 | 不是当前必选主方法；旧openai/clip-vit-base-patch32也不是IAB论文CLIP对照所用模型 |

D0—D3、A0—A3是本项目实验编号，不是八篇论文或八种互相替代的网络。D1改变训练输入，D2改变泛化问题，D3改变决策阈值；A1改变开放集任务，A2检验扰动，A3改变输出粒度，均不能用更容易任务的分数冒充A0提升。

## 主来源与原样实现程度

1. [ImageAttributionBench论文](https://arxiv.org/html/2605.12967v1)，§4、附录D；[固定作者代码](https://github.com/mttry/ImageAttributionBench/tree/19bb7d805142d30e67689dae009ece170866c5f3)。本轮重新读取了该commit源码，包括 `training/attributors/attributor_resnet50.py`、`dataset/ImageAttributionDataset/dataset_resnet50.py`、`dataset.py`、`dataloader.py`、`training/config/model/resnet50.yaml` 和训练入口。
2. [OFA AAAI2026论文](https://ojs.aaai.org/index.php/AAAI/article/view/40851/44812)；[固定作者代码](https://github.com/jumpycat/OFAAttribution/tree/3780cbffeeac48c0bf5e72a025f4803e408b9dfa)。完整架构、预训练资产及严格加载细节见 `docs/IMAGE_METHOD_AUDIT.md` 的OFA部分。
3. [实际IAB公开V3.2数据](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O4S4IV&version=3.2)，目录为22个生成来源及real；不能把新论文中扩展的标签猜测为现有目录的版本。

| 项目 | 作者论文/固定代码 | 当前项目 |
| --- | --- | --- |
| A0骨干/头 | torchvision pretrained ResNet50、backbone.fc=Identity、单独线性头 | 相同结构，显式锁ImageNet V1权重，严格加载 |
| A0优化 | CE，Adam .001，batch32，代码StepLR(1,.9) | 相同；D0额外按真假训练类数加权 |
| 轮数 | 论文附录D专门指定ResNet标准任务20轮；通用默认10轮，代码YAML30轮，README示例10轮 | 用户明确指定12轮，不能称20轮原样复现 |
| 预处理 | 论文正文概述256裁剪；固定ResNet专用代码Resize256/CenterCrop224/ImageNet归一化 | 跟随专用代码224，不悄悄改成256；训练和推理一致 |
| 来源条件裁剪 | 作者基类对grok3按真实标签切除右100/下50像素 | 未复制；所有来源统一处理。盲来源推理不应依赖未知真实标签；此差异明确保留，未证明其影响大小 |
| 类别 | 固定代码默认22生成+real共23；新论文另有扩展标签 | A0只22生成来源，D0另训二分类；任务独立 |
| 划分 | 作者标准80/10/10按标签随机划分 | 自定义70/15/15，prompt/精确重复/近重复分组，holdout封存；既不是同数据量也不是同划分 |
| 选模 | 作者配置accuracy | A0 macro-F1、D0 AUROC。虽来源数均衡仍可能选到不同epoch |
| OFA训练 | 作者先通过指纹模拟等训练提取器/学习头，再做参考验证；开放集另训T+1头 | 复用作者已训练资产，100参考/来源，验证AUC；不是从头训练复现或完整开放集复现 |

上述差异影响可比性，应先逐项报告，不能把论文数值与本工程分数直接相减。

## 已完成结果

v4-20共8,800：AI/real各4,400；每来源200，每来源每语义20。训练6,138，dev1,274，holdout1,388（未推理）。来源分类每来源139训练、28开发，不存在某个来源因下载多就进入更多训练样本。

| 已完成基线 | 实测开发结果 |
| --- | --- |
| D0 | AUROC 0.981114；accuracy91.68%；FPR@.5=88/658=13.37%，TPR=598/616=97.08% |
| A0 | Top1=263/616=42.69%，Top3=70.94%，macro-F1=.424152；训练最终Top1=98.82% |
| OFA | 来源验证宏AUC=.887708；辅助闭集Top1=36.20%，Top3=67.53%；100参考×22来源、616 dev查询 |

共同428条evaluation（208真实/220AI）比较：D0与D1-control AUROC .976246、FPR@.5 12.02%、TPR97.27%；D1 AUROC .975328、FPR8.65%、TPR93.64%。增强同时降低误报与召回，尚无整体提升结论。D3目标2%时实际校准FPR1.66%，evaluation FPR2.88%、TPR74.55%；目标FPR不是独立评估或生产保证。

逐轮数据与图表已经存在：
- `outputs/image_experiments/iab-v4-20-report/`：D0/A0全部24轮、OFA与级联分析、报告及hash。
- `outputs/image_experiments/d1-d3-v4-20/`：匹配对照与D1共24轮、共同evaluation比较、阈值区间和A0混淆分析。
- 原模型在对应 `d0-resnet50-iab-v4-20`、`a0-resnet50-iab-v4-20`、`ofa-iab-v4-20` 和 `followups-v4-20/{D1,D1-control}/train`；D3不产生新模型。

## 是否可能是代码或协议的问题

不能仅凭训练跑完排除代码问题。已核对以下证据：A0真实类排除、词表固定排序、real0/AI1语义、训练与推理RGB/归一化一致、dev不增强、预训练/训练权重严格加载、保存重载及单图/批量一致性、逐样本分数重算和匹配对照。D0/A0保存重载最大差均为0，原基线完成退出码均0；本轮未发现足以解释42.69%的标签倒置、漏载权重或错误随机配对头证据。OFA此前数值一致性问题已明确使用FP32、禁止TF32、DnCNN微批1处理，失败现场保留。

已确认的局限或需要修复的点：
- D3复用的D0 checkpoint选择看过整个父dev；其evaluation角色不是完全独立于选模。只作探索性分析。
- 小样本dev再按group-hash三分后，428 evaluation的FFHQ/cat/church没有AI查询；这些语义AUROC不可算，不能宣称覆盖10类语义的泛化。不能为看分数临时重划旧组。
- A0训练98.82%而dev42.69%，差56.13个百分点，支持泛化/过拟合问题，但不单独证明是数据量。4o→dalle3为16/28，SD3_5→SD3为9/28，janus-pro→grok3为8/28；混淆不能用来擅自合并标签。小数据、没有增强/正则、学习率及纹理/语义依赖等均需对照验证。
- 原manifest生成器在扩充后会重新全量shuffle，可能把旧holdout调入训练。这是扩充时的实际风险，本次已增加父manifest锚定：所有旧样本split/group_id保持；新增重复若桥接旧split或group则报错停止，而不是偷偷重划。该修复不改变既有v4结果。
- prompt_id按发布者文件名索引隔离，跨来源提示语对齐属于明示假设；未取得prompt正文逐条证明，不能声称消除了所有语义相关性。

本轮新增划分和监控隔离测试先失败后通过；复核发现已有manifest可能绕过父划分保护，已增加始终执行的元数据检查及两项回归测试。最终88项测试通过；shell语法、Python编译和diff检查通过，详见 `outputs/iab-v5-50-monitor/verification.json`。测试不能证明全部模型设计无误，更不能证明数据扩充一定有效。

## 扩充到每小格50张

新建 `outputs/image_experiments/iab-balanced-v5-50`，不覆盖v1/v2/v3/v4。
- AI：22×10×50=11,000，每来源500；real：每语义1,100，共11,000，总22,000，2.5倍v4。
- 取v3-100冻结清单固定前50个；包含v4全部8,800，不重新抽样。公开目录所有AI小格均足够50，当前无数据源数量缺口。
- 新selection SHA256 `375dd7978721324db4b6891746a1d99ab09a62a1ba57afe21bd83e98ed485c5c`。
- 初始逐文件SHA校验并硬链接复用6,028 AI+9,680 real=15,708；仅补6,292张约6.35GB。原图目标约14.33GB。
- 原v4 manifest SHA256 `c4adf2e0c25c4cbaba75e98bca4e29c3ae8287430ee026ab6118e7e7eab69b3e`，其8,800个样本的split/group_id必须保持；新增无旧锚点的组固定seed42分70/15/15。不使用holdout分数；只做文件完整性与身份校验。
- 新配置 `configs/image_experiments/iab_v5_50_run.json`；独立监控目录 `outputs/iab-v5-50-monitor`，4路服务器直连、签名链接内存复用、正常TLS与公开GET；正式下载数据落盘保留，不再做独立丢弃式网络验证。
- 新下载已真实启动，状态/PID/启动身份与日志在新monitor；服务器独立600秒loop负责推进，桌面每10分钟跟进。旧完成收据和旧模型不改动。
- 下载完整且父划分校验通过后，沿用D0/A0各12轮与冻结OFA，GPU0顺序执行；不自动启动D1/D2/A1/A2/A3，避免同时改方法与数据。OFA只扩参考候选与查询，不训练新指纹网络，仍100参考/来源。
- 新旧对比优先从已保存新预测筛出原v4相同dev IDs，分开报告新增dev。相同12轮下大数据优化步数也增多，不能声称严格隔离了“数据多样性”这一个因果因素。50/格仍是小规模先导，提升与否以实测为准。

## 本轮交付状态

代码尚未提交或推送，保留在独立研究工作树，HEAD `4bcbf7f53fafebd4719652c3600db5267b087485`；本轮前代码快照在新monitor的 `pre-expansion-code/`。针对本轮新增差异的独立复核已完成，发现的续跑检查入口问题已修复，无剩余重要问题。新监控不更改旧基线完成收据，桌面10分钟自动化ID为 `iab-50`。最新下载进展见 `outputs/iab-v5-50-monitor/latest-user-progress.json`，下载运行中，不能声称50张版指标已取得。
