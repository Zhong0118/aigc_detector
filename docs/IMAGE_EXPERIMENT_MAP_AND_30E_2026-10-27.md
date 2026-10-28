# 图片实验路线、分数融合与30轮对照

本批最终状态见[2026-10-29拓展验收](image_research_results/2026-10-29-final/README.md)；下文保留2026-10-27登记时的历史状态。

更新：2026-10-27。论文数值不作为项目成绩；本表区分历史已跑、当前已跑、配置就绪与排队。

## 1. 本轮实际完成的分数融合

用户选择的是融合现有ResNet50来源分类器与RGB+指纹特征融合模型的22类来源概率，未将检测概率加入。保持v6-100清单及原train/dev/holdout。alpha只在757张AI calibration上按NLL选择；固定606张AI evaluation用于报告。alpha网格0到1、步长0.05，数值平分优先原ResNet端点。50/50是预设对照，不按evaluation结果倒选校准权重。

|方法|ResNet权重|Top1|Top3|Macro-F1|
|---|---|---|---|---|
|ResNet50|1.00|64.19%|84.82%|63.57%|
|FeatureFusion|0.00|63.04%|87.29%|62.21%|
|Uniform|0.50|66.01%|93.07%|65.42%|
|Calibrated|0.70|65.18%|91.91%|64.54%|

校准组合Top1较ResNet变化+0.99pp，151个原group配对bootstrap95%区间[+0.17,+1.67]pp。原A0曾用父dev选checkpoint，仍为单seed探索性结果，不是独立测试或未知来源保证。

## 2. RGB、指纹与融合究竟采用什么

|路线|实际实现|训练范围|当前状态|
|---|---|---|---|
|D0检测|ImageNet初始化ResNet50，RGB Resize256/CenterCrop224/ImageNet归一化，真假2类头|骨干和头端到端；真实=0、AI=1|20/50/100张版已跑；当前100张20轮完成|
|A0来源|同一ResNet50架构，独立22类来源头；仅AI图|骨干和头端到端，独立于D0|20/50/100张版已跑；当前100张20轮完成|
|D-DINO检测|冻结官方DINOv2-S/14，RGB CLS384经L2，独立2类线性头|只训练头|100张20轮完成|
|A-RGB来源|冻结DINOv2-S/14的CLS384，经L2后22类线性头|只训练头|100张20轮完成；没有同时调用ConvNeXt/OpenCLIP|
|A-FP来源|作者DnCNN预测噪声残差，逐图微批/256→FFT2绝对值→log1p→作者ResNet50指纹编码器128维→L2→22类线性头|DnCNN及指纹编码器冻结，只训练新头|100张20轮完成；是OFA指纹复用，不是完整OFA从头重训|
|A-Fusion来源|384维RGB与128维指纹分别L2，拼接后整体L2，512→22线性头|两个编码器冻结，只训练融合头|100张20轮完成；验证两类特征互补|
|V1/OFA|相同冻结指纹链路＋作者已训练配对判别头；每来源100张train参考，与dev查询匹配|本轮不训练，epoch=null|20/50/100张均已跑；来源验证宏AUC单独报告|
|A-score-fusion|ResNet A0概率与A-Fusion概率加权组合|只用calibration拟合一个alpha|本轮已实际完成，见上表|

RGB特征不是独立标注的风格特征，不声称网络已经分离出风格或生成机制。指纹路线按作者公开代码和冻结权重实现，包含提示词中的DnCNN/FFT/Log链路；没有重做论文的模拟器、监督对比预训练。A-FP新分类头与原OFA配对验证是不同实验。

其他候选：ConvNeXt、OpenCLIP/UniversalAttribution、OmniDFA、DNA-Det尚未在当前100张版完成模型实验。OmniAID已准备禁用入口，但检测权重/CLIP基础资产及完整加载验证未齐，尚未运行。旧CLIP接口或代码不能当作当前100张基线成绩。GenImage为外部检测泛化候选，当前42,000张训练数据来自IAB固定子集。

## 3. D0-D3与A0-A3是实验策略，不是八种骨干

|编号|技术逻辑|需要重新训练|实际状态与适用数据|
|---|---|---|---|
|D0|真实/AI二分类；AUROC、accuracy、FPR、TPR|是|当前100张20轮完成；30轮对照排队|
|D1|训练端随机JPEG质量50/75/95及缩放0.5/0.75，各p=0.5；与同角色同seed无增强D1-control比较|是，两组匹配|仅20张版已实际跑；100张30轮模板已准备，暂未启动|
|D2|训练排除指定来源4o，或另做排除语义FFHQ的泛化；在预定未见域评估|是|代码和配置已有；尚未实际运行，不与D0同分布分数混比|
|D3|固定检测权重，用calibration真实图选择目标FPR1%/2%/5%的阈值；同时报告实际FPR和TPR|否|20张版已跑；100张新比较已做1%/5%校准；没有“D3训练30轮”|
|A0|已知AI前提下22个发布来源闭集分类|是|当前100张20轮完成；30轮对照排队|
|A1|排除未知来源后训练20类；kling只做未知校准、grok3只做未知评估；最大softmax拒识|是|50张版已完成；100张未重跑；仅少数未知样本，不能保证任意未知泛化|
|A2|同一已训练A0在JPEG95/75/50、缩放0.75/0.5和clean图上做逐样本对比；跨语义需另训练|后处理评估否；跨语义是|50张版后处理6条件完成；100张新任务未启动|
|A3|用证据核验的家族映射汇总来源概率，区分家族与具体版本|当前方案否|代码已有，映射证据仍需核验，尚未运行|

D1/D2/A1这类训练实验的新增模板预算为30轮；D3/A2后处理/A3/OFA为评估或聚合，epoch=null/不适用。本轮不因改轮数批量重跑全部D/A实验。

## 4. 30轮已经怎样安排

1. 原20轮权重、resolved配置、日志均保留，训练CLI不覆盖历史。D0/A0新输出suffix为v6-100-e30，数据及manifest完全不变。
2. D0/A0从原ImageNet初始化按同seed42、batch32、Adam0.001、StepLR(1,0.9)完整训练30轮；不能从“最佳19轮”的A0权重冒充连续训练到30轮。
3. 服务器物理GPU1目前有其他用户约15GB任务；现有monitor已记录waiting。唯一1800秒loop在GPU空闲后推进D0→A0→冻结OFA验收，物理GPU0 Ollama不动。
4. 正式四组30轮特征对照已登记在iab_research_v6_100_e30.json：同GPU1、相同旧特征硬链接、相同角色，仅增加轮数；等待30轮基线完成且GPU1空闲后通过原监督器启动一次。
5. 为避开GPU占用，曾尝试CPU缓存头：D-DINO/A-RGB完成并验收；A-FP的CPU完整图片推理与GPU缓存分数差0.000404，超过1e-5原门槛，批次标记failed且完整保留，A-Fusion未继续。该CPU尝试不冒充正式四组30轮完成。诊断与当次脚本/配置归档在outputs/image-followups-20261027/cpu-cache-attempt。
6. D1/D2/A1等30轮配置仅准备，不自动启动。先比较20/30曲线与误差，再决定后续路线或200/500扩容。

## 5. 100张版的实际磁盘空间

100张是每来源×语义一格100张：22,000 AI＋20,000真实，总42,000张。

|项目|实际数量/字节|说明|
|---|---|---|
|整数据目录占用|29,019,406,336字节，约29.02GB/27.03GiB|du实际分配，包括图像、sidecar、清单、收据、目录和catalog|
|生成图原始字节|26,453,777,925，约26.45GB|22,000张|
|真实图原始字节|2,076,935,750，约2.08GB|20,000张|
|全部图像原始字节|28,530,713,675，约28.53GB|不含清单、sidecar等|
|与旧版本硬链接共享|27,346张|旧数据保留，共享同一inode，不重复占原图空间|
|本版新增独占原图|14,654张，13,043,298,306字节，约13.04GB|不把目录du的29GB都算作本次新增|

不能直接相加各版本的du数值来算真实总占用。以上不包含模型权重、实验日志和其他研究目录；检查时全盘剩余约203GB。

## 6. 代码、配置与产物

- 分数融合入口：scripts/run_image_score_fusion.py；配置configs/image_experiments/iab_score_fusion_v6_100.json。
- 融合产物：outputs/image_experiments/resnet-feature-score-fusion-iab-v6-100-e20/，包含calibration.json、四组逐样本预测、summary、provenance。
- 30轮基线：configs/image_experiments/iab_v6_100_e30_run.json；状态outputs/iab-v6-100-e30-monitor。
- 30轮正式特征对照：configs/image_experiments/iab_research_v6_100_e30.json；输出outputs/image_experiments/independent-image-tasks-v6-100-e30-gpu。
- D/A后续模板：configs/image_experiments/iab_followups_v6_100_e30.json；automatic_execution=false。
- 本轮新增8项分数融合测试通过，与复用的训练/推理回归共41项通过。测试成绩不是模型指标。
- 原服务器生产sync树、文本服务和线上模型未修改；所有工作仍在独立attribution-pilot-baseline分支。大型模型/数据不进Git。
