# IAB 200-q179：登记拓展实验全部完成

核验时间：2026-10-29T02:04:06.691130+08:00。本批14项train/evaluate均completed/exit0；6项训练各连续完成30轮。唯一队列于2026-10-29 01:46:16（+08）记录queue=completed，原身份进程已结束。本次是已登记实验验收，不代表生产高可靠和通用未知拒识目标已经达到。

[完整指标](summary.json) · [原group配对比较](paired-comparisons.json) · [逐来源CSV](paired-per-source.csv) · [来源SHA](source-files.json) · [只读分析脚本](analysis.py)

|路线|实际逻辑|状态|
|---|---|---|
|D1-control / D1|同role/seed ResNet50，无增强与JPEG/缩放增强|各30轮及clean评估完成|
|D2|train/selection/calibration排除4o，只在未见4o AI与真实图验证|30轮及评估完成|
|D2-semantic|train/selection/calibration排除FFHQ，只在FFHQ验证|30轮及评估完成|
|D3|冻结D0，calibration真实图选择1%/2%/5%目标FPR阈值|完成，不训练|
|A1|20已知来源，kling未知校准、grok3未知评估；最大softmax拒识|30轮及评估完成|
|A2|固定A0，clean/JPEG95/75/50/缩放0.75/0.5六条件同ID比较|完成，不新增训练|
|A2-semantic|22类来源分类，训练及选模、校准排除FFHQ|30轮及评估完成|
|A3|来源到家族的映射聚合|未运行，reviewed=false|

主体100张20/30轮与200-q17930轮D0/A0、冻结OFA、D-DINO/A-RGB/A-FP/A-Fusion、来源分数融合及同ID规模比较此前已完成。主体封口summary SHA仍为fdc0fe496b09435d9a3d0e876d2623997ab70afc8c5bdb5e85e71f6d2a2445c9。有效63821=43821AI+20000真实；179新增冲突图已获批隔离，原64000文件和42000父成员保留，holdout未评分。

## 真假检测结果与局限

|evaluation|n（真实/AI）|AUROC|FPR@0.5|TPR@0.5|
|---|---:|---:|---:|---:|
|D1-control，clean|2310（979/1331）|99.6147%|3.1665%|96.2434%|
|D1增强，clean|同2310|99.5461%|5.2094%|98.1968%|
|D2，未见4o|1038（979/59）|99.6849%|4.8008%|98.3051%|
|D2-semantic，未见FFHQ|225（93/132）|88.6527%|60.2151%|98.4848%|

D1减同ID对照，原group配对bootstrap1000次：AUROC -0.07pp [-0.25,+0.11]；FPR +2.04pp [+0.72,+3.39]；TPR +1.95pp [+0.79,+3.16]。召回提升同时增加真实误报，不能称总体更好。没有增强检测器的JPEG/缩放匹配测试，不能宣称增强鲁棒性已验证。

FFHQ跨语义56/93真实被误判AI，Wilson95%约[50.05%,69.57%]，暴露明显泛化局限；不能用FFHQ evaluation调阈值/选模。4o仅59张正例，不能推广所有新来源。配对文件另列D2与同批in-domain对照ID的差异；训练域不同，不宣称纯因果或独立测试。

D3仅用calibration真实图校准：目标FPR1%在2310 evaluation实际FPR1.3279%、TPR91.4350%；5%目标对应4.6987%/98.4222%。目标不是评估保证；D3继承主D0按父完整dev选模的偏差。

## 来源分类、后处理与未知拒识

A2同1331张AI evaluation Top1：clean77.1600%、JPEG95 76.2585%、JPEG75 75.8828%、JPEG50 73.2532%、缩放0.75 76.7092%、缩放0.5 74.0045%。JPEG50减clean Top1 -3.91pp [-5.51,-2.38]；缩放0.5减clean -3.16pp [-4.55,-1.62]。Top3/MacroF1、逐来源差异与同ID原group1000次区间保存在配对结果，未混入calibration。

A2-semantic训练20986张AI，selection1190、calibration1384均不含FFHQ，选第26轮。132张未见FFHQ AI evaluation：Top1=80.30%、Top3=93.94%、MacroF1=77.85%。与包含FFHQ训练的A0在完全相同132ID上配对：Top1 -6.82pp [-11.77,-1.92]；Top3 -1.52pp [-4.88,+2.44]；MacroF1 -8.79pp [-15.27,-3.78]。这只验证一种语义，每来源3–11张；in-domain A0仍有完整dev选模偏差，不能推广全部跨语义能力。

A1仅64张kling未知calibration选阈值0.9986448884010316，cal误接收3/64=4.6875%。evaluation为1212已知＋63张grok3未知：未知AUROC=74.03%、拒识59/63=93.65%、误接收4/63=6.35%；已知保留576/1212=47.52%，错误拒绝52.4752%，正确且保留539/1212=44.4719%。不能只报未知拒识率掩盖已知覆盖损失。20类A1不等于主22类A0/Fusion，阈值不能直接移植；1-maxsoftmax不是未知概率，仅kling→grok3设定，不保证任意未知来源，不部署生产。

## 技术骨架与剩余研究目标

RGB实际用冻结DINOv2-S/14；指纹是冻结DnCNN残差→FFT2幅度→log1p→作者OFA ResNet50编码器。RGB-only、Fingerprint-only、特征Fusion和与ResNet的分数融合均已运行；真假二分类、22类来源softmax、OFA配对分数分别报告。

同原1968张AI dev的100/30→200-q179/30 A0 Top1为67.2764%→77.1341%；同606AI evaluation，特征Fusion69.1419%低于ResNet75.2475%，校准分数融合75.9076%，不能宣布复杂融合稳定最优。需针对跨语义误报、拒识覆盖损失和外部泛化另定后续协议。

ConvNeXt、OpenCLIP/UniversalAttribution、OmniAID、OmniDFA、DNA-Det没有本批模型成绩；A3映射未核验。没有自动扩500、换骨干或混入GenImage。文献数值、接口和合成测试不能冒充项目成绩。

## 验收与证据

14项status/exit、manifest/config/role/model/source SHA，6项连续1..30轮、失败样本0、重载/单图批量门槛均通过。所有角色ID/group互斥，只来自train/dev，无holdout；排除域严格不入训练、selection和calibration。effective_training_rows按split分别核对train和作为dev的selection；predictions按role/condition筛选，用保存scores重算最终指标；A1阈值只由kling calibration重算，未用evaluation调参。

9项配对比较复用仓库stats/group_bootstrap/detection_bootstrap/source_rows，不执行旧硬编码main；固定seed42、1000次原group抽样、同ID及固定词表。区间条件于已有模型/阈值，不覆盖重训或选模不确定性；单seed开发结果，不宣称独立检验、SOTA或纯数据因果。类别不平衡/权重、旧100分配偏差及新组修正、物理GPU和OFA参考ID差异仍适用。

本目录保存94份完成产物副本、汇总、配对、provenance和完成封口；模型、图像、数据manifest、角色ID全集、逐图预测、特征缓存均不进Git，原资产保留服务器。只新增独立结果目录，不覆盖旧报告，不改sync生产树、文本服务或线上模型。队列已完成，不再启动新实验；归档验证与推送后删除本定时任务。

在研究根目录可用.venv-image/bin/python运行本目录analysis.py，并用--output指定新的独占目录复核；需要原服务器证据，只读CPU分析，不能覆盖现有目录。
