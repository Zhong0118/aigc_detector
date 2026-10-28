# 100张版 20/30轮同样本比较

状态：全部保存结果核验通过。全部数据固定为 iab-balanced-v6-100；未训练、未读图片、未做holdout推理。

Manifest SHA256 `e8152a198db5b39481482cab18be9a56d173ba6a25dc330bc40cf1fe7cd4b06e`；selection SHA256 `286a3c34f5291a355259cfed7170e9b5a6d9d17ddf483f6d32c535cecc4f64b6`。

ResNet完整dev为4960张（2992真实、1968 AI）；来源完整dev为1968 AI。独立角色 evaluation 为1585张（979真实、606 AI），calibration 为1771张（1014真实、757 AI）。两预算各角色ID、标签、词表、原group完全一致。

## 保存checkpoint的同样本指标

|模型/范围|20轮最优epoch|30轮最优epoch|20轮|30轮|变化 pp|
|---|---:|---:|---:|---:|---:|
|D0/full_dev auroc|20|30|99.4483%|99.4525%|+0.0042|
|A0/full_dev top1|19|24|66.2093%|67.2764%|+1.0671|
|D-DINO/evaluation auroc|20|30|97.9165%|97.9655%|+0.0490|
|A-RGB/evaluation top1|20|26|39.6040%|39.6040%|+0.0000|
|A-FP/evaluation top1|19|27|43.5644%|44.0594%|+0.4950|
|A-Fusion/evaluation top1|19|29|63.0363%|63.5314%|+0.4950|

OFA两轮条件均为冻结验证，epoch=null；1968 AI宏AUROC均为89.56019744%，其配对分数与来源验证AUC独立于分类概率报告。

## 检测阈值取舍

D0完整4960 dev固定0.5：FPR 3.24198%→3.97727%，TPR 95.57927%→96.84959%；AUROC增加仅0.004237 pp。增加30轮带来检测灵敏度上升，也增加误报，不能只按AUROC称为实质提升。

下表只在同1014张calibration真实图选择目标FPR阈值，再在固定1585张evaluation报告。该角色仍属于曾用于开发的parent dev，目标FPR并非独立测试保证。

|模型/预算|阈值方式|阈值|calibration FPR|evaluation FPR|evaluation TPR|
|---|---|---:|---:|---:|---:|
|ResNet50/e20|固定0.5|0.5|—|3.6772%|97.3597%|
|ResNet50/e20|目标FPR 1%|0.984269023|0.9862%|1.7365%|91.9142%|
|ResNet50/e20|目标FPR 5%|0.051867954|4.9310%|5.6180%|99.0099%|
|D-DINO/e20|固定0.5|0.5|—|5.3115%|91.0891%|
|D-DINO/e20|目标FPR 1%|0.762482464|0.9862%|1.1236%|80.8581%|
|D-DINO/e20|目标FPR 5%|0.483065933|4.9310%|5.4137%|91.2541%|
|ResNet50/e30|固定0.5|0.5|—|4.4944%|97.8548%|
|ResNet50/e30|目标FPR 1%|0.999365628|0.9862%|1.1236%|90.7591%|
|ResNet50/e30|目标FPR 5%|0.118926346|4.9310%|5.8223%|98.6799%|
|D-DINO/e30|固定0.5|0.5|—|5.2094%|91.0891%|
|D-DINO/e30|目标FPR 1%|0.770934045|0.9862%|1.1236%|81.0231%|
|D-DINO/e30|目标FPR 5%|0.484368205|4.9310%|5.4137%|91.2541%|

## 来源分数融合

只融合22类来源概率；alpha网格0至1步长0.05，由757张AI calibration的最小NLL选择，数值平分倾向更大ResNet权重。606张AI evaluation仅报告；50/50为事前固定对照。两预算均选择alpha=0.70。

|预算|方法|Top1|Top3|Macro-F1|
|---|---|---:|---:|---:|
|20|ResNet50|64.1914%|84.8185%|63.5670%|
|20|FeatureFusion|63.0363%|87.2937%|62.2126%|
|20|Uniform|66.0066%|93.0693%|65.4192%|
|20|Calibrated|65.1815%|91.9142%|64.5449%|
|30|ResNet50|65.1815%|87.4587%|64.7206%|
|30|FeatureFusion|63.5314%|87.9538%|62.7371%|
|30|Uniform|66.5017%|94.0594%|66.0497%|
|30|Calibrated|66.0066%|93.0693%|65.5562%|

30轮校准融合对ResNet的Top1变化+0.8251 pp；原151个group配对bootstrap 1000次95%区间[+0.0000, +1.8034] pp，区间含0。50/50单列，不用其evaluation分数反选alpha。

## 原group配对区间

差值均为30轮减20轮；固定词表、seed42、1000次、95%百分位区间。区间以已选checkpoint为条件，不包含重新训练或重新选择模型的不确定性。

|模型/范围|指标|变化 pp|95%区间 pp|原group数|
|---|---|---:|---:|---:|
|D0/full_dev|accuracy|+0.0605|[-0.3556, +0.5427]|3476|
|D0/full_dev|macro_f1|+0.0781|[-0.3572, +0.5886]|3476|
|D0/full_dev|auroc|+0.0042|[-0.0855, +0.1022]|3476|
|D0/full_dev|fpr_0_5|+0.7353|[+0.2988, +1.1997]|3476|
|D0/full_dev|tpr_0_5|+1.2703|[+0.5197, +2.2025]|3476|
|D0/evaluation|accuracy|-0.3155|[-0.8687, +0.2751]|1128|
|D0/evaluation|macro_f1|-0.3198|[-0.9339, +0.2884]|1128|
|D0/evaluation|auroc|-0.0585|[-0.2370, +0.0819]|1128|
|D0/evaluation|fpr_0_5|+0.8172|[+0.1014, +1.6344]|1128|
|D0/evaluation|tpr_0_5|+0.4950|[-0.2749, +1.4816]|1128|
|A0/full_dev|top1|+1.0671|[-0.4475, +2.6122]|489|
|A0/full_dev|top3|+1.0163|[-0.1004, +2.1484]|489|
|A0/full_dev|macro_f1|+1.0512|[-0.4911, +2.6370]|489|
|A0/evaluation|top1|+0.9901|[-1.8053, +3.6335]|151|
|A0/evaluation|top3|+2.6403|[+0.2313, +5.1443]|151|
|A0/evaluation|macro_f1|+1.1535|[-1.7868, +4.0387]|151|
|D-DINO/evaluation|accuracy|+0.0631|[+0.0000, +0.2264]|1128|
|D-DINO/evaluation|macro_f1|+0.0646|[+0.0000, +0.2372]|1128|
|D-DINO/evaluation|auroc|+0.0490|[+0.0277, +0.0774]|1128|
|D-DINO/evaluation|fpr_0_5|-0.1021|[-0.4024, +0.0000]|1128|
|D-DINO/evaluation|tpr_0_5|+0.0000|[+0.0000, +0.0000]|1128|
|A-RGB/evaluation|top1|+0.0000|[-0.4809, +0.4870]|151|
|A-RGB/evaluation|top3|+0.4950|[-0.1849, +1.2527]|151|
|A-RGB/evaluation|macro_f1|+0.0252|[-0.5086, +0.5973]|151|
|A-FP/evaluation|top1|+0.4950|[-0.5192, +1.6212]|151|
|A-FP/evaluation|top3|+0.1650|[-0.5481, +0.9702]|151|
|A-FP/evaluation|macro_f1|+0.9492|[-0.3020, +2.2823]|151|
|A-Fusion/evaluation|top1|+0.4950|[-0.1710, +1.2723]|151|
|A-Fusion/evaluation|top3|+0.6601|[+0.1538, +1.3772]|151|
|A-Fusion/evaluation|macro_f1|+0.5245|[-0.2315, +1.3302]|151|
|score-fusion/ResNet50/evaluation|top1|+0.9901|[-1.8053, +3.6335]|151|
|score-fusion/ResNet50/evaluation|top3|+2.6403|[+0.2313, +5.1443]|151|
|score-fusion/ResNet50/evaluation|macro_f1|+1.1535|[-1.7868, +4.0387]|151|
|score-fusion/FeatureFusion/evaluation|top1|+0.4950|[-0.1710, +1.2723]|151|
|score-fusion/FeatureFusion/evaluation|top3|+0.6601|[+0.1538, +1.3772]|151|
|score-fusion/FeatureFusion/evaluation|macro_f1|+0.5245|[-0.2315, +1.3302]|151|
|score-fusion/Uniform/evaluation|top1|+0.4950|[-1.8767, +3.0903]|151|
|score-fusion/Uniform/evaluation|top3|+0.9901|[-1.1051, +3.1757]|151|
|score-fusion/Uniform/evaluation|macro_f1|+0.6305|[-1.7545, +3.3735]|151|
|score-fusion/Calibrated/evaluation|top1|+0.8251|[-1.5492, +3.2496]|151|
|score-fusion/Calibrated/evaluation|top3|+1.1551|[-1.2372, +3.5187]|151|
|score-fusion/Calibrated/evaluation|macro_f1|+1.0113|[-1.4255, +3.7751]|151|

## 前20轮曲线

|模型|数值完全相同|最大绝对差|
|---|---|---:|
|D0|True|0|
|A0|True|0|
|D-DINO|True|0|
|A-RGB|True|0|
|A-FP|True|0|
|A-Fusion|True|0|

曲线逐epoch比较train/dev或train/selection及学习率等全部数值字段；仅排除时间。各字段差异和轮数见summary.json。相同随机种子不等于多seed；曲线一致性也不增加独立样本。

## 各来源分布与指标

source-distribution.csv列出train、selection、calibration、evaluation的每来源张数和原group数；per-source.csv列出各分类模型同来源Top1/Top3/F1及20→30差异。完整dev与evaluation具有不同分母，报告分别列出，不跨分母宣称提升。

|来源|train|selection|calibration|evaluation|
|---|---:|---:|---:|---:|
|4o|520|27|34|32|
|CogView3_PLUS|520|26|34|28|
|FLUX|517|22|34|32|
|KANDINSKY|529|29|32|32|
|PIXART|528|30|37|26|
|PLAYGROUND_2_5|528|32|34|24|
|SD1_5|517|25|31|30|
|SD2_1|520|25|35|29|
|SD3|521|24|36|29|
|SD3_5|519|31|32|23|
|SDXL|522|30|32|28|
|dalle3|514|22|43|27|
|gemini|517|27|36|27|
|grok3|528|26|35|26|
|hidream|532|26|34|28|
|hunyuan|514|29|35|24|
|ideogram|504|31|32|27|
|infinity|530|30|37|26|
|janus-pro|518|27|34|27|
|kling|521|33|34|27|
|mid-5.2|534|27|33|26|
|mid-6.0|503|26|33|28|
|real|13990|999|1014|979|

## 核验证据与限制

- 核验completion、validated-summary、summary、pipeline退出码，以及四组state/status、完整真实history、最优epoch、resolved config、模型/头文件SHA和image-inference-check。
- 保存预测按sample_id、标签、词表精确对齐后重算指标。20/30 RGB和指纹cache文件SHA完全相同；所有输入在分析结束再次校验未变。
- 30轮分数融合只在CPU重算缓存向量线性头，和保存evaluation的最大分数差为 1.01327896118e-06，低于1e-5；没有重新读取图片或调用GPU。
- 无-gpu的旧CPU 30轮诊断state仍为failed，完整保留且未用于正式比较；旧progress可能停在training_head，以state/status、history及已保存验收为准。
- 单seed、探索性dev结果；不声称独立测试、任意未知来源拒识或完整OFA重训。所有旧文件保持原样，未提交或push。
- provenance.json登记输入/输出SHA、configs和分析源码；completion.json以该provenance SHA封口。compare_saved.py可复核复用已有产物，不覆盖。
