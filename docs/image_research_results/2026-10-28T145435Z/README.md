# D2跨来源与跨语义检测完成

核验时间：2026-10-28T22:54:35.409599+08:00。两项为原已登记队列的30轮ResNet50训练及显式evaluation，均completed/exit0。本次只验证已存结果和元数据，没有重训或模型推理，没有评分holdout。

|预先排除的域|实际evaluation|AUROC|FPR@0.5|TPR@0.5|
|---|---|---:|---:|---:|
|来源4o|59张4o AI＋979张真实，共1038张|99.6849%|4.8008%|98.3051%|
|语义FFHQ|132张AI＋93张真实，共225张|88.6527%|60.2151%|98.4848%|

FFHQ跨语义暴露明显真实图误报：93张真实中56张被判AI，FPR Wilson95%区间约[50.05%,69.57%]。当前检测器的跨语义泛化尚未解决，不能宣称高可靠；不能用FFHQ evaluation调阈值或选模型。4o实验仅59张正例，不能推广为所有新来源；两项目标域和分母不同，不比较总accuracy声称优劣。

D2训练36273张，4o从train/selection/calibration排除，selection AUROC选第24轮；D2-semantic训练33577张，FFHQ从上述三角色排除，选第27轮。逐ID核验角色与group互斥、有效训练ID一致、预测evaluation ID与标签一致、holdout未入角色。effective_training_rows文件同时包含train及作为dev的selection，已按split分别核对，不把selection当训练样本。真实图source_label在原manifest为null，以is_ai判断，未虚构真实来源标签。

四项train/evaluate资产清单与checkpoint SHA、config/manifest/role SHA、连续1..30轮、failed_samples=0、重载/单图批量误差均通过。AUROC/FPR/TPR由已存evaluation scores独立重算一致，统计按role筛出evaluation，没有混入calibration预测。没有新增配对bootstrap；Wilson按图片独立假设，组相关仍需最终分析说明。CSV保留原始CRLF字节，仅用本目录局部.gitattributes。

原唯一队列继续A1已知/未知来源实验，截面连续完成20/30轮，尚未完成验收；A2-semantic随后排队，A3映射仍未核验。不得因本轮低泛化结果擅自改协议、重训或新增模型。

本目录仅保存聚合指标、history、配置、混淆矩阵、逐来源统计、必要训练日志和来源SHA。模型、图片、数据manifest、角色ID全集、逐图预测与特征缓存不进Git。单seed开发结果，不宣称独立测试或SOTA。此前[主体比较](../2026-10-28/README.md)与[D1结果](../2026-10-28T0940Z/README.md)保持不覆盖。
