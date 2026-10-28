# A1未知来源拒识实验完成

核验时间：2026-10-28T23:59:01.898196+08:00。原队列A1 ResNet50训练30轮，按1179张selection Macro-F1选第27轮；train/evaluate均completed exit0，真实训练21236张AI、20个已知来源。kling与grok3均未进入训练或选模；原holdout未评分。

仅用64张kling未知calibration图按5%未知误接收目标选择最大softmax阈值0.9986448884010316，calibration实际误接收3/64=4.6875%。随后在1212张已知＋63张grok3未知的固定evaluation上报告：

|指标|结果|
|---|---:|
|未知检测AUROC|74.0302%|
|grok3未知拒识|59/63＝93.6508%|
|grok3未知误接收|4/63＝6.3492%|
|已知样本保留|576/1212＝47.5248%|
|已知被错误拒绝|636/1212＝52.4752%|
|已知闭集Top1（忽略拒识）|894/1212＝73.7624%|
|已知正确且被保留|539/1212＝44.4719%|
|被保留已知样本准确率|539/576＝93.5764%|

较高未知拒识以拒绝超过一半已知样本为代价，不能只报告93.65%便宣称可用。1-max-softmax是拒识分数，不是校准的未知概率；这里只验证kling→grok3两个发布来源之间的设定，不保证任意未知来源，也不能把20类A1阈值直接套给主22类A0/Fusion接口或部署生产。

已核验模型/资产、config/manifest/role SHA，连续1..30轮，failed_samples=0，重载与单图批量误差门槛。所有角色ID及group互斥，训练与selection只含20个已知来源；calibration未知仅kling、evaluation未知仅grok3，均为dev且不含holdout。effective_training_rows按train/dev分别核对为train和selection。predictions按role筛选，阈值只由64张kling重新计算，evaluation指标由已存scores重算一致，没有新增模型推理或训练。snapshot另存图片独立假设Wilson区间；未新增group bootstrap，单seed与语义/组相关限制保留。

原唯一队列已进入最后的A2-semantic FFHQ跨语义来源分类，截面完成5/30轮，仍未验收；A3映射未核验保持禁用。全部既定拓展完成后还需最终汇总核验，再删除定时任务。

本目录只保存已完成A1的指标、history、配置、混淆/逐来源统计、必要训练日志及来源SHA；模型、图像、数据manifest、角色ID全集、逐图预测和缓存不上传。以前的[主体快照](../2026-10-28/README.md)、[D1](../2026-10-28T0940Z/README.md)及[D2](../2026-10-28T145435Z/README.md)不覆盖。
