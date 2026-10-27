# 100张30轮阶段结果与200张扩充复测

## 已完成的同样本比较

- 100张数据固定为42,000张：22,000 AI、20,000真实；holdout未评分。
- D0与A0为独立ResNet50模型；四个新头为冻结骨干的独立实验。OFA保持作者指纹与配对头冻结，epoch不适用。
- 六个训练模型各30轮，seed42、batch32、Adam0.001、StepLR(1,0.9)。前20轮非计时记录与原20轮实验逐项一致。

|任务与比较范围|20轮|30轮|含义|
|---|---:|---:|---|
|D0 AUROC，4960 dev|99.4483%|99.4525%|变化极小|
|D0 FPR@0.5|3.2420%|3.9773%|误报增加|
|D0 TPR@0.5|95.5793%|96.8496%|检出增加|
|A0 Top1，1968 AI dev|66.2093%|67.2764%|+1.0671pp，分组95%区间[-0.4475,+2.6122]pp|
|D-DINO AUROC，1585 evaluation|97.9165%|97.9655%|冻结DINOv2-S/14，独立真假头|
|A-RGB Top1，606 evaluation|39.6040%|39.6040%|冻结DINOv2 RGB来源头|
|A-FP Top1，同606|43.5644%|44.0594%|冻结DnCNN/FFT/OFA128来源分类头|
|A-Fusion Top1，同606|63.0363%|63.5314%|两路各L2后拼接、再L2，512维来源头|
|冻结OFA来源验证宏AUROC，1968 AI dev|89.5602%|89.5602%|配对匹配分数，独立于分类概率|

A0最佳checkpoint为第24轮（按完整dev Macro-F1选模，仍有父dev选模偏差），D0第30轮。30轮对A0有小幅观察性改善，区间仍跨0；一个seed不能证明稳定收益。D0在0.5阈值下的误报与检出同时上升。

## ResNet与RGB/指纹来源分数融合

仅在757张AI calibration上按NLL选择alpha，606张evaluation不选权重。alpha=0.70表示70% ResNet来源概率加30% A-Fusion来源概率。

|30轮，固定606 evaluation|Top1|Top3|Macro-F1|
|---|---:|---:|---:|
|ResNet A0|65.1815%|87.4587%|64.7206%|
|A-Fusion|63.5314%|87.9538%|62.7371%|
|事前固定50/50分数对照|66.5017%|94.0594%|66.0497%|
|calibration选择70/30|66.0066%|93.0693%|65.5562%|

70/30相对ResNet Top1增加0.8251pp，原group配对bootstrap1000次95%区间[0,+1.8034]pp，包含0。不能因为50/50的evaluation较高而事后改权重。该闭集融合未验证未知拒识。真假softmax、来源softmax和OFA配对分数保持分离。

## 200张版本的执行协议

1. 另建`iab-balanced-v7-200`，使用旧v2-200固定清单，各AI来源×语义200张，22×10×200=44,000 AI；官方真实上限20,000，总64,000张。没有增加来源或重命名来源版本。
2. 已核验200清单逐格前缀包含100张版全部42,000张，逐文件SHA校验并硬链接复用；仅新增22,000 AI，26,514,532,504字节。正式原图总55,045,246,179字节。复用mtime不计网络吞吐。
3. 父manifest固定SHA `e8152a198db5b39481482cab18be9a56d173ba6a25dc330bc40cf1fe7cd4b06e`。所有旧ID、split、group、标签、图像hash保持；新重复桥接旧组或分区则停止并报告。
4. 仅对无父锚点的新组采用`group_and_split_extension_v2`的域分离seed/group哈希70/15/15期望比例；旧100版分配与旧holdout保持原样。
5. 下载完整、230收据及父锚点校验通过后，登记GPU空闲时顺序跑D0、A0各30轮与冻结OFA，再跑D-DINO、A-RGB、A-FP、A-Fusion各30轮。保留架构、预处理和超参；生成全新200特征缓存。OFA仍每来源100训练参考，并登记参考ID变化。
6. 仅在新calibration拟合来源分数融合，alpha0..1步长0.05、目标NLL、数值平分倾向ResNet；50/50独立报告。不能读取evaluation挑权重。
7. 对比100/30与200/30时，固定原4960真假dev、1968 AI dev；新头固定1585真假evaluation、606 AI evaluation。新200完整dev另列，报告逐epoch、逐来源和原group配对区间。

## 运行与版本边界

- 只有研究工作树、独立功能分支；不合入sync，不触碰文本服务/线上模型。
- 物理GPU由监控与研究配置共同登记，使用量超过1024MiB时等待；不停止任何服务或其他用户进程。2026-10-27用户明确授权空闲GPU0，因此200张版在启动前切至物理GPU0，CUDA_VISIBLE_DEVICES=0；GPU1上的corgi任务保留。原GPU1配置和监督进程身份已归档，训练架构、30轮预算、数据及划分不变。
- 4路直连正式落盘，TLS正常；不做下载后丢弃的验证、不绕访问拒绝、不切出口。
- 保留旧v1-v6所有清单/图片/结果和失败CPU诊断目录；新失败保留现场，不自动重训覆盖。
- D1/D2/A1/A2/A3仅模板，D3无训练轮数；本阶段不启动。500张与其他骨干不自动开始。
- 单seed、父dev曾用于选模、200新增组分配规则修正、D0类别权重随不平衡变化均列为限制，不作独立测试或纯数据量因果宣称。

## 产物位置

- 30轮完整报告与188项输入哈希封口：`outputs/image-followups-20261027/epochs20-30-comparison/`。
- 30轮融合：`outputs/image_experiments/resnet-feature-score-fusion-iab-v6-100-e30/`。
- 200采集/监督状态：`outputs/iab-v7-200-e30-monitor/`；清单SHA `8a1e4d67464d66c3afe566100e152dd2db59d15bb0afc145fbb69c358d398b67`。
- 200基线配置：`configs/image_experiments/iab_v7_200_e30_run.json`；代表模型配置：`configs/image_experiments/iab_research_v7_200_e30.json`。
- 200基线预定输出：`outputs/image_experiments/{d0-resnet50,a0-resnet50,ofa}-iab-v7-200-e30`。
- 200四组预定输出：`outputs/image_experiments/independent-image-tasks-v7-200-e30-gpu`。200训练结果尚未产生，不能将本协议当作实验成绩。

## 新增入口部署验收

人工已明确允许部署并继续。新增3个独立入口和测试已部署，85项服务器回归与38项子测试通过，shell语法及真实入口预检通过；两张已有训练图的完整性算法与实际sidecar一致，无模型推理。共享资产目录通过显式登记的权重路径读取，仍强制校验SHA，源码/配置/数据路径不能逃逸；旧核心源码哈希未变。全部配置、代码、资产与父清单绑定在新状态目录registration.json，预检修复的旧源文件/注册表保留在deployment-backup内。监督循环按download→独立extension-v2 baseline→四组特征队列推进，全部完成停在awaiting_analysis；最终融合和100/200配对分析仍需独立核验。只普通推送独立功能分支。
