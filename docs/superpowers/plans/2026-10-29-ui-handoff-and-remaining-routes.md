# 页面交接与剩余图片研究路线

用户于2026-10-29改为由另一个模型实施页面，本线程交付需求文档并继续实验。沿用已授权研究工作树与功能分支，不改线上服务。原基线、拓展、OmniAID和OpenCLIP阶段已经结束；本计划不复跑。

成功标准：页面实现者能根据文档找到真实API、模型及验收证据；每个剩余算法明确实现状态、运行状态、原方法与项目适配差异。新增实验独立登记、真实运行、同角色比较、结果Git归档；缺失资产和预算不得冒充已完成。

## 约束
- 只通过ssh school修改研究工作树。GPU0空闲显存≤1024MiB、共享gpu0-research.lock、磁盘≥20GiB；不停止其他进程。
- 冻结200-q179的63821样本和manifest 9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3。全部holdout禁止推理，旧结果和179隔离原图保留。
- 新分类头seed42、batch32、30epochs、Adam(.001)、StepLR(1,.9)，selection选模、calibration拟阈值，evaluation只报告。原论文不同训练流程单列，不能统称30轮。
- 优先复用现有partition_rows、fit_head、比较入口和已安装timm，不修改旧哈希绑定的算法源码。
- 实验审计可并行，GPU任务串行。模型/数据/特征/每图预测不入Git，代码/配置/聚合结果与SHA普通推送原功能分支。

## 任务
- [x] T1 docs/IMAGE_UI_HANDOFF_2026-10-29.md：四页面需求、现有调用链、建议响应契约、默认模型、研究能力边界、预览隔离、设计与验收。只写文档，不实施UI。
- [x] T2 官方资产审计：OmniDFA、DNA-Det、OFA全预训练/T+1、A3家族映射；记录固定revision/size/SHA、许可和缺口，不能猜来源家族。
- [x] T3 ConvNeXt：新增scripts/run_image_convnext.py、tests/test_image_convnext.py、configs/image_experiments/iab_convnext_q179_20261029.json。先测试holdout/缓存身份/资产/重复启动与实际图像重载门槛，再实现冻结ConvNeXt-Tiny 768维L2特征；完整1000类官方权重strict load后移除分类器。使用官方288测试预处理，同角色D-clean、D-heldFFHQ、A22各30轮。它是本项目冻结特征基线，不是某篇伪造检测论文复现。
- [x] T4 注册ConvNeXt独立输出convnext-q179-20261029、源/配置/模型/角色SHA，审查后唯一启动；新comparison配置复用compare_image_next_routes.py，与OpenCLIP和ResNet相同ID比较，1000次group bootstrap。
- [x] T5 OmniDFA资产若满足条件，另写协议与独立入口；冻结作者模型与参考原型法不记30训练epoch，不将cosine当未知概率。DNA-Det、OFA/T+1仅在原资产、训练数据与预算可核验后进入独立注册，不用猜测性替代伪称复现。
- [x] T6 每项完成后新建docs/image_research_results时间戳快照；核验真实epochs、预测ID/类别/角色、重载、无holdout、聚合指标和source SHA。普通提交推送；为新阶段建立每30分钟安静跟进，全部完成或明确转为待用户处理时关闭或更新。

## 验证
代码使用CPU隔离测试，真实模型只在授权GPU门槛下用selection图做全重载/缓存/单批一致性≤1e-5；每个训练头真实history连续1..30。任何failed输出原样保留，禁止重置或覆盖。既有OpenCLIP测试和比较测试作为复用接口回归。UI文档与源码、实际模型工件逐项对照，不以测试替代实验成绩。

2026-10-29补充授权：用户明确优先官方模拟器＋20+1拒识头30轮；不执行完整模拟器/提取器预训练。OFA模拟器已从官方Drive导入138627153字节，实际SHA7ee94487217a8ff9fad3b0da1bdbb60b89f923be09b377b5f1ef17d9e8eb3de8，须安全CPU严格加载与固定架构重建验收后才生成仅train派生未知样本。原图角色和holdout边界不变。

2026-10-30：ConvNeXt实际三项均完成30轮；DNA/OmniDFA/OFA入口与10阶段串行队列已测试审查，通过预检后唯一启动。T4的运行部分已完成，比较与最终Git汇总仍待执行。每30分钟新任务仅跟进此队列，不恢复旧监督器。

2026-10-30最终：原queue十阶段含七只读比较全部exit0；OFA20+1真实30轮选18轮。最终逐ID/角色/标签/模型/日志SHA与聚合分数验收见docs/image_research_results/2026-10-30T0444Z-final/，结果Git推送核验后关闭本轮定时任务。A3完整映射与OFA全预训练未执行，不冒充本轮已完成。
