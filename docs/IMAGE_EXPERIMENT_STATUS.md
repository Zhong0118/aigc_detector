# 图片 M0—M2 执行状态

> 后续状态更新（2026-10-19）：Mac实验已迁移并清理；服务器已完成48项回归和GPU链路验证。普通浏览器已核实IAB公开V3.2：22生成器＋real、350文件、528.67GB、CC BY 4.0，公开GET可用。本文原始实验记录中的403/版本未核验属于当时状态；当前数据版本、服务器任务及清理凭据以 [服务器交接记录](IMAGE_SERVER_HANDOFF.md) 为准。

日期：2026-10-19。以用户本轮消息为准：CLIP/逻辑回归不再是必选主模型，优先核验完整 OFA 和 ImageAttributionBench 端到端方法。仅做小规模开发实验，不运行正式测试，不部署。

## M0 对齐证据

- GitHub `sync`: `2b37c811f7787beba8e570cf4d5bac9b1ef27b30`，已通过 `git ls-remote` 核实。
- 原本地分支 `codex/attribution-pilot-baselines` HEAD `209935734f145179813452984fe07e43c650a3fb`，相对 sync 为 0 ahead / 9 behind；原目录有文本产物、AGENT.md、演示文件等未提交修改，未迁移或覆盖。
- school `/home/zhongxu/projects/aigc_detector` HEAD `b8adb8bc55a1a1e6896e4ba548a9982431cef354`，相对 sync 落后 1 个提交，存在 16 个已跟踪文件的未提交修改及其他未跟踪文件。服务器拥有较新的运行资产不等于代码提交最新。
- 服务器修改涉及 dashboard、config、文本阈值、provenance、reports、依赖及文本研究模块；归属未确认，未迁移。新分支基于已发布 sync，不声称包含服务器全部运行变化。
- 独立本地 worktree `/Users/zhongxu/.codex/worktrees/image-m0-m2-paper/aigc_detector`；分支 `codex/image-m0-m2-paper-20261019`。
- 原生产工作树、服务、模型、环境保持原状。未调用任何付费检测/报告/生成 API。

## 实际图片链路

`api/analysis.py` → `ingestion/loader.py` 图片 RGB 读取、metadata、fingerprint → `detection/providers.py` 配置驱动 API/demo/local → `provenance/pipeline.py` 的 C2PA、历史指纹、水印和来源分支 → `reports/generator.py` 模板/可选 LLM → 存储。

可复用：图片读取/指纹、来源 CLI JSON 边界、已有训练脚本的类标签解码与本地文件约定、独立研究目录和 pytest。`scripts/train_image_attribution.py` 已有统计或 CLIP 特征 + LogisticRegression，但随机划分、无固定 manifest、无独立真假训练，不能直接充当本次协议。

待完成/风险：`detection/image_detector.py` 是 weights=None scaffold；OFA provider 只是命令适配接口，并非论文实现；`clip_prototype` 实际使用 pHash，不是 CLIP 表示。非文本检测在无可用分数时仍回退 0/human，是现有工程风险，本轮离线实验不经过此链路，也不顺手修改共享生产模块。

服务器现有 `models/attribution/image_source_classifier.joblib` 34,313 bytes；无可追溯的新协议训练记录，保留只读，不当作新基线。服务器 data 1.9 GiB、models 6.8 GiB 主要为文本资产，未发现可直接复用的已核验图片数据集。

## 资源与环境

school 两张 RTX 4090 D；初查 GPU0 15 MiB、0%，GPU1 10,655 MiB；无进程被终止。磁盘 1.8 TiB、可用约247 GiB、86%已用，无其他数据盘，低于任务书20%保留线。已向用户请求本轮最多5 GiB且保留200 GiB的例外，在答复前不新增服务器实验资产。

生产 Python 为 3.10.20，torch 2.6.0+cu124、torchvision 0.21.0+cu124、numpy 2.2.6、sklearn 1.7.2、pyarrow 24.0.0，与 sync 约束部分不符，不修改生产环境。

本地磁盘约139 GiB可用、68%已用；独立 `.venv-image`，Python3.12、torch2.6、torchvision0.21、numpy1.26.4、sklearn1.5.2、pyarrow19.0.1；实际 MPS GPU 可用。Python3.12相对README推荐3.10/3.11属于记录在案的本地环境差异。

## M1 数据验收

实际取得Tiny-GenImage固定revision的官方训练分片10（464,008,196 bytes），从2000行中按实际generator标签分层抽700张：real350，ADM、BigGAN、GLIDE、Midjourney、SD15、VQDM、Wukong各50。SD14在这个分片中为0，不能推广为全量数据无该类。没有取得Kling、Seedream、GPT Image或Nano Banana的可核验样本。

固定manifest SHA256：`50fe1770600182182fd76c0ce595ddd615c0cd1335ce880b794a493f841d2106`。训练489、开发101、holdout110；来源任务仅用AI部分，训练245、开发49。分组检查采用原始文件SHA、解码RGB SHA及pHash距离≤4的传递闭包，本子集700组、未发现匹配的近重复对，解码失败0。没有提示词或语义标签，不能声称提示词隔离或已匹配真实/AI语义分布。

官方validation未下载；本项目holdout只做文件完整性检查，未进行模型推理、调参或选择。数据许可CC-BY-NC-SA-4.0，仅研究使用，上游图片来源/许可仍不完整。

ImageAttributionBench：论文扩展标签与公开代码23类不一致；匿名、启用TLS验证的发布页/API/文件列表均HTTP403，实际Dataverse发布版本、可取得类别和数据许可仍未核验。GitHub Release公开列出包括ResNet50在内的归因权重，未取得可替代主数据的数据包。没有运行包含凭据和关闭TLS行为的作者下载器。

## M2 实际结果（开发集，不是独立测试）

| 任务/有效run | 实际训练/参考与查询 | 实测指标 |
|---|---|---|
| D0 独立真假ResNet50 / d0-resnet50-v1 | 489训练，101开发（real52/AI49），20epoch，dev AUROC选第12epoch | AUROC 0.840659；accuracy 0.752475；macro-F1 0.746104 |
| A0 独立来源ResNet50 / a0-resnet50-v1 | 245训练，49开发，7来源，20epoch，dev Macro-F1选第10epoch | Top-1 0.591837；Macro-F1 0.586799；Top-3 0.877551 |
| V1 完整预训练OFA / ofa-v2 | 每来源35张训练参考，49张开发查询；冻结DnCNN/编码器/成对头 | 来源验证宏平均AUROC 0.881438；42个有向来源对平均AUROC 0.881438 |

三项指标任务不同，不能根据0.8814与0.5918直接排名。OFA与A0使用同样49张AI开发查询与固定标签；OFA是参考匹配验证，A0是闭集来源分类。CLIP和逻辑回归不是本轮主模型，未运行。

D0在0.5阈值下FPR=6/52=11.54%，TPR=30/49=61.22%；开发集探索性2%FPR工作点阈值0.7740306854，FPR=1/52=1.92%、TPR=21/49=42.86%。阈值与checkpoint均使用同一小开发集选择，存在选择偏差，不能据此宣称正式低误报能力。每类开发图只有7张，来源指标也不稳定。

错误分析：A0较多混淆包括GLIDE→BigGAN（3）和BigGAN→GLIDE（2）；OFA的VQDM一对其余AUROC仅0.7483，SD15/Wukong及VQDM相关来源仍有较大混淆。训练样本很少、无跨语义/跨版本/后处理/未知源独立实验，不能将结果外推至新商业模型。

## 可复现性、修复与验证

- 两个独立ResNet模型均实际训练、保存、重新加载，重载最大分数误差0；训练进程单图/批量最大误差0。
- 独立新进程CPU单图/三图推理也已执行：A0单/批最大误差1.862645149230957e-08，D0为0。对应输出和命令均保留。
- OFA资产实际包含model、pair_classifier、cls_classifier；按作者网络结构验证指纹与头的数值等价（合成测试输入误差0，非数据实验）。有效ofa-v2单/批指纹误差0。
- 首次ofa-v1参考/查询拼接顺序反转。独立审查发现后添加INVALIDATED.json，原日志/预测/指标未覆盖；修复为作者reference-first、query-second，原0.8625作废，新run为0.881438。该成对头不对称，已加入专门回归测试。
- 另修复无有限低FPR工作点时nextafter浮点舍入，以及损坏checkpoint应返回unavailable；失败测试与修复后的结果均保留。D0旧逐样本分数经修复后重新计算，所有已报指标不变。
- 相关pytest **48通过**，4条已有依赖弃用警告；覆盖分组不交叉、类别顺序、缓存身份、预处理身份、缺失/损坏模型、OFA输入顺序、低FPR边界及旧来源/报告provider回归。不是全仓库测试通过声明。
- OFA/A0/D0运行实现源码快照与提交身份已记录。首轮OFA在首次提交前执行，resolved_config如实记录dirty状态；修正版OFA在70f68dd干净提交运行。早期运行未自动记录源码哈希，另保存ed1c488的实际核心实现快照并说明范围，后续入口自动记录源码哈希。

## 交付位置和边界

代码提交：`ed1c488` 初始协议/实现；`70f68dd` OFA顺序及验证保护修复；最终指标/状态文档另见功能分支HEAD。普通push到 `codex/image-m0-m2-paper-20261019`，不合sync、不强推。

资产根目录：`/Users/zhongxu/.codex/worktrees/image-m0-m2-paper/aigc_detector/outputs/image_experiments/`。
数据：tiny-v1；预训练权重：assets；训练模型：a0-resnet50-v1/model.pth、d0-resnet50-v1/model.pth；有效OFA：ofa-v2；总清单：delivery_assets.json；精简结果：summary.json；实际命令/环境/日志见IMAGE_EXPERIMENT_RUNBOOK.md及各run。完整数据、权重、预测与缓存未进入普通Git。

school仅只读审计，未新增实验资产、创建服务器虚拟环境、训练或部署。旧图片模型SHA256为`76c7ceb2d3aa680cfe38e7b30765884783fadea2fe946cd46a97a0ccb2891631`，仍在原位置。磁盘例外尚待用户答复，不能把本地MPS结果说成school GPU结果。

## 下一阶段条件

已具备：固定开发协议、两个独立任务训练、完整预训练OFA来源验证、保存/重载/推理、逐样本结果和审计资产。当前是论文架构在自定义小数据上的工程复现，非官方全量基准复现，也未从零重训OFA指纹模拟器。

尚缺：ImageAttributionBench合法可用数据入口及发布版本、目标新模型可靠标签、充分语义/提示组信息、独立未知源协议、足量真实图和正式holdout评测、服务器磁盘预算。可以在解决数据/资源后继续OFA与端到端来源方法的受控比较；不具备直接上线或宣称新模型覆盖的条件。扩大微调/融合、正式测试、sync合并和生产切换仍需下一阶段指令。
