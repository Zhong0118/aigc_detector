# 图片模型与数据核验（2026-10-19）

本文是代码/论文核验，不是论文数值的本项目复现报告。实测结果只读独立 run 的 metrics.json。

## OFA 完整指纹路线

原文：[AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40851/44812)。代码：[OFAAttribution](https://github.com/jumpycat/OFAAttribution/tree/3780cbffeeac48c0bf5e72a025f4803e408b9dfa)，MIT。

| 项 | 核验结果 |
|---|---|
| 预处理 | 作者 extract_sim_matrix.py：RGB；短边不足256才 Resize(256)；CenterCrop256；ToTensor；0.5均值/标准差归一化；img2noise反变换回[0,1]；DnCNN网络输出除256；FFT2D绝对值后log1p。不是对RGB直接做FFT，也不是余弦原型 |
| 去噪网络 | 17层3×3 valid卷积，前16层64通道，末层3通道；中间15层BN；除末层外ReLU；输入256输出222，不做补边 |
| 骨干 | ResNet50，2048→128输出层；FFTWrapper 的 state keys 为 base.* |
| 预训练 | 作者 Drive 文件250713063611-2k.pth，实际取得286,709,636 bytes；weights_only=True加载包含 step/model/pair_classifier/cls_classifier/optimizer。DnCNN model_best.th 随作者仓库提供 |
| 成对头 | 严格按作者代码顺序拼接参考指纹S在前、查询指纹D在后；Linear256→128→128→1；两层LeakyReLU(0.2)，推理 sigmoid。该头不对称，顺序不能交换。已确认权重包含全部6个张量，缺失时本实现直接失败，绝不使用随机头 |
| 损失 | 原论文：监督对比损失＋来源配对BCE＋真实/重建BCE；三项权重均1，温度0.02。模拟器为MSE＋0.25 LPIPS；模拟器重建图不等于目标商业生成器数据 |
| 原训练 | COCO train；裁剪256；随机重建网络产生指纹监督。论文memory bank 1024，公开训练脚本默认65536；脚本batch16、label_size8、Adam lr1e-4、100000 steps，参数需区分论文与脚本，不能说完全一致 |
| 验证协议 | 查询与每张同一来源参考图经学习的成对头打分再平均；原论文默认100参考；本轮训练侧最多100，样本不足实报，dev为独立查询，holdout不参与 |
| 推理差异记录 | 论文训练使用L2归一化表示，公开extract_sim_matrix推理未再做L2归一化。本轮先复用作者实际推理规则，明确记录，未暗中增加归一化 |
| 实现范围 | 复用已训练提取器与成对头，严格加载；未从零重训指纹模拟器/编码器，未开展正式开放集和新来源实验。来源验证分数不作为真假检测分数 |

公开脚本在pair_classifier缺失时只打印提示，仍可能继续使用随机头；本轮禁止该回退。脚本的参考集内部打分阈值与本轮开发协议不同，本轮只报告无阈值AUROC，不复制未验证阈值。作者推理脚本所用成对头有充分权重，完整推理所需资产齐全，不应误报“缺成对头”。

## ImageAttributionBench：选择 ResNet50 端到端归因

论文：[ImageAttributionBench](https://arxiv.org/html/2605.12967v1)，附录D；代码：[19bb7d805142d30e67689dae009ece170866c5f3](https://github.com/mttry/ImageAttributionBench/tree/19bb7d805142d30e67689dae009ece170866c5f3)，MIT。选择该基线因为它直接在论文中评测，架构及训练定义完整，可在小数据预算内验证，并可避免未知训练细节的复杂复现。

| 项 | 作者定义与本轮实现 |
|---|---|
| 预处理 | RGB、Resize短边256、CenterCrop224、ToTensor、ImageNet mean/std；不额外添加裁剪/压缩增强。作者数据基类对grok3有特殊裁剪，本轮Tiny不含该类，未复制来源条件裁剪 |
| 骨干/权重 | torchvision ResNet50，pretrained=True 对应 IMAGENET1K_V1，文件 resnet50-0676ba61.pth；显式固定 V1，避免 DEFAULT 变成 V2 |
| 分类头 | 原backbone.fc替换Identity，单独Linear2048→K；全骨干可训练，非冻结特征或逻辑回归 |
| 损失/优化 | CrossEntropyLoss；Adam lr0.001，默认betas(0.9,0.999)、weight_decay0；代码 StepLR step1 gamma0.9 |
| 训练协议 | 论文标准任务20epoch、batch32；代码yaml nEpochs30，README示例10，三者不一致。本轮采用论文20epoch并记录代码差异；开发集选最佳，平局取最早 |
| 划分差异 | 作者标准80/10/10；本轮按任务书固定训练侧70/15/15的分组开发协议。不是官方全数据/全类别论文复现 |
| 类别差异 | 作者公开代码23类含real；论文扩展32类含real。本轮来源任务只包含实际取得的7个AI来源，real不参与来源分类；单独真假模型为2类 |
| 任务独立 | A0选择dev Macro-F1，D0选择dev AUROC；D0借用同一骨干/训练协议是本项目独立检测基线，不能称为论文已经验证的真假方法 |

原有 scripts/train_image_attribution.py 的CLIP/统计特征入口保留兼容；本轮神经网络研究放在独立CLI中，沿用现有research/scripts/tests组织，不导入生产配置。若主数据受阻，只能称“小规模工程复现”，不能宣称复现作者完整Benchmark。

## 数据发布核验

- Tiny-GenImage：[数据卡](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)，revision `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`，CC-BY-NC-SA-4.0，研究用途；上游图片许可未全部追溯。
- 卡片train28,000，validation7,000；在线statistics只计22,000且partial=true，不作为全量结论。seed42随机分片10，实际2000行：real1000，ADM/BigGAN/GLIDE/SD15/VQDM/Wukong各143，Midjourney142，SD14为0。仅代表该分片；未取得最新商业模型样本。
- ImageAttributionBench作者下载器和dataset.py均为22生成器＋real，目录名为：4o、CogView3_PLUS、FLUX、KANDINSKY、PIXART、PLAYGROUND_2_5、SD1_5、SD2_1、SD3、SD3_5、SDXL、dalle3、gemini、grok3、hidream、hunyuan、ideogram、infinity、janus-pro、kling、mid-5.2、mid-6.0、real。
- 论文扩展标签含Seedream3.0/5.0-Lite、GPT-Image1/1.5、Gemini2.5/3等，但上述当前下载器没有这些独立目录；不能把gemini重命名成Nano Banana。
- [Dataverse发布页](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O4S4IV)、dataset API及latest files API在本轮匿名TLS验证请求均返回HTTP403。实际发布版本、受限文件列表、可下载类别及数据许可仍未核验；没有下载主数据。
- 下载器存在嵌入凭据和verify=False，本轮未执行、未复制凭据、未关闭TLS；使用标准匿名HTTPS查询。需要合法可用的公开入口或用户授权的访问方式后才能补全核验。

## 第三方代码许可

研究模块的OFA网络层定义由作者MIT代码适配；ImageAttributionBench的ResNet配置按作者MIT实现。第三方权重虽由作者公开提供，但独立权重授权范围及其上游训练数据仍需单独审查，不能把代码MIT推定为所有资产可商业交付。两份MIT许可原文保存在 configs/image_experiments/THIRD_PARTY_LICENSES.txt。
