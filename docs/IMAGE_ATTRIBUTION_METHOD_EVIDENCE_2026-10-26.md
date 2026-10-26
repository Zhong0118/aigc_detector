# 来源归因方法核验

核验日期：2026-10-26。范围：用户指定 OmniDFA、DNA-Det、UniversalAttribution、OFAAttribution/OFA。只核验论文、作者仓库、官方模型卡；未启动实验、未改服务器。已按 paper-reader 的 paper-context、collection-protocol、paper-compare 做定向比较，覆盖均为 `partial: identity / task / implementation / reproducibility`，不代表全文精读。

## 可执行结论

1. **保留现有 A0 ResNet50 22 来源闭集分类和原 OFA PairHead 参考匹配**，二者回答不同问题，不能因为后者暂时不高就删除其代表路线。
2. **首选增加 UniversalAttribution 风格冻结 OpenCLIP ViT-B/16 + 中间层 CLS + L2 正则逻辑回归**。这提供成熟基础模型特征路线，24GB 单卡提取特征、CPU 拟合线性分类器合理；性能与耗时须实测。若改用统一 20 epoch PyTorch 线性头，应明确称适配基线，不能称忠实 UA 原版。
3. **OmniDFA 优先用公开权重做冻结指纹 + prototype 参考匹配**。须检查 checkpoint 已见生成器与本项目 22 来源的重叠，并分别报告 seen/unseen。原始训练的全局大批量对比学习不适宜直接缩为单张 4090D + 22,000 AI 图后仍标成原版复现。
4. **DNA-Det 作为第二优先的历史架构指纹基线**。它是浅层 CNN，但双阶段预训练、16 patch 对比训练及类别均衡批量不是简单线性头；目标是架构级不变性，和细分同家族模型版本的 22 来源标签未必一致。
5. 原先准备的 RGB 2048 / OFA 128 / 分支 L2 归一化后拼接 2176 + 同一线性头仍有价值，但只是三组本地消融，不是三种文献方法，也不能代替 UA / OmniDFA 代表路线。

以上为基于实现与预算的工程判断，不是当前数据上的效果排名。22,000 是当前 AI 图总量；训练/验证/测试划分及“100/格”的各维度须沿用主任务的真实清单，不能误写为每来源只有 100 张。

## 任务语义矩阵

| 方法 | 原始任务 | 已知来源分类 | 未知来源拒识 | 指定未知来源的参考验证 |
|---|---|---|---|---|
| A0 ResNet50（项目已有，主任务提供） | 22 来源监督分类 | 是 | 需新增校准/协议 | 无原生 PairHead |
| UniversalAttribution | 已知来源与 real 分类，并拒绝未知来源 | 是 | 是，LP 用 MSP；论文 NN 用最近邻距离 | 不是论文主要任务；可另定义参考检索适配 |
| DNA-Det | 已知 GAN 架构分类；跨 seed/loss/finetune/data 泛化 | 是，架构标签 | 原方法未提供专门未知架构拒识训练 | 否 |
| OmniDFA | real/fake 检测 + 未见生成器 N-way K-shot 归因 | 支持参考集分类 | real 超球边界处理真假；**不等于拒绝支持集之外来源** | 支持集内 prototype 归因；没有 OFA 式学习 PairHead |
| OFA / OFAAttribution | image-to-model 配对验证，并报告闭集/开集协议 | 论文报告 | 论文报告 | 是，query 与目标来源 support 的学习匹配分数 |

未知来源“拒识”与“未见来源有少量支持样本后识别”是两种不同设定。只在全部 22 来源随机划 train/test，不能宣称完成 unknown-source 泛化评估。

## 1. UniversalAttribution

身份：Dario Cioni、Christos Tzelepis、Lorenzo Seidenari、Ioannis Patras，*Are CLIP features all you need for Universal Synthetic Image Origin Attribution?*，ECCV 2024 TWYN workshop，arXiv:2408.09153。官方仓库 [ciodar/UniversalAttribution](https://github.com/ciodar/UniversalAttribution)，核验 commit `60e710c1e2191298e667b12a47c16f81e4467374`。

[论文](https://arxiv.org/html/2408.09153v1) Sec. 3.2/4.1/Appendix A.6：冻结 ViT-B backbone，从 Transformer 中间 block 提取表示；CLIP 为 B/16、DINOv2 为 B/14，12 层、768 维、224 输入。主路线 OpenCLIP 在 LAION-2B 预训练。LP 为 L2 正则逻辑回归、LBFGS、按 block/正则强度选择；NN 路线按余弦距离多数投票，论文还考察 10 epoch SupCon 线性投影。主 LP 的未知拒识使用最大 softmax 概率 MSP；评价有 closed-set accuracy、unknown AUROC、OSCR。GenImage 原协议是已见 real + 4 个已见 generator / 4 个未见 generator 的五个 split，不能把文献成绩当成 22 类成绩。

作者代码核验：

- [configs/base.py](https://github.com/ciodar/UniversalAttribution/blob/60e710c1e2191298e667b12a47c16f81e4467374/configs/base.py) 写 `vit_base_patch16_clip_224.laion2b`，但 [linear.py](https://github.com/ciodar/UniversalAttribution/blob/60e710c1e2191298e667b12a47c16f81e4467374/linear.py) 的 argparse 实际默认 `vit_base_patch16_clip_224.openai`；**必须显式传 backbone**。
- `linear.py` 使用 `timm.create_model(..., pretrained=True, num_classes=0, global_pool='')`、`resolve_data_config` 和 `create_transform(..., is_training=False)`。应保留模型自带 resize/crop/normalize，不能复用 OFA 的 [-1,1] 归一化。
- [utils/feature_extraction.py](https://github.com/ciodar/UniversalAttribution/blob/60e710c1e2191298e667b12a47c16f81e4467374/utils/feature_extraction.py) 的实际执行路径调用 `forward_intermediates(..., return_prefix_tokens=True, indices=[index])` 获取 CLS token，并非 CLIP 投影后的 image embedding。
- [models/logreg.py](https://github.com/ciodar/UniversalAttribution/blob/60e710c1e2191298e667b12a47c16f81e4467374/models/logreg.py) 是 sklearn/cuML `LogisticRegression(penalty='l2', C=C, max_iter=1000, tol=1e-12)`。标准回归目标是对数损失加 L2 正则，不应把论文“l2 loss”的文字误实现成 MSE。
- 作者代码 C sweep 为 45 个对数间隔点、10^-6 到 10^5，按 validation AUC 选择；有独立 unknown-val 和 unknown-test。当前项目若暂不做开集，可按已知类 val 指标选层/C，并明确是闭集适配。
- 上游 `linear.py` 存在可见的多 block 循环风险：loop 内把 `model` backbone 变量覆盖成 LogRegModule，下一 block 的 feature extraction 会对错误对象调用 `forward_intermediates`；且 C 一旦设置后不再为下一 block 重选。应采用独立 `encoder`/`classifier` 或一次只指定一个 block，并按自己的验证流程重选 C。这里只做静态核验，未运行上游验证这个故障。
- 当前 [KNN 实现](https://github.com/ciodar/UniversalAttribution/blob/60e710c1e2191298e667b12a47c16f81e4467374/models/knn.py) 是 Faiss inner-product on L2-normalized features，官方 README 描述 k=1..20 validation sweep。仓库 tree 未见论文可训练 SupCon projection 的完整训练入口/专用权重，不能声称已复现 LP+/NN+。

权重与许可证：[timm 官方模型卡](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b) 提供 86.2M 参数的视觉编码器、Apache-2.0 标签，源自 `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`。UA 仓库 API license=null，tree 未见 LICENSE；不能把基础模型许可自动传递为仓库源码许可。作者没有在已读 README 中提供预训练来源分类头，重训轻量头本就是该方法路径。

## 2. OmniDFA

身份：Shiyu Wu、Shuyan Li、Jing Li、Jing Liu、Yequan Wang。arXiv:2509.25682 v1 的题名为 *OmniDFA: A Unified Framework for Open Set Synthesis Image Detection and Few-Shot Attribution*（2025）；当前作者仓库题名为 *Few-Shot Synthetic Image Attribution: Identifying Unseen Generators with Limited Samples*，标 ECCV 2026。两者由同一 arXiv ID、作者和仓库链接对应，不应误判为两种方法。官方 [teheperinko541/OmniDFA](https://github.com/teheperinko541/OmniDFA)，核验 commit `35b9052e83e05436682095818693493f79da9458`。

[论文 v1](https://arxiv.org/html/2509.25682v1) Sec. 3/4：双路径局部 crop 与全局 resize，128 维表示；SupCon 加 real 球心损失；未知生成器有支持图后做 prototype 分类。原训练 8×A100 40GB，每卡 128 fake + 16 real，全局 1152；AdamW，lr=2e-5，20 epoch，温度 .07、center 权重 .01。原论文为 5-way/15-way **5-shot**，当前 README 默认 **10-shot**，不可混用指标。

作者代码核验：

- [model/model.py](https://github.com/teheperinko541/OmniDFA/blob/35b9052e83e05436682095818693493f79da9458/model/model.py)：两个独立、可训练的 timm `convnext_small`（ImageNet pretrained、num_classes=0），concat 后 MLP hidden=512、output=128。代码使用 `backbone.num_features` 动态确定输入维，不能把论文的“512维”或代码注释的“1024”当作实际 ConvNeXt-Small 输出维；实例化时应检查。
- [数据代码](https://github.com/teheperinko541/OmniDFA/blob/35b9052e83e05436682095818693493f79da9458/datasets/omnifake_episode.py)：224 输入；局部分支 `RandomCrop(224)`；全局分支 short-edge resize 到224 后 `RandomCrop(224)`，**验证时仍是 RandomCrop**。实际 transforms 到 `ToTensor()`，未加 ImageNet mean/std。应保留并固定随机种子，不能擅自替换为中心 crop 后称原生复现。
- [loss.py](https://github.com/teheperinko541/OmniDFA/blob/35b9052e83e05436682095818693493f79da9458/model/loss.py)：L2 normalize、SupCon 温度 .07；real center learnable，阈值由角度 IQR trimming + .99 momentum 更新。
- [engine_val.py](https://github.com/teheperinko541/OmniDFA/blob/35b9052e83e05436682095818693493f79da9458/model/engine_val.py)：支持特征先 normalize，平均形成 prototype 后再 normalize，query 通过内积选类。此接口不会自动对支持集外新来源返回 unknown。
- [公开权重](https://huggingface.co/MoeNew/OmniDFA/tree/main)：part1/2/3 epoch20 各约1.2GB，zero_shot epoch40 约1.2GB。后三种 fold 训练30来源/评估15来源，zero-shot 训练33来源。官方 model card 标 Apache-2.0；GitHub tree 未见 LICENSE 且 API license=null，应把权重许可与源码许可分开记录。

可复现性：已有模型适合单卡冻结推理和小样本 support enrollment；未实测4090D峰值显存。完整原训练需要大批量监督对比，梯度累积不自动等价于跨样本 contrastive negatives。当前22,000张AI数据从头适配可以探索，但应命名“小数据改训”，不与作者百万级训练算力/成绩混写。必须审计来源/图像重叠，至少固定 checkpoint hash 和 trainlist。

## 3. DNA-Det

身份：Tianyun Yang、Ziyao Huang、Juan Cao、Lei Li、Xirong Li，*Deepfake Network Architecture Attribution*，AAAI 2022，arXiv:2202.13843。官方 [ICTMCG/DNA-Det](https://github.com/ICTMCG/DNA-Det)，核验 commit `d811395b92fb22d5f82f06ee98dbde77e53cea69`。**不是**2026 的 `wangchao0708/DNA`（Dual-stage Native Attribution），也不是 DNA-DETR 生物序列检测。

[论文](https://cdn.aaai.org/ojs/20391/20391-13-24404-1-2-20220628.pdf) Figure4/Experimental Setup：两阶段，先真实图片的图像变换分类预训练，再 GAN 架构标签分类，两阶段都有 patchwise supervised contrastive 与 CE，通过 automatic weighted loss 合并。浅层8-conv encoder，通道64/64/128/128/256/256/512/512；训练先128再512放大，从图上随机取16个64×64 patch；测试整图。论文明确目标是同架构不同seed/loss/finetune/data保持归类，未提出原生 unknown-architecture 拒识。

作者代码核验：

- [models/models.py](https://github.com/ICTMCG/DNA-Det/blob/d811395b92fb22d5f82f06ee98dbde77e53cea69/models/models.py)：`Simple_CNN`、GAP、Dropout(.2)+Linear(512,class_num) 分类头；SupConNet projection 512→512→128并L2 normalize。**骨干不是 ResNet50**。
- [main.py](https://github.com/ICTMCG/DNA-Det/blob/d811395b92fb22d5f82f06ee98dbde77e53cea69/main.py)、[trainer.py](https://github.com/ICTMCG/DNA-Det/blob/d811395b92fb22d5f82f06ee98dbde77e53cea69/trainer.py)：CE + SupCon(.07) + AutomaticWeightedLoss；Adam；CelebA配置lr1e-4、StepLR500/.9、max30epoch，其他数据配置不同。
- [data/dataset.py](https://github.com/ICTMCG/DNA-Det/blob/d811395b92fb22d5f82f06ee98dbde77e53cea69/data/dataset.py)：RGB、mean/std .5；训练16个随机patch。`balance=True` 的一项数据包含**每个类别一张**，因此配置batch_size32变成32×class_num图，再乘16 patches。套22类会显著放大激活量，不能把batch_size32当作32张图。
- [变换预训练](https://github.com/ICTMCG/DNA-Det/blob/d811395b92fb22d5f82f06ee98dbde77e53cea69/data/transforms.py) 包括单/双JPEG、Gaussian/Median blur、多插值缩放、高斯噪声等；旧版 albumentations API。论文先128再512，当前 CelebA config 只有resize512，依赖原数据128；新高分辨率数据须明确是否补128阶段。
- 官方 README 提供已训练权重和数据的 Google Drive/百度链接，声明2022-03-10已发布；本次只核验链接存在，未下载验证可达性、权重键或校验和。已读 tree 无 LICENSE，API license=null；不可默认为MIT。

工程判断：保留为经典低层指纹路线，但优先级低于UA；未经双阶段/patch protocol的“DNA-Det backbone+线性头”只能叫适配消融。它更适合architecture-level对照；若22来源主要是同架构版本/供应商别名，需先核对标签是否匹配研究问题。

## 4. OFA / OFAAttribution

身份已确认：Jianwei Fei、Yunshu Dai、Peipeng Yu、Zhihua Xia、Dasara Shullani、Daniele Baracchi、Alessandro Piva，*One for All: Synthesis-Free Fingerprint Learning for Attribution of In-the-Wild Synthetic Images*，AAAI 2026。官方 [jumpycat/OFAAttribution](https://github.com/jumpycat/OFAAttribution)，核验 commit `3780cbffeeac48c0bf5e72a025f4803e408b9dfa`，MIT LICENSE（仓库API也识别MIT）。

[论文](https://ojs.aaai.org/index.php/AAAI/article/download/40851/44812) Sec. Method/Experiments：真实COCO图经过随机fingerprint simulator重建以学习模拟来源指纹，不依赖真实生成器合成训练集；ResNet50 extractor，SupCon memory bank、same/different来源配对头及real/fake辅助训练。image-to-model协议是对一个目标来源support（默认100张），对query的所有support pair score取平均，逐一目标来源计算AUC；另报告closed/open-set分类协议。它既不是通用OFA多模态基础模型，也不是独立22来源分类头。

作者代码 [extract_sim_matrix.py](https://github.com/jumpycat/OFAAttribution/blob/3780cbffeeac48c0bf5e72a025f4803e408b9dfa/extract_sim_matrix.py) 核验如下，和主任务描述的已跑通原OFA链一致：

- RGB图若短边不足256才resize，再CenterCrop256、ToTensor、normalize(.5,.5)。`img2noise` 把 [-1,1] 转回 [0,1] 后在 no_grad 内调用 `denoiser.network(x)/256.0`，sigma=1。
- 残差上 `fft2 → abs → log1p`，无fftshift；ResNet50最终fc改128维。载入checkpoint['model']。
- 作者类名 `FeaturePairClassifier`（项目可简称PairHead），concat(f1,f2)，Linear256→128→128→1，中间LeakyReLU(.2)，sigmoid得到同来源概率；必须加载checkpoint['pair_classifier']，不能随机初始化替代。
- [训练脚本](https://github.com/jumpycat/OFAAttribution/blob/3780cbffeeac48c0bf5e72a025f4803e408b9dfa/train_noiser_fft_bank.py) 中有预训练ImageNet ResNet50、128维、温度.02、对比+pair BCE+real/fake BCE，默认Adam lr1e-4、100000 steps。论文说memory bank1024，当前代码默认65536；需要固定版本，不能凭脚本默认声称逐参数复现论文。
- 作者 [README](https://github.com/jumpycat/OFAAttribution/blob/3780cbffeeac48c0bf5e72a025f4803e408b9dfa/README.md) 有clean extractor Google Drive权重链接；denoiser来自 [grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection)。当前仓库已包含encoder2.py/decoder24_with_graph.py，但README仍说到DAIR取文件。完整重训还需要 `noiser_path=250708204953-4m.pth` 对应simulator checkpoint，未核验其公开下载；不应宣称训练链完全可复现。

主任务已确认现有OFA官方权重链运行成功，本次不重复跑通验证。另加128→22的线性分类头是 “OFA frozen-feature linear probe”，与作者PairHead参考匹配分别保留。RGB+OFA拼接也是项目新增的融合消融。

## 推荐统一报告字段

- 每条路线记录：作者原任务、当前适配任务、checkpoint URL/hash、repo commit、backbone是否冻结、实际输入变换、特征层/维度、head、训练数据、选择指标、support和query划分。
- 闭集：同一22来源train/val/test及去重/语义分组，Macro-F1、balanced accuracy、每来源召回、混淆矩阵。
- 未知拒识：按来源留出，threshold只用validation拟合；known-class accuracy、unknown AUROC/OSCR、固定FPR下召回。不可用test调拒识阈值。
- 参考归因：固定support预算、query与support图像/近重复/语义组隔离；记录N-way/K-shot或全22来源support分类，另测support之外来源的拒识（这是额外协议）。
- 论文数字只能放“文献参考”表；本项目未运行的新路线没有可报告的本地准确率、速度、显存或排名。

## 尚未建立的证据

- OmniDFA官方checkpoint与本项目22来源及图像级重叠：需用本地来源清单审计。
- 三个未声明代码许可的仓库，没有证据允许按任意开源许可重新发布；这里只陈述核验状态，不代替法务判断。
- DNA-Det及OFA外链权重本次未下载校验；OFA已跑通的结论来自主任务上下文。
- OmniDFA新版论文全文未替代v1做逐项diff；所列版本冲突均保留，不猜测新版修订内容。
- 当前24GB卡的峰值/吞吐尚未实测；建议是可行性判断，不是资源保证。
