# 图片真假检测路线核验

核验日期：2026-10-26。范围：OmniAID、UnivFD/OpenCLIP、DINOv2、ConvNeXt；仅方法、作者代码、模型卡、公开下载可达性；没有运行模型、训练或测试。本文件所有运行建议都是工程判断，所有论文设置都不是本项目实验结果。

## 执行推荐

结合主任务已核验的服务器现状（torch2.6.0+cu124、torchvision0.21、transformers4.57.6、safetensors0.7、huggingface-hub0.36，尚无open_clip/timm），**优先 DINOv2-S/14 + 独立 Linear(384,1) + BCEWithLogitsLoss**，命名为 `dinov2_small_frozen_linear`。权重仅88.25 MB，不新增模型框架依赖；它应标为本项目独立线性探测基线，不能称作已经复现某篇 AIGI 检测论文。

保持骨干 `eval()`、`requires_grad_(False)`，在 `no_grad()` 下提取并缓存特征后只训练小头；如果用 `inference_mode()`，转为常规张量后再送入需反向传播的头，避免 inference tensor 不能被autograd保存的问题。4090D 24 GB 做冻结特征提取合理可行，但具体batch/吞吐必须实测；建议从batch32开始。主任务已指定仅用物理GPU1，GPU0为Ollama保护卡。

下一条有直接检测论文对应关系的代表路线是 **OpenCLIP 加载 OpenAI ViT-L/14 + Linear(768,1)**，命名 `openclip_vitl14_openai_linear`，使用 `encode_image(x, normalize=False)`，对应UnivFD方法。ConvNeXt-T可作CNN对照。OmniAID作为后续官方预训练检测器评估，初轮不要把其两阶段MoE训练混进冻结骨干基线。

## 身份与任务边界

| 对象 | 精确身份 | 实际任务与本次覆盖 |
|---|---|---|
| OmniAID | Guo et al., *OmniAID: Decoupling Semantics and Artifacts for Universal AI-Generated Image Detection in the Wild*, arXiv:2511.08423；v1 2025-11-11，当前 v4 2026-07-20；arXiv 标注 ICML 2026 accepted | **Real/Fake Detection**；已读 v4 §3、§5.1、Appendix D 与当前源码/权重配置。不是生成器 attribution |
| UnivFD | Ojha, Li, Lee, *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023, arXiv:2302.10174 | **Detection**；核验论文身份、官方图像编码器/头/损失/预处理/训练参数代码 |
| DINOv2 | Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193v2 (2024-02-02；首次 2023) | 通用视觉特征，不是现成真假检测器；本次覆盖模型卡、公开权重和分类预处理 |
| ConvNeXt | Liu et al., *A ConvNet for the 2020s*, CVPR 2022, arXiv:2201.03545v2 | 通用图像骨干；本次覆盖 Tiny 权重、预处理、特征接口 |
| OpenCLIP | 开源 CLIP 实现；Cherti et al., *Reproducible scaling laws for contrastive language-image learning*, CVPR 2023, arXiv:2212.07143 | OpenCLIP 是实现/模型家族，不能自动等同 UnivFD，也不能把 OpenAI 权重说成 LAION 训练权重 |

一手身份来源：[OmniAID v4](https://arxiv.org/abs/2511.08423v4)、[UnivFD](https://arxiv.org/abs/2302.10174)、[DINOv2](https://arxiv.org/abs/2304.07193)、[ConvNeXt](https://arxiv.org/abs/2201.03545)、[OpenCLIP 论文](https://arxiv.org/abs/2212.07143)。名称相近的 OmniDFA 是另一篇包含 few-shot attribution 的工作，不应混入 OmniAID 的身份。

## 可直接落地的骨干与头

### OpenCLIP / UnivFD 对应路线

- 精确 checkpoint：OpenAI CLIP ViT-L/14，输入 224，输出投影后的 768 维图像特征；官方 UnivFD `CLIPModel` 为 `clip.encode_image` 后接 `Linear(768,1)`，**未做 L2 特征归一化**。[模型代码](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/models/clip_models.py)
- 训练代码冻结除 `fc.weight/fc.bias` 以外全部参数，损失 BCEWithLogitsLoss，`optim=adam` 实际调用 AdamW；默认 lr=1e-4、weight_decay=0、batch=256、最多100 epoch、early-stop patience=5。这些是代码默认值，不能据此声称与论文每项训练设置完全一致。[trainer](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/networks/trainer.py)、[训练参数](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/options/train_options.py)、[基础参数](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/options/base_options.py)
- **预处理重要区别**：官方 `validate.py` 是 RGB → **直接 CenterCrop(224)，无 Resize** → ToTensor → CLIP mean `[0.48145466,0.4578275,0.40821073]` / std `[0.26862954,0.26130258,0.27577711]`。训练 Dataset 默认短边 resize256、随机 crop224、水平翻转和可配 JPEG/blur。OpenCLIP 标准返回 transform 为 bicubic 短边224+中心裁剪，不能不加说明地当作 UnivFD 官方推理预处理。[官方评估](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/validate.py)、[训练数据](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/data/datasets.py)
- 若只训练本项目新头，可以选标准 OpenCLIP transform，但必须记录为 adaptation；若用作者现成头作外部基线，优先严格匹配官方 CenterCrop-only。OpenCLIP `encode_image` 默认不归一化，建议显式 `normalize=False`。使用 OpenAI 权重需匹配 QuickGELU；HF 配置已含 `quick_gelu=true`。[OpenCLIP 实现](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py)、[权重配置](https://huggingface.co/timm/vit_large_patch14_clip_224.openai/blob/main/open_clip_config.json)
- 推荐加载 `open_clip.create_model_and_transforms('hf-hub:timm/vit_large_patch14_clip_224.openai')`，取图像编码器；实际 API 需匹配服务器安装版本。当前 upstream README 提醒 main 训练接口重构，应锁定已验证的发行版，不从 main 追最新。[OpenCLIP README](https://github.com/mlfoundations/open_clip)
- 作者独立检测头只有 4,083 bytes，公开可达；可额外评估但与本项目训练头分开命名。[fc_weights.pth](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/pretrained_weights/fc_weights.pth)

### DINOv2 独立线性基线

- 选 `facebook/dinov2-small` / ViT-S/14，CLS 特征384维；以 `last_hidden_state[:,0,:]` 接 Linear(384,1)。模型卡支持 frozen 特征加线性层/逻辑回归；真假分类头、BCE 训练及数据划分属于本项目协议，原论文没有为此提供现成真假检测头。[作者模型卡](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
- 使用该 checkpoint 的 AutoImageProcessor：RGB，bicubic 短边256，CenterCrop224，1/255，ImageNet mean/std。模型配置中的 `image_size=518` **不等于推理必须输入518**，发布的 processor 默认224。[processor 配置](https://huggingface.co/facebook/dinov2-small/blob/main/preprocessor_config.json)、[模型配置](https://huggingface.co/facebook/dinov2-small/blob/main/config.json)
- 可用 `AutoModel.from_pretrained('facebook/dinov2-small')`；PyTorch hub 同样可用，但只为抽特征不需要复现大型预训练环境。代码和原始 DINOv2 模型为 Apache-2.0。[官方仓库](https://github.com/facebookresearch/dinov2)

### ConvNeXt 独立线性基线

- 一条固定路线是 `timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True, num_classes=0)`，输出768维 pooled 特征。接 Linear(768,1)+BCE。此权重为作者 ImageNet-22K 预训练再 ImageNet-1K 微调，不是检测权重。[发布方模型卡](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k)
- 通过 `resolve_model_data_config` 和 `create_transform` 绑定权重预处理；该模型配置训练输入224，test_input288、test_crop_pct1.0，bicubic，ImageNet mean/std，不手写一个对所有权重通用的 transform。[配置](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k/blob/main/config.json)
- 更少依赖的可替代方案是现有 torchvision `convnext_tiny(weights=IMAGENET1K_V1)`，保留 LayerNorm/Flatten、把 `classifier[2]` 换 Identity 得768维；其权重预处理是 bilinear短边236+CenterCrop224，与上述 timm 权重不同。不得混称相同实验。[Torchvision 文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)
- 作者代码 MIT；timm 发布权重模型卡声明 Apache-2.0，应分别记录来源和许可证。[作者许可](https://github.com/facebookresearch/ConvNeXt/blob/main/LICENSE)

## OmniAID：值得保留，但并非冻结骨干线性头

当前 v4 论文是 CLIP ViT-L/14@336 的 SVD 残差专家 MoE；Stage1 训练专家残差+重置头，损失 CE+orthogonality；Stage2 冻结专家、训练路由器+头，损失 CE分类+CE路由+balance。论文预处理512方形再336；GenImage配置 Stage1 lr2e-4、Stage2 lr2e-5、rank4、top1；Mirage论文版本 rank8、top2、两阶段 lr2e-4。作者报告使用4张H200，不能把此预算直接承诺为单卡4090D训练预算。[论文 v4 §3、§5.1、Appendix D](https://arxiv.org/html/2511.08423v4)

**当前公开 v2 checkpoint 是另一发布设置**：CLIP variant 6专家、每专家rank1、top2、router hidden256；当前 `reward/clean_test.py` 是单次 Resize((336,336))，标准CLIP归一化；DINO variant 单次448方形、ImageNet归一化。不要拿 v1/论文 rank8 或512→336 预处理配 v2 后声称精确复现。[当前推理脚本](https://github.com/yunncheng/OmniAID/blob/main/reward/clean_test.py)、[HF配套配置](https://huggingface.co/Yunncheng/OmniAID/blob/main/config/config_omniaid_v2.json)

CLIP推理实现有独立冻结 routing feature extractor、改造的视觉网络和 Linear(hidden_size,2)，最终 `softmax[:,1]` 为 fake score；它不是给普通CLIP特征接一个已训练小头。官方 loader 使用 `strict=False`，接入时应显式检查 missing/unexpected keys，不能无条件静默成功。[self-contained 模型](https://github.com/yunncheng/OmniAID/blob/main/reward/omniaid.py)

版本选择：要对齐论文GenImage协议用 `checkpoint_omniaid_genimage_sd14.pth` + `config_omniaid_genimage_paper.json`；要评估当前作者推荐发布用 `checkpoint_omniaid_v2.pth` + HF同名v2配置。注意 GitHub `config/` 列表目前仍是 `config_omniaid_mirage_paper.json` 等名字，而 README 提到的v2配置实际在HF可直接获得，不应拼一个不存在的GitHub路径。[模型卡/配对表](https://huggingface.co/Yunncheng/OmniAID)

许可/访问：OmniAID GitHub README声明MIT，HF模型卡也是MIT；本次根目录列表未见独立LICENSE文件，记录为作者声明。**OmniAID-DINO 使用DINOv3，不是DINOv2**；其基础模型 `facebook/dinov3-vitl16-pretrain-lvd1689m` API 返回 `gated=manual`、自定义 `dinov3-license`。现成OmniAID-DINO权重公开，不代表代码调用 `from_pretrained` 所需的DINOv3基础权重无需许可。优先CLIP版可避开该访问阻塞。[基础模型卡](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

## 下载核验与可追溯版本

本机在明确允许的只读网络请求下核验；每个大权重只GET前1024 bytes，得到HTTP206，不下载整体。以下结果不能替代服务器所在网络的下载测试。

| 模型/权重 | HF commit | 大小bytes | 本次结果 |
|---|---|---:|---|
| Yunncheng/OmniAID `ckpt/checkpoint_omniaid_v2.pth` | `279cae7398ac6636f46fc4668f755f11210b36bf` | 3241518989 | ungated metadata；Range GET206 |
| facebook/dinov2-small `model.safetensors` | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | 88249960 | ungated metadata；Range GET206 |
| timm/vit_large_patch14_clip_224.openai `open_clip_model.safetensors` | `18d0535469bb561bf468d76c1d73aa35156c922b` | 1710517724 | ungated metadata；Range GET206 |
| timm/convnext_tiny.fb_in22k_ft_in1k `model.safetensors` | `bc48a87f119bae1b06418f5340bfa97ef75d4609` | 114374272 | ungated metadata；Range GET206 |
| UnivFD `pretrained_weights/fc_weights.pth` | main未固定commit | 4083 | raw GitHub GET200，只读1024bytes |

机器可读证据：服务器研究目录 `outputs/image-research-expansion-20261026/evidence/detection-evidence-access.json`。上述模型metadata API是 `https://huggingface.co/api/models/<repo>?blobs=true`；记录中的文件URL可直接用于下载，运行时应固定revision并计算完整下载hash。

## 本项目协议建议与证据缺口

- 同一 manifest、同一 train/val/test 用于所有自训头；按来源/生成器留出，避免同源图片或变体跨split。测试集不用于选超参或阈值。真实/生成标签固定0/1。
- 主报AUROC、AP、balanced accuracy及真实/生成各类召回；阈值由验证集决定，同时保留固定0.5结果。UnivFD官方脚本会计算测试集最佳阈值；不要照抄为本项目部署阈值。
- 固定骨干/预处理/特征归一化方式/头结构/seed；缓存特征时连同权重revision与图像hash。若只做固定transform特征缓存，不再声称复现原训练的每epoch随机增强。
- 本次没有验证服务器软件栈、CUDA兼容、可用显存、吞吐、完整权重加载、训练收敛或项目检测效果；这些需主任务实际运行。没有建立Mirage-Train完整下载可得性的证据，因此不把OmniAID从头训练列作立即可完成任务。
- 论文读法为 targeted comparison；没有全文精读全部候选。来源标记：论文方法/参数为原文；代码行为来自作者/发布方实现；优先顺序和统一实验协议为工程建议。没有引用论文榜单数值作为本项目结果。
