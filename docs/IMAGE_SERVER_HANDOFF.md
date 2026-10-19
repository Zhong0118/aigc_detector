# 图片实验：服务器交接与研究路线（2026-10-19）

## 执行位置与清理

后续实验使用 `ssh school`，独立工作树为 `/home/zhongxu/projects/aigc_detector/.worktrees/image-m0-m2-paper`，分支 `codex/image-m0-m2-paper-20261019`。生产 sync 工作树与文本服务不修改。原本的 Mac MPS 训练是700张工程子集实验，不是服务器训练，也不是完整论文复现；Mac 不需要继续训练。

按用户明确授权删除服务器以下目录，目标内未发现代码扩展名文件，保留代码前后 SHA-256 一致：

- `/home/zhongxu/projects/lsq_captioning/data`：47,757,266,944 bytes
- `/home/zhongxu/projects/lsq_captioning/models`：7,520,985,088 bytes
- `/home/zhongxu/projects/generate/text/models/Qwen2.5-14B-Instruct`：29,551,734,784 bytes
- `/home/zhongxu/projects/ytt/Charades_v1_480`：16,609,460,224 bytes

合计约94.47 GiB。明细与代码哈希在 `outputs/migration-20261019/server-cleanup.json`。其余ytt实验资产保留。

原Mac实验资产共769文件、749,818,855 bytes，已全部迁移到服务器 `outputs/image_experiments` 并逐文件SHA-256验证通过。迁移清单和成功凭据分别为 `outputs/migration-20261019/mac-assets-sha256.json`、`migration-verified.json`。Mac独立虚拟环境、临时下载和托管图片工作树均已移除，约2.59GiB文件；原有本地项目未清理或覆盖。本机清理凭据为 `mac-cleanup.json`，代码可从GitHub功能分支或服务器继续工作，不需要本机重跑。

独立服务器环境 `.venv-image` 从现有 Python 3.10/CUDA 包复制，包文件不依赖生产环境的site-packages软链接。仅在副本删除无关decord并将fsspec固定为2026.2.0；pip check通过，torch2.6.0+cu124、torchvision0.21.0+cu124，GPU0 RTX4090D。48项图片实验、归因接口、来源登记与报告测试通过。扩展回归发现torch先加载系统libstdc++，与Conda SQLite依赖的ICU版本不兼容；已将libstdc++.so.6和libgcc_s.so.1复制到图片环境lib中。运行包含SQLite的接口或完整回归时，进程级设置 `LD_LIBRARY_PATH="$PWD/.venv-image/lib"`；48项回归及CUDA/SQLite联合导入通过，生产运行库未修改。新环境与Mac依赖版本不同，实际锁定清单存 `outputs/migration-20261019/server-environment.txt`，不宣称跨设备数值完全一致。

## 三项实验分别定义

| 实验 | 模型与论文依据 | 数据与输出 |
|---|---|---|
| D0 真假检测 | ImageNet V1预训练ResNet50＋独立2048→2头，全网络训练；本项目工程检测对照。GenImage作为检测数据/跨生成器评测参考，当前20epoch Adam协议不等于GenImage官方200epoch timm教程复现 | Tiny完整train中的real与AI；AUROC、Macro-F1、固定阈值FPR/TPR与开发阈值；不使用来源分数替代真假分数 |
| A0 来源分类 | ImageAttributionBench论文附录D的ResNet50端到端基线，ImageNet V1骨干＋2048→K线性头 | Tiny中7个实际AI来源；Macro-F1、Top1/Top3、逐类混淆矩阵；real不参与来源分类，因此不是原论文含real的32类协议完整复现 |
| V1 指纹验证 | OFA：DnCNN噪声→FFT幅度log1p→ResNet50的128维指纹→学习的成对头，参考在前查询在后；加载完整作者权重 | 与A0同一固定划分的AI图；每类仅训练侧最多100参考，dev查询；来源匹配AUROC，不是AI真假概率；当前未重训OFA模拟器 |

A0/D0：RGB短边256、中心裁剪224、ImageNet归一化；交叉熵、Adam lr0.001、StepLR(1,0.9)、batch32、20epoch，开发集选优。OFA采用作者256裁剪及原始归一化/噪声频谱流程，完整细节与论文/代码差异见 `IMAGE_METHOD_AUDIT.md`。CLIP不是必选模型。

产品流程可以“真假检测→疑似AI进入来源分析”，但各任务独立评测；来源模型当前为封闭候选集，未知来源拒识阈值尚未验证，不能直接交付为开放世界归因。也不能只在检测通过的样本上汇报来源模型本身的准确率。

## Tiny-GenImage实际标签审计

链接：https://huggingface.co/datasets/TheKernel01/Tiny-GenImage

固定revision：`89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`。2026-10-19通过官方HTTPS Range读取所有18个parquet的label/generator列，核对了35,000条标签的一致性；没有对官方validation图像做模型推理。明细：`configs/image_experiments/tiny_full_label_counts.v1.json`。

| 类别 | 官方train | 官方validation | 合计 | 全数据占比 | AI子集占比 |
|---|---:|---:|---:|---:|---:|
| Real | 14,000 | 3,500 | 17,500 | 50% | 不适用 |
| ADM | 2,000 | 500 | 2,500 | 7.1429% | 14.2857% |
| BigGAN | 2,000 | 500 | 2,500 | 7.1429% | 14.2857% |
| GLIDE | 2,000 | 500 | 2,500 | 7.1429% | 14.2857% |
| Midjourney | 2,000 | 500 | 2,500 | 7.1429% | 14.2857% |
| SD15 | 2,000 | 500 | 2,500 | 7.1429% | 14.2857% |
| VQDM | 2,000 | 500 | 2,500 | 7.1429% | 14.2857% |
| Wukong | 2,000 | 500 | 2,500 | 7.1429% | 14.2857% |
| SD14 | 0 | 0 | 0 | 0% | 0% |

这是标签列实计，不等于35,000张图片均已下载并检查可解码。图片下载仍须逐文件核对官方LFS SHA-256、解码、去重分组。没有更具体版本证据的Midjourney等类别保持原始标签，不猜测版本。

本轮服务器下载14个train分片约6.62GB（十进制），不是700张；官方validation图片不下载。对28,000张候选执行原有像素哈希和pHash距离≤4分组，再固定70/15/15的train/dev/holdout划分，seed42。分组后的精确数量以manifest/audit为准；holdout不调参。数据集CC-BY-NC-SA-4.0，上游来源许可仍有缺口。Tiny覆盖较旧生成器，无法支撑对最新商业模型的覆盖声明。

更大检测数据来源：[GenImage官方仓库](https://github.com/GenImage-Dataset/GenImage)、[官方下载目录](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS)。正式GenImage包含SD1.4/1.5、Midjourney、ADM、GLIDE、Wukong、VQDM、BigGAN八个来源，论文规模约270万图。本轮未取得全量，不把Tiny当成全量GenImage，也不推断全量实际占比。

## ImageAttributionBench实际发布版本与访问核验

[论文](https://arxiv.org/html/2605.12967v1)；[作者代码](https://github.com/mttry/ImageAttributionBench)；[公开数据发布页](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O4S4IV)。

本轮先用普通脚本访问Dataverse API返回403，站点HEAD有AWS WAF challenge；随后通过未登录的普通浏览器成功打开网页和读取同一个公开API（HTTP200）。因此应将此前“实际发布版本未核验/主数据访问受阻”的判断更新为：**元数据已取得，公开下载GET可用，服务器脚本直连仍受访问层限制；不能据先前403推断数据不存在或一定需要授权。**

已核验公开版本 **3.2，RELEASED**，API releaseTime字段为2025-05-16T08:00:13Z，许可 **CC BY 4.0**。完整清单350文件、528,670,084,577 bytes（528.67GB十进制，约492.36GiB），所有文件restricted=false。其中一个_test_imgs示例没有核验来源标签，排除训练；其余为22个AI来源及real目录的ZIP/分卷文件。原始清单保存在 `outputs/migration-20261019/iab-public-v3.2-metadata.json`；Git版本的摘要在 `configs/image_experiments/iab_public_v3_2_inventory.json`。

公开22个来源为：GPT-4o、CogView3-Plus、FLUX.1-schnell、Kandinsky-3、PixArt-XL-2-1024-MS、Playground2.5、SD1.5、SD2.1、SD3-medium、SD3.5-medium、SDXL-base1.0、DALL-E3、Gemini2.0-flash-exp-image-generation、Grok3-image、HiDream-I1-fast、Hunyuan-DiT、Ideogram-V1-TURBO、Infinity-2B、Janus-Pro、Kling-image-V1、Midjourney5.2、Midjourney6.0。目录名与论文名称的对照保留原始版本证据，不将gemini替换为新模型名称。

发布页描述为22生成器＋real、共约460,000张。按该版本均衡设计，每个来源及real各20,000张，占全数据约4.3478%；AI子集中22来源各约4.5455%。这是发布设计数量；尚未逐包清点全部图像，不能把压缩分卷文件数当作图片数或样本占比。

新论文另列9个来源：FLUX.2-Klein、GLM-Image、Z-Image、Doubao-Seedream3.0-t2i、Doubao-Seedream5.0-lite、Gemini2.5-flash-image、Gemini3-pro-image、GPT-image1、GPT-image1.5，共31个AI来源＋real，约640,000张。论文设计每类20,000张，分别占总量3.125%、AI子集约3.2258%。**这9个新增来源没有出现在本次V3.2清单中，不能声称已取得论文的31类版本。**

普通GET已验证公开示例759,874 bytes，SHA-256为5cf71d11a8032129d468fe361d580442d7c191d5ae5567f2b0ab9a7fd6962072；只用于传输验证，不参与训练/评测。HEAD访问下载重定向曾返回403，而GET成功，不能将HEAD失败作为文件下载权限的结论。未使用下载器嵌入凭据、关闭TLS或调用付费API。

服务器正在尝试取得real/dog包（file ID11192177，79,542,874 bytes），比对发布MD5并清点ZIP内图像条目；其状态以 `outputs/migration-20261019/iab-real-download.log` 和成功时的 `iab-real-dog-audit.json` 为准。仅验证传输与文件清单，没有开展IAB训练。

完整压缩包已大于当前剩余空间，解压还需要额外空间；不能直接启动全量。后续IAB作为现代生成器来源研究的主数据候选，先依据实际22类公开版本按语义与来源分包建立平衡子集，再比较OFA与端到端来源分类器。真假检测另设real/AI平衡协议，不沿用23类自然分布；未确认的新9类保持覆盖缺口。

## 服务器执行与状态凭据

后台工作脚本：`outputs/migration-20261019/run_server_m2.sh`。阶段日志：`server-m2.log`；进程ID：`server-m2.pid`；只有产生值为0的 `server-m2.exit-code` 且三个结果目录各有metrics.json才表示流程完成。状态不能仅根据任务启动推断。

顺序：可续传下载并验证14个分片→完整manifest→D0 20epoch→A0 20epoch→冻结OFA验证。三个新结果目录分别为 `d0-resnet50-server-full-v2`、`a0-resnet50-server-full-v2`、`ofa-server-full-v2`，位于 `outputs/image_experiments`。训练开始前检查GPU0占用；若其他任务已使用超过1GiB则退出并记录，而不抢占他人任务。

截至本交接文档创建，服务器已通过独立环境测试、实际重载A0并完成CUDA单图推理，OFA完整GPU链路和D0批量推理也通过，且已启动全量下载，尚无全量训练指标。完整数据采用新v2划分，仅在v2内部比较方法，不把v1小样本结果当成相同协议的对照。此前Mac结果只保留为历史工程实验，不能冒充服务器新结果，也不能冒充原论文结果。
