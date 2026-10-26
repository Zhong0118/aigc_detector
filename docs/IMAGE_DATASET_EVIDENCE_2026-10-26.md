# 图片数据集版本与扩容证据核验

核验日期：2026-10-26。范围：只读官方论文、作者仓库及公开元数据；未修改当前下载任务、训练代码或划分，未付费生成，未下载完整图片数据集。

## 结论

**确有比当前 22 来源更大的作者公开数据，但目前核实的新增来源仅为 FLUX2_KLEIN 与 Z_IMAGE。** 作者仓库的上传脚本明确指定 Hugging Face 仓库 `multiitsuki/ImageAttributionBench`，故它不是凭名称认定的第三方镜像。公开文件树在固定 revision `a6298df1ee44868a0da1d9f6a4bebdf6d8915425` 下有 25 个模型目录（24 AI + real），每目录 10 个 Parquet，共 250 文件。[作者上传脚本](https://github.com/mttry/ImageAttributionBench/blob/clean-main/dataset_construction/t2i_generator/upload_dataset.py#L16) · [HF 固定版本](https://huggingface.co/datasets/multiitsuki/ImageAttributionBench/tree/a6298df1ee44868a0da1d9f6a4bebdf6d8915425/data)

**GPT Image 1/1.5、Seedream 3.0/5.0 Lite、Nano Banana 和 Nano Banana Pro，均已被论文明确纳入，且作者有生成代码；当前核实的公开 HF 文件树中未出现它们。** 不可据此断言作者从未发布过其他渠道，只能标为“论文/代码有据，当前已核实公开文件树未见”。本次 Dataverse 主站 API TLS 失败，因此没有把连接失败当作“数据缺失”或“数量不足”。[论文 §3.3 / 表6](https://arxiv.org/html/2605.12967v1#S3.SS3) · [作者新生成脚本](https://github.com/mttry/ImageAttributionBench/blob/clean-main/dataset_construction/t2i_generator/API/new_gen_image.py)

## 三个不同的发布口径

| 证据 | 当前实测/登记状态 | 能证明什么 |
|---|---|---|
| DataCite DOI 元数据 | version=3.2；Updated=2025-05-16；摘要22生成模型/460k图；360个文件大小条目合计528,672,555,787字节 | DOI登记仍为旧规模，不能据此代替 Dataverse 当前文件清单 |
| 作者 GitHub 默认分支 | clean-main；新生成脚本最近提交2026-04-16；下载器最近提交2026-04-12，仍列22 AI+real | 新模型生成代码已存在，但现成下载器未把新标签纳入 |
| 作者 HF 发布 | revision=a6298df...；lastModified=2026-04-24；gated=false；disabled=false；250文件；datasets-server显示559,101行、629,325,122,992字节 | 可公开获取的固定文件树确有24 AI来源，未达到论文31来源 |
| arXiv论文 | 2605.12967v1，2026-05-13，目前摘要页仅此版本；论文31模型、约640k；声明每模型每语义2,000图 | 这是论文规模与设计容量，不能替代逐文件实际计数 |
| GitHub Releases | 只有tag 1，2025-07-26，全部为归因器权重 | GitHub release不是新数据发布 |

来源：[DataCite API](https://api.datacite.org/dois/10.7910/DVN/O4S4IV)、[HF API](https://huggingface.co/api/datasets/multiitsuki/ImageAttributionBench)、[HF size API](https://datasets-server.huggingface.co/size?dataset=multiitsuki%2FImageAttributionBench)、[作者下载器](https://github.com/mttry/ImageAttributionBench/blob/clean-main/dataset/download.py)、[GitHub releases](https://github.com/mttry/ImageAttributionBench/releases)、[arXiv版本记录](https://arxiv.org/abs/2605.12967)。

## 目标模型版本对照

“可得”在本表指公开文件树有相应文件；实际成功解码、去重和本地下载完成另算。标签必须保留原值，论文/代码解释另存字段，不覆写原始来源标签。

| 目标模型 | 当前公开数据标签 | 有据的版本解释 | 状态/限制 |
|---|---|---|---|
| GPT-4o | `4o` | 作者枚举 `GPT4O=("4o","gpt-4o")`，请求实际使用代理名 `gpt-4o-all` | 原来源可得；不能映射成 GPT Image 1 或1.5，代理别名也不能证明精确后端快照 |
| GPT Image 1 | 未见 | 论文单独列 GPT-image-1；新脚本默认值为gpt-image-1 | 论文及代码有据，公开文件树未见 |
| GPT Image 1.5 | 未见 | 论文单独列GPT-image-1.5；新脚本明确支持 | 同上 |
| Seedream | 未见 | 论文/新脚本：doubao-seedream-3.0-t2i、doubao-seedream-5.0-lite | 未核实3.0、4.0、4.5、5.0等现成 IAB 公共文件；不能用其他标签替代 |
| Gemini 2.0图像实验模型 | `gemini` | 作者代码 `gemini-2.0-flash-exp-image-generation` | 原来源可得；不是 Nano Banana 的充分证据 |
| Nano Banana | 未见 | Google官方：gemini-2.5-flash-image；论文及新脚本有该名称 | 论文及代码有据，公开文件树未见 |
| Nano Banana Pro | 未见 | Google官方：gemini-3-pro-image；论文及新脚本有该名称 | 同上；更晚Nano Banana2/2.1也不能由旧gemini标签推定 |
| Kling | `kling` | 论文称Kling-image-V1；代码仅`model="kling-image"`和/v1接口路径 | 原来源可得；API路径v1不独立证明模型版本，保留“论文标称V1”，不推断2.x/3.x |
| Midjourney | `mid-5.2`、`mid-6.0` | 目录与论文明确对应V5.2、V6.0 | 两者可得；没有V7/V8证据 |
| FLUX | `FLUX` | 作者枚举指black-forest-labs/FLUX.1-schnell | 原来源可得；不是dev/pro/Flux2通用数据 |
| FLUX.2 | `FLUX2_KLEIN` | 作者枚举指black-forest-labs/FLUX.2-klein-4B | **已核实新增公开来源**，不能泛化为全部FLUX.2版本 |
| Z-Image | `Z_IMAGE` | 作者枚举指Tongyi-MAI/Z-Image-Turbo | **已核实新增公开来源** |
| GLM-Image | 未见 | 论文与生成代码存在 | 当前公开HF目录未见，不能由论文的31来源补成已下载来源 |

版本解释证据：[AIModel.py](https://github.com/mttry/ImageAttributionBench/blob/clean-main/dataset_construction/t2i_generator/API/AIModel.py)、[Gemini_model.py](https://github.com/mttry/ImageAttributionBench/blob/clean-main/dataset_construction/t2i_generator/API/Gemini_model.py#L22)、[kling_model.py](https://github.com/mttry/ImageAttributionBench/blob/clean-main/dataset_construction/t2i_generator/API/kling_model.py)、[SDModel枚举](https://github.com/mttry/ImageAttributionBench/blob/clean-main/dataset_construction/t2i_generator/diffuser_models/SDModel.py#L59)、[Google官方Nano Banana名称](https://ai.google.dev/gemini-api/docs/image-generation#model-selection)。

## 容量与最小扩容策略

**250/250格的Parquet页脚已核验成功，总行数559,101与HF size API完全一致；有效页脚读取量3,870,908字节。** 所有24个AI来源的240格均至少1,880行，因此当前22来源每格200/500及新增两来源每格200/500都具备原始发布行数容量。原有22来源共469,171 AI行，新增两来源40,000行，总AI509,171行；real共49,930行。real镜像超过原论文20,000，不能据此替换既定真实样本或划分。

实际计数来自**固定revision的Parquet footer里的num_rows**，详见 服务器 `outputs/image-research-expansion-20261026/evidence/dataset-evidence-cell-counts.json` 和本目录可读矩阵 `image_dataset_cells_20261026.csv`。这证明文件内发布行数，不证明去重、解码、质量过滤之后的有效数量。HF statistics API分类频数为partial=true、只覆盖3,994行，不能用它声称全库只有4o或两个语义。

| 来源 | 实际10格总行数 | 每格最小值 | 异常/差异 |
|---|---:|---:|---|
| 4o | 19,968 | 1,986 | 部分格略少于2,000 |
| dalle3 | 19,806 | 1,923 | 最小celebahq |
| gemini | 19,642 | 1,880 | 最小classroom，是全AI均衡采样瓶颈 |
| grok3 | 19,997 | 1,998 | 少量缺行 |
| hunyuan | 19,992 | 1,992 | ImageNet-1k略少 |
| ideogram | 19,999 | 1,999 | FFHQ少1行 |
| kling | 19,767 | 1,913 | 最小bedroom |
| SD1_5、SD2_1、SD3（各自） | 30,000 | 2,000 | COCO、ImageNet-1k各7,000，其余各2,000 |
| FLUX2_KLEIN（新增） | 20,000 | 2,000 | 10格全部2,000 |
| Z_IMAGE（新增） | 20,000 | 2,000 | 10格全部2,000 |
| 其余12个AI来源（各自） | 20,000 | 2,000 | 全部10格各2,000 |
| real | 49,930 | 4,738 | wild=4,738，dog=4,739，cat=5,153，classroom=5,300，其余5,000 |

原有22来源保持均衡时，200/格对应44,000 AI，500/格对应110,000 AI；加固定20,000 real后分别64,000/130,000总图。若新增两个来源，24×10×200/500分别48,000/120,000 AI，加既定real后68,000/140,000。以上是可用行数支撑的抽样目标，仍须通过本地校验。

最小可执行顺序：

1. 继续完成现有22×10×100=22,000 AI以及20,000 real的基线，保留现有清单和划分。
2. 需要扩大样本量时，用固定revision逐格实际行数检查200/500目标；在独立扩容清单里追加，不能把网络重试失败记成源容量不足。
3. 若目标是增加生成器覆盖，优先增加已公开的FLUX2_KLEIN、Z_IMAGE，每个仍按10语义均衡。100/格新增2,000 AI；两来源均500/格则新增10,000 AI。这是样本目标算式，不保证解码/去重后精确达到。
4. GPT Image 1/1.5、Seedream、Nano Banana 系列暂设明确缺口。后续找到带版本名的作者文件、作者清单或元数据再追加；仅有生成脚本不触发付费生成。
5. 所有新样本必须先按原图/提示词关系、hash和近重复规则核对与现有train/val/test的重叠；固定验证/测试集不能因扩容重新随机化。文件名p0_i0/p0_i1可能是同一提示词的两张图，要沿用现有分组规则。

## GenImage 的作用与边界

GenImage作者当前仓库仍为2023基准，公开下载入口是Google Drive和百度网盘，最近仓库推送为2024-04-02，GitHub无release。八个来源：Midjourney、SD1.4、SD1.5、ADM、GLIDE、Wukong、VQDM、BigGAN。论文明确Midjourney使用V5；不能把它当作V5.2/V6或更晚版本。GenImage不填补GPT Image、Seedream、Gemini、Kling、FLUX缺口。[作者仓库](https://github.com/GenImage-Dataset/GenImage) · [论文§2](https://arxiv.org/html/2306.08571v2#S2)

论文报告总计1,350,000 fake、1,331,167 real；按每个ImageNet标签，每普通生成器约162 train+6 test，SD1.5约166+8。这是**1,000个ImageNet标签粒度**，与IAB的10个语义格不同：不能直接拿GenImage的168/174来判定IAB的200/500不可行。GenImage公开压缩包内实际逐类成员数本次未下载核实。[论文§2.1](https://arxiv.org/html/2306.08571v2#S2.SS1)

## 许可

- IAB Dataverse DOI元数据登记 **CC-BY-4.0**；代码仓库 **MIT**。
- 作者HF数据卡只有 `license: apache-2.0`，README正文为空。与Dataverse CC-BY元数据不同；记录两端原始许可，不能用HF的Apache标签自动替代原始真实图像数据集各自许可。
- GenImage明确 **CC-BY-NC-SA-4.0并附非商业限制**。这与代码是否开源无关；不能把其数据当作无条件商业训练素材。[IAB DataCite](https://api.datacite.org/dois/10.7910/DVN/O4S4IV) · [HF数据卡](https://huggingface.co/datasets/multiitsuki/ImageAttributionBench) · [GenImage许可](https://github.com/GenImage-Dataset/GenImage/blob/main/License)

## 留存证据

服务器研究目录 `outputs/image-research-expansion-20261026/evidence/` 中所有 `dataset-evidence-*` 文件来自本次核验：DataCite完整JSON、HF info/tree/recursive/size/parquet/stats原始JSON、仓库README与许可快照、版本标签相关代码行、作者HF上传目标代码行、GitHub release摘要、页脚检查脚本与逐格计数。生成器源码只保存相关行，未保存任何凭据字段。
