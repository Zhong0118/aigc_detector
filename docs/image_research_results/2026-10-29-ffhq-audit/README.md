# FFHQ 误报审计（q179，已有预测）

结论：复现了 D2-semantic 的 56/93 误报；当前证据支持“未见过的 FFHQ 分布出现迁移失败”，但没有确立具体像素原因。文件格式、输入尺寸和已测量的编码头部不能解释 FFHQ 与 celebaHQ 的差异，也不能区分 FFHQ 内的误报与正确样本。后续应优先验证人脸取景/对齐、纹理与上游重采样差异，而不是直接更换阈值。

## 已有分数与配对结果

全部使用保存的 clean / evaluation 分数，固定阈值 0.5，没有重新拟合阈值或选择模型。

| 模型 | 相同 93 张真实 FFHQ 的误报 | FPR |
|---|---:|---:|
| D2-semantic | 56 | 60.22% |
| D1-control | 1 | 1.08% |
| D3，固定 0.5 | 1 | 1.08% |

D2 的 Wilson 95% 区间为 50.05%–69.57%，其 AI 分数中位数为 0.8541。55 张为 D2 独有误报，1 张三个模型共同误报，37 张都正确。D1-control 与 D3 在全部 979 张真实 evaluation 样本上的概率向量完全相同，不能把二者视为两次独立验证；两个 checkpoint 文件 SHA 不同，此处只陈述输出向量相同，没有声称其模型参数相同。

D2 的训练 33,577、selection 2,082、calibration 2,298 条样本均不含 FFHQ；evaluation 225 条全为 FFHQ，其中 93 张真实、132 张 AI。D1-control 训练中包含 1,399 张真实 FFHQ 和 2,362 张 AI FFHQ，因此配对差异伴随训练分布改变，不能把它直接归因于某一个图像特征。D3 沿用曾以完整 parent-dev 选择的 D0 checkpoint，仍有开发集选择偏差。

## 尺寸、裁剪与编码

实际读取并核对字节 SHA 的是 979 张真实 evaluation 样本，其中 FFHQ 93 张、celebaHQ 100 张。另对完整 manifest 的 train/dev 元数据做聚合；没有打开 19,294 张 holdout 图像中的任何一张。

FFHQ 与 celebaHQ 的这些实际头部属性全部一致：

- 256×256、PNG、RGB、8 位、PNG color type 2
- 仅 IHDR / IDAT / IEND chunk 类型，zlib 起始字节 789c
- 无 EXIF 标签、无 ICC profile，无 JPEG 量化表
- Resize 短边 256 在这些图像上保持原尺寸；224 中心裁剪保留面积 76.5625%，每边去除 16 像素

D2 训练与 clean 评估均走同一个 resnet_transform：Resize(256) → CenterCrop(224) → ImageNet normalization，图像先转 RGB。已对保存运行记录中的源文件 SHA 与当前实现逐个核对。这排除了“当前训练/评估预处理函数不同”这个直接解释；不能排除原始人脸在画面中的大小、位置或上游对齐过程不同。

FFHQ 的 56 个误报与 37 个正确样本在上述离散属性上也完全相同，因此这些属性无法对 FFHQ 内部失败做分层解释。PNG 的压缩字节不作为 ResNet 的输入；文件大小差异也不能被直接解释为压缩头部造成的像素变化。

## 文件大小关联

| 分组 | n | PNG 文件大小中位数 | bytes/pixel 中位数 |
|---|---:|---:|---:|
| FFHQ 全部 | 93 | 106,135 | 1.6195 |
| celebaHQ | 100 | 82,389 | 1.2572 |
| FFHQ 误报 | 56 | 105,196 | 1.6052 |
| FFHQ 正确 | 37 | 108,711 | 1.6588 |

固定描述分箱中，1.0–1.5 bytes/pixel 为 16/21 误报（76.19%，Wilson 54.91%–89.37%），1.5–2.0 为 40/72（55.56%，44.09%–66.46%）。区间重叠；分箱不是用于改动检测阈值的拟合。文件大小既受内容/纹理影响，也受编码选择影响，这仅是探索性关联。FFHQ 相对 celebaHQ 更大，而 FFHQ 内误报组又略小，单一文件大小解释并不充分。

D2 的已见域 calibration 中，真实 celebaHQ 为 0/95 误报；D1-control 的 celebaHQ evaluation 为 0/100。前者与 FFHQ evaluation 不是相同角色或相同图像，不应当作配对的模型消融证据。

## 下一步假设与边界

1. 检查非 holdout 人脸的取景、对齐、背景比例与高频纹理差异；当前头部信息不能回答这些问题。
2. 对相同开发图像做受控裁剪/重采样及解码后统一重编码，再配对比较分数，才能检查像素变化是否造成误报变化。此步骤需要另行推理，本次未执行。
3. 后续 OmniAID/OpenCLIP 实验可检验表征迁移是否改善，须沿用冻结角色与 calibration/evaluation 边界；本审计没有评估这些方法。

完成状态、每个 checkpoint 的实际文件 SHA256、manifest/roles/predictions/config/source 的 SHA256 均已验证并保存于 summary.json。三组输入运行均为 completed、exit_code 0。本审计只进行 CPU 文件与已有分数分析；没有训练、模型推理、展示图像或 holdout 图像读取。Pillow 为读取完整 PNG EXIF 可能解码非 holdout 图像，因此“检查头部”不等于完全不解码像素。

## 文件与复核

- 代码：scripts/audit_ffhq_errors.py
- 测试：tests/test_ffhq_audit.py
- 权威运行：outputs/image-next-routes-20261029/ffhq-audit/verified/
- summary.json：聚合分数、条件计数、角色、验证状态与 SHA 来源
- 运行目录的 image_features.jsonl：逐图数据，仅留服务器，不纳入此归档
- 本目录只归档 summary.json 与本文，无图像和逐图分数

首次试运行保留在 ffhq-audit 根目录；verified 子目录加入了 completed/exit-code 与实际 checkpoint 文件哈希检查，是本结论引用的版本。

验证：17 个审计测试通过，包含先失败再实现的 holdout 拒绝、角色/ID 错配、分数异常、文件 hash、未完成运行及 checkpoint hash 守卫。git diff --check 通过。

复核需指定新的独占目录，不覆盖已有结果：

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-image/bin/python scripts/audit_ffhq_errors.py --output NEW_DIRECTORY
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-image/bin/python -m pytest tests/test_ffhq_audit.py -q
```
