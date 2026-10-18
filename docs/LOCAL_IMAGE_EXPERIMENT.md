# 本地图片实验：当前结果与下一步

## 当前功能

本地服务通过 `AIGC_CONFIG_PATH` 读取 `config.local.yaml`，默认服务器配置仍是 `config.yaml`。本地配置示例在 `configs/local_image.example.yaml`。

当前本地配置仅启用图片实验分支，关闭 Hive、Sightengine、文本模型、水印模型及大模型报告。图片特征、分类和模板报告全部在本机运行。

复用的服务器模型训练记录为 Tiny-GenImage、8000 个样本、8 个类别：Real、ADM、BigGAN、GLIDE、Midjourney、SD15、VQDM、Wukong。特征来自 `openai/clip-vit-base-patch32`，分类器为 LogisticRegression。这些记录不构成准确率证明。

配套 CLIP 固定到版本 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；权重 SHA-256 为 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`。

旧 sklearn 模型的数值参数已经导出为 `models/attribution/image_source_classifier.json`，并与服务器推理的核对向量比较。以后可使用 `scripts/export_image_classifier.py` 导出自己信任的同类模型。

## 两张真实照片的检查结果

样例来自 scikit-learn 随包提供的照片，位于 `data/demo_images`；该目录中的 README 保留原照片说明和署名。

| 样例 | 已知类别 | 原始实验分数 |
| --- | --- | --- |
| real_china.png | 真实照片 | 0.9660 |
| real_flower.png | 真实照片 | 0.9726 |

这个分数是分类器在生成类别上的总权重，不是已校准的“图片由 AI 生成的概率”。沿用 0.8 阈值，会把这两张照片都误判成 AI。

因此本地配置保留 `validated: false`：结果为“不确定”，原始分数和候选只用于研究，不输出真假或来源结论。未经验证的分支也不会混入已验证分支的最终分数。需要独立评估后才能调整这一状态。

初次检查结果保存在 `outputs/local_image_smoke_results_initial.json`；启用实验结果保护后的检查记录在 `outputs/local_image_smoke_results.json`。两张样例不能用来估计总体准确率。

## 自己操作

1. 打开 `http://127.0.0.1:8501`，确认页面显示“本地图片实验模式”。
2. 进入 File，上传 `data/demo_images/real_china.png`。
3. 点击 Analyze File。首次加载模型可能稍慢，后续请求复用已加载模型。
4. 查看实验分数、结论“不确定”和中文报告。
5. 对比第二张 `real_flower.png`，记录现象。

如果所有检测分支都失败，后台现在会返回 503，页面说明“检测未完成”，不会保存一个虚假的真人判定。

## 下一步做什么

先整理独立测试集，包含真实图片和多种生成来源。记录样本来源与标签，与训练样本去重，分别统计真实图片误报、生成图片漏报及各来源表现。根据这些结果排查特征和标签流程，再决定重新训练来源分类器，或接入专门的图片真假检测模型。
