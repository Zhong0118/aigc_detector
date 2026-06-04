# AIGC 来源归因配置说明

本文档说明当前项目的候选来源归因方案。这里的“归因”输出是候选模型 Top-K，不是确定证明；确定性更强的证据仍然来自 C2PA、平台日志、可验证水印和已登记指纹库。

## 当前分支

### 文本

- `trained_classifier`：推荐主分支。使用 OpenTuringBench/MAGE/本地 JSONL 训练 `TF-IDF + LogisticRegression` 多分类器，训练产物为 `models/attribution/text_source_classifier.joblib`。
- `llmdet`：老牌文本来源概率工具，理论上支持 GPT-2、OPT、UniLM、LLaMA、BART、T5、Bloom、GPT-neo、Human-write 等候选来源。当前本机缓存缺少 `gpt2_large.npz` 和 `opt_3b.npz`，系统会显示 `data_mismatch`，不要用其他文件冒充。
- `openturingbench`：2025 OpenTuringBench 数据集/框架分支。当前以 prototype 方式接入，需要先从 Hugging Face 数据集抽样生成本地 `jsonl` 样本库。
- `embedding_prototype`：自建文本样本库分支。先用字符 TF-IDF 风格相似度做轻量 Top-K，后续可替换成 BGE/E5/SentenceTransformer embedding。

### 图片

- `trained_classifier`：推荐主分支。使用 Tiny-GenImage/本地图片目录训练 `CLIP 图像 embedding + LogisticRegression` 多分类器，训练产物为 `models/attribution/image_source_classifier.joblib`。旧的 `图像统计 + pHash/dHash` 只适合作为 baseline，不建议作为最终展示结果。
- `ofa_attribution`：预留给 2026 OFAAttribution / One for All 类开放集图片来源归因。需要你准备外部推理脚本并让它输出统一 JSON。
- `universal_attribution`：预留给 UniversalAttribution。官方流程依赖 GenImage 数据集和 split，更像实验脚本，不是单图即用 API。
- `clip_prototype`：自建图片样本库分支。当前默认使用 pHash 原型检索；如果后续准备好 CLIP/DINO 脚本，可以把 `command` 指向外部脚本。

### 视频

- `frame_image_fusion`：当前可落地方案，抽取视频帧后复用图片归因分支，再聚合 Top-K。
- `saga`：预留给 2026 SAGA 视频来源归因。该方向需要单独训练/适配权重，当前默认关闭。

## 需要下载什么

### 1. 推荐：训练 OpenTuringBench 文本来源分类器

用途：训练真正可被系统加载的文本来源多分类器。

需要联网下载：Hugging Face 数据集 `MLNTeam-Unical/OpenTuringBench`，不是模型权重。

缓存位置：`models/huggingface`。

训练产物：`models/attribution/text_source_classifier.joblib`。

命令：

```powershell
$env:HF_HOME="$PWD\models\huggingface"
$env:HF_HUB_CACHE="$PWD\models\huggingface\hub"
C:\Users\zx\.conda\envs\dl\python.exe -m pip install datasets
C:\Users\zx\.conda\envs\dl\python.exe scripts\train_text_attribution.py --dataset openturingbench --max-samples 20000 --max-per-label 1000
```

如果要补充自己的 DeepSeek、ChatGPT、Claude、Kimi、文心、Qwen 等样本，更推荐用 `--extra-jsonl`：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe scripts\train_text_attribution.py --dataset openturingbench --extra-jsonl data\attribution\text\custom_text_sources.jsonl --max-samples 30000 --max-per-label 3000
```

不建议直接把 MAGE 原始标签和 OpenTuringBench 混在一起训练，因为 MAGE 的标签包含任务类型、生成方式和模型名，类别会变得过碎。若确实要试 MAGE，请加 `--normalize-mage-labels` 做粗粒度标签规整。

训练完成后，`trained_classifier` 文本分支会自动读取：

```yaml
model_path: "models/attribution/text_source_classifier.joblib"
```

### 2. 可选：OpenTuringBench 文本 prototype

用途：给 `openturingbench` 分支生成本地候选模型样本库。

需要联网下载：Hugging Face 数据集 `MLNTeam-Unical/OpenTuringBench`，不是模型权重。

缓存位置：`models/huggingface`。

命令：

```powershell
$env:HF_HOME="$PWD\models\huggingface"
$env:HF_HUB_CACHE="$PWD\models\huggingface\hub"
C:\Users\zx\.conda\envs\dl\python.exe -m pip install datasets
C:\Users\zx\.conda\envs\dl\python.exe scripts\build_attribution_prototypes.py text-openturingbench --output data\attribution\text\openturingbench_prototypes.jsonl --max-per-model 50
```

生成后，`config.yaml` 里的路径已经指向：

```yaml
provenance:
  attribution:
    text:
      providers:
        openturingbench:
          prototypes_path: "data/attribution/text/openturingbench_prototypes.jsonl"
```

### 3. 自建文本 prototype 或自训练数据

用途：补充你关心的模型，例如 DeepSeek、Qwen、ChatGPT、Claude、Kimi、文心一言等。

不需要下载权重。你只需要准备每个候选模型生成的若干文本样本，写入：

`data/attribution/text/custom_text_prototypes.jsonl`

格式：

```jsonl
{"model":"deepseek","text":"这里放一段 DeepSeek 生成的样本文本"}
{"model":"qwen","text":"这里放一段 Qwen 生成的样本文本"}
{"model":"chatgpt","text":"这里放一段 ChatGPT 生成的样本文本"}
```

如果要用这些样本训练分类器，可以运行：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe scripts\train_text_attribution.py --dataset local-jsonl --input data\attribution\text\custom_text_prototypes.jsonl --text-column text --label-column model
```

### 4. 推荐：训练 Tiny-GenImage 图片来源分类器

用途：训练真正可被系统加载的图片来源多分类器。

需要联网下载：

- Hugging Face 数据集 `TheKernel01/Tiny-GenImage`
- CLIP 权重 `openai/clip-vit-base-patch32`

缓存位置：`models/huggingface`。

训练产物：`models/attribution/image_source_classifier.joblib`。

命令：

```powershell
$env:HF_HOME="$PWD\models\huggingface"
$env:HF_HUB_CACHE="$PWD\models\huggingface\hub"
C:\Users\zx\.conda\envs\dl\python.exe -m pip install datasets
C:\Users\zx\.conda\envs\dl\python.exe scripts\train_image_attribution.py --dataset tiny-genimage --feature-extractor clip --max-samples 10000 --max-per-label 1000
```

如果只是快速调试流程，可以用旧 baseline：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe scripts\train_image_attribution.py --dataset tiny-genimage --feature-extractor basic --max-samples 2000 --max-per-label 200
```

Tiny-GenImage 的 `generator` 标签会被自动还原为模型名：

```text
0 Real
1 ADM
2 BigGAN
3 GLIDE
4 Midjourney
5 SD14
6 SD15
7 VQDM
8 Wukong
```

训练完成后，`trained_classifier` 图片分支会自动读取：

```yaml
model_path: "models/attribution/image_source_classifier.joblib"
```

### 5. 图片 prototype

用途：给 `clip_prototype` 分支提供图片候选来源库。当前默认是 pHash 检索，不需要下载权重。

目录结构：

```text
data/attribution/image_samples/
  midjourney/
    001.png
  stable-diffusion-xl/
    001.jpg
  dalle/
    001.png
```

命令：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe scripts\build_attribution_prototypes.py image-folder --input data\attribution\image_samples --output data\attribution\image\image_prototypes.jsonl
```

如果要用这些本地图片训练来源分类器，可以运行：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe scripts\train_image_attribution.py --dataset image-folder --input data\attribution\image_samples
```

### 6. UniversalAttribution

用途：图片来源归因实验分支。

需要下载：项目代码、GenImage 数据集、官方 split。它不是单图推理模型，官方说明是跑 KNN/linear probe 实验。

当前项目接入方式：准备一个外部脚本，命令形态如下：

```powershell
python your_universal_attribution_infer.py image.png --json
```

脚本输出必须是：

```json
{
  "top_k": [
    {"model": "stable-diffusion-xl", "probability": 0.62},
    {"model": "midjourney", "probability": 0.24}
  ],
  "confidence": 0.62
}
```

然后把 `config.yaml` 中的 `universal_attribution.command` 改成该脚本路径。

### 7. OFAAttribution

用途：更接近 2026 开放集图片来源归因方向。

需要下载：项目代码、权重或训练数据，具体以作者仓库发布内容为准。

当前项目接入方式和 UniversalAttribution 一样：准备 JSON 输出脚本，并把 `ofa_attribution.command` 指过去。

### 8. SAGA 视频归因

用途：视频生成源归因。

需要下载：SAGA 复现代码、视频数据集、训练好的权重或自己训练的权重。当前没有配置成默认可跑，因为它不是 pip 包级别的一键模型。

短期建议：先用 `frame_image_fusion`，也就是抽帧后复用图片归因；后续再接 SAGA。

## 推荐测试顺序

1. 先训练文本来源分类器：`scripts/train_text_attribution.py --dataset openturingbench`。
2. 再训练图片来源分类器：`scripts/train_image_attribution.py --dataset tiny-genimage`。
3. 启动后端和前端，上传文本/图片，看 `Attribution Candidates` 是否出现 trained classifier 的 Top-K。
4. 再测视频，上传一个短视频，确认 `frame_image_fusion` 能抽帧并聚合图片分类结果。
5. 最后才考虑 UniversalAttribution、OFAAttribution、SAGA 这类研究模型，因为它们需要更多数据集、权重和环境隔离。
