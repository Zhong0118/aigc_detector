# Linux 服务器重新部署手册

这份文档用于把当前 AIGC 检测与溯源系统重新部署到 Linux 服务器。它按“先跑通，再增强”的顺序写：先启动 FastAPI + Streamlit，再准备本地模型、训练产物、C2PA、水印、来源归因数据。

## 0. 当前系统需要哪些基础东西

| 类型 | 名称 | 是否必须 | 来源 | 作用 | 默认落盘位置 |
|---|---:|---:|---|---|---|
| Python 依赖 | `requirements.txt` | 必须 | PyPI | FastAPI、Streamlit、Transformers、图像/音频处理、训练脚本基础依赖 | Python 环境 |
| 云检测 API | Hive | 推荐 | Hive 控制台 | 多模态 AIGC 检测分支之一 | 不下载模型，只需要 API Key |
| 云检测 API | Sightengine | 推荐 | Sightengine 控制台 | 图像/视频 AIGC 检测分支之一 | 不下载模型，只需要 API User/Secret |
| 本地文本检测模型 | `yuchuantian/AIGC_detector_*` | 推荐 | Hugging Face | 中文/英文、长/短文本 AIGC 检测 | `models/huggingface` |
| 报告生成 | 本地模板报告 | 必须 | 项目代码 | 不花钱，稳定输出中文解释报告 | 无 |
| 报告生成 | DeepSeek | 可选 | DeepSeek API | 把结构化证据改写成报告 | 不下载模型，只需要 API Key |
| 文本来源归因模型 | `text_source_classifier.joblib` | 推荐 | 本项目训练脚本生成 | 文本 Top-K 来源模型分类 | `models/attribution/text_source_classifier.joblib` |
| 图片来源归因模型 | `image_source_classifier.joblib` | 推荐 | 本项目训练脚本生成 | 图片 Top-K 来源模型分类；视频会抽帧复用它 | `models/attribution/image_source_classifier.joblib` |
| 图片来源归因研究框架 | UniversalAttribution | 可选 | `third_party/UniversalAttribution` | 复现实验或外接开放集图片来源归因命令 | `third_party/UniversalAttribution` |
| 文本训练数据 | OpenTuringBench | 推荐 | Hugging Face Dataset | 训练文本来源归因分类器 | `models/huggingface/datasets` |
| 图片训练数据 | Tiny-GenImage | 推荐 | Hugging Face Dataset | 训练图片来源归因分类器 | `models/huggingface/datasets` |
| 图片特征模型 | `openai/clip-vit-base-patch32` | 推荐 | Hugging Face Model | 图片来源归因的 CLIP embedding 提取器 | `models/huggingface/hub` |
| C2PA 工具 | `c2patool` Linux 可执行文件 | 推荐 | contentauth/c2patool Release | 读取内容凭证、签名、生成信息 | `tools/c2patool/c2patool` |
| 音频水印 | AudioSeal | 可选 | PyPI/Hugging Face | 检测音频水印 | `models/audioseal` |
| 图片/视频水印 | VideoSeal | 可选 | Meta/Facebook Research 项目 | 检测图片/视频水印 | `third_party/videoseal` 及其权重目录 |
| 自建文本样本 | JSONL | 可选但很有用 | 自己生成/采集 | 扩充 DeepSeek、ChatGPT、Claude、Kimi、文心等候选来源 | `data/attribution/text/*.jsonl` |
| 自建图片样本 | 分目录图片 | 可选但很有用 | 自己生成/采集 | 扩充 Midjourney、SD、DALL-E、真实图等候选来源 | `data/attribution/image_samples/<label>/` |

不要把 `data/*.db`、`data/report_cache`、`models/huggingface`、训练数据缓存提交到 Git。服务器上可以重新下载或从本机拷贝。

## 1. Linux 适配性检查结论

核心 Python 代码基本可以在 Linux 跑：

- `api/`、`ingestion/`、`detection/`、`provenance/`、`reports/` 大多使用 `pathlib.Path` 和相对路径，跨平台没问题。
- `provenance/c2pa_reader.py` 通过 `subprocess.run([tool, file])` 调用工具，Linux 只要换成 Linux 版 `c2patool` 即可。
- `provenance/metaseal_adapter.py` 对 AudioSeal/VideoSeal 是懒加载，没装依赖时不会让整个系统崩，只会返回 `dependency_missing`。
- `scripts/train_text_attribution.py` 和 `scripts/train_image_attribution.py` 是普通 Python 脚本，Linux 可用。

需要注意的 Linux 差异：

- `scripts/run_api.ps1` 和 `scripts/run_dashboard.ps1` 是 Windows PowerShell 脚本，Linux 不用它们。
- `config.yaml` 里当前 `provenance.c2pa.tool_path` 是 `tools/c2patool/c2patool.exe`，Linux 要改成 `tools/c2patool/c2patool`。
- `config.yaml` 里当前把 API Key 明文写在了 `api_key_env`、`api_user_env`、`api_secret_env`、`report.api_key_env` 字段里。Linux 部署建议改成环境变量名，而不是密钥值。
- 视频、音频依赖系统库，Linux 服务器需要安装 `ffmpeg`、`libsndfile1`、`libgl1`、`libglib2.0-0`。
- 不建议用 `uvicorn --reload`。模型下载、数据集下载、缓存写入会触发 reload，导致后端反复重启。

## 2. 准备服务器系统依赖

以下命令以 Ubuntu 22.04/24.04 为例：

```bash
sudo apt update
sudo apt install -y git curl wget build-essential ffmpeg libsndfile1 libgl1 libglib2.0-0
```

这些包的作用：

- `git/curl/wget`：拉代码、下载 c2patool 或模型相关文件。
- `build-essential`：某些 Python 包需要本地编译时使用。
- `ffmpeg`：视频、音频读取和抽帧相关依赖。
- `libsndfile1`：`soundfile/librosa` 读取音频时常用。
- `libgl1/libglib2.0-0`：无桌面 Linux 上导入 `opencv-python/cv2` 常见依赖。

## 3. 创建 Python 环境

推荐 Python 3.10 或 3.11。你现在本地是 Python 3.10，所以服务器也优先用 3.10。

```bash
cd /path/to/aigc_detector
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install joblib
```

命令作用：

- `python3.10 -m venv .venv`：在项目内创建独立 Python 环境。
- `source .venv/bin/activate`：进入这个环境。
- `pip install -r requirements.txt`：安装项目运行必需依赖。
- `datasets`、`PyArrow` 和 `tqdm`：多领域文本语料下载、磁盘缓存和进度显示所需，均已写入 `requirements.txt`，不需要另行安装。
- `joblib`：保存/加载来源归因分类器，通常会随 scikit-learn 安装，但显式安装更稳。

如果服务器有 NVIDIA GPU，请按服务器 CUDA 版本到 PyTorch 官网选择对应安装命令。不要盲目复制 Windows 或其他 CUDA 版本的命令。CPU 也能跑，只是 CLIP 图片训练和本地模型推理会慢。

## 4. 固定模型和数据缓存目录

每次开终端都建议先设置这些环境变量：

```bash
export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$PWD/models/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/models/huggingface/datasets"
export AUDIOSEAL_CACHE_DIR="$PWD/models"
export AIGC_API_BASE="http://127.0.0.1:8000/api/v1"
```

作用：

- `HF_HOME`：Hugging Face 总缓存根目录。
- `HF_HUB_CACHE`：模型权重缓存目录，例如 YuchuanTian 模型、CLIP。
- `HF_DATASETS_CACHE`：数据集缓存目录，例如 OpenTuringBench、Tiny-GenImage。
- `AUDIOSEAL_CACHE_DIR`：AudioSeal 自己的缓存根目录；代码会使用 `models/audioseal` 配置，实际 AudioSeal 还可能创建子目录。
- `AIGC_API_BASE`：Streamlit 前端请求 FastAPI 后端的地址。

如果你用 `tmux`、`screen`、`systemd` 启动服务，也要把这些环境变量写进去。

## 5. 修改 Linux 版配置

部署前请编辑 `config.yaml`，至少调整这些字段。

### 5.1 API Key 不要写明文

建议把配置改成环境变量名：

```yaml
detection:
  api_providers:
    hive:
      enabled: true
      api_version: "v3_vlm"
      api_key_env: "HIVE_API_KEY"
    sightengine:
      enabled: true
      api_user_env: "SIGHTENGINE_API_USER"
      api_secret_env: "SIGHTENGINE_API_SECRET"

report:
  provider: "template"
  llm_provider: "deepseek"
  api_key_env: "DEEPSEEK_API_KEY"
  base_url: "https://api.deepseek.com"
  model: "deepseek-v4-flash"
```

然后在 Linux 终端设置真实密钥：

```bash
export HIVE_API_KEY="你的 Hive key"
export SIGHTENGINE_API_USER="你的 Sightengine user"
export SIGHTENGINE_API_SECRET="你的 Sightengine secret"
export DEEPSEEK_API_KEY="你的 DeepSeek key"
```

说明：

- `*_env` 字段应该写环境变量名，不应该写真实密钥。
- 目前 DeepSeek 曾出现 `402 Payment Required`，所以推荐默认 `report.provider: "template"`，先不花钱。
- 如果确实要用 DeepSeek，再把 `report.provider` 改成 `"llm"`。

### 5.2 C2PA 工具路径

Linux 不使用 `.exe`：

```yaml
provenance:
  c2pa:
    enabled: true
    mode: "local"
    tool_path: "tools/c2patool/c2patool"
```

### 5.3 LLMDet 暂时保持关闭

当前配置里 `llmdet.enabled: false` 是合理的。它需要一批 n-gram 概率数据，不是普通模型权重，体积大且版本脆弱。当前主路线应使用 `trained_classifier`。

## 6. 准备 C2PA 工具

从 `contentauth/c2patool` 的 Linux release 下载可执行文件，放到：

```text
tools/c2patool/c2patool
```

然后赋予执行权限：

```bash
chmod +x tools/c2patool/c2patool
tools/c2patool/c2patool --help
```

作用：

- C2PA 是强证据，但只有文件本身携带内容凭证时才有结果。
- 没有 C2PA 不代表不是 AI，也不代表一定是真图，只是“没有可验证凭证”。

## 7. 准备本地文本 AIGC 检测模型

当前配置会使用这些 Hugging Face 模型：

```text
yuchuantian/AIGC_detector_zhv3
yuchuantian/AIGC_detector_zhv3short
yuchuantian/AIGC_detector_env3
yuchuantian/AIGC_detector_env3short
```

它们的作用：

- `zh_long`：中文长文本 AIGC 检测。
- `zh_short`：中文短文本 AIGC 检测。
- `en_long`：英文长文本 AIGC 检测。
- `en_short`：英文短文本 AIGC 检测。

可以让系统首次运行时自动下载，也可以提前预热：

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification

models = [
    "yuchuantian/AIGC_detector_zhv3",
    "yuchuantian/AIGC_detector_zhv3short",
    "yuchuantian/AIGC_detector_env3",
    "yuchuantian/AIGC_detector_env3short",
]

for name in models:
    print("download", name)
    AutoTokenizer.from_pretrained(name, cache_dir="models/huggingface")
    AutoModelForSequenceClassification.from_pretrained(name, cache_dir="models/huggingface")

print("local text detectors ready")
PY
```

这些权重会进入 `models/huggingface/hub`。

模型下载完成后，还需要生成多语言、多领域误判控制阈值。这个步骤不会训练或修改模型权重：

```bash
python -m scripts.bootstrap_text_resources \
  --profile all \
  --dataset-profile multidomain \
  --force-calibration
```

它会准备中文诗词、古文、CSL、HC3 中英文、TNEWS、M4/SemEval 和 RAID 非对抗数据，并把模型、语料、规范化样本和阈值分别保存在：

```text
models/huggingface/
data/text_false_positive/corpora/
data/text_false_positive/normalized/
artifacts/text_fp_controller/thresholds.json
```

建议为新增多领域语料和 Hugging Face 数据缓存至少预留 `4-6 GB` 磁盘空间，检测模型所需空间需要另外计算。实际占用会随数据集修订版本、Hugging Face/PyArrow 缓存格式及服务器已有缓存而变化。

下载脚本和 Hugging Face 组件会继承当前 shell 中的 `HTTP_PROXY`、`HTTPS_PROXY`（以及对应的小写变量）。服务器需要代理时，应在首次联网准备前导出这些变量；无需代理时不要设置。代理配置不会改变“先联网下载并校准、成功后再启用离线变量”的顺序。

阈值只由真人校准集分数决定；AI 留出集只用于计算召回率、FNR 和 AUROC。检查状态和数据统计：

```bash
python -m scripts.bootstrap_text_resources --profile status
python -m scripts.bootstrap_text_resources \
  --profile stats \
  --dataset-profile multidomain
```

当 `thresholds.json` 的 `artifact_schema_version` 为 `2` 时，Dashboard 和报告会展示真人误报率、AI 召回率、验证样本数、AUROC 与可靠性警告。

## 8. 准备文本来源归因分类器

这个模块不是判断“是不是 AI”，而是当内容疑似 AIGC 时，给出“更像哪个候选模型来源”的 Top-K。

### 8.1 推荐：直接拷贝本机训练产物

如果你本机已经训练好了，最稳的是把这个文件拷到服务器：

```text
models/attribution/text_source_classifier.joblib
```

### 8.2 服务器重新训练

```bash
python scripts/train_text_attribution.py \
  --dataset openturingbench \
  --max-samples 30000 \
  --max-per-label 3000
```

命令作用：

- 下载 Hugging Face 数据集 `MLNTeam-Unical/OpenTuringBench`。
- 抽取文本和来源模型标签。
- 用字符级 `TF-IDF + LogisticRegression` 训练多分类器。
- 输出 `models/attribution/text_source_classifier.joblib`。

如果你有自建候选模型样本，例如 DeepSeek、ChatGPT、Claude、Kimi、文心，可以加 JSONL：

```bash
python scripts/train_text_attribution.py \
  --dataset openturingbench \
  --extra-jsonl data/attribution/text/custom_text_sources.jsonl \
  --max-samples 30000 \
  --max-per-label 3000
```

JSONL 每行格式：

```json
{"text": "这里是一段由候选模型生成或人工写作的文本", "model": "DeepSeek"}
{"text": "另一段样本文本", "model": "ChatGPT"}
{"text": "人工写作样本", "model": "human"}
```

注意：

- 不建议直接把 MAGE 原始细标签和 OpenTuringBench 混训，会让类别过碎，之前测试效果下降。
- 如果要用 MAGE，先用 `--normalize-mage-labels` 做粗粒度归一。

## 9. 准备图片来源归因分类器

图片来源归因当前主路线是：

```text
Tiny-GenImage / 本地图片目录 -> CLIP image embedding -> LogisticRegression -> Top-K 来源分类
```

这条主路线已经在本项目代码里实现，不依赖 `third_party/UniversalAttribution`。它和 UniversalAttribution 的思想接近，都是用 CLIP 特征做来源归因；区别是本项目训练脚本已经包装成了可以直接被后端加载的 `joblib` 模型。

### 9.1 推荐：直接拷贝本机训练产物

如果本机训练好了，拷贝：

```text
models/attribution/image_source_classifier.joblib
```

同时服务器首次推理还需要 CLIP 权重，系统会自动下载到 `models/huggingface/hub`。也可以提前预热：

```bash
python - <<'PY'
from transformers import CLIPImageProcessor, CLIPModel

name = "openai/clip-vit-base-patch32"
CLIPImageProcessor.from_pretrained(name, cache_dir="models/huggingface")
CLIPModel.from_pretrained(name, cache_dir="models/huggingface")
print("CLIP ready")
PY
```

### 9.2 服务器重新训练

```bash
python scripts/train_image_attribution.py \
  --dataset tiny-genimage \
  --feature-extractor clip \
  --max-samples 10000 \
  --max-per-label 1000
```

命令作用：

- 下载 Hugging Face 数据集 `TheKernel01/Tiny-GenImage`。
- 下载 CLIP 模型 `openai/clip-vit-base-patch32`。
- 提取图片 embedding。
- 训练 LogisticRegression 来源分类器。
- 输出 `models/attribution/image_source_classifier.joblib`。

如果你有自建图片数据，目录结构应为：

```text
data/attribution/image_samples/
  Midjourney/
    001.png
    002.png
  SDXL/
    001.png
  DALL-E/
    001.png
  Real/
    001.jpg
```

训练命令：

```bash
python scripts/train_image_attribution.py \
  --dataset image-folder \
  --input data/attribution/image_samples \
  --feature-extractor clip
```

### 9.3 可选：UniversalAttribution

你当前项目里有：

```text
third_party/UniversalAttribution/
  configs/
  models/
  scripts/
  utils/
  knn.py
  linear.py
  requirements.txt
```

它的作用：

- 这是论文 `Are CLIP features all you need for Universal Synthetic Image Origin Attribution?` 的官方实验代码。
- 它主要用于复现 GenImage 上的开放集来源归因实验，例如 linear probe、KNN。
- 它不是当前后端的默认推理模块；仅仅存在这个目录，不代表系统上传图片时会自动调用它。

当前代码里的接入方式：

```yaml
provenance:
  attribution:
    image:
      providers:
        universal_attribution:
          enabled: true
          provider: "universal_attribution"
          command: null
          json_arg: true
```

其中 `command: null` 表示未启用真实调用。要让它进入后端流程，需要你额外写一个“统一 JSON 输出”的推理脚本，然后把 `command` 指向这个脚本。该脚本必须接收：

```bash
your_universal_infer_script /path/to/image --json
```

并输出类似：

```json
{
  "top_k": [
    {"model": "Midjourney", "probability": 0.42},
    {"model": "StableDiffusion", "probability": 0.31}
  ],
  "confidence": 0.42
}
```

如果只是部署当前可演示系统，优先使用 `models/attribution/image_source_classifier.joblib`。如果要复现实验或继续增强开放集归因，再单独准备 UniversalAttribution。

UniversalAttribution 自己需要的数据：

- GenImage 原始数据集。
- 官方 README 提到的 split 文件。
- 它自己的 Python 依赖，见 `third_party/UniversalAttribution/requirements.txt`。

运行示例：

```bash
cd third_party/UniversalAttribution
python -m pip install -r requirements.txt
python linear.py -c base -d GenImage:split=split1
python knn.py -c base -d GenImage:split=split1 --K 5
```

这些命令是复现实验/训练评估用，不会自动生成本项目使用的 `models/attribution/image_source_classifier.joblib`。本项目的可用训练产物仍由 `scripts/train_image_attribution.py` 生成。

## 10. 准备水印检测

水印是“主动溯源信号”，只有内容被对应工具加过水印时才可能检测出来。没检测到水印不等于人工生成。

### 10.1 AudioSeal

安装：

```bash
python -m pip install audioseal
```

预热下载检测器：

```bash
python - <<'PY'
from audioseal import AudioSeal
AudioSeal.load_detector("audioseal_detector_16bits")
print("AudioSeal detector ready")
PY
```

作用：

- 检测音频是否携带 AudioSeal 水印。
- 默认配置读取 `provenance.watermark.audio.model: audioseal_detector_16bits`。

### 10.2 VideoSeal

VideoSeal 的 Python 包和权重版本容易变化，建议按你当前能跑通的版本复制到服务器。项目配置默认：

```yaml
provenance:
  watermark:
    image:
      provider: "videoseal"
      package_root: "third_party/videoseal"
      model: "videoseal"
    video:
      provider: "videoseal"
      package_root: "third_party/videoseal"
      model: "videoseal"
```

服务器上要保证：

- `import videoseal` 成功。
- `third_party/videoseal` 存在。
- `third_party/videoseal/cards`、`configs`、`ckpts` 等模型卡和权重路径按 VideoSeal 项目要求放好。

验证：

```bash
python - <<'PY'
import videoseal
model = videoseal.load("videoseal")
print("VideoSeal ready", type(model))
PY
```

如果暂时不准备 VideoSeal，可先关闭：

```yaml
provenance:
  watermark:
    image:
      enabled: false
    video:
      enabled: false
```

## 11. 启动后端 FastAPI

不要使用 `--reload`：

```bash
source .venv/bin/activate
export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$PWD/models/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/models/huggingface/datasets"
export AIGC_API_BASE="http://127.0.0.1:8000/api/v1"

uvicorn api.main:app --host 0.0.0.0 --port 8000
```

作用：

- `api.main:app`：加载 FastAPI 应用。
- `--host 0.0.0.0`：允许服务器外部访问。
- `--port 8000`：后端 API 端口。

如果只允许本机访问，用：

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## 12. 启动前端 Streamlit

另开一个终端：

```bash
cd /path/to/aigc_detector
source .venv/bin/activate
export AIGC_API_BASE="http://127.0.0.1:8000/api/v1"

streamlit run api/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

作用：

- `api/dashboard.py`：Streamlit 演示前端。
- `AIGC_API_BASE`：告诉前端去请求哪个 FastAPI 地址。
- `8501`：前端页面端口。

如果服务器有防火墙或云安全组，需要开放 `8501`。后端 `8000` 可以只给本机访问，让前端转发即可。

## 13. 基础连通性测试

后端健康检查：

```bash
curl http://127.0.0.1:8000/health
```

查看 provider 状态：

```bash
curl http://127.0.0.1:8000/api/v1/providers
```

前端访问：

```text
http://服务器IP:8501
```

第一次上传文本或图片时，系统可能会下载 Hugging Face 模型，等待时间会比较长。之后会走本地缓存。

## 14. 推荐部署顺序

严格按这个顺序走，最少踩坑：

1. 安装系统依赖。
2. 创建 `.venv` 并安装 `requirements.txt`。
3. 设置 Hugging Face 缓存环境变量。
4. 把 `config.yaml` 里的密钥字段改成环境变量名。
5. 设置 Hive/Sightengine 环境变量。
6. 先把 `report.provider` 改成 `template`，避免 DeepSeek 继续扣费。
7. 启动 FastAPI 和 Streamlit，先测试文本/图片 API 检测是否通。
8. 下载或拷贝 YuchuanTian 文本检测模型缓存。
9. 拷贝或训练 `text_source_classifier.joblib`。
10. 拷贝或训练 `image_source_classifier.joblib`。
11. 准备 Linux 版 `c2patool` 并测试。
12. 准备 AudioSeal/VideoSeal；没准备好就先关闭水印分支。
13. 再做完整样例演示：文本、图片、视频、音频分别跑一次。

## 15. 常见问题

### 15.1 `huggingface-hub` 版本冲突

报错类似：

```text
huggingface-hub>=0.19.3,<1.0 is required ... but found huggingface-hub==1.x
```

修复：

```bash
python -m pip install "huggingface-hub>=0.24,<1.0"
```

### 15.2 DeepSeek 402

报错：

```text
402 Client Error: Payment Required
```

说明 DeepSeek 余额、额度或计费状态不可用。先改：

```yaml
report:
  provider: "template"
```

### 15.3 C2PA 一直没有结果

这是正常的。只有文件携带 C2PA manifest 时才会有结果。普通截图、微信转发图片、压缩图片通常没有 C2PA。

### 15.4 VideoSeal 加载失败

先确认：

```bash
python -c "import videoseal; print(videoseal)"
```

再确认 `third_party/videoseal` 里是否有 model card、config、checkpoint。没准备好时先关闭 image/video watermark，不影响主检测流程。

### 15.5 OpenCV 在 Linux 导入失败

常见报错是缺 `libGL.so.1`。修复：

```bash
sudo apt install -y libgl1 libglib2.0-0
```

### 15.6 第一次运行很慢

通常是在下载：

- YuchuanTian 文本检测模型。
- CLIP 图像模型。
- OpenTuringBench/Tiny-GenImage 数据集。
- AudioSeal/VideoSeal 权重。

确认缓存目录在项目内：

```bash
du -sh models/huggingface models/audioseal 2>/dev/null
```

## 16. 最小可运行版本

如果你只想先让演示跑起来，最低需要：

```text
Python 环境
requirements.txt
Hive/Sightengine API Key
report.provider: template
uvicorn 后端
streamlit 前端
```

此时能做：

- 文本/图片/视频/音频上传。
- 自动识别模态。
- API-first 检测。
- 本地模板报告。
- 数据库存储。

增强项按优先级依次加：

```text
YuchuanTian 文本检测模型
text_source_classifier.joblib
image_source_classifier.joblib
c2patool
AudioSeal
VideoSeal
自建 JSONL/图片样本库
```
