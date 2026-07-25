# AIGC Detector Linux 部署与运行文档

本文适用于将 `aigc_detector` 源码包复制到一台可以访问 Hugging Face、GitHub 和 PyPI 的 Linux 服务器，并在服务器上自动下载文本模型、校准语料和生成误判控制阈值。

本文默认项目部署目录为：

```text
/opt/aigc_detector
```

所有命令都必须从项目根目录执行，否则 `models/`、`data/`、`artifacts/`、数据库和报告缓存可能落到错误目录。

---

## 1. 部署包需要包含什么

上传到 Linux 的目录至少应包含：

```text
aigc_detector/
├── api/
├── detection/
├── ingestion/
├── provenance/
├── reports/
├── storage/
├── scripts/
├── models/
│   └── attribution/
│       ├── text_source_classifier.joblib
│       └── image_source_classifier.joblib
├── tools/
│   └── c2patool/
│       └── c2patool
├── config.yaml
├── requirements.txt
└── LINUX部署与运行_完整.md
```

如果服务器可以联网，以下内容不用从本机上传：

```text
models/huggingface/
data/text_false_positive/corpora/
data/text_false_positive/normalized/
artifacts/text_fp_controller/thresholds.json
```

它们会由文本资源引导脚本自动生成。

以下内容也不需要上传：

```text
__pycache__/
*.pyc
.pytest_cache/
.venv/
scripts/run_api.ps1
scripts/run_dashboard.ps1
```

PowerShell 脚本只用于 Windows，Linux 不运行它们。

如果服务器完全不能访问 Hugging Face、GitHub，则不能依靠自动下载，必须另外复制：

```text
models/huggingface/
data/text_false_positive/corpora/
data/text_false_positive/normalized/
artifacts/text_fp_controller/thresholds.json
```

本文后续步骤按“服务器可以联网”说明。

---

## 2. 自动下载会准备什么

运行：

```bash
python -m scripts.bootstrap_text_resources --profile all --force-calibration
```

会自动完成：

1. 下载四个 YuchuanTian 文本 AIGC 检测模型；
2. 下载中文诗词、古文和 CSL 真人语料；
3. 下载 HC3 中英文、TNEWS、M4/SemEval 中英文本和 RAID 非对抗数据；
4. 将原始资源写入 `data/text_false_positive/corpora/`；
5. 将确定性选中的规范化样本写入 `data/text_false_positive/normalized/`；
6. 按文档级哈希分离真人校准集、真人验证集和 AI 验证集；
7. 只用真人校准分数计算分语言、领域和长度的高风险阈值；
8. 用独立留出集计算真人误报率、AI 召回率/FNR、AUROC 和置信区间；
9. 生成 `artifacts/text_fp_controller/thresholds.json`（artifact v2）。

这个过程不会重新训练 YuchuanTian 模型权重。四个检测模型保持不变，脚本只下载模型、准备语料并校准判定阈值。AI 样本只用于评估召回率，不参与阈值拟合。

建议为多领域语料和 Hugging Face 数据缓存至少预留 `4-6 GB` 磁盘空间，检测模型所需空间需要另外计算。实际占用会随数据集修订版本、Hugging Face/PyArrow 缓存格式及服务器已有缓存而变化。

部署包内已有：

```text
models/attribution/text_source_classifier.joblib
models/attribution/image_source_classifier.joblib
```

这两个来源归因分类器不需要在服务器重新训练。

---

## 3. 安装 Linux 系统依赖

以下命令适用于 Ubuntu 22.04/24.04：

```bash
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  ffmpeg \
  libsndfile1 \
  libgl1 \
  libglib2.0-0 \
  curl
```

如果服务器有 NVIDIA GPU，应根据服务器 CUDA 版本安装对应的 PyTorch。不要直接复制 Windows CUDA 环境。

CPU 也可以运行，但首次阈值校准会明显更慢。

---

## 4. 创建 Python 环境

进入项目根目录：

```bash
cd /opt/aigc_detector
```

创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装依赖。当前文本模型包含 PyTorch `.bin` 权重，因此必须使用
`torch>=2.6`；项目同时将 Transformers 固定在 4.x，以避免与
`huggingface-hub<1.0` 产生依赖冲突。

先查看 NVIDIA 驱动支持的最高 CUDA 版本：

```bash
nvidia-smi
```

若顶部显示 `CUDA Version: 12.4` 或更高，先安装 CUDA 12.4 版 PyTorch：

```bash
python -m pip install -U pip setuptools wheel
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

若显示版本低于 12.4，安装兼容性更广的 CUDA 11.8 版：

```bash
python -m pip install -U pip setuptools wheel
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

然后安装其余项目依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 已包含多领域文本数据准备所需的 `datasets`、`PyArrow` 和 `tqdm`，会随上面的命令一起安装，不需要再单独执行安装命令。

两个现有来源归因 `joblib` 已在 `scikit-learn 1.5.2` 下验证可加载，因此服务器建议固定相同版本。

验证基础依赖：

```bash
python - <<'PY'
import torch
import transformers
import sklearn
import fastapi
import streamlit

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("sklearn:", sklearn.__version__)
print("basic dependencies ready")
PY
```

---

## 5. 修改 Linux 配置

### 5.1 C2PA 路径

部署包中的 `tools/c2patool/c2patool` 已经是 Linux ELF 可执行文件。

将 `config.yaml` 修改为：

```yaml
provenance:
  c2pa:
    enabled: true
    mode: "local"
    tool_path: "tools/c2patool/c2patool"
```

不要保留 `.exe` 后缀。

设置执行权限：

```bash
chmod +x tools/c2patool/c2patool
tools/c2patool/c2patool --help
```

### 5.2 API 密钥

`api_key_env`、`api_user_env`、`api_secret_env` 字段应填写环境变量名称，不能直接填写真实密钥。

建议配置为：

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
  provider: "llm"
  llm_provider: "deepseek"
  api_key_env: "DEEPSEEK_API_KEY"
```

在 Linux 设置实际值：

```bash
export HIVE_API_KEY="你的 Hive Key"
export SIGHTENGINE_API_USER="你的 Sightengine User"
export SIGHTENGINE_API_SECRET="你的 Sightengine Secret"
export DEEPSEEK_API_KEY="你的 DeepSeek Key"
```

如果暂时不使用 DeepSeek，可改为：

```yaml
report:
  provider: "template"
```

如果不使用 Hive 或 Sightengine，将对应的 `enabled` 改为 `false`。

已经直接写进旧配置的密钥应进行轮换。

### 5.3 可选水印模块

当前精简部署包不包含 `third_party/videoseal`，因此建议首次部署先关闭 AudioSeal 和 VideoSeal：

```yaml
provenance:
  watermark:
    audio:
      enabled: false
    image:
      enabled: false
    video:
      enabled: false
```

这些模块关闭后不会影响文本 AIGC 检测、误判控制、来源归因、C2PA 和报告功能。

后续需要 AudioSeal 时可单独安装：

```bash
python -m pip install audioseal
```

VideoSeal 需要额外准备源码、模型卡、配置和权重，不属于文本资源引导脚本的下载范围。

### 5.4 文本误判控制

推荐保留以下多领域配置：

```yaml
detection:
  text_false_positive_control:
    dataset_profile: "multidomain"
    target_fpr: 0.02
    bootstrap_human_calibration_limit: 5000
    bootstrap_human_validation_limit: 1000
    bootstrap_ai_validation_limit: 1000
    bootstrap_max_source_items: 500000
    min_reliable_human_validation: 200
    min_reliable_ai_validation: 200
    min_acceptable_ai_recall: 0.7
```

其中 `target_fpr` 控制真人误报目标，`min_acceptable_ai_recall` 只产生可靠性警告，不会用 AI 数据降低已由真人校准得到的阈值。

---

## 6. 设置在线下载环境

首次下载和校准前执行：

```bash
cd /opt/aigc_detector
source .venv/bin/activate

export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$PWD/models/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/models/huggingface/datasets"
export AUDIOSEAL_CACHE_DIR="$PWD/models"

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
```

首次准备阶段不能设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

这两个变量会禁止 Hugging Face 下载。

下载脚本、`datasets` 和 Hugging Face Hub 会继承当前 shell 的 `HTTP_PROXY`、`HTTPS_PROXY`（以及对应的小写变量）。服务器必须通过代理联网时，可在运行第 7 节命令前设置：

```bash
export HTTP_PROXY="http://代理地址:端口"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
```

不需要代理时不要设置这些变量。无论是否使用代理，都必须先保持在线模式完成下载和校准，确认资源可用后再按第 9 节开启离线模式。

---

## 7. 下载文本模型、语料并生成阈值

执行：

```bash
python -m scripts.bootstrap_text_resources \
  --profile all \
  --dataset-profile multidomain \
  --force-calibration
```

模型默认写入：

```text
models/huggingface/
```

校准语料默认写入：

```text
data/text_false_positive/corpora/
```

确定性抽样后的规范化样本写入：

```text
data/text_false_positive/normalized/
```

阈值产物默认写入：

```text
artifacts/text_fp_controller/thresholds.json
```

下载具有断点式幂等行为：完整资源存在时不会重复解压语料，Hugging Face 已缓存的模型文件也会被复用。

`--force-calibration` 会重新计算阈值，但不会重新下载已经完整存在的模型。

默认每个精确语言/领域/长度组最多使用 5000 条真人校准样本、1000 条真人验证样本和 1000 条 AI 验证样本。实际数量取决于数据源可用量。GPU 服务器会更快，CPU 服务器可能耗时较长。

检查结果：

```bash
python -m scripts.bootstrap_text_resources --profile status
```

不加载检测模型、只统计已经准备好的数据：

```bash
python -m scripts.bootstrap_text_resources \
  --profile stats \
  --dataset-profile multidomain
```

确认阈值文件存在：

```bash
test -f artifacts/text_fp_controller/thresholds.json \
  && echo "threshold artifact ready" \
  || echo "threshold artifact missing"
```

查看校准类型：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/text_fp_controller/thresholds.json")
payload = json.loads(path.read_text(encoding="utf-8"))

print("version:", payload.get("version"))
print("artifact_schema_version:", payload.get("artifact_schema_version"))
print("calibration_kind:", payload.get("calibration_kind"))
print("target_fpr:", payload.get("target_fpr"))
for name, group in sorted(payload.get("groups", {}).items()):
    if name.count(":") == 2:
        print(
            name,
            "threshold=", group.get("high_threshold"),
            "human_fpr=", group.get("validation_fpr"),
            "ai_recall=", group.get("ai_recall"),
            "auroc=", group.get("validation_auroc"),
            "warnings=", group.get("warnings"),
        )
PY
```

`artifact_schema_version` 应为 `2`，`calibration_kind` 应为：

```text
human_only_split_conformal
```

---

## 8. 下载图片归因所需的 CLIP

部署包中的图片来源分类器依赖：

```text
openai/clip-vit-base-patch32
```

文本资源引导脚本不会下载 CLIP，因此需要单独预热：

```bash
python - <<'PY'
from transformers import CLIPImageProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"
cache_dir = "models/huggingface"

CLIPImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)

print("CLIP ready")
PY
```

如果只使用文本检测，可以暂时跳过本节。

---

## 9. 可选：切换为 Hugging Face 离线模式

只有在第 7、8 节成功完成后，才可以开启离线模式：

```bash
export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$PWD/models/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/models/huggingface/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

开启后：

- 已下载的文本模型可以离线加载；
- 已下载的 CLIP 可以离线加载；
- 不应再运行需要下载新模型或数据集的训练任务；
- Hive、Sightengine、DeepSeek 仍属于在线 API，不受 Hugging Face 离线变量控制。

如果服务器长期可以联网，也可以不设置离线变量。

---

## 10. 启动 FastAPI 后端

进入项目根目录：

```bash
cd /opt/aigc_detector
source .venv/bin/activate
```

设置缓存：

```bash
export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$PWD/models/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/models/huggingface/datasets"
```

启动后端：

```bash
python -m uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

不要使用 `--reload`。模型缓存写入可能触发自动重启。

后端启动时会自动创建：

```text
data/aigc_detection.db
data/report_cache/
```

---

## 11. 启动 Streamlit 前端

另开一个终端：

```bash
cd /opt/aigc_detector
source .venv/bin/activate

export AIGC_API_BASE="http://127.0.0.1:8000/api/v1"

python -m streamlit run api/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

浏览器访问：

```text
http://服务器IP:8501
```

服务器防火墙或云安全组需要允许访问 `8501`。FastAPI 的 `8000` 端口可以只允许服务器本机访问。

---

## 12. 后台运行

创建日志目录：

```bash
cd /opt/aigc_detector
mkdir -p logs
```

后台启动 API：

```bash
nohup .venv/bin/python -m uvicorn api.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  > logs/api.log 2>&1 &
```

后台启动 Dashboard：

```bash
nohup env AIGC_API_BASE="http://127.0.0.1:8000/api/v1" \
  .venv/bin/python -m streamlit run api/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  > logs/dashboard.log 2>&1 &
```

查看日志：

```bash
tail -f logs/api.log
tail -f logs/dashboard.log
```

正式长期运行建议改用 `systemd` 或进程管理器，而不是长期依赖 `nohup`。

---

## 13. 服务检查

检查后端：

```bash
curl http://127.0.0.1:8000/health
```

预期：

```json
{"status":"ok"}
```

检查检测模块状态：

```bash
curl http://127.0.0.1:8000/api/v1/providers
```

重点确认：

- `local_text_detector` 已启用；
- `text_false_positive_controller` 已加载阈值产物；
- 阈值版本不是 `built-in`；
- 缺失的可选模块只显示为未配置或依赖缺失，不影响主流程。

---

## 14. 日常重新启动

首次部署成功后，日常启动不需要再次执行：

```bash
python -m scripts.bootstrap_text_resources --profile all --force-calibration
```

只需要进入项目根目录并启动两个服务：

```bash
cd /opt/aigc_detector
source .venv/bin/activate

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

以及：

```bash
cd /opt/aigc_detector
source .venv/bin/activate
export AIGC_API_BASE="http://127.0.0.1:8000/api/v1"

python -m streamlit run api/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

只有以下情况需要重新 bootstrap：

- 删除了 `models/huggingface/`；
- 删除了 `data/text_false_positive/corpora/`；
- 删除了 `artifacts/text_fp_controller/thresholds.json`；
- 修改了校准数据、抽样规则或目标误报率；
- 更换了文本检测模型；
- 明确需要重新计算阈值。

---

## 15. 最终目录结构

首次准备完成后，目录应类似：

```text
aigc_detector/
├── api/
├── detection/
├── ingestion/
├── provenance/
├── reports/
├── storage/
├── scripts/
├── artifacts/
│   └── text_fp_controller/
│       └── thresholds.json
├── data/
│   ├── aigc_detection.db
│   └── text_false_positive/
│       ├── corpora/
│       │   ├── chinese_poetry/
│       │   ├── chinese_classical_corpus/
│       │   ├── csl/
│       │   ├── hc3_en/
│       │   ├── hc3_zh/
│       │   ├── tnews/
│       │   ├── m4_multilingual/
│       │   └── raid_no_adversarial/
│       └── normalized/
│           └── *.jsonl
├── models/
│   ├── attribution/
│   │   ├── text_source_classifier.joblib
│   │   └── image_source_classifier.joblib
│   └── huggingface/
│       ├── hub/
│       └── models--yuchuantian--*/
├── tools/
│   └── c2patool/
│       └── c2patool
├── config.yaml
└── requirements.txt
```

---

## 16. 最短部署流程

已经阅读配置修改说明后，实际核心命令可以压缩为：

```bash
cd /opt/aigc_detector

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

chmod +x tools/c2patool/c2patool

export HF_HOME="$PWD/models/huggingface"
export HF_HUB_CACHE="$PWD/models/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/models/huggingface/datasets"

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

python -m scripts.bootstrap_text_resources \
  --profile all \
  --dataset-profile multidomain \
  --force-calibration

python -m uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

另一个终端启动前端：

```bash
cd /opt/aigc_detector
source .venv/bin/activate

export AIGC_API_BASE="http://127.0.0.1:8000/api/v1"

python -m streamlit run api/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

这就是当前精简部署包的完整在线部署流程。
