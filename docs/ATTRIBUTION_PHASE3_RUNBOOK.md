# 文本模型家族溯源 Phase3 增强训练运行手册

本手册对应固定的 XLM-R Phase3 管线：A0 基线、A1 辅助具体模型头、A2 家族/模型
一致性、A3 监督对比学习、A4 head/center/tail 多窗口。正式筛选始终只使用冻结的
train 与 validation；Phase3 CLI 根本不提供 test 评估入口。

## 1. 在 Mac 构建并上传覆盖包

在项目根目录执行：

```bash
python -m scripts.package_attribution_phase3_server \
  --output dist/aigc-detector-attribution-phase3-server.zip

python -m scripts.package_attribution_phase3_server \
  --verify dist/aigc-detector-attribution-phase3-server.zip
```

需要上传到服务器的只有以下三个文件：

```text
dist/aigc-detector-attribution-phase3-server.zip
data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl
data/attribution/splits/family-attribution-v5-seed42.json
```

ZIP 是项目根目录覆盖包，不含数据、日志、输出、Hugging Face 缓存、checkpoint、
tokenizer、模型权重、`.env` 或凭据。服务器已有两份 V5 数据文件时，只上传 ZIP。

```bash
scp dist/aigc-detector-attribution-phase3-server.zip \
  zhongxu@<服务器地址>:/home/zhongxu/uploads/
```

## 2. 在服务器验证并解压

```bash
conda activate zx
cd /home/zhongxu/projects/aigc_detector

python -m scripts.package_attribution_phase3_server \
  --verify /home/zhongxu/uploads/aigc-detector-attribution-phase3-server.zip

unzip -o /home/zhongxu/uploads/aigc-detector-attribution-phase3-server.zip \
  -d /home/zhongxu/projects/aigc_detector

test -f PHASE3_SERVER_MANIFEST.json
test -f scripts/run_family_attribution_phase3.py
test -f scripts/run_attribution_phase3.sh
test -f configs/text_attribution/training/family_phase3_enhancement.v1.json
```

若数据尚未在服务器项目中：

```bash
mkdir -p data/attribution/raw data/attribution/splits
mv /home/zhongxu/uploads/zhizengzeng-attribution-server-6k-v5.jsonl \
  data/attribution/raw/
mv /home/zhongxu/uploads/family-attribution-v5-seed42.json \
  data/attribution/splits/
```

## 3. Hugging Face 镜像、依赖与环境审计

网络较慢时先设置当前 shell/tmux 会话：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/zhongxu/projects/aigc_detector/models/huggingface
mkdir -p "$HF_HOME"
```

依赖清单故意不安装 PyTorch，因为 CUDA wheel 必须匹配服务器驱动。先确认当前 `zx`
环境已有 CUDA PyTorch，再安装其余依赖：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("bf16", torch.cuda.is_bf16_supported())
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else 1)
PY

python -m pip install -r requirements-attribution-phase3.txt
```

依赖清单固定 `numpy<2.0`，避免 NumPy 2 与旧 SciPy/scikit-learn 二进制轮子不兼容。
正式运行前记录环境：

```bash
mkdir -p logs/attribution-phase3/environment
python -m pip freeze --all \
  > logs/attribution-phase3/environment/pip-freeze.txt
nvidia-smi -q \
  > logs/attribution-phase3/environment/nvidia-smi-q.txt
python - <<'PY' \
  > logs/attribution-phase3/environment/runtime.txt
import platform
import torch
import transformers
import accelerate
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("bf16", torch.cuda.is_bf16_supported())
for index in range(torch.cuda.device_count()):
    print("gpu", index, torch.cuda.get_device_name(index))
PY
sha256sum requirements-attribution-phase3.txt \
  > logs/attribution-phase3/environment/requirements.sha256
```

## 4. 先跑随包测试

```bash
python -m pytest \
  tests/test_text_attribution_phase3_config.py \
  tests/test_text_attribution_phase3_data.py \
  tests/test_text_attribution_attribution_losses.py \
  tests/test_text_attribution_hierarchical_model.py \
  tests/test_text_attribution_phase3_training.py \
  tests/test_run_family_attribution_phase3.py \
  tests/test_text_attribution_phase3_selection.py \
  tests/test_run_attribution_phase3_scheduler.py \
  tests/test_attribution_phase3_server_package.py \
  -q
```

Mac 本地没有 torch 时张量测试会跳过；服务器上不应跳过这些测试。

## 5. 固定路径并执行 dry-run

```bash
export P3_DATA=data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl
export P3_SPLIT=data/attribution/splits/family-attribution-v5-seed42.json
export P3_CACHE=models/huggingface
export P3_ROOT=outputs/attribution/family-phase3

python -u -m scripts.run_family_attribution_phase3 \
  --stage A0 \
  --input "$P3_DATA" \
  --split-manifest "$P3_SPLIT" \
  --output-dir "$P3_ROOT/dry-run-A0" \
  --view closed --seed 42 --device cpu \
  --hf-cache "$P3_CACHE" --dry-run
```

Dry-run 不导入 torch/Transformers、不下载模型、不创建输出目录。应核对：10 个家族、
20 个具体模型、closed 辅助标签数 20、holdout 辅助标签数 10、冻结 split 指纹，以及
`selection_scope=validation_only`、`test_evaluated=false`。指纹不一致时不要重建 split。

## 6. 两步 smoke 与 A4 显存预检

先验证 A0 和两个 A1 配方。Smoke 只做两个 optimizer step，不能参与晋级：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_phase3 \
  --stage A0 \
  --input "$P3_DATA" --split-manifest "$P3_SPLIT" \
  --output-dir "$P3_ROOT/smoke/A0-closed-seed42" \
  --view closed --seed 42 --device cpu \
  --hf-cache "$P3_CACHE" --smoke-test

CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_phase3 \
  --stage A1-alpha01 \
  --input "$P3_DATA" --split-manifest "$P3_SPLIT" \
  --output-dir "$P3_ROOT/smoke/A1-alpha01-closed-seed42" \
  --view closed --seed 42 --device cpu \
  --hf-cache "$P3_CACHE" --smoke-test

CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_phase3 \
  --stage A1-alpha03 \
  --input "$P3_DATA" --split-manifest "$P3_SPLIT" \
  --output-dir "$P3_ROOT/smoke/A1-alpha03-closed-seed42" \
  --view closed --seed 42 --device cpu \
  --hf-cache "$P3_CACHE" --smoke-test
```

A2–A4 必须在父阶段 decision 生成后 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_phase3 \
  --stage <A2或A3或A4> \
  --parent-decision "$P3_ROOT/selection-<父阶段>/stage_decision.json" \
  --input "$P3_DATA" --split-manifest "$P3_SPLIT" \
  --output-dir "$P3_ROOT/smoke/<阶段>-closed-seed42" \
  --view closed --seed 42 --device cpu \
  --hf-cache "$P3_CACHE" --smoke-test
```

A4 正式矩阵前必须在 GPU 上做三窗口显存预检：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_phase3 \
  --stage A4 \
  --parent-decision "$P3_ROOT/selection-A3/stage_decision.json" \
  --input "$P3_DATA" --split-manifest "$P3_SPLIT" \
  --output-dir "$P3_ROOT/preflight/A4-closed-seed42" \
  --view closed --seed 42 --device cuda:0 \
  --hf-cache "$P3_CACHE" --preflight-only
```

## 7. 在 tmux 中运行 A0 与 A1 的 3×3 矩阵

```bash
tmux new -s attribution-phase3
```

在 tmux 内恢复环境变量，然后运行：

```bash
conda activate zx
cd /home/zhongxu/projects/aigc_detector
export HF_ENDPOINT=https://hf-mirror.com
export P3_DATA=data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl
export P3_SPLIT=data/attribution/splits/family-attribution-v5-seed42.json
export P3_CACHE=models/huggingface
export P3_ROOT=outputs/attribution/family-phase3

bash scripts/run_attribution_phase3.sh \
  A0 "$P3_DATA" "$P3_SPLIT" "$P3_CACHE" "$P3_ROOT/A0" \
  "42,52,62"

bash scripts/run_attribution_phase3.sh \
  A1-alpha01 "$P3_DATA" "$P3_SPLIT" "$P3_CACHE" "$P3_ROOT/A1-alpha01" \
  "42,52,62"

bash scripts/run_attribution_phase3.sh \
  A1-alpha03 "$P3_DATA" "$P3_SPLIT" "$P3_CACHE" "$P3_ROOT/A1-alpha03" \
  "42,52,62"
```

每条命令产生 3 seeds × 3 views 共 9 个正式目录。物理 GPU 0/1 各跑一个，进程内
统一为 `cuda:0`。任一任务失败后不再派发新任务，但会等待在途任务并保留日志。

从 tmux 脱离：按 `Ctrl+B`，松开，再按 `D`。重新进入并监控：

```bash
tmux attach -t attribution-phase3
watch -n 2 nvidia-smi
find "$P3_ROOT" -name COMPLETED.json | sort | wc -l
tail -f "$P3_ROOT/A0/logs/A0-closed-seed42.log"
```

## 8. A1 选择与后续 A2–A4

选择 A1：

```bash
python -u -m scripts.select_family_attribution_phase3 \
  --stage A1 \
  --parent-root "$P3_ROOT/A0" \
  --candidate-root "$P3_ROOT/A1-alpha01" \
  --candidate-root "$P3_ROOT/A1-alpha03" \
  --output-dir "$P3_ROOT/selection-A1"

cat "$P3_ROOT/selection-A1/selection_report.md"
python -m json.tool "$P3_ROOT/selection-A1/stage_decision.json" | head -80
```

选择器只接受完整 9 格矩阵，逐项执行 closed Macro-F1、2/3 seed、最弱家族、MiMo、
GLM、合并 holdout、两个 holdout 方向和零家族 F1 共 9 个门槛。

如果 A1 未晋级，A2 不训练，直接生成 skipped decision：

```bash
python -u -m scripts.select_family_attribution_phase3 \
  --stage A2 \
  --upstream-decision "$P3_ROOT/selection-A1/stage_decision.json" \
  --output-dir "$P3_ROOT/selection-A2"
```

如果 A1 已晋级，按 `selected_recipe_id` 把父矩阵设为 alpha01 或 alpha03，再运行：

```bash
export P3_A1_PARENT="$P3_ROOT/A1-alpha01"  # 若胜出 alpha03，就替换
export PHASE3_PARENT_DECISION="$P3_ROOT/selection-A1/stage_decision.json"

bash scripts/run_attribution_phase3.sh \
  A2 "$P3_DATA" "$P3_SPLIT" "$P3_CACHE" "$P3_ROOT/A2" \
  "42,52,62"

python -u -m scripts.select_family_attribution_phase3 \
  --stage A2 --parent-root "$P3_A1_PARENT" \
  --candidate-root "$P3_ROOT/A2" \
  --output-dir "$P3_ROOT/selection-A2"
```

A3 基于 `selection-A2/stage_decision.json` 中的当前胜者。把占位符替换成 A0、胜出的
A1 或 A2 根目录：

```bash
export P3_CURRENT_PARENT=<当前父实验根目录>
export PHASE3_PARENT_DECISION="$P3_ROOT/selection-A2/stage_decision.json"

bash scripts/run_attribution_phase3.sh \
  A3 "$P3_DATA" "$P3_SPLIT" "$P3_CACHE" "$P3_ROOT/A3" \
  "42,52,62"

python -u -m scripts.select_family_attribution_phase3 \
  --stage A3 --parent-root "$P3_CURRENT_PARENT" \
  --candidate-root "$P3_ROOT/A3" \
  --output-dir "$P3_ROOT/selection-A3"
```

完成 A4 预检后，把占位符替换成 A3 decision 的当前胜者：

```bash
export P3_A4_PARENT=<A4当前父实验根目录>
export PHASE3_PARENT_DECISION="$P3_ROOT/selection-A3/stage_decision.json"

bash scripts/run_attribution_phase3.sh \
  A4 "$P3_DATA" "$P3_SPLIT" "$P3_CACHE" "$P3_ROOT/A4" \
  "42,52,62"

python -u -m scripts.select_family_attribution_phase3 \
  --stage A4 --parent-root "$P3_A4_PARENT" \
  --candidate-root "$P3_ROOT/A4" \
  --output-dir "$P3_ROOT/selection-A4"
```

每个 selection 目录都包含 `stage_decision.json`、`selection_report.md` 和最终可重放的
`promoted_recipe.json`。

## 9. checkpoint 恢复

失败证据目录形如 `outputs/attribution/family-phase3/.<任务名>.failed-<随机ID>/`。

```bash
find "$P3_ROOT" -path '*/checkpoints/checkpoint-*' -type d | sort

CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_phase3 \
  --stage A3 \
  --parent-decision "$P3_ROOT/selection-A2/stage_decision.json" \
  --input "$P3_DATA" --split-manifest "$P3_SPLIT" \
  --output-dir "$P3_ROOT/recovered/A3-closed-seed42" \
  --view closed --seed 42 --device cuda:0 \
  --hf-cache "$P3_CACHE" \
  --resume-from-checkpoint \
  "$P3_ROOT/.A3-closed-seed42.failed-<随机ID>/checkpoints/checkpoint-<步数>"
```

恢复前会比较 checkpoint 与当前 `recipe_hash`，不匹配会停止。不要覆盖旧目录。

## 10. 最终 5 个补充 seed

冻结 A4 decision 后，对最终胜者和最终阶段比较对象各补跑 `12,22,32,72,82`。

```bash
bash scripts/run_attribution_phase3.sh \
  "$P3_ROOT/selection-A4/promoted_recipe.json" \
  "$P3_DATA" "$P3_SPLIT" "$P3_CACHE" \
  "$P3_ROOT/confirmation/winner" \
  "12,22,32,72,82"

find <最终比较对象的筛选根目录> -name resolved_recipe.json -print -quit

bash scripts/run_attribution_phase3.sh \
  <上一步得到的resolved_recipe.json> \
  "$P3_DATA" "$P3_SPLIT" "$P3_CACHE" \
  "$P3_ROOT/confirmation/comparator" \
  "12,22,32,72,82"
```

这 5 个 seed 是稳定性确认，不用于回头调 A1–A4。与 42/52/62 合并后，报告 8-seed
均值、样本标准差、每个 holdout 方向和最弱家族表现。

## 11. 成功契约与下载分析包

正式运行至少应包含 `COMPLETED.json`、`run_config.json`、`resolved_recipe.json`、
`metrics.json`、`predictions.jsonl`、`report.md`、`loss_components.json`、
`view_manifest.json`、`model/`、`tokenizer/` 和 `checkpoints/`。正式状态必须是
`complete`，`test_evaluated` 必须是 `false`；smoke/preflight 不能进入选择器。

只打包分析文件，不下载模型、tokenizer 或 checkpoint：

```bash
cd /home/zhongxu/projects/aigc_detector
tar \
  --exclude='*/model/*' \
  --exclude='*/tokenizer/*' \
  --exclude='*/checkpoints/*' \
  -czf attribution-phase3-analysis.tar.gz \
  outputs/attribution/family-phase3 \
  logs/attribution-phase3

sha256sum attribution-phase3-analysis.tar.gz \
  > attribution-phase3-analysis.tar.gz.sha256
```

将 `attribution-phase3-analysis.tar.gz` 发回 Mac 即可整体分析；无需逐条手工复制日志，
也不要把模型权重放进分析压缩包。
