# 模型家族溯源 Phase2 神经基座服务器运行手册

本手册对应 N0（XLM-R）、N1（mDeBERTa-v3）和 N2（CANINE）三条神经基座
筛选路线。首轮筛选只读取 train 和 validation；除“最终一次 test”一节外，所有命令
都不会解锁冻结 test。

## 1. 需要上传什么

在本机项目根目录构建服务器包：

```bash
python -m scripts.package_attribution_phase2_server \
  --output dist/aigc-detector-attribution-phase2-server.zip
```

上传以下三个文件：

1. `dist/aigc-detector-attribution-phase2-server.zip`
2. `data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl`
3. `data/attribution/splits/family-attribution-v5-seed42.json`

ZIP 是“项目根目录叠加包”，包含 Phase2 所需代码、配置、文档、依赖清单和测试。
它明确不包含原始数据、split manifest、模型权重、Hugging Face 缓存、训练输出、
日志或任何 `.env` / API key。不要把 API key 放进项目文件或压缩包。

## 2. 解压到服务器项目

```bash
mkdir -p ~/projects/aigc_detector
unzip -o aigc-detector-attribution-phase2-server.zip \
  -d ~/projects/aigc_detector
cd ~/projects/aigc_detector

python -m scripts.package_attribution_phase2_server \
  --verify /你的上传目录/aigc-detector-attribution-phase2-server.zip

mkdir -p data/attribution/raw data/attribution/splits
mv /你的上传目录/zhizengzeng-attribution-server-6k-v5.jsonl \
  data/attribution/raw/
mv /你的上传目录/family-attribution-v5-seed42.json \
  data/attribution/splits/
```

检查包清单和关键文件：

```bash
python -m json.tool PHASE2_SERVER_MANIFEST.json >/dev/null
test -f scripts/run_family_attribution_neural.py
test -f configs/text_attribution/training/family_neural_screening.v1.json
test -f data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl
test -f data/attribution/splits/family-attribution-v5-seed42.json
```

## 3. Conda 环境与依赖

```bash
conda activate zx
cd ~/projects/aigc_detector
```

`requirements-attribution-phase2.txt` 故意不安装 PyTorch，因为 PyPI 包名本身不能
表达服务器驱动、CUDA runtime 和 wheel 索引的完整组合。先检查 `zx` 环境中现有
PyTorch；若不是 CUDA 版本，则根据服务器实际 CUDA/驱动，从 PyTorch 官方索引安装
匹配版本，再安装其余依赖：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY

python -m pip install -r requirements-attribution-phase2.txt
```

确认两张卡、BF16 和关键版本：

```bash
python - <<'PY'
import torch
import transformers
import accelerate

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
print("bf16:", torch.cuda.is_bf16_supported())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

应看到 `gpu_count: 2`，且两张卡均为 4090D。若当前 PyTorch 没有 CUDA 支持，先按
服务器 CUDA 版本安装匹配的 PyTorch，再安装本清单中的其余依赖。不要让代码静默
关闭 BF16；只有硬件确实不支持时才在正式单实验命令中显式加 `--no-bf16`。

在每次正式实验前保存可审计的环境快照。它记录实际 CUDA PyTorch，而不是伪造一个
脱离服务器环境的通用 lock：

```bash
mkdir -p logs/attribution-phase2/environment-seed42
python -m pip freeze --all \
  > logs/attribution-phase2/environment-seed42/pip-freeze.txt
python - \
  > logs/attribution-phase2/environment-seed42/torch-runtime.txt <<'PY'
import platform
import torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cudnn", torch.backends.cudnn.version())
PY
nvidia-smi -q \
  > logs/attribution-phase2/environment-seed42/nvidia-smi-q.txt
sha256sum requirements-attribution-phase2.txt requirements.txt \
  > logs/attribution-phase2/environment-seed42/requirements.sha256
```

可运行随包测试（不会下载模型）：

```bash
python -m pytest \
  tests/test_text_attribution_neural_config.py \
  tests/test_text_attribution_neural_data.py \
  tests/test_text_attribution_neural_selection.py \
  tests/test_attribution_phase2_server_package.py \
  -q
```

## 4. Dry-run：先验证数据、目录和冻结切分

Dry-run 不导入 Transformers、不下载模型、不创建输出目录：

```bash
python -u -m scripts.run_family_attribution_neural \
  --experiment N0 \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --output-dir outputs/attribution/family-phase2/dry-run-N0 \
  --seed 42 \
  --device cpu \
  --hf-cache models/huggingface \
  --dry-run
```

重点核对输出中的 `clean_samples=5686`、10 个家族、20 个具体模型、split
`3981 / 912 / 793` 和固定数据指纹。任何指纹不匹配都应先停止，不要重建 split。

## 5. 两步 smoke：只验证训练链路

Smoke 会下载 N0，并在 CPU 上执行两个 optimizer step；它只验证工程链路，不能进入
正式选型：

```bash
python -u -m scripts.run_family_attribution_neural \
  --experiment N0 \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --output-dir outputs/attribution/family-phase2-smoke/N0-seed42 \
  --seed 42 \
  --device cpu \
  --hf-cache models/huggingface \
  --smoke-test
```

成功后检查：

```bash
test -f outputs/attribution/family-phase2-smoke/N0-seed42/COMPLETED.json
python -m json.tool \
  outputs/attribution/family-phase2-smoke/N0-seed42/run_config.json | head
```

`smoke_test` 必须为 `true`，`evaluate_test` 必须为 `false`。

## 6. 用 tmux 启动正式首轮筛选

创建会话：

```bash
conda activate zx
cd ~/projects/aigc_detector
tmux new -s attribution-phase2
```

在 tmux 内运行：

```bash
mkdir -p logs
bash scripts/run_attribution_phase2_screening.sh \
  data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  data/attribution/splits/family-attribution-v5-seed42.json \
  models/huggingface \
  outputs/attribution/family-phase2 \
  42 \
  2>&1 | tee logs/attribution-phase2-seed42.console.log
```

脚本会让物理 GPU 0 跑 N0、物理 GPU 1 跑 N1；两者均成功后，再让物理 GPU 0
跑 N2。每个 Python 进程只看见一张卡，因此进程内部统一使用 `cuda:0`。脚本不含
test 参数；N0 或 N1 任一失败时会等待另一个进程收尾、记录失败码并且不启动 N2。

从 tmux 脱离：按 `Ctrl+B`，全部松开，再按 `D`。此后本机断开不会终止服务器任务。

重新连接：

```bash
conda activate zx
tmux attach -t attribution-phase2
```

查看状态：

```bash
tmux ls
watch -n 2 nvidia-smi
tail -f logs/attribution-phase2/N0-seed42.log
tail -f logs/attribution-phase2/N1-seed42.log
```

## 7. 中断后从 checkpoint 继续

失败运行不会伪装成正式成功目录，而会保存在类似下面的证据目录：

```text
outputs/attribution/family-phase2/.N0-seed42.failed-<随机ID>/
```

查找最新 checkpoint：

```bash
find outputs/attribution/family-phase2 \
  -path '*/checkpoints/checkpoint-*' -type d | sort
```

从一个明确 checkpoint 恢复，并写入新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_neural \
  --experiment N0 \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --output-dir outputs/attribution/family-phase2/N0-seed42-resumed \
  --seed 42 \
  --device cuda:0 \
  --hf-cache models/huggingface \
  --resume-from-checkpoint \
  outputs/attribution/family-phase2/.N0-seed42.failed-<随机ID>/checkpoints/checkpoint-<步数>
```

不要覆盖已有正式输出目录。恢复成功后，以新目录作为该次实验的候选输入。

## 8. 汇总 validation 并选前两名

三条正式 seed42 实验完成后：

```bash
python -u -m scripts.select_family_attribution_neural \
  --inputs \
    outputs/attribution/family-phase2/N0-seed42 \
    outputs/attribution/family-phase2/N1-seed42 \
    outputs/attribution/family-phase2/N2-seed42 \
  --output-dir outputs/attribution/family-phase2/selection-seed42
```

查看：

```bash
cat outputs/attribution/family-phase2/selection-seed42/neural_screening_summary.md
```

选择器只接受正式 validation 产物：smoke、含 test 指标、数据指纹不一致或指标
support 不一致的目录都会被拒绝。

随后只对前两名分别运行 seed 52、62。下面以 N0 为例，N1/N2 只需替换实验 ID 和
物理 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_neural \
  --experiment N0 \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --output-dir outputs/attribution/family-phase2/N0-seed52 \
  --seed 52 \
  --device cuda:0 \
  --hf-cache models/huggingface
```

三随机种子都完成后再报告均值、标准差和最差家族表现；不要根据单个 seed 直接进入
增强阶段。

## 9. 最终一次 test（只有冻结选择后才运行）

先根据全部 validation 实验冻结：基座、超参数和最终 seed。之后只允许对最终胜出
方案执行一次显式 test。当前 CLI 会在新目录中按同一配置重新训练并评估 test：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_family_attribution_neural \
  --experiment <最终N0或N1或N2> \
  --input data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl \
  --split-manifest data/attribution/splits/family-attribution-v5-seed42.json \
  --output-dir outputs/attribution/family-phase2-final/<实验>-seed<种子>-test \
  --seed <冻结种子> \
  --device cuda:0 \
  --hf-cache models/huggingface \
  --evaluate-test
```

不要把此命令加入首轮筛选脚本，也不要用 test 结果回头选择基座。Phase1 test 已被
查看，因此最终研究结论仍应再用外部或时间切分数据验证。

## 10. 结果目录与下载清单

每个成功实验应包含：

```text
<实验输出>/
├── COMPLETED.json
├── model/
├── tokenizer/
├── checkpoints/
├── run_config.json
├── dataset_audit.json
├── metrics.json
├── predictions.jsonl
├── trainer_history.json
├── report.md
├── overall_metrics.png
├── confusion_matrices.png
├── per_model_diagnostics.png
├── calibration.png
└── subgroup_metrics.png
```

第一轮分析优先下载这些小文件：

- `run_config.json`
- `dataset_audit.json`
- `metrics.json`
- `predictions.jsonl`
- `trainer_history.json`
- `report.md`
- 五张 PNG 图
- `selection-seed42/neural_screening_summary.json`
- `selection-seed42/neural_screening_summary.md`
- `logs/attribution-phase2/*.log`

服务器打包结果示例：

```bash
tar -czf attribution-phase2-seed42-results.tar.gz \
  --exclude='model' \
  --exclude='tokenizer' \
  --exclude='checkpoints' \
  outputs/attribution/family-phase2 \
  logs/attribution-phase2
```

模型确定后，再单独下载胜出实验的 `model/`、`tokenizer/` 和需要保留的 checkpoint。
