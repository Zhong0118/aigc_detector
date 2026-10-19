# 图片开发实验运行记录

实际本地 worktree：`/Users/zhongxu/.codex/worktrees/image-m0-m2-paper/aigc_detector`。
分支：`codex/image-m0-m2-paper-20261019`。本轮未写入school服务器实验目录，磁盘例外授权仍待答复。

## 环境与资产

本轮独立环境 `.venv-image`，MPS GPU；CPU线程4，DataLoader worker0。运行内 environment.txt 是每次启动时的实际依赖快照，不能用生产环境版本代替。新增代码不读取config.yaml，不调用任何API。

```bash
cd /Users/zhongxu/.codex/worktrees/image-m0-m2-paper/aigc_detector
python3 -m venv .venv-image
.venv-image/bin/python -m pip install -r configs/image_experiments/requirements.txt
```

既有来源/报告回归还需独立环境安装 transformers4.50.3、huggingface-hub0.36.2、sqlalchemy2.1.5、pandas2.3.3；这些仅为旧模块测试依赖，实际版本保存在安装日志和环境锁文件。

数据从固定revision公开URL取得，仅下载随机训练分片10，没有下载官方validation。源码下载到 `/tmp/aigc-image-research-20261019/`，未执行其下载器。OFA权重通过作者README的公开Google Drive普通下载确认取得；DnCNN来自作者仓库；ImageNet权重来自torchvision官方URL。全部哈希见 `outputs/image_experiments/delivery_assets.json`。

## 已执行命令

本节命令已真实执行，输出目录不可覆盖；续跑必须使用新的run目录。parquet在 `/tmp` 是原始下载副本，正式冻结子集位于本worktree的outputs，足以重跑训练。

```bash
.venv-image/bin/python scripts/prepare_image_manifest.py \
  --parquet /tmp/aigc-image-research-20261019/train-00010-of-00014.parquet \
  --revision 89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69 \
  --output outputs/image_experiments/tiny-v1 --per-source 50

.venv-image/bin/python scripts/run_image_experiment.py \
  --manifest outputs/image_experiments/tiny-v1/manifest.jsonl \
  --output outputs/image_experiments/ofa-v2 --task ofa \
  --weights outputs/image_experiments/assets/ofa_fingerprint.pth \
  --denoiser outputs/image_experiments/assets/model_best.th \
  --device mps --batch-size 8

.venv-image/bin/python scripts/train_image_attribution.py paper \
  --manifest outputs/image_experiments/tiny-v1/manifest.jsonl \
  --output outputs/image_experiments/a0-resnet50-v1 --task attribution \
  --weights outputs/image_experiments/assets/resnet50-0676ba61.pth \
  --device mps --batch-size 32 --epochs 20

.venv-image/bin/python scripts/train_image_attribution.py paper \
  --manifest outputs/image_experiments/tiny-v1/manifest.jsonl \
  --output outputs/image_experiments/d0-resnet50-v1 --task detection \
  --weights outputs/image_experiments/assets/resnet50-0676ba61.pth \
  --device mps --batch-size 32 --epochs 20
```

运行stdout/stderr分别重定向到 outputs/image_experiments 下同名 `.log`；失败日志保留。每个run输出 resolved_config.json、environment.txt、status.json、metrics.json、asset_manifest.json。ResNet另有history、model.pth、逐样本predictions、confusion_matrix与single_prediction。OFA另有带身份的features.pt、reference_manifest、逐样本配对分数。OFA上游等价检查见ofa-author-parity.json，合成输入只用于数值测试，不进入实验结果。

首次ofa-v1曾反转非对称成对头的参考/查询顺序，已增加INVALIDATED.json并保留原文件；其0.8625不是有效交付指标。修复为作者S在前、D在后，用相同manifest重新运行ofa-v2。开发集已用于此工程排错，holdout始终未用于推理或选择。

来源模型另用独立CLI在新进程执行了单图及三图批量推理，结果为 `a0-single-standalone.jsonl` 和 `a0-batch-standalone.jsonl`；真假模型相应为d0前缀。具体输入均为固定manifest中的开发样本，命令日志见 `standalone_commands.json`。CLI可用参数已通过--help核验：

```bash
.venv-image/bin/python scripts/predict_image_experiment.py --help
```

此CLI显式加载研究checkpoint；缺模型输出unavailable、score=null，坏图输出unsupported。来源输出仅在登记候选内给出top_k，unknown_probability=null，尚不具备未知来源拒识。它不改现有provider配置，也不将来源分数放入真假score。

## 验证与续跑边界

```bash
.venv-image/bin/python -m pytest -q \
  tests/test_image_experiments.py tests/test_attribution_adapters.py \
  tests/test_provenance_pipeline.py tests/test_report_and_providers.py
```

首次原有回归因隔离环境缺transformers导入失败，日志baseline-regression.log已保留；补齐后再运行相关测试并记录实际通过数。不是全仓库测试通过声明。

继续前先核对Git、status、manifest和资产哈希，使用同一冻结划分。未授权正式测试、扩大训练/微调搜索、商业API、sync合并或线上切换。服务器运行须先解决磁盘20%保留线，并在 `/home/zhongxu/projects/aigc_detector` 下建立独立图片工作树/环境；不要在现有sync生产工作树执行本runbook。
