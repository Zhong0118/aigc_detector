# 智增增文本溯源数据生产说明

## 1. 当前数据规模

`local_pilot.json` 的 960 条是十个模型合计：

```text
10 模型 × 24 提示词 × 2 采样设置 × 2 次重复 = 960
```

每个模型生成：

```text
24 × 2 × 2 = 96 条
```

每个模型都完整覆盖相同的 24 条提示词、相同的两组采样参数和两次独立重复。
正式先导配置使用 `max_tokens=1536`。这是允许的最大输出而非强制输出长度；冒烟发现
768 会截断 Gemini 的展开式回答，而 1536 的定向复测以 `finish_reason=stop` 正常结束。

## 2. 配置和代码

当前 20 个具体版本、600 条阶段采集和 6000 条正式采集以
`docs/ATTRIBUTION_DATASET_V2_RUNBOOK.md` 为准。V2 采集器默认使用 4 个全局
worker，并限制同一具体模型只有 1 个在途请求；400/403/404 会触发该模型熔断，
空正文最多尝试 2 次，避免慢模型或无效路由拖住整轮任务。

同一家族允许使用不同具体版本，训练标签仍为 `family_id`。具体版本不能只根据
`/models` 列表选择，必须先运行 `scripts/probe_attribution_models.py` 的中英文
双探测。2026-07-29 的实测替换为 Kimi 的 `moonshot-v1-8k`、MiMo 的
`mimo-v2.5-pro`、Gemini 的 `gemini-3.1-flash-lite-preview`，以及 MiniMax 的
`abab6.5s-chat` 和 `MiniMax-Text-01`。最后三个版本还使用正式冒烟提示词与
1536/3072 输出预算进行了定向复测，六次请求均为非空、未截断、无 reasoning token
挤占，并且返回路由与请求路由一致。

- 模型目录：`configs/text_attribution/zhizengzeng_model_catalog.json`
- 24 条提示词：`configs/text_attribution/prompt_manifest.pilot-v1.json`
- 20 条冒烟：`configs/text_attribution/runs/smoke.json`
- 960 条本地先导：`configs/text_attribution/runs/local_pilot.json`
- 10,000 条服务器模板：`configs/text_attribution/runs/server_10k.json`
- 采集入口：`scripts/collect_attribution_run.py`
- 质量审计：`scripts/audit_attribution_data.py`
- 服务器入口：`scripts/run_zhizengzeng_server.ps1`

`server_10k.json` 当前有意设置为 `enabled: false`。只有提示词扩展并冻结到 100 条后
才启用，防止拿 24 条提示词重复堆出伪大数据。

## 3. 本地预演和冒烟

离线查看 20 条计划，不读取 Key、不访问网络：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_run `
  --config configs/text_attribution/runs/smoke.json `
  --dry-run
```

真实运行：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_run `
  --config configs/text_attribution/runs/smoke.json `
  --confirm
```

默认输出：

```text
data/attribution/raw/zhizengzeng-smoke-v3.jsonl
data/attribution/errors/zhizengzeng-smoke-v3.errors.jsonl
```

中断后重复执行相同命令即可续采；已经成功写入的 `request_key` 会被跳过。

## 4. 质量审计

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.audit_attribution_data `
  --config configs/text_attribution/runs/smoke.json `
  --input data/attribution/raw/zhizengzeng-smoke-v3.jsonl `
  --errors data/attribution/errors/zhizengzeng-smoke-v3.errors.jsonl `
  --output data/attribution/reports/zhizengzeng-smoke-v3.audit.json
```

报告包括模型、家族、语言、任务、采样和重复次数分布，以及缺失组合、空文本、截断、
路由不一致和重复文本数量。

## 5. 运行 960 条

只有冒烟报告满足十个家族均成功、无空输出、无截断后再运行：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_run `
  --config configs/text_attribution/runs/local_pilot.json `
  --confirm
```

数据写入：

```text
data/attribution/raw/zhizengzeng-local-pilot-v1.jsonl
```

这一步是计费操作，因此不会由脚本在冒烟后自动启动。

## 6. 上传服务器

上传项目代码和配置，但不要把 Key 写入 Git。登录服务器后，在项目根目录单独创建
`.env.local`：

```env
ZZZ_API_KEY=服务器上的真实Key
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
```

先预演：

```powershell
.\scripts\run_zhizengzeng_server.ps1 `
  -RunConfig configs/text_attribution/runs/local_pilot.json `
  -PythonExe python
```

确认计划为 960 后真实运行：

```powershell
.\scripts\run_zhizengzeng_server.ps1 `
  -RunConfig configs/text_attribution/runs/local_pilot.json `
  -PythonExe python `
  -Confirm
```

Linux 服务器可以直接使用相同的 Python 命令，不依赖 PowerShell脚本。API采集主要受
网络、RPM/TPM和账户额度限制，两张4090D在这个阶段不会显著加速；GPU用于后续特征
提取和训练。

## 7. 数据用于训练前的门槛

960 条完成后先执行：

1. 覆盖率、截断、拒答、重复和路由不一致审计；
2. 按 `prompt_id` 分组切分，禁止同提示泄漏；
3. 字符 n-gram、字节 n-gram、TF-IDF 和冻结编码器基线；
4. 提示外、领域外和时间外测试；
5. 只有基线证明存在可泛化的家族/模型差异后，才扩展 100 条提示词与 10,000 条数据。

## 8. 在服务器运行 923 条探索性基线

当前 923 条先导数据先用于验证“是否存在跨提示词可泛化的模型差异”，不能直接作为
正式十分类结论。运行器会排除空文本、截断、内容过滤、路由不一致、语言不一致和同一
采样单元内的完全重复文本；中英文语义配对提示词会被放在同一测试折，避免提示词泄漏。

先只检查数据和六折切分，不加载模型，也不创建输出目录：

```bash
python -u -m scripts.run_attribution_pilot_baselines \
  --input data/attribution/raw/zhizengzeng-local-pilot-v1.jsonl \
  --output-dir outputs/attribution/zhizengzeng-local-pilot-v1-baselines \
  --methods char_lr tfidf_svm frozen_e5 \
  --folds 6 \
  --dry-run
```

确认清洗数量和十个标签均合理后，正式运行：

```bash
python -u -m scripts.run_attribution_pilot_baselines \
  --input data/attribution/raw/zhizengzeng-local-pilot-v1.jsonl \
  --output-dir outputs/attribution/zhizengzeng-local-pilot-v1-baselines \
  --methods char_lr tfidf_svm frozen_e5 \
  --encoder-model intfloat/multilingual-e5-base \
  --device cuda:0 \
  --batch-size 16 \
  --max-length 512 \
  --hf-cache models/huggingface \
  --folds 6 \
  --seed 42
```

冻结 E5 只执行前向编码，随后在缓存向量上训练线性分类头，不会更新 E5 参数。默认
嵌入缓存为：

```text
outputs/attribution/zhizengzeng-local-pilot-v1-baselines/
  embeddings/intfloat__multilingual-e5-base.npz
```

重复运行会验证数据指纹和样本顺序后复用该缓存。运行结果包括：

```text
run_config.json
dataset_audit.json
metrics.json
metrics.csv
predictions.jsonl
report.md
figures/overall_metrics.png（以及 PDF）
figures/fold_metrics.png（以及 PDF）
figures/confusion_matrices.png（以及 PDF）
figures/subgroup_metrics.png（以及 PDF）
```

运行器先完成并保存字符 TF-IDF + Logistic Regression 与词/字符 TF-IDF + Linear
SVM，再加载 E5。即使 E5 下载或显存阶段失败，前两个基线及图表仍会保留，并在
`method_failures.json` 中记录失败原因。
