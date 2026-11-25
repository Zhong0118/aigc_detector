# 中英文文本检测服务交付手册

## 1. 交付物与校验

本服务提供中英文文本检测、分级判定和十个候选模型家族的文本归因。归因融合 A0、B0、B1、B3 四路结果。交付范围为文本，图片、音频和视频需另行部署。

| 交付物 | 放置位置 |
| --- | --- |
| 已交付版本的 Git 仓库检出目录或源码归档解压目录 | `/srv/aigc_detector/` |
| 模型资产包 `aigc-text-runtime-20261124.tar.gz` | `/srv/incoming/`，约 2.96 GB，解压约 4.29 GB |
| Python 依赖 wheelhouse | `/srv/incoming/wheelhouse/`，仅完全离线安装需要 |

用交付方经可信渠道提供的 SHA-256 核对资产包。下列值对应本手册所列版本。部署脚本还会按包内清单校验每个资产文件。

```bash
cd /srv/incoming
sha256sum aigc-text-runtime-20261124.tar.gz
sha256sum -c <<'EOF_SHA256'
c1e52dd1ccf473b6c1b0aa86a14fbb72148b1489afb07d2c6592645a03cf334a  aigc-text-runtime-20261124.tar.gz
EOF_SHA256
```

## 2. 运行环境

- Linux x86_64、Python 3.10（含 `venv` 模块）、Bash 和 curl。启动脚本依赖 Linux `/proc` 管理进程。macOS 和 Windows 可作为访问客户端。
- 磁盘至少 10 GB。若同时保留 CUDA 依赖、wheelhouse 或多个版本，需额外预留磁盘。
- CPU 部署至少 8 GB 内存，推荐 16 GB。GPU 部署可选，需安装与所选 PyTorch CUDA wheel 兼容的 NVIDIA 驱动。
- 负责部署的账号需要能写入代码目录。上传真正的旧版二进制 `.doc` 时，服务端还需安装 LibreOffice Writer，并能通过 `PATH` 找到 `soffice` 或 `libreoffice`。完全离线时请预装该系统依赖，或先将文档另存为 `.docx` / `.txt`。

## 3. 安装与启动

### 方式 A · 目标机器可访问 pip 软件源

在以下两套命令中选一套。PyTorch 与其它包需要安装传递依赖，安装命令应省略 `--no-deps`。可将第二次 pip 安装的 `--index-url` 换成接收方可访问的国内 PyPI 镜像地址。

**CPU**

```bash
cd /srv/aigc_detector
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-text-runtime.txt --index-url https://pypi.org/simple
.venv/bin/python -m pip check
PYTHON=python3.10 bash scripts/deploy_runtime.sh \
  --bundle /srv/incoming/aigc-text-runtime-20261124.tar.gz \
  --venv .venv --device cpu --api-port 8611 --ui-port 8612
```

**CUDA 12.4 示例**

```bash
cd /srv/aigc_detector
python3.10 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-gpu/bin/python -m pip install -r requirements-text-runtime.txt --index-url https://pypi.org/simple
.venv-gpu/bin/python -m pip check
.venv-gpu/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.version.cuda)'
PYTHON=python3.10 bash scripts/deploy_runtime.sh \
  --bundle /srv/incoming/aigc-text-runtime-20261124.tar.gz \
  --venv .venv-gpu --device cuda:0 --api-port 8611 --ui-port 8612
```

CUDA 构建版本由驱动兼容性决定。选用其它 PyTorch 2.6.0 wheel 时，同步调整下文 wheelhouse 的构建后缀。[PyTorch 2.6.0 安装索引](https://pytorch.org/get-started/previous-versions/#v260)

### 方式 B · 目标机器完全离线

在同为 Linux x86_64、Python 3.10，且系统兼容的**联网机**上，从交付代码目录执行。CPU 用 `cpu`，CUDA 12.4 改为 `cu124`。每种构建使用独立目录，确保一个 wheelhouse 中只有一种 PyTorch 构建。

```bash
cd /srv/aigc_detector
TORCH_BUILD=cpu
WHEELHOUSE="$PWD/wheelhouse-$TORCH_BUILD"
mkdir -p "$WHEELHOUSE"
printf 'torch==2.6.0+%s\n' "$TORCH_BUILD" > "$WHEELHOUSE/torch-constraint.txt"
python3.10 -m pip download --only-binary=:all: \
  --index-url https://pypi.org/simple \
  --extra-index-url "https://download.pytorch.org/whl/$TORCH_BUILD" \
  --constraint "$WHEELHOUSE/torch-constraint.txt" \
  -r requirements-text-runtime.txt --dest "$WHEELHOUSE"
```

将该目录完整拷贝到离线机的 `/srv/incoming/wheelhouse/`。在**离线机**执行以下命令。首次启动会创建虚拟环境，并使用 `--no-index` 从 wheelhouse 安装全部依赖。已有 venv 会直接复用。

```bash
cd /srv/aigc_detector
PYTHON=python3.10 bash scripts/deploy_runtime.sh \
  --bundle /srv/incoming/aigc-text-runtime-20261124.tar.gz \
  --wheelhouse /srv/incoming/wheelhouse --venv .venv-offline \
  --device cpu --api-port 8611 --ui-port 8612
```

使用 CUDA wheelhouse 时，将上面的 `--device cpu` 改为 `--device cuda:0`，并为该构建使用独立的 `--venv .venv-gpu`。

## 4. 验收与访问

启动成功表示 API 和看板均通过健康检查，内置中英文样本完成检测，且至少一条样本完成十家族四路归因。查看 `runtime_state/smoke.json` 可核对阈值组、分数、归因状态、耗时和本实例的报告 ID。启动烟测和下文的人工补测都会写入本实例的历史库。

```bash
cd /srv/aigc_detector
cat runtime_state/smoke.json
curl -fsS http://127.0.0.1:8611/health
curl -fsS http://127.0.0.1:8612/_stcore/health
```

API 和看板默认监听部署机的回环地址。在客户端将 `deploy@text-server.example` 替换为实际 SSH 登录地址并运行下列命令，保持终端连接，再打开 [Web 看板](http://127.0.0.1:8612)。

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8612:127.0.0.1:8612 \
  -L 127.0.0.1:8611:127.0.0.1:8611 \
  deploy@text-server.example
```

在部署机或已转发端口的客户端运行以下命令，补测一条少于 500 字的中文短文和一条超过 500 字的中文长文。预期阈值组分别为 `zh_short` 与 `zh_long`，实际判定和归因随输入而变化。

```bash
API_BASE=http://127.0.0.1:8611
SMOKE_DIR=$(mktemp -d)
cat > "$SMOKE_DIR/short.txt" <<'EOF_SHORT'
周六早上，图书馆门前新摆了两排自行车架。管理员说，过去下雨时，大家常把车挪到入口的雨棚下，带婴儿车的读者进门不太方便。这一回，车架移到了侧门，沿墙留出了一条人行通道。午后，志愿者拿着纸板写了几张指路牌，又给旧车架贴上了迁移通知。接下来两周，图书馆会记录每天的停车数量，再决定是否增加一排车架。入口的雨棚仍会开放，方便读者临时收伞和整理随身物品。
EOF_SHORT
cat > "$SMOKE_DIR/long.txt" <<'EOF_LONG'
这家社区图书馆准备调整周末开放时间。过去两年，借书台周六九点开门，周日只开半天。附近的家长反映，孩子参加完上午的活动再来，常常赶不上周日的开放时间。也有人习惯周日早一点来还书，希望保留现有时段。馆员先在入口贴了一张登记表，问读者通常几点到馆、是否愿意使用门外的还书箱。登记持续三个周末，工作人员每天收馆后录入一次，只统计到访时段和建议，不收集姓名。

从登记表看，周日午后有几个相对集中的时段，但每周人数不同。放晴的时候，多数人愿意步行过来，在阅览区多坐一会儿。遇到降雨，来访者往往只还书，很少留下阅读。馆员把每日借还册数与当天的天气分开记录，也在表上标出了讲座和亲子活动的日期。这样月底整理记录时，可以看出哪些客流来自临时活动，哪些时段在平常一周里也有稳定需求。

馆方计划先试行四周。周日早上维持还书服务，借阅和阅览窗口延长到下午四点。人员调度从周六较空闲的晚间时段挪出，原来已经预约的场次照常举办。每班留下两名工作人员，一人负责借还登记，一人整理书架和处理求助。门口另备一张简明时刻表，把前台服务、活动室和还书箱的使用时间分开标明，避免读者到馆后才发现某个窗口尚未开放。

试行期间，工作人员会每周汇总借还数量、阅览座位使用和排队时间，也会抄录读者在意见簿里写下的具体困难。比如有人建议在门外补一盏灯，方便冬天傍晚还书；也有读者希望把大字号的开放时间表贴在入口玻璃上。四周后，馆方将公布调整后的安排，并写明这次试行中实际采纳了哪些建议，方便读者按新的开放时间规划到访。
EOF_LONG

python3.10 - "$SMOKE_DIR" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
for name in ("short", "long"):
    sample = folder / f"{name}.txt"
    payload = {"text": sample.read_text(encoding="utf-8"), "filename": sample.name}
    (folder / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
PY

curl -fsS --max-time 180 "$API_BASE/api/v1/analyze/text" \
  -H 'Content-Type: application/json' --data-binary "@$SMOKE_DIR/short.json" \
  -o "$SMOKE_DIR/short-result.json"
curl -fsS --max-time 180 "$API_BASE/api/v1/analyze/text" \
  -H 'Content-Type: application/json' --data-binary "@$SMOKE_DIR/long.json" \
  -o "$SMOKE_DIR/long-result.json"
python3.10 - "$SMOKE_DIR" <<'PY'
import json
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*-result.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    detection = report["detection"]
    print(path.name, report["content_id"], detection["decision"]["threshold_group"], detection["label"])
PY
```

## 5. HTTP API 与结果字段

| 方法和路径 | 用途 |
| --- | --- |
| `GET /health` | 返回 `{"status":"ok"}`。 |
| `POST /api/v1/analyze/text` | 接收 JSON，`text` 是正文，`filename` 可省略。参见上节两条请求。 |
| `POST /api/v1/analyze/file` | 用 `multipart/form-data` 上传 `file`，支持 `.txt`、`.md`、`.doc` 和 `.docx`。文本文件建议使用 UTF-8。 |
| `GET /api/v1/report/{content_id}` | 按分析响应的 `content_id` 查询已保存的检测和归因记录。 |

沿用上节变量可直接执行文件分析和历史查询。上传 Word 或 Markdown 时，把 `-F` 后的文件路径替换为实际路径。

```bash
curl -fsS --max-time 180 "$API_BASE/api/v1/analyze/file" \
  -F "file=@$SMOKE_DIR/short.txt"
CONTENT_ID=$(python3.10 -c 'import json,sys; print(json.load(sys.stdin)["content_id"])' \
  < "$SMOKE_DIR/short-result.json")
curl -fsS "$API_BASE/api/v1/report/$CONTENT_ID"
```

分析接口的 `detection.score` 是 0–1 的风险分。已校准结果须结合 `detection.decision.threshold_group` 及该组的 `screening_threshold`、`high_threshold` 阅读。复核线提示人工复核，严格线用于高风险判定。高风险还取决于检测覆盖率，高风险长文还需满足高分窗口的覆盖和分布条件。分数不能解读为 AI 撰写的字数比例。

| 判定状态 | 如何读 |
| --- | --- |
| `high_risk_ai` | 高风险疑似 AI。分数超过严格线，且满足判定所需的覆盖条件。 |
| `uncertain` 且达到复核线 | 不确定 · 疑似 AI，需复核。 |
| `uncertain` 且未触发复核 | 严格判定不确定 · 未触发复核。 |
| `low_risk` | 低风险（更偏人工）。 |
| `insufficient_evidence` | 证据不足，界面隐藏不可比的风险分。API 调用方可读取 `uncertainty_reasons` 确认原因。 |

`decision.segments` 提供窗口区间和得分，`high_risk_coverage` 是去重叠高分字符覆盖率，`analyzed_coverage` 是检测覆盖率。可将这些局部证据与整篇判定一同交给人工复核。

归因对象位于 `provenance.attribution`。`status` 表示本次归因是否完成，例如 `ok`（完成）、`skipped`（未触发）或 `error`（运行失败）。`family_probabilities` 是候选家族分布，`top_k` 给出排序靠前的候选，`confidence` 是归因置信度，`recipe` 和 `branches` 记录融合方案及分路细节。候选分布用于溯源研判。

## 6. 停止、切换、回退与排障

```bash
cd /srv/aigc_detector
PYTHON=python3.10 bash scripts/deploy_runtime.sh --stop
```

- **停止与切换**。停机后用对应的 venv 和 `--device cpu` 或 `--device cuda:0` 重跑启动命令。相同部署目录会复用 `data/` 下的历史库。切换到 CUDA 前，用该 venv 检查 `torch.cuda.is_available()`。
- **端口冲突**。运行 `ss -ltnp '( sport = :8611 or sport = :8612 )'` 确认监听进程。可用 `--api-port 8711 --ui-port 8712` 另选两个端口，同步修改健康检查与 SSH 转发的目标端口。
- **启动或烟测失败**。检查 `runtime_state/api.log`、`runtime_state/ui.log` 与终端报错。脚本会停止本次启动的进程。若 pip 找不到离线 wheel，检查目标机的架构、Python 版本和 wheelhouse 是否完整。若提示无法读取旧版 `.doc`，检查 LibreOffice 是否可在服务环境中调用。
- **离线缓存**。脚本设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，并将 Hugging Face 缓存定位在 `runtime_state/hf-cache/`。模型由随包交付的文件加载。校验失败时应重新取得完整交付包，避免手动篡改 `runtime_assets/`。保持 `preprocessing.enabled` 关闭。启用它会改变评分协议，需要重新校准。
- **版本回退**。为每个交付版本保留独立目录、资产包和原启动命令。先停当前版本，再到前一版本目录按其原命令启动并重新核验 `/health` 和 `runtime_state/smoke.json`。回退后的历史记录来自该部署目录自己的 `data/`。
