# 双语文本服务离线迁移

本方案提供中英文文本检测、校准判定和十家族文本归因。代码使用含迁移脚本的 Git 提交，模型包与运行历史独立传输。四路归因使用 A0、B0、B1、B3，各占 25%。

## 在 school 打包

从包含本脚本的代码目录执行。命令只读取现有发布目录的校准产物与白名单内的模型文件。

```bash
bash scripts/pack_runtime_bundle.sh \
  --source-root /home/zhongxu/projects/aigc_detector \
  --output /home/zhongxu/projects/aigc_detector/dist/aigc-text-runtime-20261124.tar.gz
sha256sum /home/zhongxu/projects/aigc_detector/dist/aigc-text-runtime-20261124.tar.gz
```

归档内模型总计约 4.29 GB，压缩后约 2.96 GB。包内含 `SHA256SUMS` 和 `bundle-manifest.json`。保留校验值并使用可信渠道发送，接收方也应校验整个压缩包。部署脚本还会逐文件校验以及核对白名单。

```bash
# 从能连通 school 和新服务器的管理机执行
rsync -avP school:/home/zhongxu/projects/aigc_detector/dist/aigc-text-runtime-20261124.tar.gz ./
rsync -avP aigc-text-runtime-20261124.tar.gz new-host:/srv/incoming/
```

## 新机启动

在 Linux 上准备包含此脚本的代码目录。联网阶段可以用 Git 获取对应的 `sync` 提交；完全隔离时可以预先把对应提交打成 Git bundle 带入。验收使用 Python 3.10 和现成 venv。另需约 4.3 GB 解包空间及足够的模型推理内存。

已有兼容环境时，部署直接复用。全新机器可以提前在同架构、同 Python 次版本的机器下载依赖 wheel 到本地目录，再随包迁移。CPU 可预先下载 PyTorch 的 CPU wheel；CUDA 环境选与驱动兼容的 PyTorch wheel，然后补齐 `requirements-text-runtime.txt` 及其依赖。模型归档不承载 Python wheel。部署阶段自动创建 venv 时，pip 强制使用 `--no-index`，并从指定 wheelhouse 安装。

```bash
# 复用现有 Python 环境
bash scripts/deploy_runtime.sh \
  --bundle /srv/incoming/aigc-text-runtime-20261124.tar.gz \
  --device cpu --api-port 8611 --ui-port 8612 --venv /srv/venvs/aigc-text

# 或从已带入的 wheelhouse 创建独立 .venv
bash scripts/deploy_runtime.sh \
  --bundle /srv/incoming/aigc-text-runtime-20261124.tar.gz \
  --wheelhouse /srv/incoming/wheelhouse
```

解包至代码根目录的 `runtime_assets/`。脚本在 `runtime_state/` 渲染绝对路径配置，并根据新机权重路径重新计算评分协议签名。温度和阈值的数值保持与发布产物一致。数据库与指纹库首次启动在 `data/` 下创建。预处理开关默认关闭。显式启用预处理会改变评分协议；生成匹配的新温度和阈值产物后才可输出可比的判定。

FastAPI 与 Streamlit 都监听回环地址。启动需通过两个健康检查、中英文检测和四路归因检查才返回成功。运行时设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，使用新目录下的独立 HF 缓存位置。日志在 `runtime_state/api.log` 和 `runtime_state/ui.log`，实测摘要在 `runtime_state/smoke.json`。

```bash
# 本机访问工作台
ssh -N -o ExitOnForwardFailure=yes -L 8612:127.0.0.1:8612 new-host
```

## 切换、停止与回退

- 用 `bash scripts/deploy_runtime.sh --stop` 停止。脚本核对记录的启动时间、命令和工作目录后发送 SIGTERM，并保留资产、配置和数据库。
- 切换 GPU 时先停止，再用相同命令及 `--device cuda:0` 启动；新环境需能在 PyTorch 中检测到 CUDA。回到 CPU 时同样先停止，再用 `--device cpu`。
- 启动或烟雾测试失败时，脚本停止本次启动的进程，保留日志供排查。端口冲突时可指定另外两个端口。若要换其他版本，先停止，在另一个代码目录部署对应包；保留原目录作为回退入口。
- 该运行包聚焦文本检测与归因。图片、音视频权重及旧历史数据库由各自的迁移方案处理。
