# 文本与图片检测工作台 · 隔离预览

更新：2026-10-30。分支 `workbench-preview`（基于 `attribution-pilot-baseline` d595a91），工作树
`.worktrees/workbench-preview`。预览独立于线上服务：独立进程、端口、配置、数据库和临时目录；未重启或修改线上文本服务、sync 树、生产库或线上模型；未训练、未下载模型、未访问 holdout。

## 启动与访问

在服务器上（`ssh school`）：

```bash
cd <仓库根目录>/.worktrees/workbench-preview   # 服务器上的 aigc_detector 仓库
../attribution-pilot-baseline/.venv-image-next/bin/python scripts/workbench_preview.py start    # 先检查端口空闲
../attribution-pilot-baseline/.venv-image-next/bin/python scripts/workbench_preview.py status   # image_state=ready 即可用
../attribution-pilot-baseline/.venv-image-next/bin/python scripts/workbench_preview.py stop     # 只停止自己记录的 PID
```

在本机建隧道后打开 <http://localhost:18612>（四页：文本检测 / 图片检测 / 实验对照 / 历史报告）：

```bash
ssh -N -L 18612:127.0.0.1:8612 school
```

- 端口：API `127.0.0.1:8611`，UI `127.0.0.1:8612`（只绑定回环地址，`--api-port/--ui-port` 可改；启动前用 bind + `ss` 双重确认空闲）。
- 运行目录 `.preview/`（Git 忽略）：`config.yaml`（config.yaml + `configs/workbench/preview.overlay.example.yaml` 合并）、`preview.db`、`tmp/`、`logs/`、`pids.json`。
- 子进程使用干净环境：无任何 API key、`CUDA_VISIBLE_DEVICES=""`（登记的设备策略为 CPU，不占 GPU0/GPU1）、`HF_HUB_OFFLINE=1`、私有 `TMPDIR`。外部检测 API 与 LLM 报告在覆盖配置中关闭。
- 依赖：服务器现有 `.venv-image-next`（Python 3.10.20、torch 2.6.0+cu124、open_clip_torch 2.32.0、streamlit 1.56、fastapi 0.136、uvicorn 0.45）；未安装新包。模型与数据只读引用研究工作树 `attribution-pilot-baseline`，均不进 Git。

## 后端

- `POST /api/v1/analyze/file`：图片走 `workbench/image_engine.py`，返回旧字段并新增 `image_result`（schema `image_result.v1`）；可选表单字段 `mode=research`、`request_id`（同 ID 去重，不重复推理）。非图片仍走原文本流程。
- `GET /api/v1/capabilities`、`GET /api/v1/history`（分页与类型/状态/结论/关键词筛选）、`GET /api/v1/history/{analysis_id}`、`GET /api/v1/experiments`。
- 错误统一为 `{"error": {code, message, request_id, retryable}}`：`invalid_input` 400、`too_large` 413、`decode_failed` 422、`model_unavailable`/`artifact_mismatch` 503、`busy` 429、`inference_failed` 500、`report_not_found` 404。模型不可用时报错，绝不回退到 `weights=None` scaffold、demo、pHash 或 0 分。
- 默认检测：D-OpenCLIP（冻结 ViT-L/14 QuickGELU + 30 轮线性头，`[real, ai]`），固定阈值 0.5，显示为“模型检测分数”，不是校准概率。
- 默认来源：只有检测分数达到阈值时给出 A0 ResNet50 22 类 Top3（可展开全部），附条件说明；类名按 checkpoint 顺序。未知来源恒为 `status=unavailable, score=null`。
- 研究模式：额外给出 1%/5% 校准阈值及其 calibration/evaluation 实测 FPR；检测低于阈值时 A0 闭集分数只在研究区显示；融合与 open20 拒识只展示登记结果。
- 加载：所有产物 SHA-256 与登记值逐一核对，`torch.load(weights_only=True)`/safetensors、strict 加载；D-OpenCLIP provenance 登记的 `openclip_route.py`、`run_image_openclip.py`、`models.py` 哈希也核对，复用其 `load_backbone`/`load_head`/`load_checkpoint`/`resnet_transform`，不执行训练入口。一个常驻模型集，后台线程加载一次；一次只推理一张，最多 1 张排队，超出返回 `busy`。
- 输入：服务器端校验空文件、字节数（20 MiB）、格式（JPEG/PNG/WebP/BMP）、多帧、边长 32–8192、总像素 4000 万、像素模式与损坏。解码与研究加载器一致：`convert("RGB")`，透明通道直接丢弃，不应用 EXIF 旋转（结果中记录方向与透明通道）。
- 历史：`analysis_snapshots` 表保存版本化快照（模型哈希、阈值、预处理、用时），只存显示用文件名（去掉路径）和 SHA-256，不保存原图。

## 页面

- 图片：桌面两栏、窄屏自动堆叠；上传 → 服务端校验 → 真实分析 → 结果，可清除/替换/导出 JSON 与 Markdown。按钮在无文件、模型未就绪或分析中时禁用；分析中只显示真实已等待时间；同一图片+模式复用已存结果；切换页面保留状态；超时不自动重试，手动重试沿用同一 request_id。
- 文本：沿用原渲染（结果渲染从 `api/dashboard.py` 抽到共享的 `api/text_view.py`），旧 Dashboard 固定的 18/42/78/100% 假进度改为真实等待时间。
- 实验对照：检测 / 来源 / 未知拒识 / 路线状态 / 数据来源，四个登记结果文件逐一哈希校验；每行带分母与计数；只有同一样本的配对比较给出差值与 95% 区间，不跨分母比较“提升”。ConvNeXt、DNA-Det、OmniDFA、OFA T+1、A3 只显示状态，不提供上传推理。
- 视觉：白底黑字蓝强调，红色只用于错误；Material Symbols 开源图标，无 emoji；hover/active/focus-visible/disabled 状态，轻量卡片淡入与分数条动画，`prefers-reduced-motion` 时关闭。只注入 CSS，不做 DOM 脚本注入。

## 验证（证据见 `docs/workbench_preview/evidence/`）

- `adapter_consistency.json`（`scripts/verify_workbench_adapter.py`）：2 张登记 selection 图片 + 2 张 AI selection 图片（均为 dev，未打开 holdout），CPU。D-OpenCLIP 与 A0 的单张/批量、两次完整重载、服务端到端、离线参照（D-OpenCLIP 特征缓存 + 头；A0 已存 dev 预测）最大误差均 ≤ 2.4e-7，阈值 1e-5。
- OFA：CPU 与登记 GPU 预测的最大误差约 5.0e-4，超过 OFA 自身登记容差 1e-4，因此预览中 OFA 配对分数关闭（返回 `unavailable` 与原因，并绑定参考库/参考清单哈希），不展示近似值。
- `live_acceptance.json`（`scripts/accept_workbench_preview.py`，对运行中的预览 API）：真实图片线上分数与离线参照误差 ≤ 1e-5；低于阈值的图片不给来源；去重；研究模式；6 种错误输入的类型化错误；并发 6 请求得到 2×200 + 4×429；历史与 404；实验目录分母；文本流程；线上库 mtime 不变、上传未落盘、预览库只有哈希。
- 测试：`tests/test_workbench_image_engine.py`、`test_workbench_api.py`、`test_workbench_catalog.py`、`test_workbench_ui.py`（AppTest）与原 `test_dashboard_integration.py` 全部通过。全量测试除 2 项在未修改研究树上同样失败的环境项（离线 tnews 数据缓存缺失、sklearn 克隆测试）外全部通过。

## 局限

- 所有数值来自单 seed、已查看过的开发角色（IAB v7-200-q179，63,821 张，每格实际 187–200 张，179 张隔离），不是独立外部测试；holdout 未推理。
- 预览只在 CPU 上运行（单图约 0.2–0.5 秒）；GPU 策略未启用，因此 OFA 研究分数关闭。
- 浏览器验收用合成画布图片走完上传流程（内置浏览器无法选择服务器文件）；真实图片的正确性由 API 验收覆盖。分析速度很快，按钮“分析中”状态在浏览器里一闪而过，其行为由代码与 AppTest 覆盖。
- 文本页沿用原文本渲染，其中原有的红色风险配色保持不变，未纳入本次视觉调整。
- Streamlit 的 `data-testid` 选择器（按钮过渡、焦点样式）可能随 Streamlit 版本变化；失效时只影响样式，不影响功能。
