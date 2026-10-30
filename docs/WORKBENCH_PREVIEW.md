# 文本与图片检测工作台 · 隔离预览

更新：2026-10-30（第二轮修正）。分支 `attribution-pilot-baseline`，工作树 `.worktrees/attribution-pilot-baseline`。
实现依据：`docs/IMAGE_UI_REQUIREMENTS_2026-10-30.md`、`docs/IMAGE_MODEL_INTEGRATION_2026-10-30.md`、
`docs/IMAGE_MODEL_INVENTORY_2026-10-30.json`（只作证据，不作运行配置）、`docs/image_research_results/2026-10-30T0444Z-final/`。

预览与线上隔离：独立进程、配置、数据库、临时目录和回环端口。没有修改或重启线上文本服务、sync 生产树、生产数据库和线上模型。没有训练、下载模型或打开 holdout，也没有执行研究脚本的 `main`/`--execute`。

## 启动、停止与访问

服务器（`ssh school`）上：

```bash
cd ~/projects/aigc_detector/.worktrees/attribution-pilot-baseline
.venv-image-next/bin/python scripts/workbench_preview.py start    # 先确认端口空闲；只启动一个常驻 API worker，无 reload
.venv-image-next/bin/python scripts/workbench_preview.py status   # image_state=ready 即可用
.venv-image-next/bin/python scripts/workbench_preview.py stop     # 只停止 .preview/pids.json 记录、且身份核验一致的进程
```

启动失败处理与身份保护（`scripts/workbench_preview.py`）：

- 身份：每个子进程以独立 session 启动，派生后立即记录 PID 和 `/proc/<pid>/stat` 的启动时刻（start ticks）。满足以下全部条件才算存活：
  - 记录值与实时值都有效且相等；
  - 进程不是僵尸；
  - 进程仍是自己 session 的首进程。
  缺失、`null`、PID 不存在或已被复用都判为不存在。第一版里 `None == None` 会把不存在的进程当成存活，本轮已修复。
- 启动：子进程必须在 60 s 内保持运行并监听自己的端口，才写入 `pids.json`。若子进程提前退出（报告退出码和日志路径）、超时未监听或身份核验失败，本次启动失败：已启动的兄弟进程会被停止，不写记录，日志保留在 `.preview/logs/`。
- 残留记录：`pids.json` 中已全部失效的记录会被清除，之后可以重新启动；只要还有一条存活记录，就拒绝重复启动。
- 停止：只向身份核验一致的进程组发送 SIGTERM，最多等待 15 s。已退出或身份不符的记录只报告、不发信号。仍在运行的进程保留在记录中。
- 测试（`tests/test_workbench_preview_launcher.py`）：只使用临时目录和测试自建的子进程，不碰真实线上进程。覆盖空记录、不存在的 PID、快速退出、身份不匹配、僵尸进程、非本启动器的进程组、超时未监听，以及正常启动后停止。

本机建隧道后打开 <http://localhost:18612>（四页：文本检测 / 图片检测 / 实验对照 / 历史报告）：

```bash
ssh -N -L 18612:127.0.0.1:8612 school
```

- 端口：API `127.0.0.1:8611`，UI `127.0.0.1:8612`，只绑定回环地址，可用 `--api-port/--ui-port` 修改。
- 运行目录 `.preview/`（Git 忽略）包含：
  - `config.yaml`：主仓库 config.yaml 与 `configs/workbench/preview.overlay.example.yaml` 合并而成；
  - `preview.db`、`tmp/`、`logs/`、`pids.json`。
- 配置传递：
  - `AIGC_CONFIG=.preview/config.yaml` 供 `AnalysisService`、`ReportGenerator` 和 `init_db` 读取；
  - UI 的 `API_BASE` 来自 `AIGC_API_BASE=http://127.0.0.1:<api-port>/api/v1`。
- live 验收中的 `isolation` 检查确认四点：预览库与生产库不是同一文件；生产库 mtime 未变；`tmp/` 中没有遗留上传图片；数据库中没有原图字节。
- 子进程环境：
  - 干净环境，不带任何 API key；
  - `CUDA_VISIBLE_DEVICES=""`，只用 CPU，GPU0/GPU1 都不占；
  - `HF_HUB_OFFLINE=1`，私有 `TMPDIR`；
  - 外部检测 API 与 LLM 报告在覆盖配置中关闭。
- 依赖：服务器现有 `.venv-image-next`，未安装新包：
  - Python 3.10.20、torch 2.6.0+cu124、open_clip_torch 2.32.0；
  - streamlit 1.56、fastapi 0.136、uvicorn 0.45。
- 模型、数据集、特征缓存都只读引用工作树内 `outputs/image_experiments/`，均不进 Git。

## API（`/api/v1`）

- `POST /analyze/file`：表单字段 `file`，可选字段如下：
  - `mode`：`standard` 或 `research`；
  - `research_method_id`：仅研究模式使用，一次一个白名单方法；
  - `request_id`：UUID。

  图片返回版本化 `image_result`（`image-result-v1`），并保留旧字段兼容；非图片仍走原文本流程，行为不变。
- `GET /capabilities`：模型状态、bundle、注册表摘要、可运行和仅离线的方法、上传限制。
- `GET /reports?modality=&status=&limit=&offset=&q=&created_from=&created_to=`：历史分页。
  - 时间按 UTC 计，区间为 **[created_from, created_to)**：起点包含，终点不包含。
  - 取值为 ISO 8601 日期（`2026-10-30`，即当天 00:00）或带时间的值（`2026-10-30T08:00:00Z`、`…+08:00`）。不带时区的值按 UTC 解释，带时区的值先换算到 UTC。
  - 两端都可省略或留空，也可只填一端。
  - 非法时间，以及开始时间不早于结束时间，都返回 422 `invalid_time_range`。
  - 所有筛选条件同时生效；列表、`total` 和分页使用同一套过滤条件。
  - 参数可选，旧调用不受影响。
  - 使用现有的 `contents.created_at`（UTC），不做数据库迁移；读取历史只还原快照，不重新推理。
  - 历史页按日期选择：结束日包含当天，查询时转为次日 00:00Z 作为上界；任一筛选改变都会回到第 1 页。
- `GET /report/{content_id}`：历史快照。
- `GET /experiments`：封存聚合结果，加载前做哈希校验。
- 响应中不含绝对路径、凭据或签名 URL。
- 错误统一为 `{"error": {code, message, request_id, retryable}}`：

  | HTTP | code |
  |---|---|
  | 400 | `empty_file`、`decode_failed`、`unsupported_image_policy` |
  | 413 | `too_large` |
  | 415 | `unsupported_media_type` |
  | 422 | `unsupported_mode`、`unknown_method_id`、`invalid_request_id`、`invalid_time_range` |
  | 409 | `request_id_conflict` |
  | 503 | `model_unavailable`、`artifact_mismatch`、`busy` |
  | 500 | `inference_failed` |
  | 404 | `report_not_found` |

  出错不返回 0 分、human、demo、pHash 或随机初始化结果。检测成功而来源推理失败时 `status=partial`，检测结果保留。
- 请求去重：
  - `request_id` 绑定输入 SHA-256、mode 和方法；同 ID 同参数直接返回首个结果（`deduplicated=true`），不同参数返回 409。
  - 推理一次一张，最多 `max_pending=2` 个请求被接纳，其余立即返回 `busy`。
  - 容量准入在完整解码之前、以非阻塞方式完成。队列满时新请求直接 `busy`，不再解码（16 MP RGB 缓冲约 48 MB）。去重、冲突检查和已完成结果的复用都在准入之前，所以队列满时同一 ID 仍能取回已完成的结果。
  - 解码失败、等锁超时、推理失败和存储失败都会释放许可与 in-flight 状态。

## 默认 bundle `image-standard-q179-v1`

| 角色 | model_id | 资产 SHA-256 | 类别表 SHA-256 | 预处理 SHA-256 |
|---|---|---|---|---|
| 真假检测 | `det-openclip-q179-v1`（OpenCLIP ViT-L/14 冻结 + 线性头，`[real, ai]`） | backbone `9ce2e8a8…cd207e`，head `0432c050…4ea4bbb` | `fb8ea606…44d3a390f` | `3f716194…bf3210b` |
| 来源候选 | `attr-resnet50-q179-v1`（A0 ResNet50，22 类） | model `c3756dd5…891562b` | `36592d0f…7788b7` | `e44736b2…4e041` |
| 研究：DNA | `research-dna-lp22`（DNA-Det 冻结骨干 + 22 类线性头） | backbone `641cf077…d962d92`，head `d1e5651a…4ed12a7` | `36592d0f…7788b7` | `e0d5b6da…3d317` |

完整哈希、字节数和证据文件见 `configs/workbench/preview.overlay.example.yaml`。当前注册表摘要 `registry_sha256 = 09daa5afd087558d0f71681ddb7d3a7061de1cf97ab468a0fc754087eaf8433e`。

- 检测：`score` 是 `[real, ai]` softmax 的 ai 列（`binary_softmax_p_ai`）。阈值固定为 0.5（`fixed-0.5-v1`），没有灰区。页面上显示为“AI 生成分数”，并注明这是 AI 类的分数，不是校准概率。低分（例如 1.67%）不是“真实图片”的置信度。
- 来源：只有 `p_ai ≥ 0.5` 才运行。
  - 运行时按 checkpoint 类别顺序给出 Top3，全部 22 类可展开，分数不重新归一化，并附说明：“在图片属于AI生成、且来源属于已知22类的条件下，以下标签是模型的候选排序。”
  - 低于阈值时为 `skipped` / `below_detection_threshold`。
- 未知来源：始终为 `status=unavailable`、`score=null`，不会从 22 类分数推导“未知”。
- 加载：
  - 逐个核对资产与代码 SHA-256；
  - `weights_only=True` / safetensors，strict `load_state_dict`；
  - FP32、`eval()`、`inference_mode`；不用 AMP、不量化、不下载，也没有随机初始化回退。
  - 复用研究代码中的 `load_backbone` / `load_head` / `load_checkpoint` / `resnet_transform`，各模型预处理与研究登记逐项一致。
- 准入：只有 `docs/workbench_preview/evidence/adapter_acceptance.json` 对当前资产哈希、在 CPU 上记录为 `passed` 时才加载并提供服务。该文件的 SHA-256 也钉在配置里，任一不符即为 `artifact_mismatch`，不提供结果。
- 上传：服务器端校验，以下任一不满足即拒绝，不做猜测性转换：
  - 静态 8-bit JPEG/PNG，RGB 或 L；
  - ≤10 MiB，≤16 MP，边长 ≤8192；
  - 不接受多帧（含 MPO）、透明通道（RGBA、LA、tRNS）、其他色彩模式，以及 EXIF 方向不为 1 的图片。
- 历史：保存完整版本化快照（`image_result` 加报告快照），不保存原图，只保存文件名（去路径）和 SHA-256。存储失败时 `persistence.status=failed`，页面不会显示“已保存”，但仍可导出 JSON/Markdown。

## 研究模式

一次最多附加一个方法，结果在标准结论下方独立显示，不混入标准结论。

| 状态 | 方法 |
|---|---|
| 可在线运行（真实适配器 + CPU 设备验收通过） | `research-dna-lp22`；`research-a0-assumed-known-ai`（A0 排序，不经检测门控，用于查看“假设为已知 22 类 AI”时的排序）；`research-openclip-calibrated-thresholds`（已登记的 1%/5% 校准阈值 0.7415 / 0.2684 及其实测 FPR） |
| 已有离线结果，在线未接入（21 个） | ConvNeXt / OpenCLIP / ResNet 的 held-FFHQ 检测头、DINOv2、ResNet-control、OmniAID、ConvNeXt / OmniDFA / OpenCLIP 22 类与 KNN、A-RGB、A-FP、A-Fusion、两种分数融合、OFA 配对验证（CPU 复现误差约 5e-4，超出 1e-5 容差）、三种 open20 / OFA20+1 未知拒识 |
| 不提供入口 | 未训练检测 scaffold、demo / 哈希分数、A3（家族映射未核验）、完整 OFA 预训练（未运行） |

## 实验页元数据

- 每个结果行和“方法与接入状态”表都显示：
  - 训练范围：协议，以及 train/selection 数量；
  - 骨干是否冻结；
  - 实际训练 epochs 和选模 epoch。
- 这些值只取自盘点文件中、按哈希钉住的各方法封口证据（`metrics.json`/`provenance.json`/配置）。不重训，不重新评估，也不改原始研究结果。
- 状态分三种：
  - `registered`：显示数值；
  - `not_applicable`：显示“不适用”，用于冻结外部模型 OmniAID、KNN、K10 原型和分数融合，值保持 `null`；
  - `not_registered`：显示“未登记”，用于封口文件没有记录的情况，例如 DINOv2、A-RGB、A-FP、A-Fusion 只登记了选模 epoch，ResNet follow-up 都没有登记。
- 不按 30 轮推测，也不用 0 代替 `null`。
- 证据缺失或哈希不符时，目录为 `partial`，相关元数据显示为未登记。
- clean 与 held-FFHQ、完整 dev 与 evaluation、20 类与 22 类各自分组，分母不同；A0 等“父 dev 参与选模”的说明保留在训练范围和说明列中。
- 实验页顶部常显 4 条关键限制：FFHQ 实际误报、校准后 7/93、OFA 已知覆盖损失 58/1212 与 28/1212、父 dev 选模与单 seed。其余限制和数据范围折叠在一个展开项里，结果表因此上移。

## 验收

- A. 契约与失败测试：
  - `tests/test_workbench_image_engine.py`、`test_workbench_api.py`、`test_workbench_catalog.py`、`test_workbench_ui.py`；
  - 加上文本、报告、provider 回归测试。
- B. 真实模型数值验收：`scripts/verify_workbench_adapter.py`，证据文件 `evidence/adapter_acceptance.json`。
  - 输入：dev split 的 selection 图片 3 张（2 张 AI、1 张真实），未打开 holdout。
  - 检测与来源分别直接核对，不经门控；比较单张对批量、完整重载、登记的特征缓存加 head（或已保存的预测）、同设备研究参考实现，以及服务对直接调用。
  - 最大误差：检测 2.4e-7，A0 1.2e-7，DNA 3.6e-7，均 ≤1e-5。
  - 设备：CPU（i9-14900K），8 线程，FP32，TF32 关闭。
- live 验收：`scripts/accept_workbench_preview.py`，证据文件 `evidence/live_acceptance.json`，共 10 项：
  - capabilities、标准结果对离线结果、研究模式、去重、12 种类型化错误、并发；
  - 历史、实验、文本流程、隔离（生产库 mtime 不变）。
- C. 浏览器检查（桌面 1366×900、窄屏 375×812）：
  - 标准 AI 结果；低于阈值时来源显示“未运行”；DNA 研究项；历史快照还原；RGBA 拒绝；
  - 键盘焦点：按钮获得焦点时有 2px 蓝色描边；
  - 悬停缩放 1.015，过渡 0.18 s；在 `prefers-reduced-motion: reduce` 规则下 transition 为 none；
  - 窄屏单列，无横向溢出。
- 耗时（CPU，单常驻进程）：
  - 冷加载 6.3–8.6 s，完整重载 4.9 s；
  - 温推理：标准约 0.45–0.49 s（检测约 448 ms，来源约 32 ms），DNA 研究约 80 ms。

## 已知限制

- 真实人脸误报：held-FFHQ 同 93 张真实图上，OpenCLIP 固定 0.5 误报 15 张，ResNet 与 ConvNeXt 各 56 张。按 calibration 目标 1% 选阈值后，OpenCLIP 仍误报 7/93。
- 来源结果是条件排序：只在“图片是 AI、且来源属于已知 22 类”的前提下有意义。它不能识别未知来源，也不说明某张图一定由该模型生成。
- 未知拒识：OFA20+1 拒绝了全部 63/63 张未知图，但已知图只保留 58/1212、正确且保留只有 28/1212。因此在线不提供未知来源判断。
- 选择偏差与种子：A0 和部分模型的父 dev 集参与了选模，全部结果都是单 seed 的开发结果，不是独立测试。
- 分数不是概率：开发 evaluation 上的 FPR/TPR 不代表新上传图片上的表现。

## Streamlit 限制

- 动效只能通过 CSS 实现（悬停、按下、焦点、进度条生长），没有页面切换动画。
- 推理在后台线程运行，页面以轮询方式等待，不提供伪造的进度百分比。
- 窄屏下顶部导航折叠到侧栏，属于 Streamlit 自身行为。
- 图片上传控件用 Streamlit 1.56 的逐控件参数 `max_upload_size=10`（Streamlit 的 MB 即 MiB），控件提示与 10 MiB 策略一致。全局 `server.maxUploadSize=20` 不变，仍供文本上传使用。10 MiB 等图片策略最终由服务器端强制。
- 首屏只放输入、操作、真假倾向、AI 生成分数、条件来源候选和必要局限。模型 ID、bundle、设备、分数类型和完整方法说明放在“模型与方法说明”和“技术详情”展开项中。
- 原生控件（selectbox、date_input、tabs、expander）的动效由 Streamlit 自带，无法单独调整。这里没有用脆弱的 DOM 注入，也没有实现完整的 Morph 过渡。
- 结果保存在会话内：同一图片、同一参数在会话中复用已有结果；更换图片或点“清空”会丢弃旧结果和待处理请求，迟到的响应会被丢弃。
