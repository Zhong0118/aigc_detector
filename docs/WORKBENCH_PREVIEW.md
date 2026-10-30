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
- 启动：子进程必须在 60 s 内保持运行并监听自己的端口，才写入 `pids.json`。所有子进程启动后，返回成功前还会统一复查每个子进程：仍在运行、身份一致、仍在监听，因为较早启动的 API 可能在 UI 启动期间退出。若子进程提前退出（报告退出码和日志路径）、超时未监听、复查不通过或身份核验失败，本次启动失败：已启动的兄弟进程会被停止，不写记录，日志保留在 `.preview/logs/`。
- 残留记录：`pids.json` 中已全部失效的记录会被清除，之后可以重新启动；只要还有一条存活记录，就拒绝重复启动。
- 停止：只向身份核验一致的进程组发送 SIGTERM，最多等待 15 s。已退出或身份不符的记录只报告、不发信号。仍在运行的进程保留在记录中。
- 测试（`tests/test_workbench_preview_launcher.py`）：只使用临时目录和测试自建的子进程，不碰真实线上进程。覆盖空记录、不存在的 PID、快速退出、身份不匹配、僵尸进程、非本启动器的进程组、超时未监听、API 通过初检后在 UI 启动期间退出或停止监听、UI 启动失败时 API 被清理、启动失败不写 `pids.json`，以及正常启动后停止。

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
  - 非法时间、换算到 UTC 后超出 0001-01-01 至 9999-12-31 的时间（例如 `0001-01-01T00:00:00+14:00`），以及开始时间不早于结束时间，都返回 422 `invalid_time_range`（`retryable=false`）。不截断或替换用户给的时间。
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
- 证据缺失或哈希不符时，目录为 `partial`，并为该方法给出 `artifact_mismatch` 提示，该文件不再作为来源。仍有效的其余证据和已通过哈希校验的 inventory 中明确登记的字段继续显示，例如 A0 的 30/29 来自 inventory 本身。没有任何有效来源的字段显示“未登记”。inventory 自身哈希不符时整份不用，其中的数值不会当作登记值。SHA 校验不放宽。
- OpenCLIP open20 原型行：拒识分数是与 train 特征类原型（L2 归一化均值）的余弦，原型本身没有训练轮次，训练 / 选模 epoch 为 `null`（不适用）。该 run 登记的 30/29 属于配套 open20 线性分类头。这个分类头同时给 MSP 行和原型行提供已知类预测（“正确且保留”），因此单列为“配套分类头 epoch”，并在表下说明。MSP 行仍显示 30/29。
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

### 第二轮修正验收（2026-10-30）

上面的 A–C 是首版（f4709a5 / 842a03f）的记录，保留不改。第二轮的证据放在新的时间戳目录，旧证据文件未覆盖。

- 提交对应关系：
  - `2f5c876`：主体代码（启动器存活判定、解码前准入、历史时间筛选、实验元数据、提示文案）。
  - `54654d8`：浏览器验收中发现的小问题——快速双击“开始分析”时第二次回调替换了待处理任务，页面显示 busy 而不是结果（服务端只推理一次）；导出 Markdown 中未运行阶段的耗时显示为 None；导航链接缺少焦点描边。
  - 本节与证据所在的提交：只含文档和证据，不改代码。预览运行的代码与新报告的 `inference_code_commit` 都是 `54654d8`。
- 证据目录：
  - `evidence/2026-10-30T1454Z-round2/`：`adapter_acceptance.json`（真实模型数值复核）和 `live_acceptance.json`（在 `2f5c876` 上运行）。
  - `evidence/2026-10-30T1515Z-round2-fix/live_acceptance.json`：在 `54654d8` 上运行，`--expect-commit 54654d8…`。
- live 验收 `workbench-live-acceptance-v2` 共 13 项，两次均全部通过。新增三项：
  - `recovery`：失败后能恢复；
  - `code_version`：新结果的 `inference_code_commit` 等于期望提交，且不带 `+dirty`；
  - `history_time_filter`：边界、组合条件、反向区间。
- 真实模型回归：只用登记过、manifest 确认、SHA 已核对的非 holdout selection 图片，最大误差 ≤3.6e-7（≤1e-5），没有重跑数据集评估。
- 测试：
  - 命令：`.venv-image-next/bin/python -m pytest -q tests/test_workbench_image_engine.py tests/test_workbench_api.py tests/test_workbench_catalog.py tests/test_workbench_ui.py tests/test_dashboard_integration.py tests/test_report_and_providers.py tests/test_workbench_preview_launcher.py`
  - 结果：140 passed。
  - 全量 `tests/`：只有 `test_supports_sklearn_cloning_string_tags_and_pipeline_fit_transform`（`AttributeError: _get_tags`，sklearn 版本问题）失败。在干净的 `git archive 842a03f` 上同样失败，属于既有失败，与本轮无关，未屏蔽。
- 浏览器检查（通过 SSH 隧道访问 `http://localhost:18612`，桌面 1366×900 与 375×812）：
  - 图片：AI 图 96.45%；真实图 1.67%，来源显示“未运行”并附分数说明；双击只产生 1 次 `POST /analyze/file` 和 1 条报告；换图复用会话内结果；导航往返后结果保留；“清空”有效。
  - 输入错误：截断的 PNG 返回 `decode_failed`；11 MiB 文件在上传控件内被拒，服务器端 413 由 live 验收覆盖。
  - 研究模式：DNA 结果单独展示。
  - 实验页：表格不被说明挤到下方；epoch、训练范围列正确；FFHQ 误报和未知拒识的覆盖损失保持可见。
  - 历史：
    - 开始日和结束日都设为 2026-10-30（结束日含当天）时为 41 条、3 页；反向区间显示明确错误。
    - 组合条件：日期加类型“文本”为 4 条；日期、类型“图片”、状态“完成”加关键字 `ai` 为 5 条，与 API 一致。“部分成功”给出空结果提示。
    - 在第 2 页改变筛选会回到第 1 页。快照能还原。
    - JSON 和 Markdown 导出在服务器端用同一导出函数生成：JSON 有效，含 `inference_code_commit`；Markdown 使用“AI 生成分数”。为避免在浏览器下载文件，没有点击下载按钮。
  - 文本页能正常显示；文本流程由 live 验收的 `text_flow` 覆盖。
- reduced-motion：用本机 Google Chrome（Playwright 无头模式，`reducedMotion: 'reduce'`，375×812）实际开启。
  - `matchMedia('(prefers-reduced-motion: reduce)')` 为 true，页面上有过渡或动画的元素从 6 个降到 0 个。
  - 同次 Tab 遍历：导航链接在修复前没有可见焦点。`54654d8` 给它们加了 2px 蓝色 `:focus-visible` 描边，其余按钮和上传控件已有描边或阴影。
- 隔离：
  - 预览使用独立的 `.preview/config.yaml` 和 `.preview/preview.db`，只用 CPU（预览进程未出现在 GPU 进程列表中），外部调用关闭。
  - 生产库 `data/aigc_detection.db` 的 mtime 保持 2026-09-09 17:18:34，sync 生产树 HEAD 未变。sync 树中 16 个未提交修改的 mtime 都不晚于 2026-09-09，为既有修改，本轮未触碰。
  - 研究结果与 inventory 无改动。
- 重启方式：先用 `status` 确认 pid、start_ticks 与 `/proc` 一致、进程组首进程等于 pid，且端口由该 pid 监听；再用 `stop`、`start` 重启。没有使用 pkill，也没有按端口停止进程；8611 和 8612 未被其他进程占用。

### 第三轮定向修正验收（2026-10-30）

基线为 `217ee1a`（仅文档与证据，运行代码为 `54654d8`）。本轮只修四项已知问题，不训练、不下载、不重跑研究或数据集评估。旧证据目录全部保留。

- 提交对应关系：
  - `c6169bc`：四项修正及其回归测试，以及验收脚本新增的检查。
  - `827fb10`：真实预览中发现的文案问题。原型说明里括号套括号，已改写，UI 测试同时拒绝嵌套括号。
  - 本节与证据所在的提交：只含文档和证据。预览运行的代码、最终验收中新报告的 `inference_code_commit` 均为 `827fb10`。
- 四项问题，均先写出能复现故障的测试，在未修改的代码上确认失败，再修复：
  - P2-1 启动器：修复前，API 通过初检后在 UI 启动期间退出（退出码 7）或停止监听，`launch()` 仍返回成功（pytest：`DID NOT RAISE StartupFailed`）。修复后，返回成功前统一复查所有子进程：仍在运行、身份一致、仍在监听。任一不满足即启动失败，清理本次创建的全部子进程，不写 `pids.json`，日志保留。新增测试覆盖：退出与停止监听两种情形（UI 一并被清理）、UI 启动失败时 API 被清理、启动失败不写记录。
  - P2-2 原型 epoch：修复前，`/experiments` 的原型行显示 30/29，骨干写“冻结骨干，只训练头”。修复后，原型自身的两个 epoch 均为 `null`（`not_applicable`，类原型 = train 特征均值）。30/29 移到单独的 `support_head`，即配套 open20 线性分类头，它为原型行提供已知类预测（研究脚本中 `prototype_cosine` 的“正确且保留”使用线性头的 argmax）。页面显示为“配套分类头 epoch”列、“说明”列和表下说明。OpenCLIP MSP 行和方法表仍为 30/29。各行元数据改为深拷贝，测试证明修改一行不影响其他行，也不影响下一次构建。
  - P3-1 UTC 溢出：修复前，`created_from=0001-01-01T00:00:00+14:00` 与 `created_to=9999-12-31T23:59:59-14:00` 在实时 API 上都返回 HTTP 500。修复后只捕获 `OverflowError`，返回 422 `invalid_time_range`（`retryable=false`），提示中写明字段名和可表示范围，不暴露堆栈或路径，也不截断时间。
  - P3-2 证据回退：修复前，提示称“相关元数据按未登记显示”，但实现仍保留 inventory 中登记的 epoch。修复后提示与文档改为如实描述（见“实验页元数据”）。新增测试：子证据被篡改时 A0 仍保留 inventory 中的 30/29，无有效来源的字段显示未登记；inventory 被篡改时整份不用，篡改值 31 不出现。
- 测试：
  - 规定命令加启动器测试：**153 passed**（基线 140，新增 13）。启动器测试连续 3 次各 21 passed。
  - 全量 `tests/`：1275 passed、1 failed。失败项仍是 `test_supports_sklearn_cloning_string_tags_and_pipeline_fit_transform`（`AttributeError`，sklearn 版本），在干净的 `git archive 217ee1a` 上同样失败，未屏蔽。
  - 一条既有断言随需求更新：原型行原先断言旧说明字符串，现改为断言 `not_applicable` 加 train 特征说明，要求更严格。
- 证据：
  - `evidence/2026-10-30T1556Z-round3/`：`adapter_acceptance.json` 和 `live_acceptance.json`（在 `c6169bc` 上运行）。
  - `evidence/2026-10-30T1605Z-round3-final/live_acceptance.json`（在 `827fb10` 上运行）。
  - 真实模型数值复核使用同 3 张登记过的 dev selection 图，SHA 与 manifest 核对一致，未打开 holdout，CPU，最大误差 3.6e-7（≤1e-5）。
  - live 验收两次均 13/13 通过。`history_time_filter` 新增溢出 422 和偏移时区等价检查，`experiments` 新增原型、分类头与 MSP 的 epoch 检查。
- 浏览器检查（桌面 1366×900；375×812 用本机 Chrome 无头模式）：
  - 实验页：原型行为“不适用 / 配套分类头 30/29”，MSP 行为 30/29，ResNet MSP 为“未登记”，表下说明可见，375 下同样可见。
  - 历史：仅结束日、开始日与结束日同为一天、反向区间报错、空结果、日期加类型“文本”（7 条）、第 2 页改变筛选后回到第 1 页；文本快照与图片快照均能还原。
  - JSON 与 Markdown 导出在服务器端用同一导出函数核对（未点击下载）。
  - 图片：
    - AI 图 100.00% 和 96.45%，Top3 为原始 softmax；
    - 真实图 1.67%，来源未运行；未知来源为 null；
    - DNA 研究结果单列，不改写标准结论；
    - 截断 PNG 返回解码错误，11 MiB 文件在上传控件内被拒；
    - 双击只产生 1 条报告；
    - 待处理时上传控件锁定，不会串图；
    - 导航往返结果保留，“清空”有效。
  - 375 无横向溢出。Tab 遍历中所有可操作控件都有焦点描边或阴影，唯一例外是 Streamlit 自带的主内容滚动 `section`。
  - reduced-motion 实际开启后，有过渡或动画的元素从 6 个降到 0 个。
- 文本：检测与报告正常返回（例如 high_risk_ai 0.9997），快照能还原。文本来源分支 `window_a0_equal_fusion` 的状态为 `error`，原因是 “resolved recipe hash does not match recipe content; legacy window artifacts require an explicit audited migration”。预览中从第一条文本报告（05:55Z，f4709a5 时期）到本轮全部 7 条都相同，且 `217ee1a..HEAD` 未改动文本路径代码，属于既有问题，本轮按要求不改文本模型。因此不能宣称完整文本来源能力正常。
- 隔离与重启：
  - 每次重启前用 `/proc` 核对 pid、start_ticks、进程组和端口归属，全部一致才执行 `stop` 与 `start`。第一次重启时独立核对脚本因语法错误未输出，但 `stop` 自带的身份校验仍然生效，停止的正是本启动器记录的两个进程。
  - 生产库 mtime 仍为 2026-09-09 17:18:34。sync 树 HEAD 最后一次移动在 2026-09-08。研究结果、inventory 与旧证据无改动，预览进程不在 GPU 上。本用户可见的其他监听服务只有 2026-10-26 启动的 openclaw-gateway，未触碰。
- Streamlit 原生限制：鼠标悬停在表格上时，表格工具栏图标可能暂时盖住紧邻上方说明的末尾，移开即恢复。

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
