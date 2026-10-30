# 图片模型接入与API交接（2026-10-30）

本文件是**拟实施契约与接入要求**，不是已经实现的服务。页面规范见[页面需求](IMAGE_UI_REQUIREMENTS_2026-10-30.md)，可核对资产见[JSON清单](IMAGE_MODEL_INVENTORY_2026-10-30.json)。清单本身仅为文档证据，不是应用当前已经读取的runtime配置。

## 1. 环境与现有调用链

只通过ssh school读取或修改研究工作树 /home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline，分支attribution-pilot-baseline。本版证据基点为881618c7d9075750283fa068c188caf48ee75968；实现前再次检查Git状态，保留他人修改。不改sync生产树、线上文本服务/数据库/模型，不停服务或其他用户进程。原实验和定时任务已完成，不恢复训练监督器。

已存在：
- api/main.py：FastAPI app，挂载/api/v1；startup直接init_db()，/health仅表示进程健康。
- api/routes.py：POST /analyze/text、POST /analyze/file（当前仅file参数）、GET /providers、GET /report/{content_id}；/detect复用上传，/provenance再抽取旧响应字段。
- api/analysis.py::AnalysisService：ingestion→DetectionEngine→ProvenancePipeline→report→storage；get_analysis_service()当前固定默认config.yaml。
- detection/providers.py::detect：非文本通用分支会平均可用provider；无分数时0/human。local模型入口目前主要是文本。
- detection/image_detector.py是未训练scaffold，不是本次可上线模型。
- storage/database.py默认读根config.yaml；DetectionResult.details为JSON，可承载版本化图片快照，但当前_store_analysis只保存detection。
- api/dashboard.py复用偏文本的render_analysis，request_with_progress有固定伪进度。现有tests/test_dashboard_integration.py使用Streamlit AppTest。
- 当前没有已实现的图片capabilities、独立图片结果契约、历史分页目录或实验聚合目录端点。

最小实现建议：新增独立图片适配/编排模块，例如detection/image_service.py（**拟新增文件**），由AnalysisService在真实图片类型校验后进入；文本继续原链。routes负责HTTP，图片模块负责模型/分数与状态，dashboard使用独立图片renderer。不要直接把图片送进平均provider或旧文本/PHASH家族归因链，也不为此重构文本算法。

## 2. 默认bundle与权重身份

固定bundle_id=image-standard-q179-v1：
- det-openclip-q179-v1：D-OpenCLIP clean，classes严格为[real,ai]，score=softmax[:,1]，threshold=0.5。
- attr-resnet50-q179-v1：A0最终22类ResNet50，classes严格按checkpoint数组，不重排或自建版本映射。
- 22类主接口unknown固定unavailable/null。

| 组件 | 研究树相对路径 | SHA256 |
|---|---|---|
| OpenCLIP完整原始权重 | outputs/image_experiments/assets/openclip-vit-l-14-openai/open_clip_model.safetensors | 9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e |
| D-OpenCLIP检测头 | outputs/image_experiments/openclip-q179-20261029/D-OpenCLIP/linear/head.pth | 0432c050a2c9afcb1845e31651586032e5672f0e4096713fdd4a9f0d64ea4bbb |
| A0 ResNet50完整最终权重 | outputs/image_experiments/a0-resnet50-iab-v7-200-q179-e30/model.pth | c3756dd56fbd3d151ebda8db69b683fd8c78cd773a9967af6e70a3565891562b |

class map摘要为UTF-8的json.dumps(classes,sort_keys=True,separators=(",",":"))后SHA256，与现有openclip_route.digest一致：
- 检测：fb8ea606be8ff22177301d34bccc0184eadaab0c2a062f9bc77a81c44d3a390f
- 来源22类：36592d0f921fd1c2fdf7bd6d62ad8fd2d0a8653349e7bdf01f0a1e5e007788b7

A0 classes：
4o, CogView3_PLUS, FLUX, KANDINSKY, PIXART, PLAYGROUND_2_5, SD1_5, SD2_1, SD3, SD3_5, SDXL, dalle3, gemini, grok3, hidream, hunyuan, ideogram, infinity, janus-pro, kling, mid-5.2, mid-6.0。

资产允许使用研究树中已经登记的共享assets目录符号链接，但必须同时检查允许根、大小和SHA；不可放行任意目录逃逸、用户提供路径或..。文件缺失/哈希或类别不符即不可用，不能下载或使用名字相近权重替换。

## 3. 精确推理步骤，不能复用错预处理

### D-OpenCLIP

复用research/image_attribution/openclip_route.py::load_backbone(root,config,device)；config可从已验收D-OpenCLIP/provenance.json取得asset与dependencies并先检查文档清单哈希。该函数使用pretrained=None、safetensors及完整模型strict load，返回visual和transform。必须是ViT-L-14-quickgelu，不是普通GELU、ViT-L/14-336或其他LAION权重。

预处理：8-bit RGB → 短边Resize224/bicubic → CenterCrop224 → float32 ToTensor → mean[.48145466,.4578275,.40821073] / std[.26862954,.26130258,.27577711]。保持原RGB像素策略，模型内无增强。

visual输出768维，按行L2后进入Linear(768,2)，softmax列1为p_ai。可复用scripts/run_image_openclip.py::load_head(path,["real","ai"],device)与probabilities(head,features,device)，只调用函数，不执行main/--execute。head内preprocessing标签per-branch-l2-concat-l2-v1只说明特征处理，**不能单独代表图像Resize/mean/std**；图像变换以backbone/provenance为准。

### A0 ResNet50

复用research/image_attribution/models.py::load_checkpoint(path,device)与resnet_transform()。现有结构是backbone ResNet50的fc=Identity，外接2048→22 fc；不能把state_dict直接塞入另一个torchvision标准fc结构。最终model.pth已经包含完整训练权重，在线推理无需原ImageNet初始化文件，也不允许随机初始化兜底。

预处理：RGB → 短边Resize256（该版本torchvision默认BILINEAR）→ CenterCrop224 → ToTensor → mean[.485,.456,.406] / std[.229,.224,.225]。不套用CLIP mean/std，不在2048特征上加L2。softmax(logits)给出22类条件分数。稳定排序按(-score,checkpoint_class_index)，TopK不重归一化。

两个模型共享“已校验解码出的RGB图”即可，不能共享归一化tensor。均eval()+inference_mode()、FP32、关闭TF32；不得擅自AMP/FP16/量化/替换resize来压缩延迟。

### 门控与结果组合

1. 输入解码通过后运行默认检测。
2. 原始p_ai>=0.5才自动运行A0；p_ai<0.5时attribution.status=skipped、reason_code=below_detection_threshold、top_k=[]。
3. 检测异常/不可用返回错误；不能运行来源后伪造检测成功。检测ok而A0失败返回partial，保留真实检测分数。
4. 不融合真假分数与来源softmax，不用来源高分反改真假结论，不用1-maxsoftmax或TopK余量生成主22类unknown.score。
5. 文本原有deep_provenance_threshold=0.6不控制图片的0.5门控。图片结果不经过旧demo/API provider平均、文本家族规则或PHASH匹配来替代生成器归因。

## 4. 新增API目标契约（全部为待实现）

优先兼容扩展POST /api/v1/analyze/file，不另起v2平台。现有文本请求字段和响应保持兼容；图片新增image_result为权威对象，旧detection/provenance/report只作一致的兼容投影。

请求multipart：
| 参数 | 约束 |
|---|---|
| file | 必需，既有字段 |
| mode | 拟新增，standard默认；research需明确选择 |
| research_method_id | 拟新增，standard时省略；research时一个后端allowlist ID。一次最多一个 |
| request_id | 拟新增，可由前端生成UUID；缺省由后端生成，用于追踪/同进程去重 |

兼容别名/detect与/provenance应调用同一服务helper并显式传standard模式，不直接调用新增Form默认参数的路由函数；Python普通调用不会获得FastAPI依赖注入后的默认字符串。

图片分支在接收/解码前有界读取并检查字节/像素上限；服务端按真实JPEG/PNG内容识别，不信任filename或Content-Type。识别为图片但不符合输入策略时直接拒绝，不能退回通用demo流程。同步模型计算应通过已有FastAPI/Starlette线程池隔离，并用图片单飞锁限制并发，不阻塞async事件循环；不新建任务队列。

不接受任意checkpoint、目录、设备、alpha或数值阈值。研究阈值只能由已登记方法/策略ID选择，不在上传图上拟合。非图片请求带research模式时明确拒绝，不偷偷改变文本行为。

image_result固定字段：
| 字段 | 类型/含义 |
|---|---|
| schema_version | image-result-v1 |
| request_id / created_at | 字符串UUID / UTC ISO时间 |
| mode / status | standard或research / ok或partial。请求级错误使用非2xx错误对象 |
| bundle_id / registry_sha256 / inference_code_commit | 固定bundle名、实际runtime清单SHA、当前适配代码commit；实验训练commit另存，不混用 |
| input | sha256原字节、media_type、width、height、decode_policy=image-upload-rgb-v1 |
| detection | status、model_id、score、score_kind=binary_softmax_p_ai、label=ai或real、threshold=0.5、threshold_id=fixed-0.5-v1、class_map_sha256 |
| attribution | status=ok/skipped/error/unavailable、reason_code、model_id、label_space=known_ai_22、score_kind=known_ai_source_softmax、conditional_on、top_k[{rank,source,score}]、class_map_sha256 |
| unknown | status=unavailable、score=null、reason_code=not_calibrated_for_22_classes |
| research | []或一个独立方法对象；不得覆盖detection/attribution/unknown |
| runtime | 实际device类别、precision、依赖版本、实际阶段timing；无可用数值时null |
| limitations | 有界字符串数组，至少包含条件来源/开发数据边界；不是由LLM生成的视觉证据 |
| persistence | status=ok或error；记录实际是否保存报告，失败时content_id=null且整体partial |

成功图片响应仍含content_id、filename、modality、metadata、fingerprint、detection、provenance和report。兼容投影中image_result.label=real可映射旧label=human；用户界面使用“真实图片倾向”，不称人类作者已认证。检测没有有效分数时，绝不能投影为0/human；旧/provenance别名也必须遵守相同图片分支与错误语义。

partial示例是检测ok/A0 error；unknown unavailable与低分下来源skipped是正常标准策略，不单独让整体partial。来源模型未接入/失效时capabilities明确缺项；不自动换DNA、融合或外部API。

拟新增最小只读端点：
- GET /api/v1/capabilities：输入限制、默认bundle、方法ID/task、offline_validation、online状态、可用mode、阈值策略、score_kind、是否外传。**不能返回服务器资产路径、凭据或签名URL**。
- GET /api/v1/experiments：封口聚合实验目录，按task/scope分组，含n/epoch/指标/局限/配对报告ID。默认读取Git快照，不遍历输出图片/每图预测。
- GET /api/v1/reports?modality=&status=&limit=&offset=：分页历史摘要；现有GET /report/{content_id}继续详情。
这些端点是目标，不是已存在API；提交前OpenAPI和前后端测试要与此定稿一致。

## 5. 错误、持久化与重试

统一非2xx错误体固定error={code,message,request_id,retryable}；不含stacktrace或私有路径。建议状态码：
- 400 empty_file/decode_failed/unsupported_image_policy
- 413 too_large（字节或像素/单边超限）
- 415 unsupported_media_type
- 422 unsupported_mode/unknown_method_id/invalid_request_id
- 409 request_id_conflict（同ID不同输入/参数）
- 503 model_unavailable/artifact_mismatch/busy
- 500 inference_failed（不允许成功分数兜底）
模型加载中和busy均不返回伪造0分。客户端超时显示“结果暂未收到”，不推断后台已取消，也不自动重提交。

首版同步API、单个worker、有界并发（图片推理串行即可）。前端显式提交并锁定按钮；后端可用同进程有界的request_id结果/在途表，绑定原字节SHA+mode+method+bundle版本。重复相同请求返回同一结果或busy，同ID不同输入409。只缓存结果/在途状态，不全局缓存原图；进程重启不承诺exactly-once，不引入Redis/Celery或分布式队列。

图片完整image_result与report快照可放DetectionResult.details中的新image_result/report_snapshot键，保留原检测字段，不强制创建全新数据库模型。图片Content.file_hash应保存原字节SHA，fingerprint若保留PHASH要另作解释，二者不是生成器指纹。历史详情优先读当时快照，不按当前bundle重建或重算。没有快照的旧报告标legacy，仅渲染已存字段，不能补造当时模型版本。

存储失败不得声称“报告已保存”；可返回已完成推理和persistence.status=error、content_id=null，整体partial并允许导出，不能靠重跑模型掩盖持久化问题。日志使用request_id、耗时、模型ID和输入SHA；不记录图像base64、完整EXIF、凭据或签名地址。临时文件保持finally删除。会话之间不共享上传对象或结果引用，模型只读实例可以全局复用。

## 6. 研究方法接入规则

JSON清单列出的research_catalog均为research_only，默认online未接入。entry_artifacts是主要产物入口，不表示完整依赖闭包；若实现研究上传，必须从该run/provenance与对应已登记配置把编码器、head、类别、预处理、引用库等全部绑定，并重新做实际适配验收。不能只加载head就标ready。

- DNA：research/image_attribution/dna_route.py及run_image_dna.py的加载函数；512维L2、作者wild骨干严格原11类加载后去头、独立22类头。完整图预处理含非正方形CenterCrop128、resize128再512、mean/std .5，不能替换为224。优先研究上传候选。
- ConvNeXt/OmniDFA：复用run_image_convnext.py、omnidfa_route.py对应加载/变换；禁止把OmniDFA双view合成普通单view。OmniDFA只在SHA和静态globals匹配后局部safe_globals([argparse.Namespace])，始终weights_only=True，不能unsafe fallback。
- RGB/FP/A-Fusion：参照predict_image_research.py的验证/纯推理函数；RGB为DINOv2 CLS384，FP为原OFA128；特征融合是各分支L2→concat→L2。FP的历史CPU/GPU缓存一致性失败证据保留，不能假定CPU直接可用或放宽1e-5验收。
- 分数融合：先各自得完整22类概率并按class map对齐，再固定0.5/0.5或0.75/0.25组合；只调用mix_scores这类纯函数，不执行run_image_score_fusion.py的main/fit_weight/calibration流程。没有新的训练权重文件；两个方法有各自方法ID、固定alpha和证据。
- 原OFA配对：load_ofa、ofa_transform、ofa_features、reference_score；固定reference_bank.pt/reference_manifest.json及原fingerprint+pair+denoiser全权重。支持特征在前、查询在后，不能交换；每来源100个训练参考，输出均值配对分数不做跨来源softmax。
- 拒识：ResNet open20、OpenCLIP open20 MSP/原型、OFA20+1均有专用checkpoint和20known class map。OFA21类末列simulated_unknown；接受置信=1−该列，已知候选取前20列，不转为22类。读取原metrics中的已冻结阈值，不拟合新阈值。OFA模拟器只用于此前训练派生未知，不需要为每张上传图运行模拟器来“判断未知”。

研究响应独立结构为{method_id,task,status,score_kind,label_space,outputs,limitations}。outputs按任务分别给出：
- detection：p_ai、阈值和来源、条件结论。
- source_classification：原类序与TopK条件分数。
- source_verification：每来源原配对分数，probability=null、threshold=null（本轮没有校准拒识阈值）。
- unknown_rejection：acceptance_confidence、冻结threshold、accepted/rejected_by_research_rule、已知TopK、验证时的known_retention/correct_and_retained/unknown_false_acceptance；unknown_origin_probability=null。UI说“该研究规则拒绝归入20个已知类”，不说“证实是任意未知生成器”。

研究页面不接受用户上传参考库、训练样本或checkpoint，不扩展标签空间，不用清单名字猜架构家族。

## 7. 运行隔离与ready门槛

优先受控CPU串行预览，但**CPU结果和耗时尚需实际接入验收**，不预设实时性能。某设备未通过验收即不可用；不静默切换精度/设备来返回结果。

若显式使用GPU，只允许已授权物理GPU0，CUDA_VISIBLE_DEVICES=0、逻辑cuda:0；启动模型前实际memory.used<=1024MiB并持有共享outputs/image_experiments/gpu0-research.lock。常驻GPU预览在占用期间持锁，避免后续研究任务并发；不占GPU1，不停止其他进程/服务，忙则返回busy。不会每请求把“包含自身模型的显存”误当作外部忙碌而卡死，GPU门槛用于取得驻留权，随后使用该worker自身有界并发。不自动后台训练。

独立配置必须实际注入AnalysisService、API startup的init_db和dashboard API_BASE，不能只换端口却仍读根config.yaml。预览独立DB/上传临时目录；实现时检查可用端口后登记实际数值。单个API进程复用模型；不要启用reload导致重复加载，也不要每次Streamlit rerun加载torch模型。/health只表示进程活着，capabilities中的每个模型ready还需资产、加载、设备和数值验收通过。

研究环境.venv-image-next的已登记参考版本见清单dependencies（torch2.6.0+cu124、torchvision0.21.0+cu124、open_clip_torch2.32.0、timm1.0.15等）；页面服务环境单独核验。requirements.txt仅下限（如Streamlit>=1.25）不代表已支持某导航API；按实际安装版本选原生实现，不为样式升级线上环境。

## 8. 实施顺序、测试与完成定义

建议改动位置均以实现者实查为准：api/routes.py增加图片可选参数和只读目录；api/analysis.py加图片分支和快照；新图片适配模块封装bundle/错误；api/dashboard.py四页和图片renderer；现有storage/database.py/startup打通显式预览DB。不改研究训练源码来适应UI。

1. 固定本文契约，先增加失败用例：缺权重、哈希/class map错、无分数、损坏/超限/不支持格式、非法method、busy；断言没有0/human成功、无外部API/demo调用、文本行为未变。
2. 接两个默认适配器和门控：原始分数比较.5，低分来源skipped，高分Top3，来源失败partial，检测失败非2xx。mock只验证编排，不能代替真实模型验收。
3. 实际数值验收：至少2张原selection图片（清单给出AI selection ID，manifest锚定）；禁止holdout。分别直接测检测/来源adapter与原参考实现，不让门控跳过来源数值测试。验证原文件身份、全重载、单张/批量、最大绝对分数误差<=1e-5。记录每个实际设备和精度；失败先诊断，不放宽阈值或重训。数值验收不产生新的evaluation模型排名。
4. 使用合成小fixture测上传策略（空/损坏/大像素/多帧/透明/CMYK/EXIF方向）、会话和DB快照。真实selection图/大预测不入Git。
5. 四页AppTest：成功、部分成功、失败、超时、导航往返、换图、清空、重复提交、历史/导出；保留tests/test_dashboard_integration.py及文本/report/provider回归。浏览器实查窄屏、键盘焦点、reduced-motion和可控动效，不以单元测试替代页面验收。
6. 记录冷加载和实际暖推理耗时，分别注明CPU/GPU及是否包含来源；没有测量不承诺p95/SLA。并发测试确认单worker不会重复加载模型或抢GPU。
7. 独立预览交付：默认真实推理可用、所有离线研究结果可读、明确哪些研究上传适配实际完成。给实际启动命令、环境/端口/独立DB、测试结果、源/模型/预处理SHA。只普通提交代码/文档/脱敏证据，模型/数据/上传图不进Git。不自行切换线上流量。

这些是接手模型的待办。本文作者本轮只核对资产和编写文档，没有实现上述API、启动预览、运行新推理或修改生产服务。
