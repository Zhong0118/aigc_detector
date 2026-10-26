# RGB、指纹与融合：实验准备记录

日期：2026-10-26。状态：代码与配置已准备，三组真实实验均未启动。

## 1. 本轮准备的比较

三组回答同一问题：冻结的RGB表征和OFA指纹表征是否互补。仅在AI图上做22个已知生成来源分类，真实图不作为来源类别。真假检测仍由独立D0系列负责。

| 组别 | 输入与冻结编码器 | 分类输入维数 | 可训练头参数量 | 本轮状态 |
|---|---|---:|---:|---|
| RGB-only / rgb | RGB → ImageNet V1 ResNet50 → L2归一化 | 2048 | 45,078 | 准备完成，未运行 |
| 指纹-only / fingerprint | DnCNN残差 → FFT对数幅度谱 → 作者OFA编码器 → L2归一化 | 128 | 2,838 | 准备完成，未运行 |
| 双分支融合 / fusion | 各路先L2归一化，再拼接、整体L2归一化；不搜索分支权重 | 2176 | 47,894 | 准备完成，未运行 |

所有编码器冻结参数并保持eval模式，仅训练线性分类头。RGB组不加载已按来源微调的A0权重。各头参数量不同，比较时一并报告，不能声称容量完全相同。

这是一组项目消融设计。现有A0为端到端ResNet50微调，仍作为独立基线；OFA原基线为冻结指纹和作者配对头的参考来源验证。这里为指纹添加22分类头是新适配，不能称为OFA原论文的完整分类复现，也不能把其准确率与配对验证AUC直接比较。残差表征不保证完全排除语义信息。

## 2. 数据与划分

| 数据版本 | 总图数 / AI图数 | 准备状态 |
|---|---:|---|
| v5-50 | 22,000 / 11,000 | 绑定现有selection和manifest哈希，已固定开发角色ID |
| v6-100 | 目标42,000 / 22,000 | 仅绑定现有selection；等待下载及父划分校验完成，再由后续授权绑定最终manifest哈希 |

不生成新的train/dev/holdout划分。原train保持不变；沿用已有的`sha256(42:group_id)%3`规则，将原dev分成selection、calibration、evaluation。整个组进入同一角色。calibration预留而不使用；holdout只允许元数据核验，不读取图像、不提取特征、不产生预测。

50张版的AI角色数量：

| train | selection | calibration（预留） | evaluation |
|---:|---:|---:|---:|
| 7612 | 440 | 594 | 462 |

三组共享相同的角色ID、来源词表和逐样本顺序。100张版不猜测尚未冻结的manifest哈希；即使稍后文件出现，也保持待人工绑定，不自动启用。

## 3. 固定训练及评估协议

| 项目 | 固定设置 |
|---|---|
| 训练对象 | 仅线性头；全部编码器冻结，无训练图像增强 |
| 训练参数 | 20 epochs、batch32、seed42、Adam(lr=0.001, weight_decay=0)、StepLR(step=1, gamma=0.9) |
| 损失 | 无类别权重的交叉熵 |
| RGB预处理 | Resize256 → CenterCrop224 → RGB tensor → ImageNet标准化；复用项目现有实现 |
| OFA预处理 | 最短边不足256时缩放，中心裁剪256；DnCNN逐图microbatch1，残差/256，log1p(abs(FFT2))；FP32、关闭TF32；复用项目现有实现 |
| 提取批量 | RGB32，OFA8（DnCNN内部仍逐图处理） |
| 每轮记录 | train与selection的loss、Top1、Top3、macro-F1、类别数量、学习率、耗时 |
| 选模 | selection macro-F1最大；相等时保留较早epoch |
| 最终评估 | 仅对选出的头评估一次evaluation；输出Top1/Top3/macro-F1、逐来源/语义指标、混淆矩阵和逐样本分数 |
| 后续比较 | 严格对齐同一版本evaluation IDs/标签/词表，再比较三组；如计算差异区间，应以group_id做配对重采样；不使用holdout |
| 数值核验 | 最佳头保存、严格重新加载；同批原头/重载头误差与单图/批量推理误差 |
| 可追溯性 | resolved config、代码/权重/manifest/selection/角色ID哈希、seed、退出状态 |

特征只提取train、selection和evaluation中的AI图。原始RGB与指纹特征分别缓存，按文件哈希、预处理、代码和有序sample_id绑定；融合复用这两份缓存。身份不一致时拒绝复用，不覆盖旧输出。新模型为`head.pth`，不冒充现有生产ResNet模型。

父dev已经影响过此前的实验和选模，所以该评估仍为探索性开发结果。单seed和不同头维度也是比较的限制，不能当作独立测试或SOTA证明。

## 4. 代码与执行门槛

研究工作树：`/home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline`，分支`attribution-pilot-baseline`。

| 文件 | 用途 |
|---|---|
| research/image_attribution/feature_ablation.py | 元数据绑定、冻结/融合、线性头训练和评估、缓存身份检查 |
| scripts/run_image_feature_ablation.py | 独立入口，默认只输出元数据计划；不加入任何自动训练流水线 |
| configs/image_experiments/iab_feature_ablation.json | 三组共享的注册配置、资产与代码哈希、数据绑定、禁用开关 |
| tests/test_image_feature_ablation.py | CPU合成输入及执行门槛验证 |
| outputs/iab-feature-preparation-20261026/ | 准备快照、50张版角色IDs、100张版待绑定记录、测试和核验日志 |

默认`execution_enabled=false`且`automatic_execution=false`。即使传入`--action run --execute`也必须拒绝，不创建实验输出。未来运行还要求人工启用、明确执行参数、最终manifest哈希绑定、50/100张两条20轮基线完成标记、预训练权重/代码哈希一致，以及共享GPU锁可用。使用物理GPU1，不干扰GPU0上的Ollama。

只读查看示例（在研究工作树及研究环境中）：

```bash
.venv-image/bin/python scripts/run_image_feature_ablation.py --dataset v5-50 --variant rgb
.venv-image/bin/python scripts/run_image_feature_ablation.py --dataset v6-100 --variant fusion
```

实际运行后的独立目录约定为`outputs/image_experiments/feature-ablation-<版本>/<rgb|fingerprint|fusion>/`。目前这些训练输出目录不存在。

## 5. 本轮验证与现有任务

- 新增14项CPU测试；其中分类头训练/保存/重载仅使用合成特征与临时目录，不属于真实图片实验。
- 连同现有模型、后续协议、epoch参数及OFA microbatch回归，共57项通过。
- 元数据准备时对PIL打开图像、torch加载权重及CUDA可用性调用设为报错；准备流程通过，未读取图片或模型。
- 对12个受保护的现有脚本、配置、selection/manifest做前后哈希对照；最终核验保存在verification.json。
- 不新增依赖，不覆盖原模型、历史resolved配置或研究工作树的既有未提交修改。
- 100张版下载与50张版20轮D0/A0/OFA沿原流水线继续。30分钟定时跟进仍只推进已有基线；新三组不会因下载或基线完成而自动启动。等待100张版结果后再由用户决定先运行哪组。
