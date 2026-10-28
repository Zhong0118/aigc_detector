# IAB 200张版隔离修订与30轮复测

用户已明确批准“隔离179张新增图后继续”。原始200版的64,000张文件、230份收据及失败状态全部保留，失败原因是新增重复组件桥接旧分区或旧组。

| 项目 | 实际设置 |
|---|---|
| 原始数据 | 44,000 AI +20,000真实，名义每生成来源/语义200张 |
| 隔离对象 | 23个冲突组件中179张新增图；不排除任何旧样本 |
| 有效数据 | 43,821 AI +20,000真实 =63,821；逐格实际数量另存 |
| 父锚点 | v6-100全部42,000个ID、split、group、标签与图像哈希保留 |
| 新组划分 | extension-split-v2，仅新无锚点组按seed42域分离哈希70/15/15 |
| 计算设备 | 物理GPU0，CUDA_VISIBLE_DEVICES=0；使用量超过1024MiB等待，不停止服务或他人任务 |
| 训练 | D0/A0及D-DINO/A-RGB/A-FP/A-Fusion各30轮，batch32、seed42、Adam.001、StepLR(1,.9) |
| OFA | 原配对头与编码器冻结，每来源100训练参考；epoch不适用 |
| holdout | 仅元数据与图片完整性校验，禁止推理、调参、评分 |

采用已有收据与分组校验器，新增curate_iab_quarantine_subset.py执行离线筛选并硬链接，网络请求与新网络字节均为0。原采集证据仍在raw200目录，派生receipt明确标注offline，不能冒充再次下载。脚本先验证排除列表SHA、父成员保护及去除后组件，再创建独占输出；完整字节/像素/pHash复核和manifest冻结仍由原extension preparer执行。

- 数据：outputs/image_experiments/iab-balanced-v7-200-q179
- 状态：outputs/iab-v7-200-q179-e30-monitor；download.json登记真实离线curation子进程的退出码，非网络下载任务
- 离线配置：configs/image_experiments/iab_v7_200_q179_curation.json
- 基线配置：configs/image_experiments/iab_v7_200_q179_e30_run.json
- 特征队列配置：configs/image_experiments/iab_research_v7_200_q179_e30.json
- 基线模型suffix：v7-200-q179-e30；特征队列输出：independent-image-tasks-v7-200-q179-e30-gpu

先完成数据与父锚点验证，再按GPU0空闲门槛顺序跑D0/A0/冻结OFA及四组特征；最后独立登记来源分数融合，仅calibration选权重。100/30对本版30轮须按原dev/evaluation的相同sample_id配对，新完整dev另列，保存逐epoch、逐来源、group bootstrap95%区间。说明179图隔离、旧100分配偏差、类别权重、OFA参考ID和GPU不同等因素，不宣称纯扩容因果或独立测试。D1/D2/A1/A2/A3不自动启动，不扩500。

## 实际恢复登记

离线curation真实子进程1143418退出0，16.18秒完成，网络请求0。99项测试与46项子测试通过；独立只读审查无Critical/Important，路径别名保护已复现并修复。全部保留图像及sidecar均与raw目录同inode硬链接。实际AI小格187–200张，67格因隔离不足200；真实每格仍2000。元数据重算为train37338/dev7189/holdout19294，最终以全量完整性校验后冻结的manifest为准。新selection SHA为1c8dee2e17cc128867562186cee7bc4f8cb82042b26ddb6d5569d703ab77876b。新状态登记还绑定了completed离线curation状态文件、230个派生收据和排除列表；若丢失或篡改会预检失败，不能自动转成网络采集。原失败200版不重置。

## 结果

修订版基线、四组特征、来源分数融合及100/30同样本比较已完成并核验，见 [IMAGE_200_Q179_RESULTS_2026-10-28.md](IMAGE_200_Q179_RESULTS_2026-10-28.md)。
