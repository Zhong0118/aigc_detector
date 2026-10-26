# 图片研究恢复与20轮实验登记（2026-10-26）

- 工作树：`/home/zhongxu/projects/aigc_detector/.worktrees/attribution-pilot-baseline`；分支`attribution-pilot-baseline`。
- 原HEAD：`4bcbf7f53fafebd4719652c3600db5267b087485`。原有未提交图片代码保留；预变更快照在`outputs/iab-resume-20261026/pre-change-code/`。
- 本轮不操作sync生产工作树，不停用Ollama，不改历史模型、resolved配置或指标。

## 固定实验矩阵

| 条件 | 数据与划分 | D0/A0实际训练配置 | OFA | 状态目录 |
| --- | --- | --- | --- | --- |
| 50张/格，12轮 | 旧v5-50，22000图 | 历史完成记录 | 原冻结结果 | `outputs/iab-v5-50-monitor`，不改 |
| 50张/格，20轮 | 相同v5 manifest与全部样本 | 独立新输出v5-50-e20；20轮、batch32、seed42、Adam .001、StepLR(1,.9) | 冻结，每来源100训练参考 | `outputs/iab-v5-50-e20-monitor` |
| 100张/格，20轮 | 新v6-100，42000图；父v5锚定 | 独立新输出v6-100-e20；其余同配方 | 冻结，每来源100训练参考 | `outputs/iab-v6-100-monitor` |

- 本轮物理GPU1用于图片研究，物理GPU0现有Ollama保留；进程内逻辑cuda:0由CUDA_VISIBLE_DEVICES=1映射。
- 比较50/12与50/20观察轮数变化；比较50/20与100/20观察同20轮下的数据扩充变化。历史GPU0与本轮GPU1、单seed等差异需报告，不作严格因果或独立检验宣称。
- D0真假与A0来源分别训练评估，真实图不进入来源标签；OFA验证AUC与分类准确率分开。
- D1/D2/A1及A2重训变体的共享/专用训练配置也更新为20，但本轮不自动启动这些后续实验；D3/A2冻结评估/A3/OFA没有训练epoch。

## 数据与复用

- 22来源×10语义×100＝22000 AI；官方真实目录每语义2000张，共20000真实，合42000。不是44000或88000。
- 固定前缀选择，包含原v5全部22000样本，不重新抽样。新selection SHA256：`286a3c34f5291a355259cfed7170e9b5a6d9d17ddf483f6d32c535cecc4f64b6`。
- 父v5 manifest SHA256：`0dd207abf3fb477a89d8c767144a4869b425722415258a3e031ff2bedbf26553`。
- 逐文件SHA核验、硬链接复用27346张（11409 AI、15937真实），仅补14654张约13.04GB。复用没有网络请求，不作为下载吞吐。
- 原v1-v5清单和文件保留；旧split/group_id不变，新增重复桥接旧分区/旧组时fail closed，holdout不推理、不调参。
- 下载：4路服务器直连、真实/生成公平并发、公开GET签名S3链接进程内缓存、入口共享节流、TLS校验；不再下载后丢弃验证图。拒绝响应/校验失败保留现场，不盲重试、不换出口绕过。

## 启动与核验

- 全部新启动均经现有monitor和锁。服务器100张版loop每1800秒检查并推进，下载完成且父划分校验通过后才启动新D0/A0/冻结OFA。
- 图片共享锁仍沿用`outputs/image_experiments/gpu0-research.lock`文件名，该历史名字不表示实际使用物理GPU0。
- shell支持显式epochs与GPU索引；已完成输出的epoch数不符时拒绝复用，必须换新suffix，防止把12轮旧模型当20轮结果。
- 53项相关测试通过，包含5项离线参数/作用域/历史输出保护用例；bash语法、Python编译及git diff --check通过。测试通过不等于模型训练或下载完成。
- 每条流水线保存退出状态、配置、逐epoch历史、逐样本预测、模型与哈希。报告完成前核对completion、validated-summary和汇总一致。

## 完成后的分析

- 同原v5的3140条D0 dev、1496条AI dev按sample_id配对；核对标签、词表和group_id，新完整dev另列。
- 报告D0 AUROC/准确率/FPR/TPR，A0 Top1/Top3/macroF1与每来源变化、分组bootstrap区间，OFA来源验证宏AUC与独立的辅助分类值。
- 可复用`outputs/image_experiments/iab-v5-50-comparison/compare.py`的align/stats/group_bootstrap函数。原main硬编码旧20/50版和12轮，不能直接当本轮汇总；新的三条件报告使用独立目录，保留旧报告。
- 依据结果提出D2/A1/A2/A3后续建议，不自动启动。当前仅登记方案与启动状态，不填入尚未取得的最终指标。
