# 多家族文本溯源：现状、对照与后续路线

日期：2026-09-08  
范围：本仓库 Phase1–Phase4 实测 + 公开任务设定对照。  
限制：本机当时无法直连 arXiv / GitHub 检索页，公开论文数字只引用广泛复现的任务设定，不编造不可核对的分数。

## 1. 我们现在在做什么

任务不是「这段是不是 AI」，而是 **闭集 10 家族分类**：给定一段文本，判断来自哪个模型家族。

| 家族 | 每家两个版本（fast / balanced） |
| --- | --- |
| openai, claude, gemini, deepseek, qwen, glm, minimax, kimi, doubao, mimo | 见 `configs/text_attribution/zhizengzeng_model_catalog.v5.json` |

数据：`zhizengzeng-attribution-server-6k-v5.jsonl`，清洗后约 5686 条。冻结划分 train 3981 / validation 912 / test 793。正式筛选只用 validation；Phase3/4 不碰 test。另有两个迁移视图：`fast-to-balanced`、`balanced-to-fast`（同家族换版本）。

用户面目标是家族，不是精确到 `gpt-5.4` 这种具体模型。

## 2. 当前模型与策略

### 2.1 神经主干

Phase2 筛过三个 encoder：

| 实验 | 模型 | 8-seed 验证 macro-F1 | 结论 |
| --- | --- | ---: | --- |
| N0 | `FacebookAI/xlm-roberta-base`，max_length 512 | 0.557 | **入选** |
| N1 | `microsoft/mdeberta-v3-base`，512 | 0.492 | 单 seed 曾赢，8-seed 整体输 N0 |
| N2 | `google/canine-c`，1024 | seed42 仅 0.422 | 淘汰 |

N0 对 N1 的同 seed 差：均值 +0.066，8 局里 7 局 N0 赢。所以当前神经主干就是 **XLM-R base + 单窗口 512 + 线性分类头**。

Phase3 在这个主干上叠加配方，门控后 **没有晋级**：

| 配方 | 内容 | 结果 |
| --- | --- | --- |
| A0 | 纯家族分类 | **胜出** |
| A1 | 辅助具体模型头 | 未过门 |
| A2 | 家族/模型一致性 | `skipped_by_gate` |
| A3 | 监督对比学习 | 未作为最终 winner |
| A4 | head-center-tail 三窗口 | closed macro-F1 只比 A0 高 0.0009，门槛 0.01，**不晋级** |

A4 仍作为 Phase4 的 comparator（不是生产 winner）。

### 2.2 经典分支与融合

Phase1 经典基线（validation）：

| 实验 | 做法 | macro-F1 |
| --- | --- | ---: |
| B0 | 字符 TF-IDF + Logistic | 0.557 |
| B1 | 文体特征 + Logistic | 0.449 |
| B2 | 字符+文体 | 0.519 |
| B3 | 词+字符 TF-IDF + 校准 LinearSVC | **0.582**（test 也是 0.582） |

Phase4 Fast Track 实际融合的是 **A4 + B0 + B1**（不是 B3）：温度缩放后按权重混合概率，5 折交叉拟合，禁止单支 macro-F1 掉超过 0.005。再在 E1 上做拒识（max-prob / margin / entropy / 分歧）。

### 2.3 评估策略（这部分并不落伍）

- 冻结 split，筛选不用 test
- 8 seed 确认
- 闭集 + 两个版本迁移视图
- 晋级门：closed 增益、最差家族、holdout 不崩
- 部署另设拒识门槛：precision ≥ 0.9、coverage ≥ 0.4、ECE ≤ 0.05

## 3. 目前效果（Phase4，8-seed 验证，未看 test）

Closed：

| 系统 | Accuracy | Macro-F1 | 最差家族 F1 | ECE |
| --- | ---: | ---: | ---: | ---: |
| A0 XLM-R | 0.582 | 0.564 | 0.198 | 0.050 |
| A4 三窗口 | 0.602 | 0.587 | 0.220 | 0.045 |
| **E1 融合** | **0.668** | **0.646** | **0.313** | **0.106** |

版本迁移会掉一截：E1 在 fast-to-balanced / balanced-to-fast 上 macro-F1 约 **0.472 / 0.474**（A0 约 0.37）。随机 10 类是 0.10，所以迁移仍高于瞎猜，但明显不稳。

E1 closed 分家族 F1：

| 家族 | F1 | 家族 | F1 |
| --- | ---: | --- | ---: |
| doubao | 0.975 | minimax | 0.608 |
| openai | 0.878 | deepseek | 0.513 |
| claude | 0.782 | glm | 0.348 |
| kimi | 0.696 | mimo | 0.332 |
| qwen | 0.669 | gemini | 0.657 |

拒识未达标：accepted precision 0.891、coverage 0.388、ECE 0.112。决策是：E1 可当研究模型，**不能当部署模型**。

互补性：A4 若能在 B0/B1 里神谕选对支，closed macro-F1 还能再涨约 **0.147**。说明三支确实有不同错误，融合吃到了一部分，但远没吃干。

## 4. 和公开做法比，落伍在哪、不落伍在哪

公开文献里要分清三件事，不能拿二分类检测器的 95%+ 来比我们的 10 类 F1：

1. **AI vs 人**（RAID、MAGE、DetectGPT、Binoculars）：数字高，任务更容易。
2. **多生成器闭集溯源**（SemEval-2024 Task 8 Subtask B 最接近）：多模型、多域、多语言黑盒溯源，encoder 微调 + 集成是主流。
3. **开集 / 未见模型 / 对抗改写**：准确率会掉到接近随机；这才是部署时的真实难度。

对照我们的设定：

**仍然站得住**

- 闭集家族分类用 encoder 微调，仍是 SemEval-8B 类任务的标准底座，不是过时套路。
- 字符 n-gram / TF-IDF 在 LLM 溯源里一直强，尤其短文本和中文。我们 B0≈A0、B3 单点甚至略高于单 seed XLM-R，符合这个领域的老结论。
- 概率融合 + 拒识，方向对。部署不该输出一个硬标签就完事。
- 冻结 test、多 seed、版本迁移视图，比很多论文只报一个 lucky seed 更严谨。

**偏旧、且已经被我们自己证伪的部分**

- 死磕 XLM-R base 512。2024–2026 同类任务更常见 DeBERTa-v3、更强多语 encoder、或 embedding + 线性探针；更长上下文也常见。我们 N1 8-seed 输了，所以「换 DeBERTa 就赢」不成立，但「base + 512 已经到头」很像。
- A1–A4 的辅助头 / 对比学习 / 三窗口，是 2022–2024 分类器论文里的常规加法。在 **6k 样本、10 类、家族标签已经很粗** 的条件下过不了 1 个点的门，说明不是缺这些花活，是表征容量和家族可分性不够。
- 融合用了 B0/B1，没用 Phase1 最强的 B3。E1 还有一截经典分支没接上。
- 没有开集「未知家族」类，没有改写/翻译/摘要攻击评测。closed 0.65 不能外推到网上随手贴的文本。

**不要做的**

- 把 YuChuan 那套 **AI/人检测器** 拿来当家族溯源。检测器和溯源器任务不同。
- 用 GPT-4 当 judge 做主分类器：贵、不稳、难复现，当诊断可以，当生产头不合适。
- 指望水印：你们的数据是网关生成文本，没有可靠跨厂商水印。

## 5. 后续路线（按性价比）

原则：评估协议已经够好，下一步不要再堆 A5/A6 损失项。优先换表征、补最强经典支、把部署定义改成「能拒识的系统」。

1. **产品推理：复用检测高分块逐块溯源，相邻才合并；页面每块 Top-K + 总加权。**  
   已接到工作台：切窗 A0 seed42 + B0 + B1 + B3 等权。不要把不相邻高分块拼成一篇再切。seed62 只对照。不要再训 A1–A4。

2. **把 B3 接进融合，做 E1-B3 对照**  
   成本最低。Phase1 里 B3 验证 0.582，已经接近 A4。神谕增益 0.147 说明支路还有余量。仍只看 validation。

3. **（可选，下一轮训练）窗级扩增：一条长文切成多条 512 窗，family_id 等标签跟着复制。**  
   这不是 A4。A0 现在是扔掉尾巴；非重叠切窗合计约 2.06 万条，但同一 `request_key` 的窗相关，val/test 必须按篇聚合。过 8-seed 门再谈换骨干。

4. **换更强/更长的中英编码器，但用现有 8-seed 门，不许单 seed 晋级**  
   候选（需实测，不预设赢家）：多语 DeBERTa 更大档、GTE/E5-large 冻结 embedding + 分类头、中英都强的 0.6B–1.5B decoder 当分类器。先做 closed + 两个迁移视图。6k 数据不该一上来就全参训 7B。

5. **针对 glm / mimo / deepseek 做错误分析，而不是再加全局损失**  
   这三家拖着 macro-F1。先看是中英混、长度、提示模板，还是家族内 fast/balanced 互相抢。若是近亲混淆，再考虑「家族超类 + 难家族细分」，而不是再给全体加 contrastive。

6. **拒识门槛按家族分开，或先放宽 ECE**  
   现在全局 0.90 / 0.40 / 0.05 过不了，是 doubao/openai 很稳、glm/mimo 很糊叠在一起。可部署形态更像：高置信才报家族，否则报「AI 但家族不明」。那是产品决策，不是再训一轮就能 magically 达标。

7. **补开集与轻量攻击再谈部署**  
   至少：留一个未见家族当 unknown、同义改写、中英翻译一轮。closed 0.65 而 holdout 0.47，说明版本一变就掉；网上文本只会更脏。

明确不优先：继续 A2/A3/A4 配方搜索、把 test 解冻看一眼、上大模型 judge 当主模型。

## 6. 对照 2024–2026「模型谱系」文献（外部清单，本机未打开原文）

2026-09-08 用户从另一条能上网的链路给出一组 USENIX/ICLR/CVPR 2026 论文。本服务器当时仍打不开 `usenix.org`、`iclr.cc`、`arxiv.org`、`github.com` 网页，**下面不把那些 AUC/TPR 当已复核数字**，只对照问题定义和我们仓库里实际在跑的东西。

对方把溯源分成四层，和本项目的对应关系：

| 层次 | 对方清单里的代表 | 我们现在实际在做 |
| --- | --- | --- |
| 内容级归因：这段是哪个生成器 | SemEval-8B 类闭集分类；图片侧 IncreFA / LIDA | **主线**。10 家族闭集分类，E1 closed macro-F1 0.65 |
| 模型谱系：这个权重/API 从谁微调来 | USENIX 文生图父模型、Lineage attestation、LLM DNA、AWM、SeedPrints、Stemma | **没做**。数据是网关生成文本，没有目标模型的权重，也没有对 API 做 challenge-response |
| 主动溯源：水印/指纹/C2PA | ICLR watermark 用户级归因、C2PA 2.4、SynthID | **脚手架有，证据弱**。`provenance/c2pa_reader.py` 是占位解析；工作台能显示 C2PA 状态；文本网关样本没有跨厂商水印 |
| 数据溯源：训练样本是否进入模型 | 未在本次清单展开 | **没做** |

对方 6 篇优先阅读里，**只有内容级归因那一层和当前 Phase1–4 同题**；其余五篇问的是「模型从哪来」，不是「这段话是谁写的」。IncreFA / LIDA / Face-trace 还是**图像**生成器，不能直接搬到 6k 中英文本上。

和我们现状对齐后的判断：

- 对方说的转向（closed-set → open-set / retrieval / lineage）**方向对**，但那是下一个产品形态，不是否定当前闭集分类。没有能用的闭集底座，开集和谱系都没有东西可 register。
- 我们的 holdout 已经从 0.65 掉到 0.47，等于用「同家族换版本」做了一半开集预演。这比再堆 A5 损失更接近 2026 的问题。
- `AGENT.md` / `ATTRIBUTION_SETUP.md` 里写的 C2PA + watermark + Top-K 融合，和对方说的工业三层（C2PA + watermark + passive forensic）一致。缺口是：**文本家族分类还停在闭集 classifier，C2PA 还是 placeholder，没有未知家族发现。**
- 不要把课题改成 LLM DNA / AWM。那些要白盒权重或对 300+ checkpoint 抽功能指纹；我们只有生成文本和两张 4090。

因此：**总路线不用推倒，题面要升一层，实现仍先吃闭集余量。**

具体调整（相对第 5 节，只改优先级表述）：

1. 短期仍做内容级：接 B3、换表征、难家族分析。这是 IncreFA/LIDA 之前必须有的 generator fingerprint。
2. 中期把拒识改成开集接口：高置信报家族，否则 `unknown`，而不是再调一个全局 0.90 阈值。这是 Face-trace / IncreFA 的最小文本版，不需要 57 万张图。
3. 系统层按 `AGENT.md` 把 C2PA / 水印 / 被动分类分开显示，不要把 E1 的 0.65 包装成谱系证明。
4. 谱系（父模型、DNA、权重指纹）单独立项，等有 API probe 或开源权重审计需求再做；不塞进当前 6k JSONL 训练循环。

## 7. 一句话判断

做法的 **实验纪律不落伍**，**模型底座偏保守**，**题面比 2026 谱系文献窄一层**。当前最好的系统是 E1（XLM-R 三窗口 + 字符/文体 logistic 融合），闭集 macro-F1 0.65、准确率 0.67；换版本掉到 0.47；拒识未达部署线。不要改去做模型族谱；下一刀仍是更强内容级表征 + B3 + 把拒识做成 unknown，为以后的开集/检索留接口。
