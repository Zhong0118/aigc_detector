# Public Source Matcher Hardening Design

## Goal

把联网公开来源匹配建设成“来源证据分支”，用于降低经典文学、论文、技术文档等公开文本被 AIGC 模型误伤的概率，同时不把“能在网上找到”错误解释成“必然由人类创作”。

## First Principles

1. 公开来源匹配回答的是“文本是否与公开页面重合”，不是“文本由谁创作”。
2. AI 风险分数和来源覆盖度是两种不同量纲，不能直接加权平均。
3. 短文本缺少唯一性。常见短句即使完全命中，也不足以支持强来源结论。
4. 单个网页不可访问不代表其他来源不存在，失败必须局部隔离。
5. 已确认的公开片段不应继续参与 AI 风格检测；不确定的弱匹配不能随意删除。
6. 外部网页和搜索引擎都属于不可信边界，必须限制协议、地址、重定向、正文大小和超时。
7. 用户文本可能敏感。系统只发送必要查询片段，不在检测详情中额外持久化未匹配原文。

## Decision Model

| Matcher status | Meaning | Downstream AI input | Final behavior |
| --- | --- | --- | --- |
| `too_short` | 规范化文本不足 80 字 | 完整文本 | 正常 AI 检测 |
| `no_match` | 有效覆盖度低于 0.15 | 完整文本 | 正常 AI 检测 |
| `weak` | 覆盖度在 0.15 到 0.45 附近 | 完整文本 | 正常 AI 检测并记录弱来源证据 |
| `partial` | 单页相似度达到 0.75，或聚合覆盖度达到 0.45 | 只检测未匹配文本 | 报告分别解释来源片段和 AI 检测范围 |
| `matched` | 单页相似度至少 0.90 且覆盖度至少 0.75 | 不执行 AI 模型 | 最终标签为 `public_source`，AI 属性未评估 |
| `error` | 没有可用页面且搜索/抓取均失败 | 完整文本 | 正常 AI 检测并披露来源分支失败 |

## Components

### PublicSourceMatcher

- 新增 `min_input_chars`，短文本在联网前返回 `too_short`。
- 每个查询和每个页面分别捕获异常，写入结构化 `issues`，继续处理其他候选。
- 只有没有成功搜索、没有可用页面且存在错误时才返回整体 `error`。
- 记录 `searched_page_count`、`failed_page_count`，便于判断证据质量。
- 使用规范化字符到原文字符的索引映射，从原文中删除匹配区间，生成仅供当前检测调用的未匹配文本。
- `to_dict()` 不输出未匹配原文，只输出字符数量和匹配区间。

### Safe Page Fetching

- 仅允许 `http` 和 `https`。
- 拒绝 localhost、私有、回环、链路本地、保留地址和无法解析的主机。
- 关闭自动重定向，每次重定向前重新验证目标，最多跟随三次。
- 使用流式读取并设置 `max_page_bytes`，防止超大响应占用内存。
- 只接受文本类 Content-Type；单页失败不会终止整个任务。
- 使用 `max_total_seconds` 阻止新的联网工作超过总时间预算。
- 默认拒绝非公网 DNS；可信代理若把公网域名映射到 `198.18.0.0/15`，可显式启用 `allow_proxy_dns`。

### Provider Integration

- `matched` 分支返回 `label=public_source`，provider 本身不提供 AI score。
- `partial` 分支通过 `text_override` 把未匹配文本传给 Hive V2、Hive V3、demo 和本地文本模型。
- 非文本模态继续使用原始文件路径，不改变接口行为。
- 下游 provider 详情增加 `text_scope`、原始规范化字符数、实际分析字符数，报告可以说明检测范围。
- 未匹配文本过短时不调用 AI 模型，返回 `insufficient_unmatched_text`，避免短样本分数失真。

### Reporting

- 报告覆盖 `matched`、`partial`、`weak`、`too_short` 和 `error`。
- `public_source` 的摘要明确写成“公开来源高度重合，AI 属性未评估”。
- 部分匹配报告明确说明 AI 分数只基于未匹配部分。

## Error Handling

- 搜索查询失败：记录 query/stage/error，继续下一查询。
- 页面 403、超时、正文过大或非文本响应：记录 URL/stage/error，继续下一页面。
- 所有来源均失败：来源 provider 为 `error`，原 AI 流程继续。
- 部分页面失败但仍有可用页面：正常计算结果，并在详情中保留 issues。

## Testing Strategy

- 单元测试：短文本、规范化映射、未匹配文本提取、阈值状态。
- 容错测试：一个页面失败但后续页面成功；所有页面失败。
- 安全测试：拒绝私有 URL、超大正文和过多重定向。
- 集成测试：强命中产生 `public_source`；部分命中只把剩余文本送给本地和 Hive provider。
- 报告测试：强、部分、弱、错误状态均有可解释文字。
- 回归测试：完整 unittest suite，随后使用经典公开文本执行一次真实联网 best-effort 验证。

## Known Limits

- DuckDuckGo HTML 后端仍属于 demo/best-effort 接口，生产环境应替换为正式搜索 API。
- 当前匹配以精确字符覆盖为主，不把语义改写当成可靠公开来源命中。
- 联网查询会把选定片段发送给搜索服务，部署方应在产品隐私说明中披露，并允许关闭该分支。
