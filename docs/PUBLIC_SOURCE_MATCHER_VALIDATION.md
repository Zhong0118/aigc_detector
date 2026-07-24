# Public Source Matcher Validation Record

## Objective

本次修改要解决的不是“让搜索命中率看起来更高”，而是保证以下结论成立：

1. 公开来源匹配只提供来源证据，不冒充 AI 概率或人工创作证明。
2. 一个网页失败不能让全部候选失效。
3. 短文本、部分匹配和剩余文本过短时不能制造虚假的确定结论。
4. 只有未被可靠公开来源覆盖的内容才进入 AI 风格检测。
5. 外部 URL、响应大小、重定向和总等待时间都有明确边界。

## Baseline

修改前执行：

```text
python -m unittest discover
Ran 45 tests
OK
```

## Reproduced Problems

### One inaccessible page aborted the branch

使用《红楼梦》公开文本进行真实联网测试时，搜索成功返回候选，但第一个百度文库页面返回 HTTP 403。旧实现把整个 matcher 变为 `error`，没有继续检查其他页面。

根因：查询、候选遍历和页面抓取被包在同一个异常边界内。

### Short phrases became strong matches

输入“今天天气很好”，候选正文包含该短句时，旧实现输出：

```text
status=matched coverage=1.0 max_page_similarity=1.0
```

根因：不足 80 字的完整输入仍被作为精确窗口；比例为 100% 不代表文本具有来源唯一性。

### Public source was mislabeled as human

旧 provider 对强来源命中返回 `score=0.05, label=human`。

根因：把“可检索来源”错误换算成了“人工创作概率”。公开网页也可能包含 AI 文本，两者不能等价。

### Partial match still analyzed the full text

旧流程记录 `partial`，但 Hive 和本地检测器继续读取原文件或完整 `raw_data`。

根因：匹配区间只存在于证据详情，没有形成传给下游 provider 的临时文本范围。

### A 75-character match could have zero coverage

100 字输入连续匹配 75 字时，状态可因 `max_page_similarity=0.75` 成为 `partial`，但固定 80 字的 span 门槛导致 `coverage=0`，无法排除已匹配部分。

根因：状态阈值和区间记录阈值不一致。

### Strict DNS validation conflicted with the local proxy

当前运行环境把公网域名解析到 `198.18.0.0/15`，这是代理/基准测试保留网段。严格公网地址校验会在请求前拒绝所有候选。

根因：安全校验看到的是代理合成地址，而不是目标站点的真实公网地址。

## Implemented Logic

### Matcher states

| State | Condition | AI handling |
| --- | --- | --- |
| `too_short` | 规范化输入少于 80 字 | 完整文本正常检测 |
| `no_match` | 未达到弱匹配阈值 | 完整文本正常检测 |
| `weak` | 覆盖度至少 0.15 | 完整文本正常检测，并披露弱证据 |
| `partial` | 单页相似度至少 0.75、覆盖度至少 0.45，或多来源支持弱覆盖 | 删除可靠匹配区间，只检测未匹配部分 |
| `matched` | 覆盖度至少 0.75 且单页相似度至少 0.90 | 不运行 AI detector，标签为 `public_source` |
| `error` | 没有可用页面且存在搜索/抓取错误 | 完整文本正常检测，并披露来源分支失败 |

### Failure isolation

- 每个搜索 query 单独捕获错误。
- 每个候选 URL 单独验证和抓取。
- HTTP 403、超时、PDF、过大响应等仅增加 `issues` 和 `failed_page_count`。
- 有可用页面时继续计算；全部候选失败时才返回整体 `error`。

### Unmatched-text routing

- 规范化函数同时保留“规范化字符索引 → 原文字符索引”映射。
- 匹配 span 先映射回原文区间，再从原文删除。
- 未匹配文本只在当前 `detect()` 调用内传递，不写入 provider details 或数据库。
- Hive V2、Hive V3、demo provider 和本地文本 detector 都支持该临时文本范围。
- 下游详情只记录 `text_scope` 和字符数量。
- 未匹配内容不足 80 字时不输出短文本 AI 分数，最终标签为 `uncertain`。

### Security and resource limits

- 仅允许 HTTP/HTTPS。
- 拒绝 localhost、私有、回环、链路本地、保留和无法解析的地址。
- 手动验证每次重定向，最多三次。
- 仅接受文本类响应，正文最多 2,000,000 bytes。
- 单请求超时 6 秒；总任务预算 20 秒，预算耗尽后不再启动新请求。
- `allow_proxy_dns` 只允许显式信任 `198.18.0.0/15` 代理映射，不放开 RFC1918 私网。

## TDD Evidence

按行为分组，先增加测试并确认失败，再实现最小修复：

```text
short-text/mapping RED:
TypeError: unexpected keyword argument 'min_input_chars'
AttributeError: no attribute 'unmatched_text'

failure-isolation RED:
TypeError: unexpected keyword argument 'page_fetcher'
expected matched, got error

provider-routing RED:
expected unmatched text, detector received full text
expected public_source/0.0, got human/0.05

partial-span RED:
expected coverage 0.75, got 0.0

proxy compatibility RED:
TypeError: unexpected keyword argument 'allow_proxy_dns'
```

分组通过结果：

```text
python -m unittest tests.test_public_source_matcher
Ran 14 tests - OK

python -m unittest tests.test_report_and_providers
Ran 19 tests - OK
```

## Live Network Verification

运行日期：2026-07-10。

### Mixed classic-literature passage

输入组合了《红楼梦》不同位置的句子。8 个候选中 3 个页面返回 403，5 个页面成功抓取。最高单页连续覆盖约 17.39%，没有达到有效 span 阈值，结果为 `no_match`。

这验证了“搜到短句或零散页面不等于强来源命中”。

### Longer constitutional passage

```text
status=partial
coverage=0.6713
max_page_similarity=0.6713
independent_source_count=4
attempted_pages=8
failed_pages=2
```

失败页面分别为 PDF Content-Type 和 HTTP 403；其他政府/人大页面仍参与计算。

### Continuous 96-character constitutional passage

```text
status=matched
coverage=1.0
max_page_similarity=1.0
independent_source_count=4
attempted_pages=8
failed_pages=2
```

该结果验证了完整链路：搜索、失败隔离、安全抓取、正文抽取、单页相似度、聚合覆盖度和独立来源计数。

## Remaining Limits

1. DuckDuckGo HTML 是无密钥 demo 后端，页面结构和限流策略可能变化；正式交付建议替换为官方搜索 API。
2. 当前算法强调精确连续覆盖，不把语义改写当成可靠来源证据。
3. JavaScript 动态正文、登录墙、robots 限制和 PDF 不会被强行绕过。
4. DNS 校验降低 SSRF 风险，但不能消除所有 DNS rebinding 和代理侧风险；生产环境仍应配置网络出口白名单。
5. 联网查询会向搜索服务发送选定文本片段，产品应披露并允许关闭该模块。
6. `config.yaml` 当前还包含与本功能无关的凭据/阈值改动，交付前应清理明文密钥并使用环境变量。

## Delivery Scope

运行代码：

- `detection/public_source_matcher.py`
- `detection/providers.py`
- `reports/generator.py`
- `config.yaml` 中的 `detection.public_source_match` 配置块

验证和设计记录：

- `tests/test_public_source_matcher.py`
- `tests/test_report_and_providers.py`
- `docs/superpowers/specs/2026-07-10-public-source-hardening-design.md`
- `docs/superpowers/plans/2026-07-10-public-source-hardening.md`
- `docs/PUBLIC_SOURCE_MATCHER_VALIDATION.md`
