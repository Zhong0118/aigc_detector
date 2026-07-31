# LLM 来源溯源本地小数据操作说明

本阶段只验证 API 协议、模型标签和提示词设计，不训练深度模型，也不使用两张 4090D。
当前生产网关只使用智增增；CloseAI 因密钥无效暂不参与采集。

十家族配置驱动的最新操作方式见 `docs/ZHIZENGZENG_DATA_PRODUCTION.md`。本文件后续
内容保留为早期双网关协议验证记录。

## 1. 文件位置

- 密钥模板：`.env.local.example`
- 真实密钥：`.env.local`，已被 Git 忽略
- 网关配置：`research/text_attribution/config.py`
- OpenAI 兼容客户端：`research/text_attribution/gateway.py`
- 提示词库：`research/text_attribution/prompts.py`
- 请求计划和安全写入：`research/text_attribution/collection.py`
- 采集命令：`scripts/collect_attribution_pilot.py`
- 默认结果：`data/attribution/raw/pilot-<gateway>.jsonl`

## 2. 放置 API Key

已经贴入聊天的旧 CloseAI Key 必须先在平台轮换。

在项目根目录执行：

```powershell
Copy-Item .env.local.example .env.local
```

然后只在本地编辑 `.env.local`：

```env
CLOSEAI_API_KEY=轮换后的CloseAI密钥
CLOSEAI_BASE_URL=https://api.openai-proxy.org/v1

ZZZ_API_KEY=智增增密钥
ZZZ_BASE_URL=https://api.zhizengzeng.com/v1
```

代码只会在发 HTTP 请求时创建 Authorization header。Key 不会写入 JSONL、日志、模型产物或对象字符串。

## 3. 第一步：只查看请求计划

不需要 Key，也不会访问网络：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_pilot `
  --gateway closeai `
  --models model-a,model-b `
  --prompt-limit 2 `
  --max-requests 4 `
  --dry-run
```

本地安全限制默认为最多 100 个请求。超过 100 必须显式增加
`--confirm-large-run`，防止误刷费用。

## 4. 第二步：查询模型列表

Key 配置完成后：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_pilot `
  --gateway closeai `
  --list-models
```

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_pilot `
  --gateway zhizengzeng `
  --list-models
```

如果某个平台不支持 `/models`，记录错误格式，改用平台控制台的模型清单进行最小调用验证。

## 5. 第三步：每站一个模型、两个请求

从模型列表选择一个低成本模型，用准确 model ID：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_pilot `
  --gateway closeai `
  --models accurate-model-id `
  --prompt-limit 1 `
  --max-requests 2 `
  --max-tokens 256
```

智增增使用完全相同的参数：

```powershell
C:\Users\zx\.conda\envs\dl\python.exe -m scripts.collect_attribution_pilot `
  --gateway zhizengzeng `
  --models accurate-model-id `
  --prompt-limit 1 `
  --max-requests 2 `
  --max-tokens 256
```

结果同时保存：

- 请求模型别名 `request_model`
- 响应模型字段 `response_model`
- 网关 request ID
- prompt ID、语言、任务、领域和目标长度
- temperature、top_p、max_tokens
- 文本、finish reason、usage 和耗时

同一个 `request_key` 已经写入后，再次运行会跳过，避免重复计费。

## 6. 提示词设计

### 6.1 配对原则

同一个 `prompt_id` 必须原样发送给：

- 所有候选模型
- 两个网关
- 多个采样设置
- 后续时间复采

训练和测试按照 `prompt_id` 分组，不能把同一 prompt 的不同模型回答随机拆到训练和测试。

### 6.2 当前 pilot-v1

当前包含 24 条提示：

- 中文 12 条、英文 12 条
- short、medium、long 三种长度
- science、technology、business、education、health、office、literature、history、data_analysis、general 等领域
- explanation、procedure、comparison、argument、summary、rewrite、creative、analysis、data_interpretation、diagnosis、document_writing、question_answering 等任务

提示词避免：

- 询问模型身份
- 指定某家模型习惯用语
- 依赖近期新闻
- 敏感或违规内容
- 所有任务都要求相同 Markdown 模板

### 6.3 为什么不马上做 100 条

先用 24 条验证：

- 网关是否更改 system prompt
- 模型 ID 是否真实稳定
- 输出是否大量拒答或截断
- 两站是否返回相同协议字段
- 提示长度是否与 max_tokens 匹配

通过后再将 prompt bank 扩展到 100 条，形成约 3,200 条数据冒烟：

```text
8 模型 × 2 网关 × 100 prompt × 2 采样设置
```

## 7. 什么时候上传服务器

本地完成以下检查后再上传：

- 两站协议可调用
- request/response model 标签可解释
- 3,200 条内的小数据没有明显网关和模板捷径
- prompt/group manifest 已冻结

第一次服务器实验只做冻结特征、线性探针、Byte/TF-IDF 基线和网关泄漏审计。得到前两名骨干后，才生成 20,000–72,000 条试验数据并开展 LoRA、全量微调、对比学习和开放集实验。
