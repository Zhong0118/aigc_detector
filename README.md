# AIGC Detector

Python 文本检测与模型家族溯源工作台。统一开发基线是 GitHub `sync`；部署版本以明确的提交哈希记录，服务器未提交文件不自动成为发布版本。

## 代码入口

| 入口 | 职责 |
| --- | --- |
| `api/main.py`、`api/routes.py` | FastAPI 接口 |
| `api/dashboard.py` | Streamlit 工作台 |
| `api/analysis.py` | 输入、检测、溯源、报告、存储的编排 |
| `ingestion/` | 文本及文档提取、元数据、指纹 |
| `detection/` | 检测适配器、文本切分、误报控制 |
| `provenance/` | 高分块选取、家族融合及其他来源证据 |
| `research/text_attribution/`、`scripts/` | 数据准备、训练、验证与打包 |
| `tests/` | 无需真实 API 的回归测试 |

先读 `api/analysis.py`，再按需要进入检测或溯源链路。图像、音频、视频分支的可用性取决于额外模型、第三方工具和服务配置；接口存在不表示已经验收。

## 开发环境

Python 3.10/3.11。推荐在独立虚拟环境安装 `requirements.txt`，不要直接升级服务器现有 Conda 环境。GPU PyTorch 应匹配服务器驱动与清单版本。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
```

`requirements.txt` 对 NumPy、scikit-learn、Transformers 的约束是有意设置的。不要用整机已经安装的版本替代依赖清单后宣称结果等价。研究阶段的额外依赖见相应 `requirements-attribution-*.txt`。

## 配置与启动

复制 `.env.local.example` 到私有的 `.env.local`，填入轮换后的凭据。Python 入口不自动加载此文件；启动前显式导出可信的本地文件：

```bash
set -a
. ./.env.local
set +a
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

另一个终端启动页面：

```bash
AIGC_API_BASE=http://127.0.0.1:8001/api/v1 python -m streamlit run api/dashboard.py --server.port 8501
```

`/health` 只证明 API 存活，不证明模型、数据、凭据或溯源产物可用。`scripts/start_workbench.sh` 是学校机器专用脚本，包含原账户路径，其他机器使用上面的命令。

历史版本曾把凭据写入配置/文档。本次移除了工作树里的已识别字面值，但没有重写 Git 历史或轮换供应商凭据；维护者必须在供应商后台轮换，不能继续信任旧值。

## 数据、模型和实验

`data/`、`models/`、`outputs/` 中的数据、权重、缓存及校准采样记录不随源码自动交付。拿到代码不等于拿到完整运行环境。

文本家族融合支持窗口 A0 与 B0/B1/B3。公共配置默认等权，属于可复现基线，不表示通过生产效果验收。模型文件缺失或配方哈希不符时明确失败，不再绕过校验。旧服务器模型与调权缓存不能凭同名目录认定兼容。

详见 [服务器改动整合与迁移说明](docs/SYNC_INTEGRATION.md)。Phase 1–4 的旧运行手册是历史流程参考；涉及新切窗与调权协议时，以整合说明和当前 CLI `--help` 为准。

## 验证

```bash
python -m pytest -q tests
```

测试使用临时目录、小型模型或 mock，不需要真实 API 密钥或正式训练。完整测试包含专门模拟在线到离线切换的用例，不要给整个测试进程强制设置 `HF_HUB_OFFLINE=1`，否则该测试的首次缓存构建会被跳过。

新增功能从 `sync` 建功能分支，完成测试与审查后合回 `sync`。服务器只部署明确提交，并另记数据集、模型、校准文件及环境版本。
