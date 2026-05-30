"""FastAPI 后端入口文件。

这个文件只负责创建后端应用、挂载路由、初始化数据库和提供健康检查。
真正的检测/溯源/报告流程不写在这里，而是在 `api.analysis.AnalysisService`
和 `api.routes` 里完成。
"""

import sys
from pathlib import Path

# Streamlit 或命令行从不同目录启动时，可能找不到项目根目录。
# 这里把项目根目录加入 import 路径，保证 `api/`, `storage/` 等包能被导入。
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from storage.database import init_db
from api.routes import router

# 创建 FastAPI 应用对象。这个对象会被 uvicorn 加载：
# `uvicorn api.main:app --reload`
app = FastAPI(
    title="AIGC Detection & Attribution API",
    version="0.1.0",
    description="Multi-modal AI content detection and provenance analysis",
)

# 把 api/routes.py 里的所有业务接口挂到 `/api/v1` 下面。
# 例如 routes.py 里定义 `/analyze/text`，最终访问路径就是 `/api/v1/analyze/text`。
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup():
    """应用启动时初始化数据库表。

    这里会调用 storage/database.py 里的 init_db。
    如果 SQLite 数据库文件或表不存在，会自动创建。
    """
    init_db()


@app.get("/health")
def health():
    """健康检查接口。

    用来确认 FastAPI 服务是否正常启动。
    浏览器或命令行访问 `/health`，返回 `{"status": "ok"}` 就说明后端活着。
    """
    return {"status": "ok"}
