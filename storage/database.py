"""数据库连接与初始化。

这个文件负责：
- 从 config.yaml 读取 SQLite 数据库路径
- 创建 SQLAlchemy engine
- 创建 SessionFactory
- 初始化数据库表

当前是本地 SQLite，适合快速演示。
后续建议：
- 使用 PostgreSQL 作为生产数据库
- 使用 Alembic 管理 schema migration
- 增加连接池配置
- 区分 dev/test/prod 数据库
"""

from pathlib import Path

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

_engine = None
_SessionFactory = None


def _load_db_path() -> str:
    """从 config.yaml 读取数据库路径。

    如果配置文件不存在或没有 database.path，就使用默认路径。
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("database", {}).get("path", "data/aigc_detection.db")
    return "data/aigc_detection.db"


def init_db(db_path: str | None = None) -> None:
    """初始化数据库 engine、session factory 和表结构。

    FastAPI 启动时会调用这个函数。
    如果数据库文件或目录不存在，会自动创建。
    """
    global _engine, _SessionFactory

    if db_path is None:
        db_path = _load_db_path()

    db_file = Path(db_path)
    # SQLite 文件所在目录需要先存在。
    db_file.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    _SessionFactory = sessionmaker(bind=_engine)
    # 根据 storage/models.py 中的 SQLAlchemy model 创建表。
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    """获取一个数据库 session。

    调用方用完后要 close。当前代码一般放在 try/finally 里关闭。
    """
    global _engine, _SessionFactory

    if _SessionFactory is None:
        # 如果外部忘了 init_db，这里懒初始化一次，降低使用成本。
        init_db()

    return _SessionFactory()
