from pathlib import Path

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

_engine = None
_SessionFactory = None


def _load_db_path() -> str:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("database", {}).get("path", "data/aigc_detection.db")
    return "data/aigc_detection.db"


def init_db(db_path: str | None = None) -> None:
    global _engine, _SessionFactory

    if db_path is None:
        db_path = _load_db_path()

    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    _SessionFactory = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    global _engine, _SessionFactory

    if _SessionFactory is None:
        init_db()

    return _SessionFactory()
