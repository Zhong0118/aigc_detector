"""storage 包的统一导出入口。

storage 层负责持久化：
- Content: 输入内容的基础信息和 fingerprint
- DetectionResult: 检测分数、标签、provider 详情
- ProvenanceRecord: C2PA、水印、归因等溯源结果

当前使用 SQLite + SQLAlchemy，适合 MVP 和本地演示。
后续如果要做正式服务，可以迁移到 PostgreSQL，并加入 Alembic 数据库迁移。
"""

from .database import init_db, get_session
from .models import Content, DetectionResult, ProvenanceRecord
