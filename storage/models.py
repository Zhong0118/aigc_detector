"""数据库表模型。

这里定义 SQLAlchemy ORM model，对应 SQLite/PostgreSQL 中的表。

当前三张核心表：
- contents: 内容主表
- detection_results: 检测结果表
- provenance_records: 溯源结果表

后续建议：
- 增加 ProviderCall 表，记录每个外部 API 调用耗时、状态、错误
- 增加 Report 表，保存 LLM/模板生成的最终报告
- 增加 Dataset/ModelVersion 表，管理本地模型训练版本
- 增加 Alembic migration，避免直接 create_all 管理生产 schema
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Content(Base):
    """内容主表。

    一条记录代表一次被分析的输入内容。
    这里保存文件身份信息、模态、fingerprint、大小和创建时间。
    """

    __tablename__ = "contents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(512))
    modality = Column(String(32))
    file_hash = Column(String(64), index=True)
    fingerprint = Column(String(256), index=True)
    source_model = Column(String(128), nullable=True)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 一个 Content 可以对应多条检测结果。
    detection_results = relationship("DetectionResult", back_populates="content")
    # 一个 Content 可以对应多条溯源记录。
    provenance_records = relationship("ProvenanceRecord", back_populates="content")


class DetectionResult(Base):
    """检测结果表。

    保存最终 score/label，以及 provider 详情 JSON。
    目前一次分析通常写一条；后续多 provider 或多版本模型可写多条。
    """

    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_id = Column(Integer, ForeignKey("contents.id"), index=True)
    modality = Column(String(32))
    score = Column(Float)
    label = Column(String(16))
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 多对一关联回 Content。
    content = relationship("Content", back_populates="detection_results")


class ProvenanceRecord(Base):
    """溯源记录表。

    保存 C2PA metadata、水印信息、Top-K 模型归因和归因置信度。
    """

    __tablename__ = "provenance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_id = Column(Integer, ForeignKey("contents.id"), index=True)
    c2pa_metadata = Column(JSON, nullable=True)
    watermark_info = Column(JSON, nullable=True)
    attribution_top_k = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 多对一关联回 Content。
    content = relationship("Content", back_populates="provenance_records")


class AnalysisSnapshot(Base):
    """预览工作台的版本化结果快照（文本/图片），不保存原始图片字节。

    snapshot 保存当时的模型哈希、阈值和预处理，历史页按它还原，不重新推理。
    """

    __tablename__ = "analysis_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(64), unique=True, index=True)
    kind = Column(String(16), index=True)
    schema_version = Column(String(32))
    status = Column(String(16), index=True)
    label = Column(String(64), nullable=True)
    score = Column(Float, nullable=True)
    display_name = Column(String(256))
    input_sha256 = Column(String(64), index=True, nullable=True)
    snapshot = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
