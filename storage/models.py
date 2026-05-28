from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Content(Base):
    __tablename__ = "contents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(512))
    modality = Column(String(32))
    file_hash = Column(String(64), index=True)
    fingerprint = Column(String(256), index=True)
    source_model = Column(String(128), nullable=True)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    detection_results = relationship("DetectionResult", back_populates="content")
    provenance_records = relationship("ProvenanceRecord", back_populates="content")


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_id = Column(Integer, ForeignKey("contents.id"), index=True)
    modality = Column(String(32))
    score = Column(Float)
    label = Column(String(16))
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    content = relationship("Content", back_populates="detection_results")


class ProvenanceRecord(Base):
    __tablename__ = "provenance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_id = Column(Integer, ForeignKey("contents.id"), index=True)
    c2pa_metadata = Column(JSON, nullable=True)
    watermark_info = Column(JSON, nullable=True)
    attribution_top_k = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    content = relationship("Content", back_populates="provenance_records")
