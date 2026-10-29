"""Versioned result snapshots in the preview DB; history restores them without re-inference."""
from __future__ import annotations

import uuid
from pathlib import PurePath
from typing import Any

from storage.database import get_session
from storage.models import AnalysisSnapshot
from workbench.image_engine import WorkbenchError, utc_now

TEXT_SCHEMA = "text_result.v1"


def display_name(filename: str | None) -> str:
    """Keep only a short base name; never store client or server paths."""
    name = PurePath(str(filename or "").replace("\\", "/")).name.strip()
    return (name or "未命名")[:128]


def wrap_text(payload: dict[str, Any], filename: str, sha256: str | None) -> dict[str, Any]:
    return dict(schema_version=TEXT_SCHEMA, analysis_id=uuid.uuid4().hex, created_at=utc_now(), kind="text",
                display_name=display_name(filename), input_sha256=sha256, result=payload)


def save(kind: str, snapshot: dict[str, Any], name: str | None = None) -> str:
    if kind == "image":
        detection = snapshot["detection"]
        label, score, sha = detection.get("label"), detection.get("score"), snapshot["input"]["sha256"]
    else:
        detection = snapshot["result"].get("detection") or {}
        label, score, sha = detection.get("label"), detection.get("score"), snapshot.get("input_sha256")
    session = get_session()
    try:
        session.add(AnalysisSnapshot(
            analysis_id=snapshot["analysis_id"], kind=kind, schema_version=snapshot["schema_version"],
            status=snapshot.get("status", "ok"), label=label, score=score,
            display_name=display_name(name if name is not None else snapshot.get("display_name")),
            input_sha256=sha, snapshot=snapshot))
        session.commit()
    finally:
        session.close()
    return snapshot["analysis_id"]


def _summary(row: AnalysisSnapshot) -> dict[str, Any]:
    return dict(analysis_id=row.analysis_id, kind=row.kind, schema_version=row.schema_version, status=row.status,
                label=row.label, score=row.score, display_name=row.display_name,
                input_sha256=row.input_sha256, created_at=row.created_at.isoformat(timespec="seconds") + "Z")


def list_snapshots(page: int = 1, page_size: int = 20, kind: str | None = None, status: str | None = None,
                   label: str | None = None, query: str | None = None) -> dict[str, Any]:
    page, page_size = max(1, page), min(max(1, page_size), 100)
    session = get_session()
    try:
        rows = session.query(AnalysisSnapshot)
        if kind:
            rows = rows.filter(AnalysisSnapshot.kind == kind)
        if status:
            rows = rows.filter(AnalysisSnapshot.status == status)
        if label:
            rows = rows.filter(AnalysisSnapshot.label == label)
        if query:
            like = f"%{query.strip()}%"
            rows = rows.filter(AnalysisSnapshot.display_name.like(like) | AnalysisSnapshot.input_sha256.like(like))
        total = rows.count()
        items = rows.order_by(AnalysisSnapshot.id.desc()).offset((page - 1) * page_size).limit(page_size).all()
        return dict(items=[_summary(r) for r in items], total=total, page=page, page_size=page_size)
    finally:
        session.close()


def get_snapshot(analysis_id: str) -> dict[str, Any]:
    session = get_session()
    try:
        row = session.query(AnalysisSnapshot).filter(AnalysisSnapshot.analysis_id == analysis_id).first()
        if row is None:
            raise WorkbenchError("report_not_found", "找不到该分析记录")
        return dict(_summary(row), snapshot=row.snapshot)
    finally:
        session.close()
