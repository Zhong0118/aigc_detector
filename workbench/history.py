"""Report persistence in the existing tables (Content / DetectionResult / ProvenanceRecord).

Images: DetectionResult.details holds the complete image_result and report snapshot taken at
analysis time; history restores them and never re-runs or rebuilds with the current bundle.
No original image bytes, client paths or server paths are stored.
"""
from __future__ import annotations

import copy
from pathlib import PurePath
from typing import Any

from storage.database import get_session
from storage.models import Content, DetectionResult, ProvenanceRecord
from workbench.image_engine import WorkbenchError

IMAGE_REPORT = "image-report-snapshot-v1"
TEXT_REPORT = "text-report-snapshot-v1"
LEGACY_LABEL = {"ai": "ai", "real": "human"}


def display_name(filename: str | None) -> str:
    """Keep only a short base name; never store client or server paths."""
    name = PurePath(str(filename or "").replace("\\", "/")).name.strip()
    return (name or "未命名")[:128]


def image_report(result: dict[str, Any]) -> dict[str, Any]:
    """Deterministic template text from the structured result (no LLM, no visual 'evidence')."""
    d, a = result["detection"], result["attribution"]
    tendency = "AI 生成倾向" if d["label"] == "ai" else "真实图片倾向"
    summary = (f"{tendency}：模型检测分数 {d['score'] * 100:.2f}%（{d['model_id']}，固定阈值 {d['threshold']}）。"
               "该分数不是校准概率。")
    evidence = [f"检测：{d['model_id']} score={d['score']:.6f}，threshold_id={d['threshold_id']}"]
    if a["status"] == "ok":
        top = "、".join(f"{t['source']} {t['score'] * 100:.2f}%" for t in a["top_k"][:3])
        evidence.append(f"来源候选（条件排序，{a['model_id']}）：{top}")
    elif a["status"] == "skipped":
        evidence.append("来源：检测低于阈值，未运行来源模型（不是“已验证未知来源”）")
    else:
        evidence.append("来源：来源模型推理失败，结果为部分成功")
    evidence.append("未知来源：不可用（score = null）")
    return dict(schema_version=IMAGE_REPORT, provider="template", status="ok", summary=summary, evidence=evidence,
                limitations=list(result["limitations"]), recommendation="仅作辅助参考，结合人工复核与其他来源证据。")


def legacy_image_payload(result: dict[str, Any], filename: str) -> dict[str, Any]:
    """Old /analyze/file keys kept as a consistent projection; `image_result` is authoritative."""
    d, a, i = result["detection"], result["attribution"], result["input"]
    return {
        "content_id": result["persistence"]["content_id"], "filename": display_name(filename), "modality": "image",
        "metadata": {"size_bytes": i["bytes"], "width": i["width"], "height": i["height"], "media_type": i["media_type"]},
        "fingerprint": i["sha256"],  # SHA-256 of the original bytes, not a generator fingerprint
        "detection": {"score": d["score"], "label": LEGACY_LABEL[d["label"]], "threshold": d["threshold"],
                      "modality_scores": {"image": d["score"]}, "model_scores": {d["model_id"]: d["score"]},
                      "decision": {"threshold_id": d["threshold_id"], "score_kind": d["score_kind"]},
                      "providers": [{"name": d["model_id"], "status": d["status"], "score": d["score"],
                                     "details": {"score_kind": d["score_kind"]}}]},
        "provenance": {"status": a["status"], "c2pa": {"status": "not_checked"}, "watermark": {"status": "not_checked"},
                       "attribution": {"status": a["status"], "reason_code": a.get("reason_code"),
                                       "top_k": [{"model": t["source"], "score": t["score"]} for t in a["top_k"][:3]]}},
        "report": image_report(result),
        "image_result": result,
    }


def save_image(result: dict[str, Any], filename: str) -> int:
    session = get_session()
    try:
        content = Content(filename=display_name(filename), modality="image", file_hash=result["input"]["sha256"],
                          fingerprint=None, file_size=result["input"]["bytes"])
        session.add(content)
        session.flush()
        result["persistence"] = dict(status="ok", content_id=int(content.id))
        snapshot = copy.deepcopy(result)
        d, a = snapshot["detection"], snapshot["attribution"]
        session.add(DetectionResult(content_id=content.id, modality="image", score=d["score"],
                                    label=LEGACY_LABEL[d["label"]],
                                    details=dict(image_result=snapshot, report_snapshot=image_report(snapshot))))
        session.add(ProvenanceRecord(content_id=content.id, c2pa_metadata={"status": "not_checked"},
                                     watermark_info={"status": "not_checked"},
                                     attribution_top_k=[{"model": t["source"], "score": t["score"]} for t in a["top_k"][:3]],
                                     confidence=None))
        session.commit()
        return int(content.id)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def text_snapshot(analysis: dict[str, Any]) -> dict[str, Any]:
    from fastapi.encoders import jsonable_encoder
    return dict(schema_version=TEXT_REPORT,
                analysis=jsonable_encoder({k: v for k, v in analysis.items() if k != "content_id"}))


def _status(details: dict | None) -> str:
    details = details or {}
    if "image_result" in details:
        return details["image_result"].get("status", "ok")
    return "ok" if "report_snapshot" in details else "legacy"


def _summary(content: Content, result: DetectionResult | None) -> dict[str, Any]:
    details = (result.details if result else None) or {}
    image = details.get("image_result")
    return dict(content_id=content.id, filename=content.filename, modality=content.modality,
                created_at=content.created_at.isoformat(timespec="seconds") + "Z" if content.created_at else None,
                status=_status(details), snapshot="versioned" if ("image_result" in details or "report_snapshot" in details)
                else "legacy", label=result.label if result else None, score=result.score if result else None,
                bundle_id=image.get("bundle_id") if image else None, mode=image.get("mode") if image else None,
                input_sha256=content.file_hash if content.modality == "image" else None)


def list_reports(modality: str | None, status: str | None, limit: int, offset: int,
                 query: str | None = None) -> dict[str, Any]:
    """Status lives inside the JSON snapshot, so filtering happens after the modality query
    (single-user preview scale)."""
    session = get_session()
    try:
        rows = session.query(Content)
        if modality:
            rows = rows.filter(Content.modality == modality)
        if query:
            like = f"%{query.strip()}%"
            rows = rows.filter(Content.filename.like(like) | Content.file_hash.like(like))
        contents = rows.order_by(Content.id.desc()).all()
        results = {}
        if contents:
            for r in session.query(DetectionResult).filter(DetectionResult.content_id.in_([c.id for c in contents])):
                results.setdefault(r.content_id, r)
        items = [_summary(c, results.get(c.id)) for c in contents]
        if status:
            items = [i for i in items if i["status"] == status]
        return dict(items=items[offset:offset + limit], total=len(items), limit=limit, offset=offset)
    finally:
        session.close()


def report_detail(content_id: int) -> dict[str, Any]:
    session = get_session()
    try:
        content = session.query(Content).filter(Content.id == content_id).first()
        if content is None:
            raise WorkbenchError("report_not_found", "找不到该报告")
        results = session.query(DetectionResult).filter(DetectionResult.content_id == content_id).all()
        provenance = session.query(ProvenanceRecord).filter(ProvenanceRecord.content_id == content_id).all()
        details = (results[0].details if results else None) or {}
        return {
            "content": {"id": content.id, "filename": content.filename, "modality": content.modality,
                        "fingerprint": content.fingerprint},
            "detection_results": [{"score": r.score, "label": r.label, "details": r.details} for r in results],
            "provenance": [{"c2pa": p.c2pa_metadata, "watermark": p.watermark_info,
                            "attribution": p.attribution_top_k, "confidence": p.confidence} for p in provenance],
            "summary": _summary(content, results[0] if results else None),
            "image_result": details.get("image_result"),
            "report_snapshot": details.get("report_snapshot"),
        }
    finally:
        session.close()
