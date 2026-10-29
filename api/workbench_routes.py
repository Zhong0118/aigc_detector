"""Workbench API: image branch of /analyze/file, capabilities, history and experiment catalog.

Images never reach the legacy scaffold/demo path: they go to the real image adapter or
fail with a typed error (code, message, request_id, retryable).
"""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from workbench import snapshots
from workbench.catalog import build_catalog
from workbench.image_engine import ImageService, WorkbenchError
from workbench.settings import workbench_settings

router = APIRouter()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".heic", ".heif", ".avif"}
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM", b"II*\x00", b"MM\x00*")
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

_image_service: ImageService | None = None


def get_image_service() -> ImageService:
    global _image_service
    if _image_service is None:
        _image_service = ImageService(workbench_settings())
    return _image_service


def looks_like_image(filename: str, content: bytes) -> bool:
    if Path(filename).suffix.lower() in IMAGE_SUFFIXES:
        return True
    return content.startswith(IMAGE_MAGIC) or (content[:4] == b"RIFF" and content[8:12] == b"WEBP")


def error_response(exc: WorkbenchError, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"error": exc.payload(request_id), "detail": exc.message})


def legacy_image_payload(filename: str, result: dict[str, Any]) -> dict[str, Any]:
    """Keep the old /analyze/file keys; the authoritative data is `image_result`."""
    d, a = result["detection"], result["attribution"]
    return {
        "filename": snapshots.display_name(filename), "modality": "image", "content_id": None,
        "analysis_id": result["analysis_id"],
        "metadata": {"size_bytes": result["input"]["bytes"], "width": result["input"]["width"],
                     "height": result["input"]["height"]},
        "fingerprint": result["input"]["sha256"],
        "detection": {"score": d["score"], "label": d["label"], "threshold": d["threshold"],
                      "modality_scores": {"image": d["score"]}, "model_scores": {}, "decision": {},
                      "providers": [{"name": d["model_id"], "status": d["status"], "score": d["score"],
                                     "details": {"score_kind": d["score_kind"]}}]},
        "provenance": {"status": "not_run", "c2pa": {"status": "not_checked"}, "watermark": {"status": "not_checked"},
                       "attribution": {"status": a["status"], "top_k": [
                           {"model": x["label"], "score": x["score"]} for x in a["top_k"][:3]]}},
        "report": {"provider": "none", "status": "disabled", "summary": "图片结果见 image_result；未生成 LLM 解释。",
                   "evidence": [], "limitations": result["limitations"], "recommendation": "结合人工复核使用。"},
        "image_result": result,
    }


def analyze_image_upload(filename: str, content: bytes, research: bool, request_id: str | None):
    request_id = request_id if request_id and REQUEST_ID.match(request_id) else uuid.uuid4().hex
    try:
        result = get_image_service().analyze(content, filename, research=research, request_id=request_id)
    except WorkbenchError as exc:
        return error_response(exc, request_id)
    if not result.get("deduplicated"):
        try:
            snapshots.save("image", result, filename)
            result["history_saved"] = True
        except Exception:
            result["history_saved"] = False
    return legacy_image_payload(filename, result)


def record_text(payload: dict[str, Any], filename: str, content: bytes) -> dict[str, Any]:
    try:
        snapshot = snapshots.wrap_text(payload, filename, hashlib.sha256(content).hexdigest())
        payload["analysis_id"] = snapshots.save("text", snapshot)
    except Exception:
        payload["analysis_id"] = None
    return payload


@router.get("/capabilities")
def capabilities():
    from api.analysis import get_analysis_service
    settings = workbench_settings()
    text = get_analysis_service().providers()
    return {
        "schema_version": "capabilities.v1", "preview": bool(settings),
        "text": {"available": True, "mode": text["mode"],
                 "providers": [{k: p.get(k) for k in ("name", "type", "enabled", "configured")} for p in text["providers"]]},
        "image": get_image_service().capabilities(),
        "history": {"stores_original_images": False},
        "experiments": {"available": bool(settings.get("catalog"))},
    }


@router.get("/history")
def history(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), kind: str | None = None,
            status: str | None = None, label: str | None = None, q: str | None = Query(None, max_length=128)):
    return snapshots.list_snapshots(page, page_size, kind, status, label, q)


@router.get("/history/{analysis_id}")
def history_item(analysis_id: str):
    request_id = uuid.uuid4().hex
    try:
        return snapshots.get_snapshot(analysis_id)
    except WorkbenchError as exc:
        return error_response(exc, request_id)


@router.get("/experiments")
def experiments():
    settings = workbench_settings()
    if not settings.get("catalog"):
        return error_response(WorkbenchError("model_unavailable", "未配置实验目录"), uuid.uuid4().hex)
    return build_catalog(settings)
