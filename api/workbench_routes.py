"""Workbench API: image branch of /analyze/file, capabilities, report history and experiment catalog.

Images never reach the legacy demo/provider-average/text-family path: they go to the real
image service or fail with error={code, message, request_id, retryable} (no stack traces,
no server paths).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from workbench import history
from workbench.catalog import build_catalog
from workbench.image_engine import ImageService, WorkbenchError, sniff_format
from workbench.settings import workbench_settings

router = APIRouter()
log = logging.getLogger("workbench.image")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".heic", ".heif",
                  ".avif", ".jfif"}

_image_service: ImageService | None = None


def get_image_service() -> ImageService:
    global _image_service
    if _image_service is None:
        _image_service = ImageService(workbench_settings())
    return _image_service


def is_image_upload(filename: str, head: bytes) -> bool:
    """Route by real content first; an image-like name with other bytes is still rejected as an image."""
    return sniff_format(head) is not None or Path(filename).suffix.lower() in IMAGE_SUFFIXES


def error_response(exc: WorkbenchError, request_id: str | None) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"error": exc.payload(request_id)})


async def analyze_upload_request(file: UploadFile, mode: str, method_id: str | None, request_id: str | None):
    """Shared by /analyze/file, /detect and /provenance (explicit arguments, no Form defaults)."""
    from api.analysis import get_analysis_service
    filename = file.filename or "upload.bin"
    head = await file.read(64)
    if is_image_upload(filename, head):
        service = get_image_service()
        try:
            request_id = service.validate_request(mode, method_id, request_id)
        except WorkbenchError as exc:
            return error_response(exc, request_id if isinstance(request_id, str) and len(request_id) <= 64 else None)
        limit = service.limits["max_bytes"]
        content = head + await file.read(limit + 1 - len(head))  # bounded read
        if len(content) > limit:
            return error_response(WorkbenchError("too_large", f"文件超过 {limit // (1024 * 1024)} MiB 上限"), request_id)
        return await run_in_threadpool(analyze_image_upload, filename, content, mode, method_id, request_id)
    if mode != "standard" or method_id:
        return error_response(WorkbenchError("unsupported_mode", "研究模式只适用于图片；文本流程保持原行为"), None)
    content = head + await file.read()
    if not content:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return get_analysis_service().analyze_upload(filename, content)


def analyze_image_upload(filename: str, content: bytes, mode: str, method_id: str | None, request_id: str):
    try:
        result, deduplicated = get_image_service().analyze(
            content, mode=mode, method_id=method_id, request_id=request_id,
            persist=lambda r: history.save_image(r, filename))
    except WorkbenchError as exc:
        log.info("image request_id=%s error=%s", request_id, exc.code)
        return error_response(exc, request_id)
    log.info("image request_id=%s sha256=%s mode=%s method=%s status=%s timing_ms=%s dedup=%s", request_id,
             result["input"]["sha256"], mode, method_id, result["status"], result["runtime"]["timing_ms"], deduplicated)
    payload = history.legacy_image_payload(result, filename)
    payload["deduplicated"] = deduplicated
    return payload


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    from api.analysis import get_analysis_service
    settings = workbench_settings()
    text = get_analysis_service().providers()
    return {
        "schema_version": "capabilities-v1", "preview": bool(settings),
        "text": {"available": True, "mode": text["mode"], "research_mode": False,
                 "providers": [{k: p.get(k) for k in ("name", "type", "enabled", "configured")} for p in text["providers"]]},
        "image": get_image_service().capabilities(),
        "history": {"stores_original_images": False, "endpoint": "/api/v1/reports"},
        "experiments": {"available": bool(settings.get("catalog"))},
    }


@router.get("/reports")
def reports(modality: str | None = Query(None, pattern="^(text|image|audio|video)$"),
            status: str | None = Query(None, pattern="^(ok|partial|legacy)$"),
            limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
            q: str | None = Query(None, max_length=128)):
    return history.list_reports(modality, status, limit, offset, q)


@router.get("/experiments")
def experiments():
    settings = workbench_settings()
    if not settings.get("catalog"):
        return error_response(WorkbenchError("model_unavailable", "未配置实验目录"), None)
    return build_catalog(settings)
