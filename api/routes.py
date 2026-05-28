import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from ingestion import load_content, extract_metadata, compute_fingerprint
from detection import TextDetector, ImageDetector, AudioDetector, VideoDetector, FusionClassifier
from provenance import read_c2pa_metadata, decode_watermark, ModelAttribution
from storage.database import get_session
from storage.models import Content, DetectionResult, ProvenanceRecord

router = APIRouter()

fusion = FusionClassifier()
text_detector = None
image_detector = None
audio_detector = None
video_detector = None
attribution = None


def _get_text_detector():
    global text_detector
    if text_detector is None:
        text_detector = TextDetector()
    return text_detector


def _get_image_detector():
    global image_detector
    if image_detector is None:
        image_detector = ImageDetector()
    return image_detector


def _get_audio_detector():
    global audio_detector
    if audio_detector is None:
        audio_detector = AudioDetector()
    return audio_detector


def _get_video_detector():
    global video_detector
    if video_detector is None:
        video_detector = VideoDetector()
    return video_detector


def _get_attribution():
    global attribution
    if attribution is None:
        attribution = ModelAttribution()
    return attribution


@router.post("/detect")
async def detect_content(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        item = load_content(tmp_path)
        meta = extract_metadata(tmp_path)
        fp = compute_fingerprint(tmp_path, item.modality)

        scores = {}
        details = {}

        if item.modality == "text":
            result = _get_text_detector().detect(item.raw_data)
            scores["text"] = result.score
            details["text"] = result.details
        elif item.modality == "image":
            result = _get_image_detector().detect(item.raw_data)
            scores["image"] = result.score
            details["image"] = result.details
        elif item.modality == "audio":
            result = _get_audio_detector().detect(item.raw_data)
            scores["audio"] = result.score
            details["audio"] = result.details
        elif item.modality == "video":
            result = _get_video_detector().detect(tmp_path)
            scores["video"] = result.score
            details["video"] = result.details

        fused = fusion.fuse(scores)

        session = get_session()
        try:
            db_content = Content(
                filename=file.filename,
                modality=item.modality,
                file_hash=fp,
                fingerprint=fp,
                file_size=len(content),
            )
            session.add(db_content)
            session.flush()

            db_result = DetectionResult(
                content_id=db_content.id,
                modality=item.modality,
                score=fused.score,
                label=fused.label,
                details=details,
            )
            session.add(db_result)
            session.commit()
        finally:
            session.close()

        return {
            "filename": file.filename,
            "modality": item.modality,
            "metadata": meta,
            "fingerprint": fp,
            "detection": {
                "score": fused.score,
                "label": fused.label,
                "modality_scores": fused.modality_scores,
                "details": details,
            },
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/provenance")
async def analyze_provenance(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        item = load_content(tmp_path)

        c2pa = None
        watermark = None

        if item.modality == "image":
            c2pa = read_c2pa_metadata(tmp_path)
            watermark = decode_watermark(item.raw_data)

        attr = _get_attribution()
        attr_result = attr.attribute({"score": 0.5}, top_k=3)

        session = get_session()
        try:
            db_content = Content(
                filename=file.filename,
                modality=item.modality,
                fingerprint="",
                file_size=len(content),
            )
            session.add(db_content)
            session.flush()

            prov = ProvenanceRecord(
                content_id=db_content.id,
                c2pa_metadata=c2pa,
                watermark_info=watermark,
                attribution_top_k=[r for r in attr_result.top_k],
                confidence=attr_result.confidence,
            )
            session.add(prov)
            session.commit()
        finally:
            session.close()

        return {
            "filename": file.filename,
            "c2pa": c2pa,
            "watermark": watermark,
            "attribution": {
                "top_k": attr_result.top_k,
                "confidence": attr_result.confidence,
            },
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/report/{content_id}")
async def get_report(content_id: int):
    session = get_session()
    try:
        content = session.query(Content).filter(Content.id == content_id).first()
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        results = session.query(DetectionResult).filter(
            DetectionResult.content_id == content_id
        ).all()
        provenance = session.query(ProvenanceRecord).filter(
            ProvenanceRecord.content_id == content_id
        ).all()

        return {
            "content": {
                "id": content.id,
                "filename": content.filename,
                "modality": content.modality,
                "fingerprint": content.fingerprint,
            },
            "detection_results": [
                {"score": r.score, "label": r.label, "details": r.details}
                for r in results
            ],
            "provenance": [
                {
                    "c2pa": p.c2pa_metadata,
                    "watermark": p.watermark_info,
                    "attribution": p.attribution_top_k,
                    "confidence": p.confidence,
                }
                for p in provenance
            ],
        }
    finally:
        session.close()
