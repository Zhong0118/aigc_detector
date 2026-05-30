"""FastAPI 路由定义。

这个文件负责声明“外部怎么调用后端”，也就是 HTTP API 层。
它不直接写检测模型逻辑，而是把请求交给 `AnalysisService`。

后续如果新增接口，优先保持这个原则：
- HTTP 参数校验写在 routes.py
- 业务编排写在 analysis.py
- 具体模型/API 调用写在 detection/ 或 provenance/
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from api.analysis import get_analysis_service
from storage.database import get_session
from storage.models import Content, DetectionResult, ProvenanceRecord

# APIRouter 是 FastAPI 的子路由容器。
# main.py 会把这个 router 挂载到 `/api/v1` 前缀下。
router = APIRouter()


class TextAnalyzeRequest(BaseModel):
    """文本检测请求体。

    前端调用 `/analyze/text` 时会传 JSON：
    {
        "text": "...",
        "filename": "input.txt"
    }
    filename 是虚拟文件名，用来复用 ingestion 里的文件类型判断逻辑。
    """

    text: str = Field(min_length=1)
    filename: str = "input.txt"


@router.get("/providers")
async def get_providers():
    """返回当前检测 provider 和端口配置。

    前端 Dashboard 顶部的 provider 状态栏就是调用这个接口。
    它会告诉用户 demo_api、Hive、Sightengine、本地模型是否启用/配置。
    后续新增 OpenAI 报告、c2patool、Meta Seal 等 provider 状态时，
    也可以统一挂在 AnalysisService.providers() 返回值里。
    """
    return get_analysis_service().providers()


@router.post("/analyze/text")
async def analyze_text(request: TextAnalyzeRequest):
    """分析用户直接输入的文本。

    这个接口会把文本临时保存成一个 .txt 文件，再进入统一分析流程：
    ingestion -> detection -> provenance -> report -> storage。
    """
    return get_analysis_service().analyze_text(request.text, filename=request.filename)


@router.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """分析用户上传的文件。

    支持文本、图片、音频、视频等文件。这里先读取上传内容，
    然后交给 AnalysisService.analyze_upload 统一处理。
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return get_analysis_service().analyze_upload(file.filename or "upload.bin", content)


@router.post("/detect")
async def detect_content(file: UploadFile = File(...)):
    """兼容旧版本的检测接口。

    早期前端调用的是 `/detect`。为了不让旧调用失效，
    这里直接复用新的 `/analyze/file` 逻辑。
    """
    return await analyze_file(file)


@router.post("/provenance")
async def analyze_provenance(file: UploadFile = File(...)):
    """兼容旧版本的溯源接口。

    当前实现会先跑完整分析，再只抽取 provenance 相关字段返回。
    这样可以避免检测流程和溯源流程分叉成两套逻辑。
    """
    analysis = await analyze_file(file)
    return {
        "filename": analysis["filename"],
        "modality": analysis["modality"],
        "detection_score": analysis["detection"]["score"],
        "provenance": analysis["provenance"],
    }


@router.get("/report/{content_id}")
async def get_report(content_id: int):
    """按数据库 ID 读取历史报告。

    content_id 是 storage.models.Content 的主键。
    这个接口会把内容信息、检测结果和溯源记录一起查出来返回。
    """
    session = get_session()
    try:
        # 先查内容主表；如果不存在，直接返回 404。
        content = session.query(Content).filter(Content.id == content_id).first()
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # 一个内容可以有多条检测记录，这里按 content_id 全部取出。
        results = session.query(DetectionResult).filter(
            DetectionResult.content_id == content_id
        ).all()
        # 一个内容也可以有多条溯源记录，这里同样全部取出。
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
        # SQLAlchemy session 用完必须关闭，避免连接泄漏。
        session.close()
