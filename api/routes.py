"""FastAPI 路由定义。

这个文件负责声明“外部怎么调用后端”，也就是 HTTP API 层。
它不直接写检测模型逻辑，而是把请求交给 `AnalysisService`。

后续如果新增接口，优先保持这个原则：
- HTTP 参数校验写在 routes.py
- 业务编排写在 analysis.py
- 具体模型/API 调用写在 detection/ 或 provenance/
"""

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.analysis import get_analysis_service
from api.workbench_routes import analyze_upload_request
from workbench import history
from workbench.image_engine import WorkbenchError

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
async def analyze_file(
    file: UploadFile = File(...),
    mode: str = Form("standard"),
    research_method_id: str | None = Form(None),
    request_id: str | None = Form(None),
):
    """分析用户上传的文件。

    图片（按真实 JPEG/PNG 内容识别）走独立图片服务：D-OpenCLIP 检测 + 门控后的 A0 来源候选，
    返回旧字段与权威的 image_result；模型不可用、输入不合规时返回类型化错误，不回退到演示分数。
    其他文件仍交给 AnalysisService.analyze_upload，行为不变；research 模式只适用于图片。
    """
    return await analyze_upload_request(file, mode, research_method_id, request_id)


@router.post("/detect")
async def detect_content(file: UploadFile = File(...)):
    """兼容旧版本的检测接口：与 /analyze/file 共用同一 helper，并显式使用 standard 模式。"""
    return await analyze_upload_request(file, "standard", None, None)


@router.post("/provenance")
async def analyze_provenance(file: UploadFile = File(...)):
    """兼容旧版本的溯源接口。

    先跑完整分析（显式 standard 模式），再只抽取 provenance 相关字段返回；
    图片错误按同样的类型化错误返回，不投影为 0 分或 human。
    """
    analysis = await analyze_upload_request(file, "standard", None, None)
    if not isinstance(analysis, dict):
        return analysis
    return {
        "filename": analysis["filename"],
        "modality": analysis["modality"],
        "detection_score": analysis["detection"]["score"],
        "provenance": analysis["provenance"],
    }


@router.get("/report/{content_id}")
async def get_report(content_id: int):
    """按数据库 ID 读取历史报告。

    保留原有 content / detection_results / provenance 字段；有版本化快照时另给出当时的
    image_result 或 report_snapshot，历史不按当前模型重算。
    """
    try:
        return history.report_detail(content_id)
    except WorkbenchError as exc:
        # 保留旧的 detail 字段，同时给出统一错误体。
        return JSONResponse(status_code=exc.status, content={"error": exc.payload(None), "detail": "Content not found"})
