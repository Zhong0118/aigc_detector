"""统一分析编排服务。

这个文件是当前后端最核心的“业务流程层”。
它不关心 HTTP 细节，也不关心前端展示，只负责把一次分析任务串起来：

1. ingestion: 识别类型、读取内容、提取 metadata、计算 fingerprint
2. detection: 调用 API-first 检测适配器
3. provenance: 根据阈值决定是否做 C2PA / watermark / attribution
4. reports: 生成解释报告
5. storage: 写入数据库

后续扩展预留点：
- 接真实检测 API：主要改 `detection/providers.py`，本文件不用大改。
- 接本地小模型：仍然通过 detector 返回统一 detection 结构。
- 接 LLM 报告：主要改 `reports/generator.py`，本文件只继续调用 reporter.generate。
- 换数据库：主要改 `storage/`，本文件只保留 `_store_analysis` 的统一写入入口。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from detection.providers import ApiFirstDetectionEngine
from ingestion import compute_fingerprint, extract_metadata, load_content
from provenance.pipeline import ProvenancePipeline
from reports.generator import ReportGenerator
from storage.database import get_session
from storage.models import Content, DetectionResult, ProvenanceRecord


class AnalysisService:
    """AIGC 内容分析总服务。

    FastAPI 路由不要直接调用 detector/provenance/report/storage，
    而是统一通过这个类。这样后续换 API、换模型、换数据库时，
    外部接口不用跟着大改。
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """初始化分析服务。

        config_path 指向项目配置文件。这里会加载：
        - 检测阈值
        - 深度溯源阈值
        - provider 配置
        - 端口和报告设置
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        # 检测器是一个“适配器入口”。现在默认是 demo_api，
        # 未来 Hive/Sightengine/本地模型都应该藏在 ApiFirstDetectionEngine 后面。
        self.detector = ApiFirstDetectionEngine(config_path=config_path)
        deep_threshold = float(
            self.config.get("detection", {}).get("deep_provenance_threshold", 0.6)
        )
        provenance_cfg = self.config.get("provenance", {})
        c2pa_tool_path = provenance_cfg.get("c2pa", {}).get("tool_path")
        # 溯源 pipeline 只关心检测结果和阈值，不直接关心检测来自哪个模型/API。
        self.provenance = ProvenancePipeline(
            deep_threshold=deep_threshold,
            c2pa_tool_path=c2pa_tool_path,
            watermark_config=provenance_cfg.get("watermark", {}),
        )
        # 报告生成器现在是模板版，后续可以在 reports/generator.py 内接 LLM。
        self.reporter = ReportGenerator()

    def analyze_file_path(self, path: str | Path, filename: str | None = None) -> dict[str, Any]:
        """分析一个已经存在于本地磁盘上的文件。

        这是最核心的方法。文本输入和文件上传最后都会转成临时文件，
        然后进入这个统一入口。
        """
        path = Path(path)
        # load_content 会判断 text/image/audio/video，并读取原始内容。
        item = load_content(path)
        # metadata 是文件大小、扩展名、时间戳、图片 EXIF 等信息。
        metadata = extract_metadata(path)
        # fingerprint 是内容指纹，用于去重、历史匹配和未来指纹库检索。
        fingerprint = compute_fingerprint(path, item.modality)
        # detector 当前走 API-first 适配器；默认是 demo_api，后续可接 Hive/Sightengine/本地模型。
        # 约定：不管后面接几个 API 或模型，都要归一化成 DetectionPackage。
        detection = self.detector.detect(path, item.modality, item.raw_data)
        # provenance 根据检测分数判断是否触发深度溯源。
        # 约定：C2PA、Meta Seal、指纹库、模型归因都放在 provenance 层内部扩展。
        provenance = self.provenance.analyze(path, item.modality, detection, fingerprint=fingerprint)

        # analysis 是统一返回给前端的 JSON 主体。
        analysis = {
            "filename": filename or path.name,
            "modality": item.modality,
            "metadata": metadata,
            "fingerprint": fingerprint,
            "detection": {
                "score": detection.score,
                "label": detection.label,
                "threshold": detection.threshold,
                "modality_scores": detection.modality_scores,
                "model_scores": detection.model_scores,
                # providers 保存每个检测来源的原始归一化结果，方便前端解释和后续调试。
                "providers": [result.__dict__ for result in detection.provider_results],
            },
            "provenance": provenance,
        }
        # 报告生成器把结构化证据转成用户可读的 summary/evidence/limitations。
        # 后续接 LLM 时，也建议只把 analysis 这种结构化证据传给 LLM，而不是让 LLM 凭空判断。
        analysis["report"] = self.reporter.generate(analysis)
        # 把本次分析写入数据库，并把数据库主键回填给前端。
        content_id = self._store_analysis(analysis)
        analysis["content_id"] = content_id
        return analysis

    def analyze_text(self, text: str, filename: str = "input.txt") -> dict[str, Any]:
        """分析用户在前端文本框里输入的内容。

        为了复用文件型 pipeline，这里会把文本写入临时 .txt 文件，
        分析结束后再删除临时文件。
        """
        suffix = Path(filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        try:
            return self.analyze_file_path(tmp_path, filename=filename)
        finally:
            tmp_path.unlink(missing_ok=True)

    def analyze_upload(self, filename: str, content: bytes) -> dict[str, Any]:
        """分析用户上传的二进制文件。

        FastAPI 的 UploadFile 读出来是 bytes。这里先写入临时文件，
        再交给 analyze_file_path 处理，避免每种输入写一套流程。
        """
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            return self.analyze_file_path(tmp_path, filename=filename)
        finally:
            tmp_path.unlink(missing_ok=True)

    def providers(self) -> dict[str, Any]:
        """返回系统当前 provider 和端口状态。

        Streamlit Dashboard 顶部会调用这个方法展示：
        - 当前检测模式
        - 检测阈值
        - 深度溯源阈值
        - API/Dashboard 端口
        - demo_api/Hive/Sightengine/local_models 的状态
        """
        return {
            "mode": self.config.get("detection", {}).get("mode", "api_first"),
            "threshold": self.config.get("detection", {}).get("threshold", 0.5),
            "deep_provenance_threshold": self.config.get("detection", {}).get(
                "deep_provenance_threshold", 0.6
            ),
            "providers": self.detector.provider_status(),
            "ports": {
                "api": self.config.get("api", {}).get("port", 8000),
                "dashboard": self.config.get("dashboard", {}).get("port", 8501),
            },
        }

    def _store_analysis(self, analysis: dict[str, Any]) -> int:
        """把一次分析结果写入数据库。

        当前会写三张表：
        - Content: 文件本身的信息和 fingerprint
        - DetectionResult: AI 分数、标签、provider 详情
        - ProvenanceRecord: C2PA、水印、Top-K attribution 等溯源信息
        """
        session = get_session()
        try:
            # 内容主表：记录文件名、模态、指纹、文件大小。
            db_content = Content(
                filename=analysis["filename"],
                modality=analysis["modality"],
                file_hash=analysis["fingerprint"],
                fingerprint=analysis["fingerprint"],
                file_size=analysis["metadata"].get("size_bytes"),
            )
            session.add(db_content)
            session.flush()

            # 检测结果表：记录最终分数、标签和完整 detection JSON。
            db_result = DetectionResult(
                content_id=db_content.id,
                modality=analysis["modality"],
                score=analysis["detection"]["score"],
                label=analysis["detection"]["label"],
                details=analysis["detection"],
            )
            session.add(db_result)

            # 溯源记录表：记录 C2PA、水印和模型来源提示。
            prov = ProvenanceRecord(
                content_id=db_content.id,
                c2pa_metadata=analysis["provenance"].get("c2pa"),
                watermark_info=analysis["provenance"].get("watermark"),
                attribution_top_k=analysis["provenance"].get("attribution", {}).get("top_k"),
                confidence=analysis["provenance"].get("attribution", {}).get("confidence"),
            )
            session.add(prov)
            session.commit()
            return int(db_content.id)
        finally:
            # 不管写入是否成功，都关闭 session。
            session.close()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """读取 YAML 配置文件。

        如果配置文件不存在，就返回空字典，让各模块使用自己的默认值。
        """
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


_service: AnalysisService | None = None


def get_analysis_service() -> AnalysisService:
    """获取全局单例 AnalysisService。

    FastAPI 每次请求都可以调用这个函数。第一次调用时创建服务实例，
    之后复用同一个实例，避免重复加载配置和初始化 provider。
    """
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service
