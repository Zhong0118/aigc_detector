"""溯源流程编排。

这个文件负责在检测结果之后，决定是否运行 provenance 检查。
它不会自己做 AI 检测，而是消费 detection 层给出的 DetectionPackage。

当前实现：
- 分数超过 deep_threshold 才跑深度溯源
- C2PA 和指纹库属于轻量检查，会尽量总是运行
- C2PA 优先调用 c2patool，缺工具时 fallback 轻量 parser
- image watermark 按 deep_threshold 调用 LSB placeholder
- text/audio/video watermark 只返回 reserved
- 指纹库会对历史内容做 exact/near lookup
- attribution 优先使用 detection provider 给出的 model_scores

后续需要补充：
- c2patool adapter：支持更多文件和签名验证
- Meta Seal adapter：支持 TextSeal/AudioSeal/VideoSeal/image watermark
- FingerprintRegistry 真正接入 lookup
- provenance 结果置信度融合
- 记录每个溯源模块耗时和错误
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detection.providers import DetectionPackage
from ingestion import compute_fingerprint
from .c2pa_reader import read_c2pa_metadata
from .fingerprint_registry import FingerprintRegistry
from .metaseal_adapter import MetaSealWatermarkDetector


class ProvenancePipeline:
    """统一溯源 pipeline。

    api/analysis.py 会在 detection 之后调用这个类。
    它的输出会进入前端展示、报告生成和数据库存储。
    """

    def __init__(
        self,
        deep_threshold: float = 0.6,
        c2pa_tool_path: str | None = None,
        watermark_config: dict[str, Any] | None = None,
    ) -> None:
        """初始化深度溯源阈值。

        detection.score >= deep_threshold 时，才触发 C2PA/watermark 等深度检查。
        """
        self.deep_threshold = deep_threshold
        self.c2pa_tool_path = c2pa_tool_path
        self.registry = FingerprintRegistry()
        self.watermark_config = watermark_config or {}
        self.watermark_detector = MetaSealWatermarkDetector(self.watermark_config)

    def analyze(
        self,
        path: str | Path,
        modality: str,
        detection: DetectionPackage,
        fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """执行一次溯源分析。

        path: 临时文件路径
        modality: text/image/audio/video
        detection: detection 层的统一检测结果
        fingerprint: ingestion 层已计算的内容指纹；不传时本方法兜底计算
        """
        path = Path(path)
        fingerprint = fingerprint or compute_fingerprint(path, modality)
        deep_triggered = detection.score >= self.deep_threshold
        result: dict[str, Any] = {
            "deep_triggered": deep_triggered,
            "threshold": self.deep_threshold,
            "fingerprint": fingerprint,
            "c2pa": self._run_c2pa(path, modality),
            "watermark": self._skipped("watermark") if not deep_triggered else self._run_watermark(path, modality),
            "fingerprint_registry": self._run_fingerprint_registry(fingerprint, modality),
            "attribution": self._attribution_from_detection(detection, modality),
        }
        return result

    def _run_c2pa(self, path: Path, modality: str) -> dict[str, Any]:
        """运行 C2PA 检查。

        当前只调用轻量 parser。后续应该替换/补充为 c2patool：
        `c2patool file --json`
        并解析签名、claim generator、assertions 等字段。
        """
        if modality not in {"image", "video", "audio"}:
            return {
                "status": "not_applicable",
                "found": False,
                "note": "C2PA is mainly used here for media files and documents.",
            }

        metadata = read_c2pa_metadata(path, tool_path=self.c2pa_tool_path)
        return {
            "status": "ok",
            "found": metadata is not None,
            "metadata": metadata,
            "mode": metadata.get("parser") if isinstance(metadata, dict) else "not_found",
            "note": (
                "C2PA metadata parsed with c2patool or local fallback."
                if metadata
                else "No C2PA manifest was found, or c2patool/local parser could not parse one."
            ),
        }

    def _run_fingerprint_registry(self, fingerprint: str, modality: str) -> dict[str, Any]:
        """运行历史指纹库检索。

        这个检查不依赖 deep_threshold，低分内容也可以用于查重和历史命中。
        """
        try:
            matches = self.registry.lookup_as_dicts(
                fingerprint,
                modality=modality,
                max_distance=8 if modality == "image" else 0,
                limit=5,
            )
            return {
                "status": "ok",
                "matches": matches,
                "match_count": len(matches),
                "note": (
                    "Found historical or known-content fingerprint matches."
                    if matches
                    else "No exact or near fingerprint match found in the local registry."
                ),
            }
        except Exception as exc:  # noqa: BLE001 - provenance failure should not break detection.
            return {
                "status": "error",
                "matches": [],
                "match_count": 0,
                "error": str(exc),
                "note": "Fingerprint registry lookup failed.",
            }

    def _run_watermark(self, path: Path, modality: str) -> dict[str, Any]:
        """运行水印检查。

        当前优先走 Meta Seal 家族适配器：
        - image/video: VideoSeal
        - audio: AudioSeal
        - text: TextSeal CLI（配置 command 后启用）
        """
        if not self.watermark_config.get("enabled", True):
            return {
                "status": "disabled",
                "provider": "Meta Seal",
                "result": None,
                "note": "Watermark detection is disabled in config.",
            }
        return self.watermark_detector.detect(path, modality)

    def _attribution_from_detection(self, detection: DetectionPackage, modality: str) -> dict[str, Any]:
        """生成模型来源提示。

        当前优先复用 detection provider 返回的 model_scores。
        如果真实 API 给出 Midjourney/SDXL/GPT family 等分数，就会在这里转成 Top-K。
        如果没有，则返回 local attribution reserved。
        """
        if detection.model_scores:
            top_k = [
                {"model": model, "probability": score}
                for model, score in list(detection.model_scores.items())[:3]
            ]
            return {
                "status": "ok",
                "source": "detection_provider_model_scores",
                "top_k": top_k,
                "confidence": top_k[0]["probability"] if top_k else 0.0,
            }

        return {
            "status": "reserved",
            "source": "local_attribution_model",
            "top_k": [],
            "confidence": 0.0,
            "note": f"Local model attribution is reserved for {modality}.",
        }

    def _skipped(self, name: str) -> dict[str, Any]:
        """生成某个溯源模块被跳过时的统一结果。"""
        return {
            "status": "skipped",
            "found": False,
            "note": f"{name} check skipped because the AI score did not reach the deep provenance threshold.",
        }
