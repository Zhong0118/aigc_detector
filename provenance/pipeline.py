"""溯源流程编排。

这个文件负责在检测结果之后，决定是否运行更深的 provenance 检查。
它不会自己做 AI 检测，而是消费 detection 层给出的 DetectionPackage。

当前实现：
- 分数超过 deep_threshold 才跑深度溯源
- C2PA 调用 c2pa_reader.py 的轻量 parser
- image watermark 调用 LSB placeholder
- text/audio/video watermark 只返回 reserved
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

from PIL import Image

from detection.providers import DetectionPackage
from .c2pa_reader import read_c2pa_metadata
from .watermark_decoder import decode_watermark


class ProvenancePipeline:
    """统一溯源 pipeline。

    api/analysis.py 会在 detection 之后调用这个类。
    它的输出会进入前端展示、报告生成和数据库存储。
    """

    def __init__(self, deep_threshold: float = 0.6) -> None:
        """初始化深度溯源阈值。

        detection.score >= deep_threshold 时，才触发 C2PA/watermark 等深度检查。
        """
        self.deep_threshold = deep_threshold

    def analyze(self, path: str | Path, modality: str, detection: DetectionPackage) -> dict[str, Any]:
        """执行一次溯源分析。

        path: 临时文件路径
        modality: text/image/audio/video
        detection: detection 层的统一检测结果
        """
        path = Path(path)
        deep_triggered = detection.score >= self.deep_threshold
        result: dict[str, Any] = {
            "deep_triggered": deep_triggered,
            "threshold": self.deep_threshold,
            "c2pa": self._skipped("c2pa") if not deep_triggered else self._run_c2pa(path, modality),
            "watermark": self._skipped("watermark") if not deep_triggered else self._run_watermark(path, modality),
            "fingerprint_registry": {
                "status": "reserved",
                "matches": [],
                "note": "Hash/fingerprint registry lookup is reserved for known-content matching.",
            },
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

        metadata = read_c2pa_metadata(path)
        return {
            "status": "ok",
            "found": metadata is not None,
            "metadata": metadata,
            "mode": "local",
            "note": "Local C2PA parser executed. c2patool can replace this parser for broader media support.",
        }

    def _run_watermark(self, path: Path, modality: str) -> dict[str, Any]:
        """运行水印检查。

        当前只有 image 会跑本地 LSB placeholder。
        text/audio/video 只返回 reserved，表示接口位置已经预留。
        """
        if modality == "image":
            try:
                image = Image.open(path).convert("RGB")
                return {
                    "status": "ok",
                    "provider": "local_lsb_placeholder",
                    "result": decode_watermark(image),
                    "note": "Image placeholder decoder ran. Meta Seal can replace this adapter later.",
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "provider": "local_lsb_placeholder",
                    "result": None,
                    "error": str(exc),
                }

        provider = {
            "text": "TextSeal",
            "audio": "AudioSeal",
            "video": "VideoSeal",
        }.get(modality, "Meta Seal")
        return {
            "status": "reserved",
            "provider": provider,
            "result": None,
            "note": f"{provider} interface is reserved for local watermark detection.",
        }

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
