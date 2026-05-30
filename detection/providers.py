"""API-first 检测 provider 适配层。

这个文件是当前检测主路径。FastAPI 不直接调用模型，
而是通过 `ApiFirstDetectionEngine` 得到统一的检测结果。

当前真实状态：
- demo_api 已实现，用于快速跑通端到端流程
- Hive/Sightengine 只预留接口，还没有真实 HTTP 请求
- local_models 只预留开关，还没有接入到主流程

后续真实接 API 时，建议保持 ProviderResult 这个统一返回格式，
这样前端、报告、数据库都不用改。

detection 后续需要补充的内容：
- 真实 API 请求：Hive、Sightengine 或其他服务的 HTTP client
- 超时/重试/限流：避免外部 API 卡住整个后端
- provider registry：按 modality 自动选择可用 provider
- 分数校准：不同 provider 的 0.8 不一定含义相同，需要统一校准
- 本地模型权重管理：模型路径、版本、device、懒加载、缓存
- 评估集脚本：用固定测试集评估准确率、召回率、误报率
- 审计日志：保存每个 provider 的请求 ID、耗时、错误原因
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml


@dataclass
class ProviderResult:
    """单个检测来源的归一化结果。

    不管检测来自 demo_api、Hive、Sightengine 还是本地模型，
    都应该被转换成这个结构。
    """

    provider: str
    provider_type: str
    status: str
    score: float | None
    label: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DetectionPackage:
    """一次检测的总结果。

    provider_results 保存所有检测来源；
    score/label 是融合后的最终结果；
    model_scores 是可能的来源模型提示，例如 Midjourney、SDXL、GPT family。
    """

    score: float
    label: str
    threshold: float
    provider_results: list[ProviderResult]
    modality_scores: dict[str, float]
    model_scores: dict[str, float]


class ApiFirstDetectionEngine:
    """API-first 检测引擎。

    当前由 api/analysis.py 调用。它负责：
    - 根据 config 判断启用哪些 provider
    - 运行 demo_api 或未来真实 API
    - 汇总 provider 分数
    - 输出统一 DetectionPackage
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """读取检测配置。

        关键配置项：
        - detection.threshold
        - detection.demo_provider_enabled
        - detection.api_providers.hive
        - detection.api_providers.sightengine
        - detection.local_models.enabled
        """
        self.config = self._load_config(config_path)
        detection_cfg = self.config.get("detection", {})
        self.threshold = float(detection_cfg.get("threshold", 0.5))
        self.demo_enabled = bool(detection_cfg.get("demo_provider_enabled", True))
        self.api_providers = detection_cfg.get("api_providers", {})
        self.local_enabled = bool(detection_cfg.get("local_models", {}).get("enabled", False))
        # 后续可以在这里初始化 provider registry，例如：
        # {"image": [SightengineProvider(), LocalImageProvider()], "text": [HiveProvider()]}
        # 当前为了 MVP 简化为 if/else 分发。

    def detect(self, path: str | Path, modality: str, raw_data: object | None = None) -> DetectionPackage:
        """执行一次内容检测。

        path 是临时文件路径；
        modality 来自 ingestion.detect_modality；
        raw_data 是 ingestion.load_content 读取出的对象。
        """
        path = Path(path)
        results: list[ProviderResult] = []

        # 如果 config 中打开 Hive/Sightengine，这里会进入对应 provider。
        # provider 失败不会中断全流程，而是返回 status=error，方便前端展示。
        results.extend(self._run_configured_api_providers(path, modality))

        if self.demo_enabled:
            # demo_api 是当前唯一真正返回分数的 provider。
            results.append(self._run_demo_provider(path, modality, raw_data))

        if self.local_enabled:
            # 本地模型开关预留。后续可把 text/image/audio/video detector 接到这里。
            results.append(
                ProviderResult(
                    provider="local_models",
                    provider_type="local_model",
                    status="reserved",
                    score=None,
                    label="not_run",
                    details={
                        "note": "Local model adapters are reserved; enable concrete wrappers after model selection."
                    },
                )
            )

        # 只融合 status=ok 且有 score 的 provider。
        # reserved/not_configured provider 不参与最终分数。
        usable_scores = [result.score for result in results if result.status == "ok" and result.score is not None]
        final_score = sum(usable_scores) / len(usable_scores) if usable_scores else 0.0
        label = "ai" if final_score >= self.threshold else "human"
        model_scores = self._merge_model_scores(results)

        return DetectionPackage(
            score=round(final_score, 4),
            label=label,
            threshold=self.threshold,
            provider_results=results,
            modality_scores={modality: round(final_score, 4)},
            model_scores=model_scores,
        )

    def provider_status(self) -> list[dict[str, Any]]:
        """返回 provider 配置状态。

        Streamlit Dashboard 会调用 `/api/v1/providers` 展示这些信息。
        这里不运行检测，只汇报当前 provider 是否启用、是否配置 key。
        """
        providers = [
            {
                "name": "demo_api",
                "type": "api_demo",
                "enabled": self.demo_enabled,
                "configured": True,
                "purpose": "Fast end-to-end demo without external keys.",
            }
        ]

        hive_cfg = self.api_providers.get("hive", {})
        providers.append(
            {
                "name": "hive",
                "type": "detection_api",
                "enabled": bool(hive_cfg.get("enabled", False)),
                "configured": bool(os.getenv(hive_cfg.get("api_key_env", "HIVE_API_KEY"))),
                "purpose": "Cloud AI-generated content detection for multiple modalities.",
            }
        )

        sight_cfg = self.api_providers.get("sightengine", {})
        providers.append(
            {
                "name": "sightengine",
                "type": "detection_api",
                "enabled": bool(sight_cfg.get("enabled", False)),
                "configured": bool(
                    os.getenv(sight_cfg.get("api_user_env", "SIGHTENGINE_API_USER"))
                    and os.getenv(sight_cfg.get("api_secret_env", "SIGHTENGINE_API_SECRET"))
                ),
                "purpose": "Cloud image/video AI detection and generator hints.",
            }
        )

        providers.append(
            {
                "name": "local_models",
                "type": "local_model",
                "enabled": self.local_enabled,
                "configured": False,
                "purpose": "Reserved adapters for text/image/audio/video local models.",
            }
        )
        return providers

    def _run_configured_api_providers(self, path: Path, modality: str) -> list[ProviderResult]:
        """运行配置中启用的外部 API provider。

        当前 Hive/Sightengine 已经会按配置真实请求。
        这里做统一分发，每个 provider 单独处理自己的鉴权、endpoint 和返回格式。
        """
        results: list[ProviderResult] = []
        hive_cfg = self.api_providers.get("hive", {})
        if hive_cfg.get("enabled", False):
            results.append(self._run_hive_provider(path, modality, hive_cfg))

        sight_cfg = self.api_providers.get("sightengine", {})
        if sight_cfg.get("enabled", False):
            results.append(self._run_sightengine_provider(path, modality, sight_cfg))
        return results

    def _not_configured_result(self, provider: str, missing_env: str | list[str]) -> ProviderResult:
        """生成“启用了 provider，但还没有配置凭据”的结果。"""
        missing = [missing_env] if isinstance(missing_env, str) else missing_env
        return ProviderResult(
            provider=provider,
            provider_type="detection_api",
            status="not_configured",
            score=None,
            label="not_run",
            details={
                "missing_env": missing,
                "note": "Provider is enabled, but credentials are missing from environment variables.",
            },
        )

    def _run_hive_provider(self, path: Path, modality: str, cfg: dict[str, Any]) -> ProviderResult:
        """调用 Hive 同步检测接口。

        Hive 的文本检测用 JSON `text_data`；图片/视频/音频用 multipart `media`。
        返回后统一抽取 ai_generated 概率，后续前端不用理解 Hive 原始 JSON。
        """
        key_env = str(cfg.get("api_key_env", "HIVE_API_KEY"))
        api_key = os.getenv(key_env)
        if not api_key:
            return self._not_configured_result("hive", key_env)

        endpoint = str(cfg.get("endpoint", "https://api.thehive.ai/api/v2/task/sync"))
        timeout = float(cfg.get("timeout_seconds", 45))
        model = str(cfg.get("models", {}).get(modality, self._hive_model_for_modality(modality)))
        headers = {"Authorization": f"token {api_key}"}

        try:
            if modality == "text":
                payload = {
                    "models": [model],
                    "text_data": self._read_text_payload(path),
                    "user_id": "local-aigc-detector",
                    "post_id": path.stem,
                }
                response = requests.post(
                    endpoint,
                    headers={**headers, "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
            else:
                with open(path, "rb") as media:
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        data={
                            "models": json.dumps([model]),
                            "user_id": "local-aigc-detector",
                            "post_id": path.stem,
                        },
                        files={
                            "media": (
                                path.name,
                                media,
                                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                            )
                        },
                        timeout=timeout,
                    )
            response.raise_for_status()
            payload = response.json()
            score = self._extract_ai_score(payload)
            model_scores = self._extract_model_scores(payload)
            if score is None:
                return ProviderResult(
                    provider="hive",
                    provider_type="detection_api",
                    status="error",
                    score=None,
                    label="not_run",
                    details={
                        "modality": modality,
                        "model": model,
                        "raw": payload,
                        "note": "Hive response did not contain a recognizable ai_generated score.",
                    },
                    error="Missing ai_generated score in Hive response.",
                )
            return ProviderResult(
                provider="hive",
                provider_type="detection_api",
                status="ok",
                score=round(score, 4),
                label="ai" if score >= self.threshold else "human",
                details={
                    "modality": modality,
                    "model": model,
                    "model_scores": model_scores,
                    "score_source": "max ai_generated-like score found in Hive response",
                    "raw": payload,
                },
            )
        except (requests.RequestException, OSError, ValueError, TypeError) as exc:
            return ProviderResult(
                provider="hive",
                provider_type="detection_api",
                status="error",
                score=None,
                label="not_run",
                details={"modality": modality, "model": model, "endpoint": endpoint},
                error=str(exc),
            )

    def _run_sightengine_provider(self, path: Path, modality: str, cfg: dict[str, Any]) -> ProviderResult:
        """调用 Sightengine 图片/视频 AIGC 检测接口。

        Sightengine 主要覆盖 image/video；文本和音频会返回 skipped，
        避免误把不支持的模态送过去。
        """
        if modality not in {"image", "video"}:
            return ProviderResult(
                provider="sightengine",
                provider_type="detection_api",
                status="skipped",
                score=None,
                label="not_run",
                details={
                    "modality": modality,
                    "note": "Sightengine AI-generation API is used here only for image/video.",
                },
            )

        user_env = str(cfg.get("api_user_env", "SIGHTENGINE_API_USER"))
        secret_env = str(cfg.get("api_secret_env", "SIGHTENGINE_API_SECRET"))
        api_user = os.getenv(user_env)
        api_secret = os.getenv(secret_env)
        missing = [name for name, value in [(user_env, api_user), (secret_env, api_secret)] if not value]
        if missing:
            return self._not_configured_result("sightengine", missing)

        endpoint_key = "video_endpoint" if modality == "video" else "image_endpoint"
        default_endpoint = (
            "https://api.sightengine.com/1.0/video/check-sync.json"
            if modality == "video"
            else "https://api.sightengine.com/1.0/check.json"
        )
        endpoint = str(cfg.get(endpoint_key, default_endpoint))
        timeout = float(cfg.get("timeout_seconds", 45))

        try:
            with open(path, "rb") as media:
                response = requests.post(
                    endpoint,
                    data={
                        "models": "genai",
                        "api_user": api_user,
                        "api_secret": api_secret,
                    },
                    files={
                        "media": (
                            path.name,
                            media,
                            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        )
                    },
                    timeout=timeout,
                )
            response.raise_for_status()
            payload = response.json()
            score = self._extract_ai_score(payload)
            model_scores = self._extract_model_scores(payload)
            if score is None:
                return ProviderResult(
                    provider="sightengine",
                    provider_type="detection_api",
                    status="error",
                    score=None,
                    label="not_run",
                    details={
                        "modality": modality,
                        "raw": payload,
                        "note": "Sightengine response did not contain a recognizable ai_generated score.",
                    },
                    error="Missing ai_generated score in Sightengine response.",
                )
            return ProviderResult(
                provider="sightengine",
                provider_type="detection_api",
                status="ok",
                score=round(score, 4),
                label="ai" if score >= self.threshold else "human",
                details={
                    "modality": modality,
                    "model_scores": model_scores,
                    "score_source": "max ai_generated-like score found in Sightengine response",
                    "raw": payload,
                },
            )
        except (requests.RequestException, OSError, ValueError, TypeError) as exc:
            return ProviderResult(
                provider="sightengine",
                provider_type="detection_api",
                status="error",
                score=None,
                label="not_run",
                details={"modality": modality, "endpoint": endpoint},
                error=str(exc),
            )

    def _hive_model_for_modality(self, modality: str) -> str:
        """按模态选择 Hive 模型名。"""
        return {
            "text": "ai_generated_text",
            "image": "ai_generated_media",
            "video": "ai_generated_media",
            "audio": "ai_generated_audio",
        }.get(modality, "ai_generated_media")

    def _read_text_payload(self, path: Path) -> str:
        """读取文本内容给 Hive 文本检测接口。"""
        return path.read_text(encoding="utf-8", errors="ignore")

    def _extract_ai_score(self, payload: Any) -> float | None:
        """从不同 provider 的 JSON 中抽取 AI 生成概率。

        外部 API 的 JSON 层级经常变化，所以这里使用递归抽取：
        看到 `ai_generated` 或 classes 中的 `ai_generated` 就收集分数。
        多个片段时取最高分，偏向召回可疑内容。
        """
        scores: list[float] = []
        self._collect_ai_scores(payload, scores)
        valid_scores = [score for score in scores if 0.0 <= score <= 1.0]
        if not valid_scores:
            return None
        return max(valid_scores)

    def _collect_ai_scores(self, value: Any, scores: list[float]) -> None:
        """递归收集 ai_generated-like score。"""
        if isinstance(value, dict):
            label_value = value.get("class") or value.get("label") or value.get("name")
            if self._is_ai_label(label_value) and self._is_number(value.get("score")):
                scores.append(float(value["score"]))

            for key, item in value.items():
                if self._is_ai_label(key) and self._is_number(item):
                    scores.append(float(item))
                else:
                    self._collect_ai_scores(item, scores)
        elif isinstance(value, list):
            for item in value:
                self._collect_ai_scores(item, scores)

    def _extract_model_scores(self, payload: Any) -> dict[str, float]:
        """抽取来源模型提示分数，例如 midjourney/stable_diffusion 等。"""
        scores: dict[str, float] = {}
        self._collect_model_scores(payload, scores)
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def _collect_model_scores(self, value: Any, scores: dict[str, float]) -> None:
        """递归收集可解释的模型来源分数。"""
        if isinstance(value, dict):
            label_value = value.get("class") or value.get("label") or value.get("name")
            if (
                isinstance(label_value, str)
                and not self._is_ai_label(label_value)
                and label_value.lower() not in {"human", "real", "natural"}
                and self._is_number(value.get("score"))
            ):
                scores[label_value] = max(scores.get(label_value, 0.0), float(value["score"]))

            type_value = value.get("type")
            if isinstance(type_value, dict):
                for key, item in type_value.items():
                    if (
                        not self._is_ai_label(key)
                        and key.lower() not in {"human", "real", "natural"}
                        and self._is_number(item)
                    ):
                        scores[key] = max(scores.get(key, 0.0), float(item))

            for item in value.values():
                self._collect_model_scores(item, scores)
        elif isinstance(value, list):
            for item in value:
                self._collect_model_scores(item, scores)

    def _is_ai_label(self, value: Any) -> bool:
        """判断字段名或类别名是否表示 AI 生成。"""
        if not isinstance(value, str):
            return False
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        return normalized in {"ai_generated", "generated", "synthetic", "is_ai"}

    def _is_number(self, value: Any) -> bool:
        """判断值是否能作为 0-1 概率分数。"""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _run_demo_provider(self, path: Path, modality: str, raw_data: object | None) -> ProviderResult:
        """运行 demo provider。

        注意：这个不是 AI 检测模型，只是为了让 MVP 有稳定输出。
        它根据内容 hash 生成确定性分数，方便调试 UI、API、报告和数据库。
        """
        payload = self._payload_for_scoring(path, raw_data)
        digest = hashlib.sha256(payload).hexdigest()
        base = int(digest[:8], 16) / 0xFFFFFFFF
        modality_bias = {"text": 0.08, "image": 0.13, "audio": 0.10, "video": 0.16}.get(modality, 0.0)
        score = min(max((base * 0.72) + modality_bias, 0.02), 0.98)
        label = "ai" if score >= self.threshold else "human"

        return ProviderResult(
            provider="demo_api",
            provider_type="api_demo",
            status="ok",
            score=round(score, 4),
            label=label,
            details={
                "modality": modality,
                "model_scores": self._demo_model_scores(modality, score),
                "note": "Deterministic demo provider for fast product validation. Replace with Hive/Sightengine or local models.",
            },
        )

    def _payload_for_scoring(self, path: Path, raw_data: object | None) -> bytes:
        """把不同类型输入转成 bytes，供 demo provider 计算 hash。

        文本直接编码为 UTF-8；其他文件优先读取原始文件 bytes。
        """
        if isinstance(raw_data, str):
            return raw_data.encode("utf-8", errors="ignore")
        try:
            return path.read_bytes()
        except OSError:
            return str(path).encode("utf-8")

    def _demo_model_scores(self, modality: str, score: float) -> dict[str, float]:
        """生成演示用来源模型分数。

        这些不是可靠溯源结果，只是让前端提前具备 Top-K source hints 的展示结构。
        后续真实来源模型分数应该来自检测 API、provenance 或 attribution 模型。
        """
        candidates = {
            "text": ["gpt-family", "claude-family", "gemini-family"],
            "image": ["midjourney", "stable-diffusion-xl", "flux"],
            "audio": ["elevenlabs", "bark", "audioseal-watermarked"],
            "video": ["kling", "wan-video", "runway"],
        }.get(modality, ["unknown-generator"])

        return {
            name: round(max(score - (idx * 0.13), 0.01), 4)
            for idx, name in enumerate(candidates)
        }

    def _merge_model_scores(self, results: list[ProviderResult]) -> dict[str, float]:
        """合并多个 provider 返回的 model_scores。

        如果多个 provider 都给出同一个模型的分数，保留最高值。
        """
        merged: dict[str, float] = {}
        for result in results:
            model_scores = result.details.get("model_scores", {})
            if not isinstance(model_scores, dict):
                continue
            for model, score in model_scores.items():
                try:
                    merged[model] = max(merged.get(model, 0.0), float(score))
                except (TypeError, ValueError):
                    continue
        return dict(sorted(merged.items(), key=lambda item: item[1], reverse=True))

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """读取 YAML 配置。

        如果配置不存在，返回空字典，调用方使用默认配置。
        """
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
