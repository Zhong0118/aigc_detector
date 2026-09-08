"""API-first 检测 provider 适配层。

这个文件是当前检测主路径。FastAPI 不直接调用模型，
而是通过 `ApiFirstDetectionEngine` 得到统一的检测结果。

"""

from __future__ import annotations

import hashlib
import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

from detection.false_positive_controller import FalsePositiveController
from detection.text_detector import RoutedTextAigcDetector


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
    decision_details: dict[str, Any] = field(default_factory=dict)


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
        self.text_fp_cfg = detection_cfg.get("text_false_positive_control", {})
        self.local_cfg = detection_cfg.get("local_models", {})
        self.local_enabled = bool(self.local_cfg.get("enabled", False))
        self._text_fp_controller = FalsePositiveController(self.text_fp_cfg)
        self._local_text_detector: RoutedTextAigcDetector | None = None
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
        text_payload = raw_data if isinstance(raw_data, str) else None
        if modality == "text" and text_payload is None:
            from ingestion.document_text import extract_document_text, is_document_path

            if is_document_path(path):
                text_payload = extract_document_text(path)
            else:
                text_payload = path.read_text(encoding="utf-8", errors="ignore")

        # 如果 config 中打开 Hive/Sightengine，这里会进入对应 provider。
        # provider 失败不会中断全流程，而是返回 status=error，方便前端展示。
        api_results = self._run_configured_api_providers(
            path,
            modality,
            text_override=text_payload if modality == "text" else None,
        )
        results.extend(api_results)

        if self.demo_enabled:
            # demo_api 是当前唯一真正返回分数的 provider。
            demo_result = self._run_demo_provider(
                path,
                modality,
                raw_data,
            )
            results.append(demo_result)

        if self.local_enabled:
            # 本地模型作为额外检测分支，不改变 FastAPI 对外接口。
            local_results = self._run_local_model_providers(
                path,
                modality,
                raw_data,
            )
            results.extend(local_results)

        if modality == "text" and bool(self.text_fp_cfg.get("enabled", False)):
            text = text_payload if text_payload is not None else ""
            decision = self._text_fp_controller.evaluate(text, results)
            final_score = float(decision.score or 0.0)
            high_threshold = float(decision.high_threshold or self.threshold)
            return DetectionPackage(
                score=round(final_score, 4),
                label=decision.decision,
                threshold=high_threshold,
                provider_results=results,
                modality_scores={modality: round(final_score, 4)},
                model_scores=self._merge_model_scores(results),
                decision_details=decision.to_dict(),
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
                "configured": bool(
                    self._credential_from_config(
                        hive_cfg,
                        env_field="api_key_env",
                        value_field="api_key",
                        default_env="HIVE_API_KEY",
                    )
                ),
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
                    self._credential_from_config(
                        sight_cfg,
                        env_field="api_user_env",
                        value_field="api_user",
                        default_env="SIGHTENGINE_API_USER",
                    )
                    and self._credential_from_config(
                        sight_cfg,
                        env_field="api_secret_env",
                        value_field="api_secret",
                        default_env="SIGHTENGINE_API_SECRET",
                    )
                ),
                "purpose": "Cloud image/video AI detection and generator hints.",
            }
        )

        providers.append(
            {
                "name": "local_models",
                "type": "local_model",
                "enabled": self.local_enabled,
                "configured": bool(self.local_cfg.get("text_yuchuan", {}).get("enabled", False)),
                "purpose": "Local text detector branch with Hugging Face weights cached under models/huggingface.",
            }
        )
        fp_status = self._text_fp_controller.status()
        providers.append(
            {
                "name": "text_false_positive_controller",
                "type": "decision_controller",
                "enabled": bool(self.text_fp_cfg.get("enabled", True)),
                "configured": fp_status["artifact_loaded"],
                "purpose": "Offline segmented aggregation with calibrated low/uncertain/high-risk decisions.",
                "details": fp_status,
            }
        )
        return providers

    def _run_local_model_providers(
        self,
        path: Path,
        modality: str,
        raw_data: object | None,
    ) -> list[ProviderResult]:
        """运行本地模型 provider。

        当前先接入文本 AIGC 检测分支；图片/音频/视频本地模型仍然预留。
        """
        if modality != "text":
            return [
                ProviderResult(
                    provider="local_models",
                    provider_type="local_model",
                    status="skipped",
                    score=None,
                    label="not_run",
                    details={
                        "modality": modality,
                        "note": "Local model branch is currently enabled for text only.",
                    },
                )
            ]

        text_cfg = self.local_cfg.get("text_yuchuan", {})
        if not text_cfg.get("enabled", False):
            return [
                ProviderResult(
                    provider="local_text_detector",
                    provider_type="local_model",
                    status="disabled",
                    score=None,
                    label="not_run",
                    details={"note": "Local YuchuanTian text detector is disabled in config."},
                )
            ]

        try:
            from ingestion.document_text import extract_document_text, is_document_path

            if isinstance(raw_data, str):
                text = raw_data
            elif is_document_path(path):
                text = extract_document_text(path)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
            detector = self._get_local_text_detector(text_cfg)
            result = detector.detect(text)
            return [
                ProviderResult(
                    provider="local_text_detector",
                    provider_type="local_model",
                    status="ok",
                    score=result.score,
                    label=result.label,
                    details=result.details,
                )
            ]
        except Exception as exc:  # noqa: BLE001 - provider errors must not break the whole API request.
            return [
                ProviderResult(
                    provider="local_text_detector",
                    provider_type="local_model",
                    status="error",
                    score=None,
                    label="not_run",
                    details={
                        "cache_dir": text_cfg.get("cache_dir", "models/huggingface"),
                        "note": "Local text detector failed. Check model download, cache path, or device config.",
                    },
                    error=str(exc),
                )
            ]

    def _get_local_text_detector(self, cfg: dict[str, Any]) -> RoutedTextAigcDetector:
        """懒加载本地文本检测器，避免后端启动时下载模型。"""
        if self._local_text_detector is None:
            self._local_text_detector = RoutedTextAigcDetector(
                models=cfg.get("models", {}),
                cache_dir=cfg.get("cache_dir", "models/huggingface"),
                short_text_chars=int(cfg.get("short_text_chars", 200)),
                threshold=float(cfg.get("threshold", self.threshold)),
                device=str(cfg.get("device", "auto")),
                mixed_strategy=str(cfg.get("mixed_strategy", "mean")),
                max_length=int(cfg.get("max_length", 512)),
                segment_target_chars=int(cfg.get("segment_target_chars", 480)),
                segment_max_chars=int(cfg.get("segment_max_chars", 640)),
                en_segment_target_chars=int(cfg.get("en_segment_target_chars", 1400)),
                en_segment_max_chars=int(cfg.get("en_segment_max_chars", 1800)),
                segment_overlap_sentences=int(cfg.get("segment_overlap_sentences", 1)),
            )
        return self._local_text_detector

    def _run_configured_api_providers(
        self,
        path: Path,
        modality: str,
        text_override: str | None = None,
    ) -> list[ProviderResult]:
        """运行配置中启用的外部 API provider。

        当前 Hive/Sightengine 已经会按配置真实请求。
        这里做统一分发，每个 provider 单独处理自己的鉴权、endpoint 和返回格式。
        """
        results: list[ProviderResult] = []
        hive_cfg = self.api_providers.get("hive", {})
        if hive_cfg.get("enabled", False):
            results.append(self._run_hive_provider(path, modality, hive_cfg, text_override=text_override))

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
                "note": "Provider is enabled, but credentials are missing from config or environment variables.",
            },
        )

    def _run_hive_provider(
        self,
        path: Path,
        modality: str,
        cfg: dict[str, Any],
        text_override: str | None = None,
    ) -> ProviderResult:
        """调用 Hive 同步检测接口。

        Hive 的文本检测用 JSON `text_data`；图片/视频/音频用 multipart `media`。
        返回后统一抽取 ai_generated 概率，后续前端不用理解 Hive 原始 JSON。
        """
        api_version = str(cfg.get("api_version", "v2")).lower()
        if api_version in {"v3", "v3_vlm", "vlm"}:
            return self._run_hive_v3_vlm_provider(path, modality, cfg, text_override=text_override)
        return self._run_hive_v2_provider(path, modality, cfg, text_override=text_override)

    def _run_hive_v2_provider(
        self,
        path: Path,
        modality: str,
        cfg: dict[str, Any],
        text_override: str | None = None,
    ) -> ProviderResult:
        """调用 Hive V2 task/sync 检测接口。

        这个分支需要 Hive Project/legacy key；Playground API Key 通常无权访问 V2。
        """
        key_env = str(cfg.get("api_key_env", "HIVE_API_KEY"))
        api_key = self._credential_from_config(
            cfg,
            env_field="api_key_env",
            value_field="api_key",
            default_env="HIVE_API_KEY",
        )
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
                    "text_data": self._read_text_payload(path, text_override=text_override),
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

    def _run_hive_v3_vlm_provider(
        self,
        path: Path,
        modality: str,
        cfg: dict[str, Any],
        text_override: str | None = None,
    ) -> ProviderResult:
        """调用 Hive V3 VLM，用提示词返回结构化 AIGC 判断。

        V3 Playground Key 走 OpenAI-compatible Chat Completions。
        它不是 V2 专用 detector 的同款接口，而是把 Hive VLM 当成可提示的分类器使用。
        """
        if modality == "audio":
            return ProviderResult(
                provider="hive",
                provider_type="detection_api",
                status="skipped",
                score=None,
                label="not_run",
                details={
                    "modality": modality,
                    "api_version": "v3_vlm",
                    "note": "Hive V3 VLM docs cover text/image/video inputs; use V2 ai_generated_audio for audio detection.",
                },
            )

        key_env = str(cfg.get("api_key_env", "HIVE_API_KEY"))
        api_key = self._credential_from_config(
            cfg,
            env_field="api_key_env",
            value_field="api_key",
            default_env="HIVE_API_KEY",
        )
        if not api_key:
            return self._not_configured_result("hive", key_env)

        endpoint = str(cfg.get("v3_endpoint", "https://api.thehive.ai/api/v3/chat/completions"))
        timeout = float(cfg.get("timeout_seconds", 45))
        model = str(cfg.get("v3_model", "hive/vision-language-model"))

        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=self._build_hive_v3_payload(path, modality, model, text_override=text_override),
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            parsed = self._parse_hive_v3_message(payload)
            score = self._extract_ai_score(parsed)
            if score is None:
                score = self._extract_ai_score(payload)
            model_scores = self._extract_model_scores(parsed)
            if score is None:
                return ProviderResult(
                    provider="hive",
                    provider_type="detection_api",
                    status="error",
                    score=None,
                    label="not_run",
                    details={
                        "modality": modality,
                        "api_version": "v3_vlm",
                        "model": model,
                        "raw": payload,
                        "parsed": parsed,
                        "note": "Hive V3 VLM response did not contain ai_generated_score.",
                    },
                    error="Missing ai_generated_score in Hive V3 VLM response.",
                )
            return ProviderResult(
                provider="hive",
                provider_type="detection_api",
                status="ok",
                score=round(score, 4),
                label="ai" if score >= self.threshold else "human",
                details={
                    "modality": modality,
                    "api_version": "v3_vlm",
                    "model": model,
                    "model_scores": model_scores,
                    "explanation": parsed.get("explanation"),
                    "score_source": "Hive V3 VLM structured ai_generated_score",
                    "raw": payload,
                    "parsed": parsed,
                },
            )
        except (requests.RequestException, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            return ProviderResult(
                provider="hive",
                provider_type="detection_api",
                status="error",
                score=None,
                label="not_run",
                details={"modality": modality, "api_version": "v3_vlm", "model": model, "endpoint": endpoint},
                error=str(exc),
            )

    def _build_hive_v3_payload(
        self,
        path: Path,
        modality: str,
        model: str,
        text_override: str | None = None,
    ) -> dict[str, Any]:
        """构造 Hive V3 Chat Completions 请求体。"""
        content: str | list[dict[str, Any]]
        instruction = self._hive_v3_instruction(modality)

        if modality == "text":
            content = (
                f"{instruction}\n\nContent to analyze:\n"
                f"{self._read_text_payload(path, text_override=text_override)}"
            )
        else:
            media_part = self._hive_v3_media_part(path, modality)
            content = [media_part, {"type": "text", "text": instruction}]

        return {
            "model": model,
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "aigc_detection_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ai_generated_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "label": {
                                "type": "string",
                                "enum": ["ai_generated", "not_ai_generated", "uncertain"],
                            },
                            "explanation": {"type": "string"},
                            "model_scores": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                        },
                        "required": ["ai_generated_score", "label", "explanation", "model_scores"],
                        "additionalProperties": False,
                    },
                },
            },
            "messages": [{"role": "user", "content": content}],
        }

    def _hive_v3_instruction(self, modality: str) -> str:
        """给 Hive V3 VLM 的分类提示词。"""
        return (
            "You are an AIGC detection classifier. "
            f"Analyze this {modality} content and estimate whether it is AI-generated. "
            "Return JSON only. "
            "`ai_generated_score` must be a probability from 0 to 1. "
            "`label` must be ai_generated, not_ai_generated, or uncertain. "
            "`explanation` must briefly cite observable evidence and uncertainty. "
            "`model_scores` can include likely generator families when there is evidence, otherwise return an empty object. "
            "Do not claim certainty from metadata alone."
        )

    def _hive_v3_media_part(self, path: Path, modality: str) -> dict[str, Any]:
        """把本地图片/视频转成 Hive V3 支持的 data URI。"""
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data_uri = f"data:{mime_type};base64,{encoded}"
        if modality == "video":
            return {
                "type": "media_url",
                "media_url": {
                    "url": data_uri,
                    "sampling": {"strategy": "fps", "fps": 1},
                    "prompt_scope": "once",
                },
            }
        return {"type": "image_url", "image_url": {"url": data_uri}}

    def _parse_hive_v3_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        """解析 Hive V3 message.content 中的 JSON。"""
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content
        cleaned = str(content).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}

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
        api_user = self._credential_from_config(
            cfg,
            env_field="api_user_env",
            value_field="api_user",
            default_env="SIGHTENGINE_API_USER",
        )
        api_secret = self._credential_from_config(
            cfg,
            env_field="api_secret_env",
            value_field="api_secret",
            default_env="SIGHTENGINE_API_SECRET",
        )
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

    def _read_text_payload(self, path: Path, text_override: str | None = None) -> str:
        """读取文本内容给 Hive 文本检测接口。"""
        if text_override is not None:
            return text_override
        from ingestion.document_text import extract_document_text, is_document_path

        if is_document_path(path):
            return extract_document_text(path)
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
            explicit_model_scores = value.get("model_scores")
            if isinstance(explicit_model_scores, dict):
                for key, item in explicit_model_scores.items():
                    if self._is_number(item):
                        scores[str(key)] = max(scores.get(str(key), 0.0), float(item))

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
        return normalized in {
            "ai_generated",
            "ai_generated_score",
            "generated",
            "synthetic",
            "is_ai",
        }

    def _is_number(self, value: Any) -> bool:
        """判断值是否能作为 0-1 概率分数。"""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _credential_from_config(
        self,
        cfg: dict[str, Any],
        env_field: str,
        value_field: str,
        default_env: str,
    ) -> str | None:
        """从配置里解析 API 凭据。

        推荐写法：
        - `api_key: "真实密钥"` / `api_user: "真实 user"` / `api_secret: "真实 secret"`
        - 或 `api_key_env: "HIVE_API_KEY"` 这种环境变量名

        为了兼容早期配置，如果 `_env` 字段里看起来不像环境变量名，
        也会把它当成真实凭据使用。
        """
        direct_value = cfg.get(value_field)
        if direct_value:
            return str(direct_value)

        env_or_value = str(cfg.get(env_field, default_env))
        env_value = os.getenv(env_or_value)
        if env_value:
            return env_value
        if env_or_value != default_env and not self._looks_like_env_name(env_or_value):
            return env_or_value
        return None

    def _looks_like_env_name(self, value: str) -> bool:
        """判断一个字符串是否像环境变量名，而不是实际密钥。"""
        return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", value))

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
