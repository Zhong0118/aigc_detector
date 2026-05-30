from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import yaml


class ReportGenerator:
    """解释报告生成器。

    默认会生成模板报告；当 config.yaml 中 `report.provider: llm` 时，
    会调用 DeepSeek / OpenAI-compatible Chat Completions 接口生成中文解释。
    注意：LLM 只负责“解释已有证据”，不负责直接判断真伪。
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        config: dict[str, Any] | None = None,
    ) -> None:
        """初始化报告生成器。

        config 可由测试直接传入；线上默认读取 config_path。
        """
        self.config = config or self._load_config(config_path)
        self.report_cfg = self.config.get("report", {})

    def generate(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """生成用户可读的解释报告。

        返回结构固定为 provider/status/summary/evidence/limitations/recommendation。
        前端可以不用关心报告来自模板还是 LLM。
        """
        provider = str(self.report_cfg.get("provider", "template")).lower()
        if provider == "llm":
            return self._generate_llm_report(analysis)
        return self._generate_template_report(analysis)

    def _generate_template_report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """生成不依赖外部服务的模板报告。"""
        detection = analysis["detection"]
        provenance = analysis["provenance"]
        modality = analysis["modality"]
        score = detection["score"]
        label = detection["label"]

        evidence = [
            f"Primary detection score is {score:.2%}.",
            f"Detected modality is {modality}.",
        ]

        if provenance["deep_triggered"]:
            evidence.append("Deep provenance checks were triggered by the score threshold.")
        else:
            evidence.append("Deep provenance checks were skipped because the score is below threshold.")

        c2pa = provenance.get("c2pa", {})
        if c2pa.get("found"):
            evidence.append("C2PA metadata was found.")
        else:
            evidence.append("No C2PA metadata was found in the current check.")

        watermark = provenance.get("watermark", {})
        watermark_result = watermark.get("result") or {}
        if watermark_result.get("detected"):
            evidence.append("A watermark-like signal was detected.")
        else:
            evidence.append("No watermark signal was confirmed.")

        return {
            "provider": "template",
            "status": "ok",
            "summary": self._summary(label, score, modality),
            "evidence": evidence,
            "limitations": [
                "API or demo-provider scores are probabilistic and should be reviewed with provenance evidence.",
                "Absence of C2PA or watermark data does not prove human authorship.",
                "Reserved providers need API keys or local model weights before they become authoritative.",
            ],
            "recommendation": self._recommendation(label, score),
        }

    def _generate_llm_report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """调用 OpenAI-compatible LLM 生成解释报告。

        DeepSeek 兼容 OpenAI 的 Chat Completions 格式，所以这里不写死 SDK，
        直接用 requests 发 HTTP，后续换 OpenAI、通义千问等兼容接口也更简单。
        """
        llm_provider = str(self.report_cfg.get("llm_provider", "deepseek")).lower()
        key_label = str(self.report_cfg.get("api_key_env", "DEEPSEEK_API_KEY"))
        api_key = self._credential_from_config(
            self.report_cfg,
            env_field="api_key_env",
            value_field="api_key",
            default_env="DEEPSEEK_API_KEY",
        )

        if not api_key:
            template = self._generate_template_report(analysis)
            template.update(
                {
                    "provider": llm_provider,
                    "status": "not_configured",
                    "error": f"Missing API key. Set report.api_key or environment variable: {key_label}",
                }
            )
            return template

        base_url = str(self.report_cfg.get("base_url", "https://api.deepseek.com")).rstrip("/")
        endpoint = str(self.report_cfg.get("endpoint", f"{base_url}/chat/completions"))
        model = str(self.report_cfg.get("model", "deepseek-v4-flash"))
        timeout = float(self.report_cfg.get("timeout_seconds", 30))

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 AIGC 检测系统的解释报告生成器。"
                        "只能基于用户提供的结构化证据写结论，不要编造检测结果。"
                        "报告面向普通评审用户，使用中文，语气专业但不要机械。"
                        "不要出现任何第三方厂商、API、接口版本、供应商名称或内部字段名。"
                        "统一把检测来源称为“检测分支A/B/C”或“多路检测模块”。"
                        "不要声称模型是自训练或自研，只能说“系统集成的检测模块”。"
                        "低风险时也要强调概率结论不是身份证明，避免绝对化。"
                        "必须输出严格 JSON，字段为 summary、evidence、limitations、recommendation。"
                        "evidence 和 limitations 必须是字符串数组。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        self._report_context(analysis),
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": float(self.report_cfg.get("temperature", 0.2)),
            "max_tokens": int(self.report_cfg.get("max_tokens", 700)),
            "stream": False,
        }

        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = self._parse_llm_json(content)
            return {
                "provider": llm_provider,
                "status": "ok",
                "model": model,
                "summary": parsed.get("summary") or self._summary(
                    analysis["detection"]["label"],
                    float(analysis["detection"]["score"]),
                    analysis["modality"],
                ),
                "evidence": self._ensure_string_list(parsed.get("evidence")),
                "limitations": self._ensure_string_list(parsed.get("limitations")),
                "recommendation": parsed.get("recommendation")
                or self._recommendation(
                    analysis["detection"]["label"],
                    float(analysis["detection"]["score"]),
                ),
            }
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            template = self._generate_template_report(analysis)
            template.update(
                {
                    "provider": llm_provider,
                    "status": "error",
                    "model": model,
                    "error": str(exc),
                }
            )
            return template

    def _report_context(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """抽取给 LLM 的最小证据包，避免把大文件或敏感内容发给报告模型。

        这里会匿名化 provider 名称。用户报告只需要知道“多路检测分支”
        的分数和状态，不需要暴露具体供应商、API 版本或调试字段。
        """
        detection = analysis.get("detection", {})
        provenance = analysis.get("provenance", {})
        return {
            "filename": analysis.get("filename"),
            "modality": analysis.get("modality"),
            "detection": {
                "score": detection.get("score"),
                "label": detection.get("label"),
                "threshold": detection.get("threshold"),
                "branches": self._provider_branches(detection.get("providers", [])),
                "source_hints": self._public_score_map(detection.get("model_scores", {})),
            },
            "provenance": {
                "deep_triggered": provenance.get("deep_triggered"),
                "content_credentials_status": provenance.get("c2pa", {}).get("status"),
                "watermark_status": provenance.get("watermark", {}).get("status"),
                "fingerprint_registry": {
                    "status": provenance.get("fingerprint_registry", {}).get("status"),
                    "match_count": provenance.get("fingerprint_registry", {}).get("match_count"),
                    "matches": provenance.get("fingerprint_registry", {}).get("matches", []),
                },
                "attribution_confidence": provenance.get("attribution", {}).get("confidence")
                if isinstance(provenance.get("attribution"), dict)
                else None,
            },
        }

    def _provider_branches(self, providers: Any) -> list[dict[str, Any]]:
        """把 provider 证据匿名化成检测分支。"""
        if not isinstance(providers, list):
            return []
        branches: list[dict[str, Any]] = []
        for idx, provider in enumerate(providers):
            if not isinstance(provider, dict):
                continue
            details = provider.get("details") if isinstance(provider.get("details"), dict) else {}
            branches.append(
                {
                    "name": f"检测分支{chr(ord('A') + idx)}",
                    "status": provider.get("status"),
                    "score": provider.get("score"),
                    "label": provider.get("label"),
                    "evidence": self._public_evidence_text(details),
                }
            )
        return branches

    def _public_evidence_text(self, details: dict[str, Any]) -> str:
        """从 provider details 中抽取可公开展示的证据，过滤内部供应商和版本信息。"""
        candidates = [
            details.get("explanation"),
            details.get("note"),
            details.get("score_source"),
        ]
        for item in candidates:
            if isinstance(item, str) and item.strip():
                return self._sanitize_public_text(item)
        return "该检测分支返回了结构化分数，用于参与综合判定。"

    def _public_score_map(self, scores: Any) -> dict[str, float]:
        """清理来源提示分数，避免把内部模型名直接暴露给报告。"""
        if not isinstance(scores, dict):
            return {}
        public_scores: dict[str, float] = {}
        for idx, score in enumerate(scores.values(), start=1):
            try:
                public_scores[f"候选来源{idx}"] = float(score)
            except (TypeError, ValueError):
                continue
        return public_scores

    def _sanitize_public_text(self, text: str) -> str:
        """删除报告里不该暴露的供应商/API 词汇。"""
        replacements = {
            "Hive V3 VLM": "检测分支",
            "Hive": "检测分支",
            "Sightengine": "检测分支",
            "sightengine": "检测分支",
            "hive": "检测分支",
            "v3_vlm": "检测模块",
            "API": "检测接口",
            "api": "检测接口",
        }
        cleaned = text
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        return cleaned

    def _parse_llm_json(self, content: str) -> dict[str, Any]:
        """解析 LLM 输出的 JSON，兼容模型偶尔包一层 Markdown code fence。"""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}

    def _ensure_string_list(self, value: Any) -> list[str]:
        """把 LLM 返回值规整成字符串列表，避免前端遇到奇怪类型。"""
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        if value:
            return [str(value)]
        return []

    def _credential_from_config(
        self,
        cfg: dict[str, Any],
        env_field: str,
        value_field: str,
        default_env: str,
    ) -> str | None:
        """从配置里解析密钥。

        推荐写法：`api_key: "真实密钥"` 或 `api_key_env: "环境变量名"`。
        为了兼容早期配置，如果 `_env` 字段里看起来不像环境变量名，
        也会把它当成真实密钥使用。
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

    def _summary(self, label: str, score: float, modality: str) -> str:
        if label == "ai":
            return f"The {modality} content is likely AI-generated based on the current detection evidence."
        return f"The {modality} content is not strongly indicated as AI-generated by the current checks."

    def _recommendation(self, label: str, score: float) -> str:
        if label == "ai" and score >= 0.8:
            return "Treat this as high-risk AI-generated content and review provenance or source files."
        if label == "ai":
            return "Review the content manually and compare provider/model evidence before making a final decision."
        return "Keep the analysis record; run deeper checks if external context suggests the content is suspicious."

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """读取 YAML 配置文件。"""
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
