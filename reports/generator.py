from __future__ import annotations

import hashlib
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
            cached = self._load_cached_report(analysis)
            if cached:
                cached = dict(cached)
                cached["status"] = "cached"
                cached["cache_hit"] = True
                return cached
            return self._generate_llm_report(analysis)
        return self._generate_template_report(analysis)

    def _generate_template_report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """生成不依赖外部服务的模板报告。"""
        detection = analysis["detection"]
        provenance = analysis["provenance"]
        modality = analysis["modality"]
        score = float(detection["score"])
        label = str(detection["label"])
        threshold = float(detection.get("threshold", self.config.get("detection", {}).get("threshold", 0.5)))
        modality_name = self._modality_name(modality)

        evidence = [
            f"综合检测分数为 {score:.2%}，判定阈值为 {threshold:.2f}，当前标签为“{self._label_name(label)}”。",
            f"系统识别输入类型为“{modality_name}”。",
        ]

        if provenance["deep_triggered"]:
            evidence.append("检测分数达到深层溯源阈值，已继续执行内容凭证、水印和指纹库检查。")
        else:
            evidence.append("检测分数未达到深层溯源阈值，深层溯源检查未作为主要证据触发。")

        c2pa = provenance.get("c2pa", {})
        if c2pa.get("found"):
            evidence.append("检测到 C2PA/内容凭证元数据，可作为来源链路或编辑历史的强线索。")
        else:
            evidence.append("未检测到 C2PA/内容凭证元数据；这只能说明文件未携带可验证声明，不能反向证明其为人工创作。")

        watermark = provenance.get("watermark", {})
        watermark_result = watermark.get("result") or {}
        if watermark_result.get("detected"):
            confidence = watermark_result.get("confidence")
            confidence_text = f"，置信度约为 {float(confidence):.2%}" if confidence is not None else ""
            evidence.append(f"本地水印检测确认存在水印信号{confidence_text}。")
        else:
            watermark_status = watermark.get("status", "unknown")
            evidence.append(f"本地水印检查状态为“{watermark_status}”，当前未确认有效水印信号。")

        registry = provenance.get("fingerprint_registry", {})
        match_count = int(registry.get("match_count") or 0)
        if match_count:
            evidence.append(f"指纹库命中 {match_count} 条历史相似记录，可用于重复传播或已登记样本追踪。")
        else:
            evidence.append("指纹库未命中历史相似样本；空库或未命中不参与最终真伪证明。")

        attribution = provenance.get("attribution")
        if isinstance(attribution, dict) and attribution:
            status = attribution.get("status", "unknown")
            confidence = attribution.get("confidence")
            top_k = attribution.get("top_k") if isinstance(attribution.get("top_k"), list) else []
            top_text = ""
            if top_k:
                first = top_k[0]
                top_text = f"，首位候选来源为“{first.get('model')}”"
            if confidence is not None:
                evidence.append(f"来源归因模块状态为“{status}”{top_text}，候选来源置信度约为 {float(confidence):.2%}。")
            else:
                evidence.append(f"来源归因模块状态为“{status}”，当前未形成可用候选来源。")
            if status == "data_mismatch":
                evidence.append("实验性 LLMDet 分支的数据缓存不完整或版本不匹配，本次未采用其候选来源结果。")

        return {
            "provider": "template",
            "status": "ok",
            "summary": self._summary(label, score, modality),
            "evidence": evidence,
            "limitations": [
                "检测分数属于概率证据，不能单独作为最终结论，需要结合来源凭证、水印、指纹库和人工复核。",
                "未发现 C2PA 或水印并不等于内容一定由人类创作，只表示当前文件没有携带这些可验证信号。",
                "模型归因、重建误差、PPL 等被动法证特征只能提供候选来源或风险线索，不能替代签名凭证。",
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
            "max_tokens": int(self.report_cfg.get("max_tokens", 300)),
            "thinking": {"type": str(self.report_cfg.get("thinking", "disabled"))},
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
            parsed = self._parse_llm_response(content)
            report = {
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
            self._save_cached_report(analysis, report)
            return report
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            template = self._generate_template_report(analysis)
            template["evidence"].insert(0, "解释模型调用失败，系统已改用本地中文模板生成报告。")
            template["limitations"].insert(0, "本次报告不是大语言模型生成结果，而是规则模板回退结果；请检查报告模型配置、网络或返回格式。")
            template.update(
                {
                    "provider": llm_provider,
                    "status": "error",
                    "model": model,
                    "error": str(exc),
                }
            )
            return template

    def _load_cached_report(self, analysis: dict[str, Any]) -> dict[str, Any] | None:
        """按内容指纹读取报告缓存，减少重复 LLM 扣费。"""
        if not self.report_cfg.get("cache_enabled", True):
            return None
        path = self._cache_path(analysis)
        if not path or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _save_cached_report(self, analysis: dict[str, Any], report: dict[str, Any]) -> None:
        """保存成功生成的 LLM 报告。"""
        if not self.report_cfg.get("cache_enabled", True):
            return
        path = self._cache_path(analysis)
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = dict(report)
            cache_payload["cache_hit"] = False
            path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return

    def _cache_path(self, analysis: dict[str, Any]) -> Path | None:
        """生成报告缓存路径。"""
        fingerprint = analysis.get("fingerprint") or analysis.get("provenance", {}).get("fingerprint")
        if not fingerprint:
            return None
        provider = str(self.report_cfg.get("llm_provider", "deepseek")).lower()
        model = str(self.report_cfg.get("model", "deepseek-v4-flash"))
        modality = str(analysis.get("modality", "unknown"))
        key = hashlib.sha256(f"{provider}|{model}|{modality}|{fingerprint}".encode("utf-8")).hexdigest()
        cache_dir = Path(str(self.report_cfg.get("cache_dir", "data/report_cache")))
        return cache_dir / f"{key}.json"

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
                "provider_hints_top_k": provenance.get("provider_hints", {}).get("top_k", [])
                if isinstance(provenance.get("provider_hints"), dict)
                else [],
                "fingerprint_registry": {
                    "status": provenance.get("fingerprint_registry", {}).get("status"),
                    "match_count": provenance.get("fingerprint_registry", {}).get("match_count"),
                    "matches": provenance.get("fingerprint_registry", {}).get("matches", []),
                },
                "attribution_confidence": provenance.get("attribution", {}).get("confidence")
                if isinstance(provenance.get("attribution"), dict)
                else None,
                "attribution_top_k": provenance.get("attribution", {}).get("top_k", [])
                if isinstance(provenance.get("attribution"), dict)
                else [],
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

    def _parse_llm_response(self, content: str) -> dict[str, Any]:
        """解析 LLM 响应，JSON 失败时降级解析普通文本。

        DeepSeek 偶尔会返回被截断或未严格转义的 JSON。报告模型只是解释层，
        不应该因为格式问题让整个分析显得失败，所以这里保留可读内容。
        """
        try:
            return self._parse_llm_json(content)
        except json.JSONDecodeError:
            return self._parse_plaintext_report(content)

    def _parse_plaintext_report(self, content: str) -> dict[str, Any]:
        """把非 JSON 报告文本整理成统一字段。"""
        lines = [line.strip(" -\t") for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("LLM response is empty.")

        summary = lines[0]
        evidence: list[str] = []
        limitations: list[str] = []
        recommendation = ""

        for line in lines[1:]:
            lowered = line.lower()
            if line.startswith(("证据", "Evidence", "evidence")):
                evidence.append(line.split("：", 1)[-1].split(":", 1)[-1].strip() or line)
            elif line.startswith(("限制", "局限", "Limitations", "limitations")):
                limitations.append(line.split("：", 1)[-1].split(":", 1)[-1].strip() or line)
            elif line.startswith(("建议", "Recommendation", "recommendation")):
                recommendation = line.split("：", 1)[-1].split(":", 1)[-1].strip() or line
            elif "limitation" in lowered:
                limitations.append(line)
            elif "recommend" in lowered:
                recommendation = line
            else:
                evidence.append(line)

        return {
            "summary": summary.replace("总结：", "").replace("Summary:", "").strip(),
            "evidence": evidence or ["解释模型返回了非 JSON 文本，系统已保留可读内容。"],
            "limitations": limitations or ["该报告由非严格 JSON 响应恢复生成，建议结合结构化检测结果复核。"],
            "recommendation": recommendation or "建议结合检测分支、溯源证据和人工复核做最终判断。",
        }

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
        modality_name = self._modality_name(modality)
        if label == "ai":
            if score >= 0.8:
                return f"当前多路检测结果认为该{modality_name}高度疑似 AIGC 内容，综合风险分数为 {score:.2%}。"
            return f"当前多路检测结果认为该{modality_name}存在 AIGC 风险，综合风险分数为 {score:.2%}。"
        return f"当前证据未显示该{modality_name}具有明显 AIGC 特征，综合风险分数为 {score:.2%}。"

    def _recommendation(self, label: str, score: float) -> str:
        if label == "ai" and score >= 0.8:
            return "建议将该内容标记为高风险样本，优先复核检测分支证据，并继续查看 C2PA、水印、指纹库和来源归因结果。"
        if label == "ai":
            return "建议保留当前检测记录，结合人工复核和后续溯源模块再做最终判断。"
        return "建议保留分析记录；如果外部上下文仍然可疑，再补充更深层的溯源或人工复核。"

    def _modality_name(self, modality: str) -> str:
        """把内部模态名转换成报告用中文。"""
        return {
            "text": "文本",
            "image": "图片",
            "audio": "音频",
            "video": "视频",
        }.get(str(modality), str(modality))

    def _label_name(self, label: str) -> str:
        """把内部标签转换成报告用中文。"""
        return {
            "ai": "疑似 AIGC",
            "human": "未明显检出 AIGC",
        }.get(str(label), str(label))

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """读取 YAML 配置文件。"""
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
