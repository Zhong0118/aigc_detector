"""Conservative document decisions built from existing detector evidence."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class FalsePositiveDecision:
    score: float | None
    decision: str
    low_threshold: float | None
    high_threshold: float | None
    threshold_group: str | None
    domain: str
    domain_confidence: float
    length_bucket: str
    analyzed_coverage: float
    high_risk_coverage: float
    adjacent_high_segments: int
    threshold_validation_fpr: float | None = None
    threshold_validation_samples: int = 0
    threshold_reliability_status: str = "not_evaluated"
    threshold_warnings: list[str] = field(default_factory=list)
    uncertainty_reasons: list[str] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    artifact_version: str = "built-in"
    score_kind: str = "calibrated_ai_risk"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FalsePositiveController:
    """Apply routing, calibrated thresholds and a four-way decision policy."""

    _LITERATURE_MARKERS = (
        "宝玉",
        "黛玉",
        "贾母",
        "悟空",
        "唐僧",
        "八戒",
        "诗曰",
        "词曰",
        "却说",
        "只见",
        "公子",
        "姑娘",
    )
    _ACADEMIC_MARKERS = ("本文", "研究", "实验", "方法", "结果表明", "算法", "模型", "数据集", "关键词")
    _OFFICIAL_MARKERS = ("通知", "公告", "决定", "根据", "现将", "有关事项", "记者", "会议指出")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.min_effective_chars = int(self.config.get("min_effective_chars", 120))
        self.short_text_chars = int(self.config.get("short_text_chars", 200))
        self.default_low = float(self.config.get("default_low_threshold", 0.4))
        self.default_high = float(self.config.get("default_high_threshold", 0.8))
        self.min_high_coverage = float(self.config.get("min_high_coverage", 0.3))
        self.min_adjacent = int(self.config.get("min_adjacent_high_segments", 2))
        self.domain_thresholds = self.config.get("domain_thresholds", {})
        self.artifact = self._load_artifact(self.config.get("artifact_path"))
        self.artifact_version = str(self.artifact.get("version", "built-in"))

    def evaluate(self, text: str, provider_results: Iterable[Any]) -> FalsePositiveDecision:
        effective_chars = len(re.sub(r"\s+", "", text))
        language = self._language(text)
        domain, domain_confidence = self._domain(text)
        length_bucket = "short" if effective_chars <= self.short_text_chars else "long"
        group, low_threshold, high_threshold = self._thresholds(language, domain, length_bucket)
        usable = [
            result
            for result in provider_results
            if getattr(result, "status", None) == "ok" and getattr(result, "score", None) is not None
        ]

        if not usable:
            return self._decision(
                score=None,
                decision="insufficient_evidence",
                low=low_threshold,
                high=high_threshold,
                group=group,
                domain=domain,
                domain_confidence=domain_confidence,
                length_bucket=length_bucket,
                reasons=["no_usable_detector_score"],
            )

        provider_scores = [float(result.score) for result in usable]
        score = float(statistics.median(provider_scores))
        segments = self._segments(usable)
        coverage, high_coverage, adjacent = self._segment_evidence(text, segments, high_threshold)
        reasons: list[str] = []

        if effective_chars < self.min_effective_chars:
            reasons.append("text_too_short")
            decision = "insufficient_evidence"
        elif score <= low_threshold:
            decision = "low_risk"
        elif score > high_threshold:
            distributed = length_bucket == "short" or not segments
            if segments and length_bucket == "long":
                distributed = high_coverage >= self.min_high_coverage
                if len(segments) > 1:
                    distributed = distributed and adjacent >= self.min_adjacent
            if distributed and coverage >= float(self.config.get("min_analyzed_coverage", 0.7)):
                decision = "high_risk_ai"
            else:
                decision = "uncertain"
                if len(segments) > 1 and adjacent < self.min_adjacent:
                    reasons.append("isolated_high_segment")
                if high_coverage < self.min_high_coverage:
                    reasons.append("insufficient_high_risk_coverage")
                if coverage < float(self.config.get("min_analyzed_coverage", 0.7)):
                    reasons.append("insufficient_analyzed_coverage")
        else:
            decision = "uncertain"
            reasons.append("score_in_uncertain_interval")

        return self._decision(
            score=score,
            decision=decision,
            low=low_threshold,
            high=high_threshold,
            group=group,
            domain=domain,
            domain_confidence=domain_confidence,
            length_bucket=length_bucket,
            coverage=coverage,
            high_coverage=high_coverage,
            adjacent=adjacent,
            reasons=reasons,
            segments=segments,
        )

    def status(self) -> dict[str, Any]:
        path = self.config.get("artifact_path")
        return {
            "enabled": bool(self.config.get("enabled", False)),
            "artifact_path": path,
            "artifact_loaded": bool(self.artifact),
            "artifact_version": self.artifact_version,
            "target_fpr": float(self.config.get("target_fpr", 0.02)),
        }

    def _decision(
        self,
        *,
        score: float | None,
        decision: str,
        low: float,
        high: float,
        group: str,
        domain: str,
        domain_confidence: float,
        length_bucket: str,
        coverage: float = 0.0,
        high_coverage: float = 0.0,
        adjacent: int = 0,
        reasons: list[str] | None = None,
        segments: list[dict[str, Any]] | None = None,
    ) -> FalsePositiveDecision:
        diagnostics = self._threshold_diagnostics(group)
        return FalsePositiveDecision(
            score=None if score is None else round(score, 4),
            decision=decision,
            low_threshold=round(low, 4),
            high_threshold=round(high, 4),
            threshold_group=group,
            domain=domain,
            domain_confidence=round(domain_confidence, 4),
            length_bucket=length_bucket,
            analyzed_coverage=round(coverage, 4),
            high_risk_coverage=round(high_coverage, 4),
            adjacent_high_segments=adjacent,
            threshold_validation_fpr=diagnostics[0],
            threshold_validation_samples=diagnostics[1],
            threshold_reliability_status=diagnostics[2],
            threshold_warnings=diagnostics[3],
            uncertainty_reasons=reasons or [],
            segments=segments or [],
            artifact_version=self.artifact_version,
        )

    def _thresholds(self, language: str, domain: str, length_bucket: str) -> tuple[str, float, float]:
        groups = self.artifact.get("groups", {})
        calibrated_candidates = (
            f"{language}:{domain}:{length_bucket}",
            f"{language}:{domain}",
        )
        for candidate in calibrated_candidates:
            entry = groups.get(candidate)
            if isinstance(entry, dict):
                return (
                    candidate,
                    float(entry.get("low_threshold", self.default_low)),
                    float(entry.get("high_threshold", self.default_high)),
                )
        entry = self.domain_thresholds.get(domain)
        if isinstance(entry, dict):
            return (
                f"built-in:{domain}:{length_bucket}",
                float(entry.get("low_threshold", self.default_low)),
                float(entry.get("high_threshold", self.default_high)),
            )
        for candidate in (language, "global"):
            entry = groups.get(candidate)
            if isinstance(entry, dict):
                return (
                    candidate,
                    float(entry.get("low_threshold", self.default_low)),
                    float(entry.get("high_threshold", self.default_high)),
                )
        return f"built-in:{domain}:{length_bucket}", self.default_low, self.default_high

    def _threshold_diagnostics(self, group: str) -> tuple[float | None, int, str, list[str]]:
        entry = self.artifact.get("groups", {}).get(group)
        if not isinstance(entry, dict):
            return None, 0, "not_evaluated", []
        validation_fpr = entry.get("validation_fpr")
        return (
            None if validation_fpr is None else round(float(validation_fpr), 6),
            int(entry.get("validation_samples", 0)),
            str(entry.get("reliability_status", "not_evaluated")),
            [str(item) for item in entry.get("warnings", [])],
        )

    def _segments(self, usable: list[Any]) -> list[dict[str, Any]]:
        for result in usable:
            details = getattr(result, "details", {}) or {}
            raw_segments = details.get("segments")
            if isinstance(raw_segments, list) and raw_segments:
                return [segment for segment in raw_segments if isinstance(segment, dict) and segment.get("score") is not None]
        return []

    def _segment_evidence(
        self,
        text: str,
        segments: list[dict[str, Any]],
        high_threshold: float,
    ) -> tuple[float, float, int]:
        if not segments:
            return 1.0, 0.0, 0
        text_length = max(1, len(text))
        covered: set[int] = set()
        high: set[int] = set()
        high_flags: list[bool] = []
        for segment in sorted(segments, key=lambda item: int(item.get("index", 0))):
            start = max(0, int(segment.get("start", 0)))
            end = min(text_length, int(segment.get("end", start)))
            positions = range(start, max(start, end))
            covered.update(positions)
            is_high = float(segment["score"]) > high_threshold
            high_flags.append(is_high)
            if is_high:
                high.update(positions)
        longest = 0
        current = 0
        for flag in high_flags:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        return len(covered) / text_length, len(high) / text_length, longest

    def _domain(self, text: str) -> tuple[str, float]:
        if any(marker in text for marker in self._LITERATURE_MARKERS):
            return "literature", 0.9
        if any(marker in text for marker in self._ACADEMIC_MARKERS):
            return "academic_technical", 0.8
        if any(marker in text for marker in self._OFFICIAL_MARKERS):
            return "official_news", 0.8
        return "general_qa", 0.55

    def _language(self, text: str) -> str:
        chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        english = sum(1 for char in text if char.isascii() and char.isalpha())
        if chinese and english and min(chinese, english) / max(chinese, english) >= 0.1:
            return "mixed"
        return "zh" if chinese >= english else "en"

    def _load_artifact(self, artifact_path: str | None) -> dict[str, Any]:
        if not artifact_path:
            return {}
        path = Path(artifact_path)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}
