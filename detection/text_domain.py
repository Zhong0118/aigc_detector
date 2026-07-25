"""Explicit bilingual marker routing for canonical text domains."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainDecision:
    domain: str
    confidence: float
    evidence: tuple[str, ...]


class TextDomainRouter:
    """Classify text using deterministic marker rules, not semantic inference."""

    _TIE_BREAK_PRIORITY = {
        "literature": 1,
        "official_news": 2,
        "academic_technical": 3,
    }
    _DOMAIN_MARKERS = (
        (
            "literature",
            (
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
                "once upon a time",
                "novel",
                "poem",
                "poetry",
                "stanza",
                "protagonist",
                "narrator",
            ),
            0.85,
        ),
        (
            "academic_technical",
            (
                "本文",
                "研究",
                "实验",
                "方法",
                "结果表明",
                "算法",
                "模型",
                "数据集",
                "关键词",
                "study",
                "research",
                "experiment",
                "method",
                "results",
                "algorithm",
                "model",
                "dataset",
                "neural",
                "analysis",
                "abstract",
                "we propose",
                "findings",
            ),
            0.8,
        ),
        (
            "official_news",
            (
                "通知",
                "公告",
                "决定",
                "根据",
                "现将",
                "有关事项",
                "记者",
                "会议指出",
                "政府",
                "government",
                "ministry",
                "official statement",
                "public notice",
                "press release",
                "announced",
                "authorities",
                "agency",
            ),
            0.8,
        ),
    )

    def classify(self, text: str) -> DomainDecision:
        candidates: list[tuple[int, float, int, str, tuple[str, ...]]] = []
        for domain, markers, base_confidence in self._DOMAIN_MARKERS:
            evidence = tuple(marker for marker in markers if self._contains(text, marker))
            if evidence:
                candidates.append(
                    (
                        len(evidence),
                        base_confidence,
                        self._TIE_BREAK_PRIORITY[domain],
                        domain,
                        evidence,
                    )
                )
        if candidates:
            evidence_count, base_confidence, _, domain, evidence = max(candidates)
            confidence = min(0.98, base_confidence + 0.03 * (evidence_count - 1))
            return DomainDecision(domain, confidence, evidence)
        return DomainDecision("general_qa", 0.25, ())

    @staticmethod
    def _contains(text: str, marker: str) -> bool:
        if marker.isascii():
            return re.search(
                rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])",
                text.casefold(),
            ) is not None
        return marker in text
