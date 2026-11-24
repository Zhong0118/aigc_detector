"""Presentation of calibrated text decisions, shared by reports and both dashboards."""
from dataclasses import dataclass
from typing import Any

SCORE_LABEL = "模型校准风险分"
SCORE_EXPLANATION = "用于对照本分支阈值，非 AI 撰写字数比例"
REVIEW_LABEL = "疑似 AI，需复核"


@dataclass(frozen=True)
class TextRiskPresentation:
    conclusion: str
    score: float | None
    group: str
    screening: float | None
    strict: float | None
    review: bool
    high_windows: int
    windows: int
    high_coverage: float | None

    def evidence(self) -> list[str]:
        rows = [
            f"阈值组 {self.group} · 复核线 {format_percent(self.screening)} · 严格线 {format_percent(self.strict)}。",
            f"{SCORE_LABEL} {format_percent(self.score)}（{SCORE_EXPLANATION}）。",
        ]
        if self.windows and self.strict is not None:
            rows.append(f"分段复核线索有 {self.high_windows}/{self.windows} 个窗口超过严格线，"
                        f"去重叠高分字符覆盖率 {format_percent(self.high_coverage)}。")
        return rows


def format_percent(value: float | None) -> str:
    return "不可用" if value is None else f"{value:.2%}"


def text_risk(analysis: dict[str, Any]) -> TextRiskPresentation | None:
    if analysis.get("modality") != "text":
        return None
    detection = analysis.get("detection") or {}
    decision = detection.get("decision") or {}
    label = decision.get("decision") or detection.get("label")
    if label not in {"high_risk_ai", "uncertain", "low_risk"}:
        return None

    providers = [p.get("details") or {} for p in detection.get("providers") or []
                 if p.get("status") == "ok"]
    cutoff = decision.get("screening_threshold")
    if cutoff is None:
        # Older stored results advertised screening on the executed provider.
        cutoff = min((float(p["screening_threshold"]) for p in providers
                      if p.get("screening_threshold") is not None), default=None)
    score = detection.get("score")
    review = label == "uncertain" and score is not None and cutoff is not None and score >= cutoff
    labels = {"high_risk_ai": "高风险疑似 AI", "low_risk": "低风险（更偏人工）",
              "uncertain": f"不确定 · {REVIEW_LABEL}" if review else "严格判定不确定 · 未触发复核"}
    strict = decision.get("high_threshold")
    segments = decision.get("segments") or next((p["segments"] for p in providers if p.get("segments")), [])
    scored = [s["score"] for s in segments if s.get("score") is not None]
    return TextRiskPresentation(
        conclusion=labels[label], score=score, group=decision.get("threshold_group") or "未登记",
        screening=cutoff, strict=strict, review=review,
        high_windows=sum(s > strict for s in scored) if strict is not None else 0, windows=len(scored),
        high_coverage=decision.get("high_risk_coverage"),
    )
