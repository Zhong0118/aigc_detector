"""Text result header used by the standalone dashboard and online workbench."""
import plotly.graph_objects as go
import streamlit as st

from reports.generator import UNCALIBRATED_TEXT_LABEL, UNCALIBRATED_TEXT_SCORE, uncalibrated_text_reason
from reports.text_risk import SCORE_EXPLANATION, SCORE_LABEL, TextRiskPresentation, format_percent


def render_uncalibrated_text(data: dict) -> bool:
    detection = data.get("detection") or {}
    label = (detection.get("decision") or {}).get("decision") or detection.get("label")
    if data.get("modality") != "text" or label != "insufficient_evidence":
        return False
    reason = uncalibrated_text_reason(data)
    st.subheader(UNCALIBRATED_TEXT_LABEL if reason else "证据不足")
    st.caption(UNCALIBRATED_TEXT_SCORE)
    st.info(reason or "当前没有足够的有效模型证据，无法给出可靠判定。")
    return True


def render_text_risk(view: TextRiskPresentation, analyzed_coverage: float | None = None) -> None:
    st.subheader(view.conclusion)
    threshold_cols = st.columns(4)
    threshold_cols[0].metric(SCORE_LABEL, format_percent(view.score), help=SCORE_EXPLANATION)
    threshold_cols[1].metric("复核线", format_percent(view.screening))
    threshold_cols[2].metric("严格线", format_percent(view.strict))
    threshold_cols[3].metric("阈值组", view.group)
    if view.windows or view.high_coverage is not None or analyzed_coverage is not None:
        window_cols = st.columns(3)
        window_count = f"{view.high_windows} / {view.windows}" if view.windows and view.strict is not None else "不可用"
        window_cols[0].metric("严格线以上窗口", window_count)
        window_cols[1].metric("去重叠高分字符覆盖率", format_percent(view.high_coverage))
        window_cols[2].metric("检测覆盖率", format_percent(analyzed_coverage))
    st.caption("模型校准风险分用于对照本分支复核线与严格线，不是 AI 撰写字数比例。")


def render_score_gauge(
    score: float,
    low_threshold: float = 0.4,
    high_threshold: float = 0.8,
    screening_threshold: float | None = None,
    *,
    title: str = "模型校准风险分（非 AI 字数比例）",
) -> None:
    """Show the score against the thresholds for the selected branch."""
    has_screening_band = screening_threshold is not None and low_threshold < screening_threshold < high_threshold
    steps = [{"range": [0, low_threshold], "color": "#dbeafe"}]
    if has_screening_band:
        steps.extend([
            {"range": [low_threshold, screening_threshold], "color": "#e5e7eb"},
            {"range": [screening_threshold, high_threshold], "color": "#fbbf24"},
        ])
    else:
        steps.append({"range": [low_threshold, high_threshold], "color": "#fde68a"})
    steps.append({"range": [high_threshold, 1], "color": "#fecaca"})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 15}},
        number={"valueformat": ".2%"},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".0%"},
            "bar": {"color": "#1d4ed8", "thickness": 0.28},
            "steps": steps,
            "threshold": {"line": {"color": "#991b1b", "width": 3}, "value": high_threshold},
        },
    ))
    fig.update_layout(height=250, margin={"l": 28, "r": 28, "t": 40, "b": 16})
    st.plotly_chart(fig, width="stretch")
    if has_screening_band:
        st.caption(f"琥珀区间起点 · 复核线 {format_percent(screening_threshold)}　|　红色标线 · 严格线 {format_percent(high_threshold)}")
