"""Text result header used by the standalone dashboard and online workbench."""
import streamlit as st

from reports.generator import UNCALIBRATED_TEXT_LABEL, UNCALIBRATED_TEXT_SCORE, uncalibrated_text_reason
from reports.text_risk import SCORE_EXPLANATION, SCORE_LABEL, TextRiskPresentation, format_percent


def render_uncalibrated_text(data: dict) -> bool:
    reason = uncalibrated_text_reason(data)
    if not reason:
        return False
    st.subheader(UNCALIBRATED_TEXT_LABEL)
    st.caption(UNCALIBRATED_TEXT_SCORE)
    st.info(reason)
    with st.expander("Raw JSON"):
        st.json(data)
    return True


def render_text_risk(view: TextRiskPresentation) -> None:
    st.subheader(view.conclusion)
    threshold_cols = st.columns(3)
    threshold_cols[0].metric("阈值组", view.group)
    threshold_cols[1].metric("复核线", format_percent(view.screening))
    threshold_cols[2].metric("严格线", format_percent(view.strict))
    st.caption(f"{SCORE_LABEL} {format_percent(view.score)}（{SCORE_EXPLANATION}）")
    if view.windows and view.strict is not None:
        window_cols = st.columns(2)
        window_cols[0].metric("严格线以上窗口", f"{view.high_windows} / {view.windows}")
        window_cols[1].metric("去重叠高分字符覆盖率", format_percent(view.high_coverage))
