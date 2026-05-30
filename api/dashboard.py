"""Streamlit 前端演示工作台。

这个文件不是传统 HTML/React 前端，而是用 Streamlit 写的 Python 页面。
它负责让用户输入文本或上传文件，然后调用 FastAPI 接口，把返回的 JSON
可视化成分数、证据、溯源和报告。
"""

import os
import sys
from pathlib import Path

# Streamlit 从 `api/dashboard.py` 启动时，默认 import 路径可能只包含 api/。
# 这里把项目根目录加入 sys.path，保证可以导入同级项目包。
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import requests
import streamlit as st


API_BASE = os.getenv("AIGC_API_BASE", "http://localhost:8000/api/v1")

# 页面基础配置：宽屏布局更适合展示分数、证据和报告三类信息。
st.set_page_config(page_title="AIGC Detection Workbench", layout="wide")

# Streamlit 允许注入少量 CSS。这里仅调整页面宽度、metric 字号和状态块样式。
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1440px;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    .status-row {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: #f8fafc;
    }
    .small-muted {color: #64748b; font-size: 0.88rem;}
    .evidence-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 0.95rem;
        margin-bottom: 0.7rem;
        background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "is_analyzing" not in st.session_state:
    st.session_state["is_analyzing"] = False


def public_branch_name(index: int) -> str:
    """把 provider 名称匿名化为检测分支名称。

    Raw JSON 仍然保留真实字段，方便开发调试；主页面面向演示和普通用户，
    使用更中性的“检测分支A/B”。
    """
    return f"检测分支{chr(ord('A') + index)}"


def public_evidence_text(text: str) -> str:
    """清理主页面证据文字中的供应商和接口细节。"""
    replacements = {
        "Hive V3 VLM": "检测分支",
        "Hive": "检测分支",
        "hive": "检测分支",
        "Sightengine": "检测分支",
        "sightengine": "检测分支",
        "AI-generation API": "AIGC 检测模块",
        "api_version": "检测版本",
        "v3_vlm": "检测模块",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def request_json(method: str, url: str, **kwargs):
    """统一封装前端到后端的 HTTP 请求。

    method 是 GET/POST 等 HTTP 方法，url 是 FastAPI 地址。
    如果后端没启动或接口报错，直接在 Streamlit 页面显示错误。
    """
    try:
        resp = requests.request(method, url, timeout=120, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Backend unavailable: {exc}")
        return None
    if resp.status_code >= 400:
        st.error(resp.text)
        return None
    return resp.json()


def request_with_progress(method: str, url: str, **kwargs):
    """带进度提示的 HTTP 请求封装。

    Streamlit 不能知道后端每个内部步骤的实时进度，所以这里展示的是
    前端侧的阶段性进度：提交任务、等待检测、整理结果。
    """
    progress = st.progress(0, text="准备提交检测任务...")
    status = st.status("正在分析内容，请稍候...", expanded=True)
    try:
        st.session_state["is_analyzing"] = True
        progress.progress(18, text="正在读取输入并识别内容类型...")
        status.write("已接收输入，正在进入检测流程。")
        progress.progress(42, text="正在运行多路检测模块...")
        payload = request_json(method, url, **kwargs)
        progress.progress(78, text="正在汇总分数与溯源证据...")
        if payload:
            status.write("检测完成，正在生成解释报告。")
            progress.progress(100, text="分析完成。")
            status.update(label="分析完成", state="complete", expanded=False)
        else:
            status.update(label="分析失败，请查看错误信息", state="error", expanded=True)
        return payload
    finally:
        st.session_state["is_analyzing"] = False


def render_score_gauge(score: float):
    """渲染 AI 概率仪表盘。

    score 范围是 0 到 1。颜色区间用于帮助用户快速判断风险：
    低分灰色，中段黄色，高分红色。
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".1%"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [0, 0.5], "color": "#e2e8f0"},
                    {"range": [0.5, 0.75], "color": "#fde68a"},
                    {"range": [0.75, 1], "color": "#fecaca"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 3}, "value": 0.5},
            },
        )
    )
    fig.update_layout(height=260, margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


def render_analysis(data: dict):
    """渲染一次完整分析结果。

    data 是 FastAPI `/analyze/text` 或 `/analyze/file` 返回的 JSON。
    这里会展示：
    - 顶部核心指标
    - provider 检测证据
    - 可能来源模型提示
    - C2PA/watermark 溯源状态
    - 解释报告
    - 原始 JSON
    """
    detection = data["detection"]
    provenance = data["provenance"]
    report = data["report"]

    # 顶部四个 metric 用来快速回答：分数多少、标签是什么、类型是什么、记录 ID 是多少。
    top = st.columns([1.2, 1, 1, 1])
    top[0].metric("AI Score", f"{detection['score']:.2%}")
    top[1].metric("Label", detection["label"].upper())
    top[2].metric("Modality", data["modality"].upper())
    top[3].metric("Content ID", data["content_id"])

    left, right = st.columns([0.92, 1.08])
    with left:
        # 左侧主要展示检测层证据。
        render_score_gauge(detection["score"])
        st.subheader("Detection Evidence")
        for idx, provider in enumerate(detection["providers"]):
            cols = st.columns([1.1, 0.8, 0.8, 1.4])
            cols[0].write(public_branch_name(idx))
            cols[1].write(provider["status"])
            score = provider["score"]
            cols[2].write("n/a" if score is None else f"{score:.2%}")
            details = provider.get("details", {})
            caption = details.get("explanation") or details.get("note") or details.get("score_source", "")
            cols[3].caption(public_evidence_text(str(caption)))

        if detection["model_scores"]:
            # model_scores 当前来自 demo provider；未来可来自 Hive/Sightengine 或本地 attribution。
            st.subheader("Source Hints")
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(detection["model_scores"].values()),
                        y=list(detection["model_scores"].keys()),
                        orientation="h",
                        marker_color="#0f766e",
                    )
                ]
            )
            fig.update_layout(
                height=240,
                xaxis_range=[0, 1],
                margin={"l": 20, "r": 20, "t": 10, "b": 20},
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        # 右侧主要展示溯源层和报告层。
        st.subheader("Provenance")
        pcols = st.columns(4)
        pcols[0].metric("Deep Check", "YES" if provenance["deep_triggered"] else "NO")
        pcols[1].metric("C2PA", provenance["c2pa"].get("status", "unknown"))
        pcols[2].metric("Watermark", provenance["watermark"].get("status", "unknown"))
        registry = provenance.get("fingerprint_registry", {})
        pcols[3].metric("Fingerprint Hits", registry.get("match_count", 0))

        if registry.get("matches"):
            with st.expander("Fingerprint Registry Matches", expanded=False):
                for match in registry["matches"]:
                    st.write(
                        f"content_id={match.get('content_id')} | "
                        f"type={match.get('match_type')} | "
                        f"similarity={match.get('similarity'):.2%}"
                    )
                    if match.get("filename"):
                        st.caption(match["filename"])

        st.subheader("Report")
        st.write(report["summary"])
        st.write("Evidence")
        for item in report["evidence"]:
            st.write(f"- {item}")
        st.write("Limitations")
        for item in report["limitations"]:
            st.write(f"- {item}")
        st.info(report["recommendation"])

    with st.expander("Raw JSON"):
        # 原始 JSON 方便调试，也方便后续确认字段结构是否满足前端/报告需要。
        st.json(data)


st.title("AIGC Detection Workbench")

# 页面加载时先请求 provider 状态，让用户知道当前是 demo_api 还是已接真实 API。
provider_payload = request_json("GET", f"{API_BASE}/providers")
if provider_payload:
    status_cols = st.columns([1, 1, 1, 1])
    status_cols[0].metric("Mode", provider_payload["mode"])
    status_cols[1].metric("API Port", provider_payload["ports"]["api"])
    status_cols[2].metric("Dashboard Port", provider_payload["ports"]["dashboard"])
    status_cols[3].metric("Deep Threshold", f"{provider_payload['deep_provenance_threshold']:.2f}")

    with st.expander("Provider Status", expanded=False):
        for provider in provider_payload["providers"]:
            st.write(
                f"**{provider['name']}** | {provider['type']} | "
                f"enabled={provider['enabled']} | configured={provider['configured']}"
            )
            st.caption(provider["purpose"])

text_tab, file_tab, history_tab = st.tabs(["Text", "File", "Report"])

with text_tab:
    # 文本输入页：直接调用 `/api/v1/analyze/text`。
    text = st.text_area("Text Input", height=220)
    filename = st.text_input("Virtual Filename", value="input.txt")
    analyzing = st.session_state["is_analyzing"]
    if st.button(
        "Analyzing..." if analyzing else "Analyze Text",
        type="primary",
        use_container_width=True,
        disabled=analyzing,
    ):
        if not text.strip():
            st.warning("Text input is empty.")
        else:
            payload = request_with_progress(
                "POST",
                f"{API_BASE}/analyze/text",
                json={"text": text, "filename": filename or "input.txt"},
            )
            if payload:
                render_analysis(payload)

with file_tab:
    # 文件上传页：上传文件后调用 `/api/v1/analyze/file`。
    uploaded_file = st.file_uploader(
        "File Upload",
        type=["txt", "md", "csv", "json", "html", "jpg", "jpeg", "png", "bmp", "webp", "tiff", "wav", "mp3", "flac", "ogg", "m4a", "mp4", "avi", "mov", "mkv", "webm"],
    )
    analyzing = st.session_state["is_analyzing"]
    if st.button(
        "Analyzing..." if analyzing else "Analyze File",
        type="primary",
        use_container_width=True,
        disabled=analyzing,
    ):
        if uploaded_file is None:
            st.warning("No file selected.")
        else:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            payload = request_with_progress("POST", f"{API_BASE}/analyze/file", files=files)
            if payload:
                render_analysis(payload)

with history_tab:
    # 报告页：根据数据库 content_id 读取历史分析记录。
    content_id = st.number_input("Content ID", min_value=1, step=1)
    if st.button("Load Report", use_container_width=True):
        payload = request_json("GET", f"{API_BASE}/report/{content_id}")
        if payload:
            st.json(payload)
