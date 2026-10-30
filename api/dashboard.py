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

import requests
import streamlit as st

from api.text_view import (  # noqa: E402  文本结果渲染已抽到共享模块
    TEXT_CSS,
    render_analysis,
    run_with_elapsed,
)


API_BASE = os.getenv("AIGC_API_BASE", "http://localhost:8000/api/v1")

# 页面基础配置：宽屏布局更适合展示分数、证据和报告三类信息。
st.set_page_config(page_title="AIGC Detection Workbench", layout="wide")

# Streamlit 允许注入少量 CSS。这里仅调整页面宽度、metric 字号和状态块样式。
st.markdown(TEXT_CSS, unsafe_allow_html=True)

if "is_analyzing" not in st.session_state:
    st.session_state["is_analyzing"] = False


def http_call(method: str, url: str, **kwargs):
    """纯 HTTP 调用（可在后台线程运行，不调用 st.*）。返回 (payload, error)。"""
    try:
        resp = requests.request(method, url, timeout=600, **kwargs)
    except requests.RequestException as exc:
        return None, f"Backend unavailable: {exc}"
    if resp.status_code >= 400:
        return None, resp.text
    return resp.json(), None


def request_json(method: str, url: str, **kwargs):
    """统一封装前端到后端的 HTTP 请求。

    method 是 GET/POST 等 HTTP 方法，url 是 FastAPI 地址。
    如果后端没启动或接口报错，直接在 Streamlit 页面显示错误。
    """
    payload, error = http_call(method, url, **kwargs)
    if error:
        st.error(error)
    return payload


def request_with_progress(method: str, url: str, **kwargs):
    """带等待提示的 HTTP 请求封装。

    后端不报告内部步骤进度，所以这里只显示真实的已等待时间，
    不再展示固定的 18/42/78/100% 阶段百分比。
    """
    st.session_state["is_analyzing"] = True
    try:
        (payload, error), elapsed = run_with_elapsed(lambda: http_call(method, url, **kwargs), "正在分析内容")
    finally:
        st.session_state["is_analyzing"] = False
    if error:
        st.error(error)
        return None
    st.caption(f"分析完成，用时 {elapsed:.1f} 秒。")
    return payload


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
        "上传文档或媒体文件（Word .doc / .docx 可直接上传，无需先转 txt）",
        type=["txt", "md", "csv", "json", "html", "doc", "docx", "jpg", "jpeg", "png", "bmp", "webp", "tiff", "wav", "mp3", "flac", "ogg", "m4a", "mp4", "avi", "mov", "mkv", "webm"],
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
