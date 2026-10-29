"""Shared UI helpers: API client with typed errors, styles, small HTML fragments."""
from __future__ import annotations

import html
import os

import requests
import streamlit as st

API_BASE = os.getenv("AIGC_API_BASE", "http://127.0.0.1:8611/api/v1")

ERROR_TITLES = {
    "invalid_input": "输入无效", "too_large": "文件过大", "decode_failed": "图片解码失败",
    "model_unavailable": "模型不可用", "artifact_mismatch": "模型产物校验失败", "busy": "服务繁忙",
    "inference_failed": "推理失败", "report_not_found": "找不到记录", "timeout": "等待超时",
    "backend_unreachable": "后端不可达",
}
ROLE_NAMES = {"default": "默认", "comparison": "对照", "registered_only": "仅登记结果"}

CSS = """
<style>
:root { --wb-blue:#1d4ed8; --wb-ink:#111827; --wb-muted:#5b6475; --wb-line:#e4e7ee; --wb-soft:#f4f6fa; }
.block-container { max-width: 1280px; padding-top: 4.2rem; }  /* clear the top navigation bar */
.wb-title { font-size: 1.35rem; font-weight: 650; color: var(--wb-ink); margin: 0 0 .2rem 0; }
.wb-sub { color: var(--wb-muted); font-size: .9rem; margin-bottom: .6rem; }
.wb-card { border: 1px solid var(--wb-line); border-radius: 10px; padding: .95rem 1.05rem; background: #fff;
  margin-bottom: .75rem; transition: border-color .18s ease, box-shadow .18s ease; animation: wb-in .22s ease-out both; }
.wb-card:hover { border-color: #c7d2fe; box-shadow: 0 2px 10px rgba(29, 78, 216, .08); }
.wb-kicker { font-size: .78rem; letter-spacing: .03em; color: var(--wb-muted); margin-bottom: .25rem; }
.wb-score { font-size: 2.3rem; font-weight: 650; color: var(--wb-ink); font-variant-numeric: tabular-nums; line-height: 1.1; }
.wb-line { color: var(--wb-ink); font-size: .95rem; margin-top: .35rem; }
.wb-muted { color: var(--wb-muted); font-size: .86rem; line-height: 1.5; }
.wb-chip { display: inline-block; padding: .08rem .55rem; margin: 0 .3rem .3rem 0; border-radius: 999px; font-size: .78rem;
  border: 1px solid var(--wb-line); color: var(--wb-ink); background: var(--wb-soft); }
.wb-chip.blue { background: #eff4ff; border-color: #c7d7fe; color: var(--wb-blue); }
.wb-track { position: relative; height: 8px; background: #eef1f6; border-radius: 4px; margin: .55rem 0 .2rem; }
.wb-fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--wb-blue); border-radius: 4px;
  transform-origin: left; animation: wb-grow .35s ease-out both; }
.wb-mark { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--wb-ink); }
.wb-row { display: grid; grid-template-columns: 1.6rem minmax(0, 1fr) 4.6rem; gap: .6rem; align-items: center; padding: .28rem 0; }
.wb-row .wb-track { margin: .2rem 0 0; height: 6px; }
.wb-rank { color: var(--wb-muted); font-variant-numeric: tabular-nums; }
.wb-num { text-align: right; font-variant-numeric: tabular-nums; }
@keyframes wb-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
@keyframes wb-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
  transition: background-color .15s ease, border-color .15s ease, transform .08s ease; }
div[data-testid="stButton"] button:active:not(:disabled) { transform: translateY(1px); }
div[data-testid="stButton"] button:focus-visible, div[data-testid="stDownloadButton"] button:focus-visible {
  outline: 2px solid var(--wb-blue); outline-offset: 2px; }
div[data-testid="stButton"] button:disabled { cursor: not-allowed; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def api(method: str, path: str, timeout: float = 30, **kwargs):
    """Plain HTTP (safe in a worker thread). Returns (payload, error) with the API's error shape."""
    try:
        response = requests.request(method, API_BASE + path, timeout=timeout, **kwargs)
    except requests.Timeout:
        return None, dict(code="timeout", message="等待超时；不会自动重试，可手动再次提交（同一请求会去重）",
                          retryable=True, request_id=None)
    except requests.RequestException:
        return None, dict(code="backend_unreachable", message="无法连接预览后端", retryable=True, request_id=None)
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code >= 400:
        error = body.get("error") if isinstance(body, dict) else None
        return None, error or dict(code=f"http_{response.status_code}", message=str(body.get("detail", ""))[:300],
                                   retryable=response.status_code >= 500, request_id=None)
    return body, None


def show_error(error: dict) -> None:
    title = ERROR_TITLES.get(error.get("code"), error.get("code"))
    hint = "可稍后重试。" if error.get("retryable") else "请修改输入后再提交。"
    rid = f"（请求 ID {error['request_id']}）" if error.get("request_id") else ""
    st.error(f"{title}：{error.get('message', '')} {hint}{rid}")


@st.cache_data(ttl=5, show_spinner=False)
def capabilities():
    return api("GET", "/capabilities", timeout=10)


def header(title: str, subtitle: str) -> dict | None:
    caps, error = capabilities()
    st.markdown(f'<div class="wb-title">{esc(title)}</div><div class="wb-sub">{esc(subtitle)}</div>',
                unsafe_allow_html=True)
    if error:
        show_error(error)
        return None
    image_state = {"ready": "图片模型就绪", "loading": "图片模型加载中", "not_loaded": "图片模型未加载",
                   "disabled": "图片模型未配置", "unavailable": "图片模型不可用",
                   "artifact_mismatch": "图片模型校验失败"}.get(caps["image"]["state"], caps["image"]["state"])
    chips = ["隔离预览" if caps.get("preview") else "非预览配置", f"设备 {caps['image'].get('device') or '—'}",
             image_state, "不保存原图"]
    st.markdown("".join(f'<span class="wb-chip{" blue" if i == 0 else ""}">{esc(c)}</span>'
                        for i, c in enumerate(chips)), unsafe_allow_html=True)
    return caps


def score_bar(score: float, threshold: float | None = None) -> str:
    mark = f'<div class="wb-mark" style="left:{threshold * 100:.2f}%"></div>' if threshold is not None else ""
    return f'<div class="wb-track"><div class="wb-fill" style="width:{score * 100:.2f}%"></div>{mark}</div>'


def ranked_rows(items: list[dict], limit: int) -> str:
    rows = [f'<div class="wb-row"><span class="wb-rank">{t["rank"]}</span><div><div>{esc(t["label"])}</div>'
            f'{score_bar(t["score"])}</div><span class="wb-num">{t["score"]:.4f}</span></div>' for t in items[:limit]]
    return "".join(rows)


def persist(key: str):
    """Copy a widget value into a non-widget key so it survives page navigation."""
    def callback():
        st.session_state[f"{key}_saved"] = st.session_state[key]
    return callback


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"
