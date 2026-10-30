"""Shared UI helpers: API client with typed errors, background jobs, styles, small HTML fragments."""
from __future__ import annotations

import html
import os
import threading
import time

import requests
import streamlit as st

API_BASE = os.getenv("AIGC_API_BASE", "http://127.0.0.1:8611/api/v1")

ERROR_TITLES = {
    "empty_file": "文件为空", "decode_failed": "图片无法解码", "unsupported_image_policy": "图片不符合输入要求",
    "too_large": "文件或像素超限", "unsupported_media_type": "不支持的文件类型",
    "unsupported_mode": "不支持的模式", "unknown_method_id": "未知研究方法", "invalid_request_id": "请求 ID 无效", "invalid_time_range": "时间范围无效",
    "request_id_conflict": "请求 ID 冲突", "model_unavailable": "模型不可用", "artifact_mismatch": "模型产物校验失败",
    "busy": "服务繁忙", "inference_failed": "推理失败", "report_not_found": "找不到报告",
    "timeout": "结果暂未收到", "backend_unreachable": "后端不可达",
}
ROLE_NAMES = {"default": "默认", "comparison": "对照", "registered_only": "仅登记结果", "research_online": "研究可运行"}

CSS = """
<style>
:root { --wb-blue:#1d4ed8; --wb-ink:#111827; --wb-muted:#4b5563; --wb-line:#e5e7eb; --wb-soft:#f5f7fb; }
.block-container { max-width: 1280px; padding-top: 4.2rem; }  /* clear the top navigation bar */
.wb-title { font-size: 1.35rem; font-weight: 650; color: var(--wb-ink); margin: 0 0 .2rem 0; }
.wb-sub { color: var(--wb-muted); font-size: .9rem; margin-bottom: .6rem; }
.wb-card { border: 1px solid var(--wb-line); border-radius: 10px; padding: .95rem 1.05rem; background: #fff;
  margin-bottom: .75rem; transition: border-color .2s ease, box-shadow .2s ease; animation: wb-in .22s ease-out both; }
.wb-card:hover { border-color: #c7d2fe; box-shadow: 0 2px 10px rgba(29, 78, 216, .08); }
.wb-card.empty { min-height: 9rem; display: flex; flex-direction: column; justify-content: center; }
.wb-kicker { font-size: .78rem; letter-spacing: .03em; color: var(--wb-muted); margin-bottom: .25rem; }
.wb-verdict { font-size: 1.55rem; font-weight: 650; color: var(--wb-ink); line-height: 1.2; }
.wb-verdict.ai { color: var(--wb-blue); }
.wb-score { font-size: 1.05rem; color: var(--wb-ink); font-variant-numeric: tabular-nums; margin-top: .2rem; }
.wb-line { color: var(--wb-ink); font-size: .95rem; margin-top: .35rem; }
.wb-muted { color: var(--wb-muted); font-size: .86rem; line-height: 1.5; }
.wb-chip { display: inline-block; padding: .08rem .55rem; margin: 0 .3rem .3rem 0; border-radius: 999px; font-size: .78rem;
  border: 1px solid var(--wb-line); color: var(--wb-ink); background: var(--wb-soft); }
.wb-chip.blue { background: #eff4ff; border-color: #c7d7fe; color: var(--wb-blue); }
.wb-track { position: relative; height: 8px; background: #eef1f6; border-radius: 4px; margin: .55rem 0 .2rem; }
.wb-fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--wb-blue); border-radius: 4px;
  transform-origin: left; animation: wb-grow .24s ease-out both; }
.wb-mark { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--wb-ink); }
.wb-row { display: grid; grid-template-columns: 1.6rem minmax(0, 1fr) 4.6rem; gap: .6rem; align-items: center; padding: .28rem 0; }
.wb-row .wb-track { margin: .2rem 0 0; height: 6px; }
.wb-rank { color: var(--wb-muted); font-variant-numeric: tabular-nums; }
.wb-num { text-align: right; font-variant-numeric: tabular-nums; }
@keyframes wb-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
@keyframes wb-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
  transition: background-color .18s ease, border-color .18s ease, transform .18s ease, box-shadow .18s ease; }
div[data-testid="stButton"] button:hover:not(:disabled), div[data-testid="stDownloadButton"] button:hover:not(:disabled) {
  transform: scale(1.015); }
div[data-testid="stButton"] button:active:not(:disabled), div[data-testid="stDownloadButton"] button:active:not(:disabled) {
  transform: scale(.985); }
div[data-testid="stButton"] button:focus-visible, div[data-testid="stDownloadButton"] button:focus-visible,
div[data-testid="stFileUploader"] button:focus-visible { outline: 2px solid var(--wb-blue); outline-offset: 2px; }
div[data-testid="stButton"] button:disabled, div[data-testid="stDownloadButton"] button:disabled {
  cursor: not-allowed; opacity: .55; }
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
        return None, dict(code="timeout", message="在等待时间内没有收到结果；无法确认后台是否已完成，不会自动重新提交",
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


def start_job(fn) -> dict:
    """Run fn (no st.* calls) in a thread; the job dict survives reruns and page switches."""
    job = dict(started=time.monotonic(), done=False, value=None)

    def work():
        try:
            job["value"] = fn()
        except BaseException as exc:  # never crash the page; report as a client-side error
            job["value"] = (None, dict(code="backend_unreachable", message=type(exc).__name__, retryable=True,
                                       request_id=None))
        finally:
            job["done"] = True
    threading.Thread(target=work, daemon=True).start()
    return job


def wait_job(job: dict, label: str):
    """Spinner + real elapsed time only (no fake percentage/ETA). Safe to interrupt by rerun."""
    box = st.empty()
    while not job["done"]:
        box.info(f"{label}… 已等待 {time.monotonic() - job['started']:.1f} 秒（后端不报告内部进度）",
                 icon=":material/hourglass_top:")
        time.sleep(0.25)
    box.empty()
    return job["value"]


def show_error(error: dict) -> None:
    title = ERROR_TITLES.get(error.get("code"), error.get("code"))
    if error.get("code") == "timeout":
        hint = "可点击“重新查询”，会沿用同一请求 ID，不会重复分析同一张图。"
    else:
        hint = "可稍后重试。" if error.get("retryable") else "请修改输入后再提交。"
    rid = f"（请求 ID {error['request_id']}）" if error.get("request_id") else ""
    st.error(f"{title}：{error.get('message', '')} {hint}{rid}", icon=":material/error:")


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
    image = caps["image"]
    image_state = {"ready": "图片模型就绪", "loading": "图片模型加载中", "not_loaded": "图片模型未加载",
                   "disabled": "图片模型未配置", "unavailable": "图片模型不可用",
                   "artifact_mismatch": "图片模型校验失败"}.get(image["state"], image["state"])
    chips = ["隔离预览" if caps.get("preview") else "非预览配置", image_state, "不保存原图"]
    st.markdown("".join(f'<span class="wb-chip{" blue" if i == 0 else ""}">{esc(c)}</span>'
                        for i, c in enumerate(chips)), unsafe_allow_html=True)
    return caps


def score_bar(score: float, threshold: float | None = None) -> str:
    mark = f'<div class="wb-mark" style="left:{threshold * 100:.2f}%"></div>' if threshold is not None else ""
    return f'<div class="wb-track"><div class="wb-fill" style="width:{score * 100:.2f}%"></div>{mark}</div>'


def ranked_rows(items: list[dict], limit: int) -> str:
    return "".join(f'<div class="wb-row"><span class="wb-rank">{t["rank"]}</span><div><div>{esc(t["source"])}</div>'
                   f'{score_bar(t["score"])}</div><span class="wb-num">{t["score"] * 100:.2f}%</span></div>'
                   for t in items[:limit])


def persist(key: str):
    """Copy a widget value into a non-widget key so it survives page navigation."""
    def callback():
        st.session_state[f"{key}_saved"] = st.session_state[key]
    return callback


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"
