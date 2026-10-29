"""Four workbench pages. Results live in session_state, so navigation never re-runs inference."""
from __future__ import annotations

import hashlib
import uuid

import pandas as pd
import streamlit as st

from api.text_view import TEXT_CSS, render_analysis, run_with_elapsed
from workbench import export
from workbench.ui.common import ROLE_NAMES, api, header, pct, persist, show_error
from workbench.ui.image_view import render_image_result

S = st.session_state
TEXT_TYPES = ["txt", "md", "csv", "json", "html", "doc", "docx"]
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]


# ---------------------------------------------------------------- text
def _submit_text(kind: str) -> None:
    S["txt_pending"] = dict(kind=kind)
    S.pop("txt_error", None)


def text_page() -> None:
    header("文本检测", "沿用现有文本流程：分块检测、阈值判定与家族溯源；不使用外部检测 API 与 LLM 解释。")
    st.markdown(TEXT_CSS, unsafe_allow_html=True)
    pending = bool(S.get("txt_pending"))
    source = st.segmented_control("输入方式", ["粘贴文本", "上传文档"], default=S.get("txt_source_saved", "粘贴文本"),
                                  key="txt_source", on_change=persist("txt_source"), disabled=pending)
    if source == "上传文档":
        upload = st.file_uploader("上传文档（txt / md / docx / doc 等）", type=TEXT_TYPES, key="txt_upload",
                                  disabled=pending)
        if upload is not None:
            S["txt_doc"] = dict(name=upload.name, data=upload.getvalue())
        doc = S.get("txt_doc")
        if doc:
            st.caption(f"当前文档：{doc['name']}（{len(doc['data']):,} 字节）")
        st.button("检测中…" if pending else "开始检测", type="primary", icon=":material/play_arrow:",
                  disabled=pending or not doc, on_click=_submit_text, args=("doc",), width="stretch")
    else:
        st.text_area("文本内容", value=S.get("txt_value_saved", ""), height=220, key="txt_value",
                     on_change=persist("txt_value"), disabled=pending)
        st.text_input("虚拟文件名", value=S.get("txt_name_saved", "input.txt"), key="txt_name",
                      on_change=persist("txt_name"), disabled=pending)
        text = S.get("txt_value", S.get("txt_value_saved", ""))
        st.button("检测中…" if pending else "开始检测", type="primary", icon=":material/play_arrow:",
                  disabled=pending or not text.strip(), on_click=_submit_text, args=("text",), width="stretch")
    if pending:
        job = S["txt_pending"]
        if job["kind"] == "doc":
            doc = S["txt_doc"]
            call = lambda: api("POST", "/analyze/file", timeout=600, files={"file": (doc["name"], doc["data"])})
        else:
            payload = dict(text=S.get("txt_value_saved", S.get("txt_value", "")),
                           filename=S.get("txt_name_saved", "input.txt") or "input.txt")
            call = lambda: api("POST", "/analyze/text", timeout=600, json=payload)
        (result, error), _ = run_with_elapsed(call, "正在分析文本")
        S.pop("txt_pending", None)
        if error:
            S["txt_error"] = error
        elif result.get("image_result"):
            S["txt_error"] = dict(code="invalid_input", message="这是图片文件，请在“图片检测”页分析", retryable=False)
        else:
            S["txt_result"] = result
        st.rerun()
    if S.get("txt_error"):
        show_error(S["txt_error"])
    if S.get("txt_result"):
        render_analysis(S["txt_result"])


# ---------------------------------------------------------------- image
def _on_image_upload() -> None:
    upload = S.get(f"img_uploader_{S.get('img_nonce', 0)}")
    if upload is not None:
        data = upload.getvalue()
        S["img_file"] = dict(name=upload.name, data=data, sha256=hashlib.sha256(data).hexdigest())
        S.pop("img_error", None)


def _clear_image() -> None:
    S["img_nonce"] = S.get("img_nonce", 0) + 1
    for key in ("img_file", "img_current", "img_error", "img_pending"):
        S.pop(key, None)


def _submit_image() -> None:
    file, research = S.get("img_file"), bool(S.get("img_research_saved", False))
    key = (file["sha256"], "research" if research else "default")
    S.pop("img_error", None)
    if key in S.setdefault("img_results", {}):
        S["img_current"] = key  # same image + mode: reuse, never re-infer
        return
    request_id = S.setdefault("img_request_ids", {}).get(key) or uuid.uuid4().hex
    S["img_request_ids"][key] = request_id
    S["img_pending"] = dict(key=key, request_id=request_id)


def image_page() -> None:
    caps = header("图片检测", "D-OpenCLIP 检测分数 + A0 ResNet50 22 类来源候选；未知来源在主接口不可用。")
    ready = bool(caps and caps["image"]["state"] == "ready")
    pending = bool(S.get("img_pending"))
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.file_uploader("选择图片（JPEG / PNG / WebP / BMP，≤20 MB）", type=IMAGE_TYPES,
                         key=f"img_uploader_{S.get('img_nonce', 0)}", on_change=_on_image_upload, disabled=pending)
        file = S.get("img_file")
        if file:
            try:
                st.image(file["data"], caption=f"{file['name']} · {len(file['data']):,} 字节", width="stretch")
            except Exception:  # a broken file still goes to the server for typed validation
                st.caption(f"{file['name']} · {len(file['data']):,} 字节 · 浏览器端无法预览，提交后由服务器校验")
        st.toggle("研究模式", value=S.get("img_research_saved", False), key="img_research",
                  on_change=persist("img_research"), disabled=pending,
                  help="额外显示 1%/5% 校准阈值与 OFA 状态；分数语义彼此独立，不做融合。")
        cols = st.columns([3, 2])
        cols[0].button("分析中…" if pending else "开始分析", type="primary", icon=":material/play_arrow:",
                       disabled=pending or not file or not ready, on_click=_submit_image, width="stretch")
        cols[1].button("清除", icon=":material/close:", disabled=pending or not file, on_click=_clear_image,
                       width="stretch")
        if caps and not ready:
            st.info({"loading": "图片模型正在加载（首次约 1 分钟），完成后即可分析。"}.get(
                caps["image"]["state"], "图片模型不可用，分析按钮已禁用；不会使用演示或占位分数。"))
        st.caption("原始图片只用于本次分析，历史记录只保存结果快照与哈希。")
    with right:
        if pending:
            job, file = S["img_pending"], S["img_file"]
            (result, error), _ = run_with_elapsed(lambda: api(
                "POST", "/analyze/file", timeout=180, files={"file": (file["name"], file["data"])},
                data=dict(mode=job["key"][1], request_id=job["request_id"])), "正在分析图片")
            S.pop("img_pending", None)
            if error:
                S["img_error"] = error
                if error["code"] not in ("timeout", "backend_unreachable"):
                    S["img_request_ids"].pop(job["key"], None)  # only a lost reply reuses the request id
            else:
                S["img_results"][job["key"]] = result["image_result"]
                S["img_current"] = job["key"]
                S["img_request_ids"].pop(job["key"], None)
                capabilities_refresh()
            st.rerun()
        if S.get("img_error"):
            show_error(S["img_error"])
        current = S.get("img_current")
        if current and current in S.get("img_results", {}):
            render_image_result(S["img_results"][current], key="img-live")
        elif not S.get("img_error"):
            st.markdown('<div class="wb-card"><div class="wb-kicker">结果</div><div class="wb-muted">'
                        '上传图片并点击“开始分析”。结果包含模型检测分数、阈值、来源候选与局限说明。</div></div>',
                        unsafe_allow_html=True)


def capabilities_refresh() -> None:
    from workbench.ui.common import capabilities
    capabilities.clear()


# ---------------------------------------------------------------- experiments
@st.cache_data(ttl=60, show_spinner=False)
def _catalog():
    return api("GET", "/experiments", timeout=30)


def _paired_table(paired: list[dict]) -> pd.DataFrame:
    names = {"auroc": "AUROC", "fpr_0_5": "FPR@0.5", "tpr_0_5": "TPR@0.5", "top1": "Top1", "top3": "Top3",
             "macro_f1": "Macro-F1", "accuracy": "Accuracy"}
    return pd.DataFrame([{"配对比较（同一样本）": p["comparison"], "指标": names.get(m, m),
                          "差值（百分点）": f"{v['delta'] * 100:.2f}",
                          "95% 区间（百分点）": f"[{v['ci95'][0] * 100:.2f}, {v['ci95'][1] * 100:.2f}]",
                          "样本 / group / 重复": f"{p['n']} / {p['group_n']} / {p['repeats']}"}
                         for p in paired for m, v in p["deltas"].items()])


def experiments_page() -> None:
    header("实验对照", "只读展示已登记结果（哈希校验）；不同分母不比较“提升”，只有同一样本的配对比较给出差值与区间。")
    catalog, error = _catalog()
    if error:
        show_error(error)
        return
    for problem in catalog["errors"]:
        st.error(f"登记结果 {problem['source']} 不可用：{problem['message']}")
    ds = catalog["dataset"]
    st.caption(f"IAB v7-200-q179 · {ds['effective_images']:,} 张有效图片 · 每格实际 "
               f"{ds['actual_generated_per_cell'][0]}–{ds['actual_generated_per_cell'][1]} 张 · seed {ds['seed']} · "
               f"{ds['training_epochs']} 轮 · holdout 未评分")
    views = {v["id"]: v for v in catalog["views"]}
    tabs = st.tabs(["检测", "来源归因", "未知拒识", "路线状态", "数据来源与局限"])
    with tabs[0]:
        if "detection" in views:
            for group in views["detection"]["groups"]:
                st.markdown(f"**{group['title']}**")
                st.dataframe(pd.DataFrame([{
                    "模型": r["model"], "角色": ROLE_NAMES[r["role"]], "AUROC": pct(r["auroc"]),
                    "FPR@0.5（误报/真实）": f"{pct(r['fpr'])}（{r['fp']}/{r['negative_n']}）",
                    "TPR@0.5（检出/AI）": f"{pct(r['tpr'])}（{r['tp']}/{r['positive_n']}）"} for r in group["rows"]]),
                    hide_index=True, width="stretch")
                st.dataframe(_paired_table(group["paired"]), hide_index=True, width="stretch")
            with st.expander("研究模式：D-OpenCLIP 校准阈值"):
                st.dataframe(pd.DataFrame([{
                    "目标 FPR": f"{t['target_fpr']:.0%}", "阈值": round(t["threshold"], 6),
                    "calibration FPR（真实图）": f"{pct(t['calibration_fpr'])}（{t['calibration_negative_n']}）",
                    "evaluation FPR（真实图）": f"{pct(t['evaluation_fpr'])}（{t['evaluation_negative_n']}）",
                    "evaluation TPR": pct(t["evaluation_tpr"])} for t in views["detection"]["research_thresholds"]]),
                    hide_index=True, width="stretch")
            for note in views["detection"]["notes"]:
                st.caption(note)
    with tabs[1]:
        if "attribution" in views:
            group = views["attribution"]["groups"][0]
            st.markdown(f"**{group['title']}**")
            st.dataframe(pd.DataFrame([{
                "模型": r["model"], "角色": ROLE_NAMES[r["role"]],
                "Top1（正确/总数）": f"{pct(r['top1'])}（{r['top1_correct']}/{r['n']}）",
                "Top3（命中/总数）": f"{pct(r['top3'])}（{r['top3_correct']}/{r['n']}）",
                "Macro-F1": pct(r["macro_f1"])} for r in group["rows"]]), hide_index=True, width="stretch")
            st.dataframe(_paired_table(group["paired"]), hide_index=True, width="stretch")
            for note in views["attribution"]["notes"]:
                st.caption(note)
    with tabs[2]:
        if "rejection" in views:
            group = views["rejection"]["groups"][0]
            st.markdown(f"**{group['title']}**")
            st.dataframe(pd.DataFrame([{
                "方法": r["model"],
                "已知保留": f"{r['known_retention_count']}/{r['known_n']}（{pct(r['known_retention'])}）",
                "正确且保留": f"{r['known_correct_and_retained_count']}/{r['known_n']}（{pct(r['known_correct_and_retained'])}）",
                "未知误接收": f"{r['unknown_false_acceptance_count']}/{r['unknown_n']}（{pct(r['unknown_false_acceptance'])}）",
                "未知拒绝": f"{r['unknown_rejection_count']}/{r['unknown_n']}",
                "未知 AUROC": pct(r["unknown_auroc"])} for r in group["rows"]]), hide_index=True, width="stretch")
            for note in views["rejection"]["notes"]:
                st.caption(note)
    with tabs[3]:
        st.dataframe(pd.DataFrame([{"路线": r["id"], "运行状态": r["run_status"],
                                    "验收": "未验收" if r["acceptance"] == "not_accepted" else r["acceptance"],
                                    "上传推理": "否", "说明": r["note"]} for r in catalog["routes"]]),
                     hide_index=True, width="stretch")
        st.caption("未验收路线只显示状态，不提供上传图片推理。")
    with tabs[4]:
        st.dataframe(pd.DataFrame([{"登记结果": v["label"], "sha256": v["sha256"]} for v in catalog["sources"].values()]),
                     hide_index=True, width="stretch")
        st.markdown("\n".join(f"- {x}" for x in catalog["limitations"]))


# ---------------------------------------------------------------- history
def _reset_page() -> None:
    S["hist_page"] = 1


def history_page() -> None:
    header("历史报告", "从预览数据库还原当时的结果快照（模型哈希、阈值、预处理），不重新推理、不保存原图。")
    cols = st.columns([1.2, 1.2, 2.6, 1])
    kinds = {"全部": None, "图片": "image", "文本": "text"}
    statuses = {"全部": None, "完成": "ok", "部分结果": "partial"}
    kind = cols[0].selectbox("类型", list(kinds), key="hist_kind", on_change=_reset_page)
    status = cols[1].selectbox("状态", list(statuses), key="hist_status", on_change=_reset_page)
    query = cols[2].text_input("名称或 SHA-256", key="hist_query", on_change=_reset_page)
    if cols[3].button("刷新", icon=":material/refresh:", width="stretch"):
        st.rerun()
    page = S.setdefault("hist_page", 1)
    params = {k: v for k, v in dict(page=page, page_size=15, kind=kinds[kind], status=statuses[status],
                                    q=query.strip() or None).items() if v is not None}
    listing, error = api("GET", "/history", params=params)
    if error:
        show_error(error)
        return
    total, pages = listing["total"], max(1, -(-listing["total"] // listing["page_size"]))
    if not listing["items"]:
        st.info("暂无记录。")
        return
    labels = {"at_or_above_threshold": "达到阈值", "below_threshold": "低于阈值", **export.TEXT_LABELS}
    frame = pd.DataFrame([{"时间（UTC）": i["created_at"], "类型": "图片" if i["kind"] == "image" else "文本",
                           "名称": i["display_name"], "状态": i["status"], "结论": labels.get(i["label"], i["label"]),
                           "分数": None if i["score"] is None else round(i["score"], 4)} for i in listing["items"]])
    selection = st.dataframe(frame, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key=f"hist_table_{page}")
    nav = st.columns([1, 2, 1])
    nav[0].button("上一页", icon=":material/chevron_left:", disabled=page <= 1, width="stretch",
                  on_click=lambda: S.update(hist_page=page - 1))
    nav[1].markdown(f"<div class='wb-muted' style='text-align:center'>第 {page} / {pages} 页 · 共 {total} 条</div>",
                    unsafe_allow_html=True)
    nav[2].button("下一页", icon=":material/chevron_right:", disabled=page >= pages, width="stretch",
                  on_click=lambda: S.update(hist_page=page + 1))
    rows = selection.selection.rows if selection else []
    if not rows:
        st.caption("选择一行查看快照详情与导出。")
        return
    item = listing["items"][rows[0]]
    detail, error = api("GET", f"/history/{item['analysis_id']}")
    if error:
        show_error(error)
        return
    snapshot = detail["snapshot"]
    st.divider()
    if detail["kind"] == "image":
        render_image_result(snapshot, key=f"hist-{item['analysis_id']}")
    else:
        st.caption(f"{snapshot['display_name']} · {snapshot['created_at']} · 文本快照")
        cols = st.columns(2)
        cols[0].download_button("导出 JSON", export.to_json(snapshot), file_name=f"text-{item['analysis_id']}.json",
                                icon=":material/data_object:", width="stretch")
        cols[1].download_button("导出 Markdown", export.to_markdown("text", snapshot),
                                file_name=f"text-{item['analysis_id']}.md", icon=":material/description:",
                                width="stretch")
        st.markdown(TEXT_CSS, unsafe_allow_html=True)
        render_analysis(snapshot["result"])
