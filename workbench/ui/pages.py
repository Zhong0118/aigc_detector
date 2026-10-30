"""Four workbench pages. Results and in-flight jobs live in session_state, so navigation,
widget reruns and refreshes never trigger another inference."""
from __future__ import annotations

import hashlib
import uuid

import pandas as pd
import streamlit as st

from api.text_view import TEXT_CSS, render_analysis, run_with_elapsed
from workbench import export
from workbench.ui.common import (ROLE_NAMES, api, capabilities, esc, header, pct, persist, show_error, start_job,
                                 wait_job)
from workbench.ui.image_view import render_image_result

S = st.session_state
TEXT_TYPES = ["txt", "md", "csv", "json", "html", "doc", "docx"]
IMAGE_TYPES = ["jpg", "jpeg", "png"]
DEFAULT_MAX_BYTES = 10 * 1024 * 1024


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
            S["txt_error"] = dict(code="unsupported_media_type", message="这是图片文件，请在“图片检测”页分析",
                                  retryable=False)
        else:
            S["txt_result"] = result
        st.rerun()
    if S.get("txt_error"):
        show_error(S["txt_error"])
    if S.get("txt_result"):
        render_analysis(S["txt_result"])


# ---------------------------------------------------------------- image
def _forget_image_state(keep_file: bool) -> None:
    for key in ("img_current", "img_error", "img_pending") + (() if keep_file else ("img_file",)):
        S.pop(key, None)


def _on_image_upload() -> None:
    upload = S.get(f"img_uploader_{S.get('img_nonce', 0)}")
    _forget_image_state(keep_file=False)  # a new file invalidates the old result and any in-flight request
    if upload is not None:
        data = upload.getvalue()
        S["img_file"] = dict(name=upload.name, data=data, sha256=hashlib.sha256(data).hexdigest())


def _clear_image() -> None:
    S["img_nonce"] = S.get("img_nonce", 0) + 1
    _forget_image_state(keep_file=False)
    S["img_results"], S["img_request_ids"] = {}, {}  # drop this session's upload-derived caches only


def _params() -> tuple[str, str | None]:
    mode = "research" if S.get("img_mode_saved") == "研究模式" else "standard"
    return mode, (S.get("img_method_saved") if mode == "research" else None)


def _key() -> tuple | None:
    file = S.get("img_file")
    return (file["sha256"], *_params()) if file else None


def _submit_image() -> None:
    key, file = _key(), S["img_file"]
    S.pop("img_error", None)
    if key in S.setdefault("img_results", {}):
        S["img_current"] = key  # same image + parameters: reuse, never re-infer
        return
    request_id = S.setdefault("img_request_ids", {}).get(key) or str(uuid.uuid4())
    S["img_request_ids"][key] = request_id  # bound to input SHA + parameters
    data = dict(mode=key[1], request_id=request_id)
    if key[2]:
        data["research_method_id"] = key[2]
    S["img_pending"] = dict(key=key, request_id=request_id, job=start_job(lambda: api(
        "POST", "/analyze/file", timeout=180, files={"file": (file["name"], file["data"])}, data=data)))


def _finish_pending() -> None:
    pending = S["img_pending"]
    result, error = wait_job(pending["job"], "已提交，等待服务器分析")
    if S.get("img_pending") is not pending:  # superseded while waiting: drop the late response
        return
    S.pop("img_pending", None)
    key = pending["key"]
    if error:
        S["img_error"] = dict(key=key, error=error)
        if not error.get("retryable"):
            S["img_request_ids"].pop(key, None)
        return
    image_result = result["image_result"]
    if image_result["input"]["sha256"] != key[0] or image_result["request_id"] != pending["request_id"]:
        S["img_error"] = dict(key=key, error=dict(code="request_id_conflict", message="返回结果与当前图片不匹配，已丢弃",
                                                  retryable=False, request_id=pending["request_id"]))
        return
    S["img_results"][key] = image_result
    S["img_request_ids"].pop(key, None)
    if _key() == key:
        S["img_current"] = key


def _research_controls(caps: dict | None, pending: bool) -> None:
    image = (caps or {}).get("image") or {}
    st.segmented_control("模式", ["标准模式", "研究模式"], default=S.get("img_mode_saved", "标准模式"), key="img_mode",
                         on_change=persist("img_mode"), disabled=pending)
    if S.get("img_mode_saved") != "研究模式":
        return
    runnable = [m for m in image.get("research_methods", []) if m["run_allowed"]]
    labels = {m["id"]: m["label"] for m in runnable}
    if not runnable:
        st.info("当前没有通过验收、可在线运行的研究方法。", icon=":material/info:")
        S.pop("img_method_saved", None)
    else:
        if S.get("img_method_saved") not in labels:
            S["img_method_saved"] = runnable[0]["id"]
        st.selectbox("研究方法（一次只附加一个）", list(labels), format_func=labels.get, key="img_method",
                     index=list(labels).index(S["img_method_saved"]), on_change=persist("img_method"), disabled=pending)
        spec = next(m for m in runnable if m["id"] == S["img_method_saved"])
        st.caption(f"{spec['note']}  标签空间 {spec['label_space']} · 分数 {spec['score_kind']}")
    offline = image.get("offline_only_methods", [])
    unavailable = [m for m in image.get("research_methods", []) if not m["run_allowed"]]
    with st.expander(f"其他研究方法：{len(offline) + len(unavailable)} 个只有离线结果，在线未接入"):
        st.dataframe(pd.DataFrame(
            [{"方法": m["label"], "任务": m["task"], "离线实验": "已完成", "在线": m["online_integration"]} for m in unavailable]
            + [{"方法": m["label"], "任务": m["task"], "离线实验": "已完成", "在线": "未接入（仅可查看实验对照）"}
               for m in offline]), hide_index=True, width="stretch")


def image_page() -> None:
    caps = header("图片检测", "默认：D-OpenCLIP 真假检测（固定阈值 0.5）；达到阈值时 A0 ResNet50 给出已知 22 类来源候选。"
                              "未知来源在标准模式不可用。")
    image_caps = (caps or {}).get("image") or {}
    ready = image_caps.get("state") == "ready"
    policy = image_caps.get("input_policy") or {}
    max_bytes = int(policy.get("max_bytes", DEFAULT_MAX_BYTES))
    pending = bool(S.get("img_pending"))  # the job keeps running across reruns and page switches
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.file_uploader(f"选择图片（静态 8-bit JPEG/PNG，RGB 或灰度，≤{max_bytes // (1024 * 1024)} MiB，"
                         f"≤{policy.get('max_pixels', 16_000_000):,} 像素）", type=IMAGE_TYPES,
                         key=f"img_uploader_{S.get('img_nonce', 0)}", on_change=_on_image_upload, disabled=pending)
        file = S.get("img_file")
        too_big = bool(file and len(file["data"]) > max_bytes)
        if file:
            try:
                st.image(file["data"], width="stretch")
            except Exception:  # a broken file still goes to the server for typed validation
                st.caption("浏览器端无法预览，提交后由服务器校验。")
            st.caption(f"{esc(file['name'])} · {len(file['data']):,} 字节 · SHA-256 {file['sha256'][:12]}… · "
                       "预览为浏览器解码；模型随后按各自预处理缩放并中心裁剪，不自动旋转、不叠底透明。")
            if too_big:
                st.warning(f"文件超过 {max_bytes // (1024 * 1024)} MiB，请压缩或缩小后再上传。", icon=":material/warning:")
        _research_controls(caps, pending)
        cols = st.columns([3, 2])
        cols[0].button("分析中…" if pending else "开始分析", type="primary", icon=":material/play_arrow:",
                       disabled=pending or not file or not ready or too_big, on_click=_submit_image, width="stretch")
        cols[1].button("清空", icon=":material/close:", disabled=pending or not file, on_click=_clear_image,
                       width="stretch")
        if caps and not ready:
            st.info({"loading": "图片模型正在加载，完成后即可分析。"}.get(
                image_caps.get("state"), "图片模型不可用，分析按钮已禁用；不会使用演示或占位分数。"), icon=":material/info:")
        st.caption("原图只用于本次分析，不写入数据库；历史只保存结果快照与 SHA-256。")
    with right:
        if pending:
            _finish_pending()
            st.rerun()
        key = _key()
        failure = S.get("img_error")
        if failure and failure["key"] == key:
            show_error(failure["error"])
            if failure["error"].get("code") in ("timeout", "busy", "backend_unreachable") and key in S.get(
                    "img_request_ids", {}):
                st.button("重新查询（同一请求 ID）", icon=":material/refresh:", on_click=_submit_image)
        if key and key in S.get("img_results", {}):
            render_image_result(S["img_results"][key], key="img-live")
        elif not (failure and failure["key"] == key):
            st.markdown('<div class="wb-card empty"><div class="wb-kicker">结果</div><div class="wb-muted">'
                        + ("已选择图片，点击“开始分析”。只在显式提交时推理。" if file else
                           "上传图片后点击“开始分析”。结果包含真假倾向、模型检测分数、条件来源候选和局限说明。")
                        + "</div></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- experiments
@st.cache_data(ttl=60, show_spinner=False)
def _catalog():
    return api("GET", "/experiments", timeout=30)


METRIC_NAMES = {"auroc": "AUROC", "fpr_0_5": "FPR@0.5", "tpr_0_5": "TPR@0.5", "top1": "Top1", "top3": "Top3",
                "macro_f1": "Macro-F1", "accuracy": "Accuracy", "fpr": "FPR", "tpr": "TPR",
                "known_retention": "已知保留", "known_correct_and_retained": "正确且保留",
                "unknown_false_acceptance": "未知误接收", "unknown_rejection": "未知拒绝", "unknown_auroc": "未知 AUROC"}


def _ci(ci) -> str:
    return "—" if not ci else f"[{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"


def _paired_table(paired: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"配对比较（同一批样本）": p["comparison"], "指标": METRIC_NAMES.get(m, m),
                          "差值（百分点）": "—" if v["delta"] is None else f"{v['delta'] * 100:.2f}",
                          "95% 区间（百分点）": _ci(v["ci95"]),
                          "样本 / group / 重复": f"{p['n']} / {p['group_n']} / {p['repeats']}"}
                         for p in paired for m, v in p["deltas"].items()])


def _count(k, n, rate) -> str:
    """Counts only when the sealed overview registered them; never back-derived from a rate."""
    return f"{k}/{n}（{pct(rate)}）" if k is not None and n is not None else f"计数未登记（{pct(rate)}）"


def _detection_table(rows: list[dict], calibrated: bool = False) -> pd.DataFrame:
    out = []
    for r in rows:
        row = {"方法": r["method"]}
        if r.get("role"):
            row["角色"] = ROLE_NAMES.get(r["role"], r["role"])
        if calibrated:
            row["阈值（cal 目标）"] = f"{r['threshold']:.4f}（{r['target_fpr']:.0%}，cal 实际 {pct(r['calibration_fpr'])}）"
        row.update({"n": r["n"], "AUROC": pct(r["auroc"]),
                    "误报 / 真实": _count(r["fp"], r["real_n"], r["fpr"]),
                    "检出 / AI": _count(r["tp"], r["ai_n"], r["tpr"]),
                    "FPR Wilson 95%": _ci(r.get("fpr_wilson95"))})
        out.append(row)
    return pd.DataFrame(out)


def _source_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "方法": r["method"], **({"角色": ROLE_NAMES.get(r["role"], r["role"])} if r.get("role") else {}),
        "n": r["n"], "Top1": pct(r["top1"]) + (f"（{r['top1_correct']}/{r['n']}）" if r.get("top1_correct") is not None else ""),
        "Top3": pct(r["top3"]), "Macro-F1": pct(r.get("macro_f1")),
        "训练/选模 epoch": "—" if r.get("training_epochs") is None and r.get("selected_epoch") is None
        else f"{r.get('training_epochs') or 'null'} / {r.get('selected_epoch') or 'null'}"} for r in rows])


def _unknown_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "方法": r["method"], "已知保留": f"{r['known_retention_count']}/{r['known_n']}（{pct(r['known_retention'])}）",
        "正确且保留": f"{r['known_correct_and_retained_count']}/{r['known_n']}（{pct(r['known_correct_and_retained'])}）",
        "未知误接收": f"{r['unknown_false_acceptance_count']}/{r['unknown_n']}（{pct(r['unknown_false_acceptance'])}）",
        "误接收 Wilson 95%": _ci(r["unknown_false_acceptance_wilson95"]),
        "未知拒绝": f"{r['unknown_rejection_count']}/{r['unknown_n']}", "未知 AUROC": pct(r["unknown_auroc"]),
        "OSCR": pct(r.get("oscr"))} for r in rows])


def _summary_table(rows: list[dict]) -> pd.DataFrame:
    names = {"method": "方法", "n": "n", "auroc": "AUROC", "fpr": "FPR@0.5", "tpr": "TPR@0.5", "top1": "Top1",
             "top3": "Top3", "macro_auroc": "宏 AUROC", "selected_epoch": "选模 epoch", "train_n": "训练 n"}
    return pd.DataFrame([{names[k]: (pct(v) if k in ("auroc", "fpr", "tpr", "top1", "top3", "macro_auroc") else
                                     ("null" if v is None else v)) for k, v in r.items() if k in names} for r in rows])


def _group(group: dict) -> None:
    st.markdown(f"**{group['title']}**")
    table = {"detection": _detection_table, "source": _source_table, "unknown": _unknown_table,
             "summary": _summary_table}[group["kind"]]
    st.dataframe(table(group["rows"]), hide_index=True, width="stretch")
    if group.get("calibrated_rows"):
        st.markdown("按非 FFHQ calibration 真实图选择的阈值（只报告 FFHQ 结果，不是跨域保证）")
        st.dataframe(_detection_table(group["calibrated_rows"], calibrated=True), hide_index=True, width="stretch")
    if group.get("threshold_refit"):
        with st.expander("阈值重拟合 group bootstrap（1000 次）"):
            st.dataframe(pd.DataFrame([{"方法": r["method"], "cal 目标": f"{r['target_fpr']:.0%}",
                                        "阈值 95%": f"[{r['threshold_ci95'][0]:.4f}, {r['threshold_ci95'][1]:.4f}]",
                                        "FFHQ FPR 95%": _ci(r["evaluation_fpr_ci95"]),
                                        "FFHQ TPR 95%": _ci(r["evaluation_tpr_ci95"])} for r in group["threshold_refit"]]),
                         hide_index=True, width="stretch")
    if group["paired"]:
        st.dataframe(_paired_table(group["paired"]), hide_index=True, width="stretch")
    for note in group.get("notes", []):
        st.caption(note)


def _method_status(catalog: dict, caps: dict | None) -> pd.DataFrame:
    image = (caps or {}).get("image") or {}
    online = {m["model_id"]: m["online_integration"] for m in image.get("defaults", [])}
    online.update({m["id"]: m["online_integration"] for m in image.get("research_methods", [])})
    names = {"ready": "可用", "loading": "加载中", "unavailable": "不可用", "unvalidated": "未通过验收",
             "artifact_mismatch": "校验失败", "disabled": "未配置"}
    rows = [{"方法 ID": m["id"], "任务": m["task"], "暴露": "标准" if m["exposure"] == "standard" else "仅研究",
             "离线实验": "已完成" if m["offline_validation"] == "completed" else m["offline_validation"],
             "在线接入": names.get(online.get(m["id"]), online.get(m["id"])) if m["id"] in online else "未接入（已有离线结果）"}
            for m in catalog["methods"]]
    rows += [{"方法 ID": m["id"], "任务": m["task"], "暴露": "研究", "离线实验": "—",
              "在线接入": "可用（复用已验收默认模型）" if m["run_allowed"] else names.get(m["online_integration"])}
             for m in image.get("research_methods", []) if m["id"] not in {r["方法 ID"] for r in rows}]
    rows += [{"方法 ID": n, "任务": "—", "暴露": "不提供", "离线实验": "未核验/未运行/未训练", "在线接入": "不提供入口"}
             for n in catalog["not_exposed"]]
    return pd.DataFrame(rows)


def experiments_page() -> None:
    caps = header("实验对照", "只读展示封口的聚合实验结果（哈希校验）；不同分母不比较“提升”，只有同一批样本的配对报告给出差值与区间。")
    catalog, error = _catalog()
    if error:
        show_error(error)
        return
    for problem in catalog["errors"]:
        st.error(f"登记结果 {problem['source']} 不可用：{problem['message']}", icon=":material/error:")
    sc = catalog.get("scope") or {}
    if sc:
        st.caption(f"IAB v7-200-q179 · 有效 {sc['effective_images']:,} 张（AI {sc['ai']:,} / 真实 {sc['real']:,}）· "
                   f"train {sc['train']:,} / dev {sc['dev']:,} / holdout {sc['holdout']:,}（未推理）· 每格实际 "
                   f"{sc['actual_generated_per_cell'][0]}–{sc['actual_generated_per_cell'][1]} 张 · 隔离 "
                   f"{sc['quarantined_new_members']} 张 · seed {sc['seed']}")
    with st.container(border=True):
        st.markdown("**必须同时看到的限制**")
        st.markdown("\n".join(f"- {x}" for x in catalog["key_facts"]))
    groups = {g["id"]: g for g in catalog["groups"]}
    tabs = st.tabs(["真假检测", "22 来源分类", "未知拒识", "OFA 配对验证", "方法与接入状态", "局限与数据来源"])
    for tab, ids in zip(tabs[:4], (("detection-clean", "detection-ffhq", "detection-extensions", "detection-full-dev"),
                                   ("source-1331", "source-robustness", "source-full-dev"),
                                   ("unknown-1275",), ("ofa-pair",))):
        with tab:
            for gid in ids:
                if gid in groups:
                    _group(groups[gid])
    with tabs[4]:
        st.dataframe(_method_status(catalog, caps), hide_index=True, width="stretch")
        st.caption("离线实验完成不等于在线服务就绪；只有通过本设备数值验收的方法才在图片页出现运行入口。")
    with tabs[5]:
        st.markdown("\n".join(f"- {x}" for x in catalog["limitations"]))
        st.dataframe(pd.DataFrame([{"来源": k, "路径（研究树相对）": v["path"], "sha256": v["sha256"]}
                                   for k, v in catalog["sources"].items()]), hide_index=True, width="stretch")


# ---------------------------------------------------------------- history
PAGE_SIZE = 15


def _reset_page() -> None:
    S["hist_page"] = 0


def history_page() -> None:
    header("历史报告", "还原当时保存的结果快照（模型包、阈值、预处理与报告），不重新推理、不保存原图。")
    cols = st.columns([1.2, 1.2, 2.6, 1])
    kinds = {"全部": None, "图片": "image", "文本": "text"}
    statuses = {"全部": None, "完成": "ok", "部分成功": "partial", "旧记录（无快照）": "legacy"}
    kind = cols[0].selectbox("类型", list(kinds), key="hist_kind", on_change=_reset_page)
    status = cols[1].selectbox("状态", list(statuses), key="hist_status", on_change=_reset_page)
    query = cols[2].text_input("文件名或 SHA-256", key="hist_query", on_change=_reset_page)
    cols[3].button("刷新", icon=":material/refresh:", width="stretch")
    page = S.setdefault("hist_page", 0)
    params = {k: v for k, v in dict(limit=PAGE_SIZE, offset=page * PAGE_SIZE, modality=kinds[kind],
                                    status=statuses[status], q=query.strip() or None).items() if v is not None}
    listing, error = api("GET", "/reports", params=params)
    if error:
        show_error(error)
        return
    total, pages = listing["total"], max(1, -(-listing["total"] // PAGE_SIZE))
    if not listing["items"]:
        st.info("暂无记录。", icon=":material/info:")
        return
    tendency = {"ai": "AI 倾向", "human": "真实/未明显检出", **export.TEXT_LABELS}
    frame = pd.DataFrame([{"报告 ID": i["content_id"], "时间（UTC）": i["created_at"],
                           "类型": {"image": "图片", "text": "文本"}.get(i["modality"], i["modality"]),
                           "名称": i["filename"], "状态": {"ok": "完成", "partial": "部分成功", "legacy": "旧记录"}[i["status"]],
                           "结论": tendency.get(i["label"], i["label"]),
                           "分数": None if i["score"] is None else round(i["score"], 4),
                           "模型包": i["bundle_id"] or "—"} for i in listing["items"]])
    selection = st.dataframe(frame, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key=f"hist_table_{page}")
    nav = st.columns([1, 2, 1])
    nav[0].button("上一页", icon=":material/chevron_left:", disabled=page <= 0, width="stretch",
                  on_click=lambda: S.update(hist_page=page - 1))
    nav[1].markdown(f"<div class='wb-muted' style='text-align:center'>第 {page + 1} / {pages} 页 · 共 {total} 条</div>",
                    unsafe_allow_html=True)
    nav[2].button("下一页", icon=":material/chevron_right:", disabled=page + 1 >= pages, width="stretch",
                  on_click=lambda: S.update(hist_page=page + 1))
    rows = selection.selection.rows if selection else []
    if not rows:
        st.caption("选择一行查看当时的完整结果与导出。")
        return
    item = listing["items"][rows[0]]
    detail, error = api("GET", f"/report/{item['content_id']}")
    if error:
        show_error(error)
        return
    st.divider()
    if detail.get("image_result"):
        render_image_result(detail["image_result"], key=f"hist-{item['content_id']}", snapshot=True)
        st.caption("快照记录分析当时的模型包、registry、阈值和预处理；之后更换默认模型不会改写或重算它。")
    elif (detail.get("report_snapshot") or {}).get("analysis"):
        analysis = detail["report_snapshot"]["analysis"]
        cols = st.columns(2)
        cols[0].download_button("导出 JSON", export.to_json(detail["report_snapshot"]),
                                file_name=f"text-{item['content_id']}.json", icon=":material/data_object:", width="stretch")
        cols[1].download_button("导出 Markdown", export.text_markdown(analysis, item),
                                file_name=f"text-{item['content_id']}.md", icon=":material/description:", width="stretch")
        st.markdown(TEXT_CSS, unsafe_allow_html=True)
        render_analysis(analysis)
    else:
        st.info("旧记录：没有分析当时的版本快照，只显示已保存字段，不补造模型版本。", icon=":material/history:")
        st.download_button("导出 JSON", export.to_json(detail), file_name=f"report-{item['content_id']}.json",
                           icon=":material/data_object:")
        st.json(detail["detection_results"], expanded=False)


__all__ = ["text_page", "image_page", "experiments_page", "history_page", "capabilities"]
