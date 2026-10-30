"""Render an image_result (live result or restored snapshot); never recomputes anything."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from workbench import export
from workbench.ui.common import esc, pct, ranked_rows, score_bar

TENDENCY = {"ai": "AI 生成倾向", "real": "真实图片倾向"}


def card(kicker: str, body: str) -> None:
    st.markdown(f'<div class="wb-card"><div class="wb-kicker">{esc(kicker)}</div>{body}</div>', unsafe_allow_html=True)


def render_image_result(r: dict, key: str, snapshot: bool = False) -> None:
    d, a, rt = r["detection"], r["attribution"], r["runtime"]
    chips = ["部分成功" if r["status"] == "partial" else "完成", "研究模式" if r["mode"] == "research" else "标准模式",
             r["bundle_id"], f"{rt['device'].upper()} · {rt['precision'].upper()}",
             f"服务端用时 {rt['timing_ms']['total_ms'] / 1000:.2f} 秒"]
    if snapshot:
        chips.insert(0, "历史快照")
    st.markdown("".join(f'<span class="wb-chip{" blue" if i == 0 else ""}">{esc(c)}</span>'
                        for i, c in enumerate(chips)), unsafe_allow_html=True)
    if r["status"] == "partial":
        st.warning("部分成功：检测结果有效；失败的组件显示为不可用，没有填充默认值。", icon=":material/warning:")

    ev = d.get("threshold_evidence") or {}
    card(f"真假检测 · {d['model_id']}", (
        f'<div class="wb-verdict {d["label"]}">{TENDENCY[d["label"]]}</div>'
        f'<div class="wb-score">模型检测分数 {d["score"] * 100:.2f}%</div>{score_bar(d["score"], d["threshold"])}'
        f'<div class="wb-muted">固定阈值 {d["threshold"]}（{esc(d["threshold_id"])}）；分数不是校准概率，接近阈值时也没有经过标定的灰区。'
        f'开发 evaluation {ev.get("n")} 张上该阈值 FPR {pct(ev.get("fpr"))}、TPR {pct(ev.get("tpr"))}，'
        f'不代表新上传图片的性能。</div>'))

    kicker = f"来源候选 · {a['model_id']} · 已知 22 类"
    if a["status"] == "ok":
        card(kicker, f'<div class="wb-muted">{esc(a["conditional_statement"])}</div>{ranked_rows(a["top_k"], 3)}'
                     '<div class="wb-muted">Top3 为原 22 类 softmax 分数，未重新归一化。</div>')
        with st.expander("展开全部 22 类"):
            st.dataframe(pd.DataFrame([{"排名": t["rank"], "来源（checkpoint 类名）": t["source"],
                                        "22 类 softmax 分数": round(t["score"], 6)} for t in a["top_k"]]),
                         hide_index=True, width="stretch")
    elif a["status"] == "skipped":
        card(kicker, '<div class="wb-line">未运行：检测分数低于阈值，标准模式不做来源归因。</div>'
                     '<div class="wb-muted">这不是“已验证的未知来源”。如需查看“假设为已知 22 类 AI”时的排序，'
                     '可在研究模式选择“A0 来源排序（不经检测门控）”。</div>')
    else:
        card(kicker, '<div class="wb-line">来源模型推理失败，来源候选不可用。</div>'
                     '<div class="wb-muted">检测结果仍然有效；没有用其他模型替代。</div>')
    card("未知来源", '<div class="wb-line">不可用 · score = null</div>'
                    '<div class="wb-muted">尚无经验证的通用未知来源拒识能力；不从 22 类分数推导“未知”。</div>')

    for item in r.get("research") or []:
        render_research_item(item)

    persisted = r["persistence"]
    if persisted["status"] == "ok":
        st.caption(f"报告已保存（报告 ID {persisted['content_id']}），历史页可还原当时的版本。")
    else:
        st.warning("报告未保存（存储失败）。本次推理结果仍可在下方导出。", icon=":material/save:")

    with st.expander("技术详情：模型、预处理、设备与耗时"):
        i, t = r["input"], rt["timing_ms"]
        st.markdown(
            f"- 输入 SHA-256：`{i['sha256']}`；{i['media_type']}，{i['width']}×{i['height']}，{i['bytes']:,} 字节，"
            f"{i['color_mode']}；解码策略 `{i['decode_policy']}`\n"
            f"- 检测：`{d['model_id']}`，类别图 `{d['class_map_sha256'][:16]}…`，预处理 `{rt['preprocessing']['detection'][:16]}…`"
            "（短边 bicubic 224 → 中心裁剪 224 → CLIP 均值方差）\n"
            f"- 来源：`{a['model_id']}`，类别图 `{a['class_map_sha256'][:16]}…`，预处理 `{rt['preprocessing']['attribution'][:16]}…`"
            "（短边 bilinear 256 → 中心裁剪 224 → ImageNet 均值方差）\n"
            f"- 模型包 `{r['bundle_id']}`，registry `{r['registry_sha256'][:16]}…`，推理代码 `{r['inference_code_commit']}`\n"
            f"- 设备 {rt['device']}，{rt['precision']}，TF32 关闭；外部调用：无\n"
            f"- 服务端耗时（毫秒）：解码 {t.get('decode_ms')} · 排队 {t.get('queue_ms')} · 检测 {t.get('detection_ms')} · "
            f"来源 {t.get('attribution_ms') if t.get('attribution_ms') is not None else '未运行'} · "
            f"研究 {t.get('research_ms') if t.get('research_ms') is not None else '未运行'} · 合计 {t.get('total_ms')}")
    with st.expander("局限"):
        st.markdown("\n".join(f"- {x}" for x in r["limitations"]))
    cols = st.columns(2)
    name = f"image-{r['request_id']}"
    cols[0].download_button("导出 JSON", export.to_json(r), file_name=f"{name}.json", mime="application/json",
                            icon=":material/data_object:", key=f"{key}-json", width="stretch")
    cols[1].download_button("导出 Markdown", export.image_markdown(r), file_name=f"{name}.md", mime="text/markdown",
                            icon=":material/description:", key=f"{key}-md", width="stretch")


def render_research_item(item: dict) -> None:
    out = item.get("outputs") or {}
    st.markdown(f"**研究对照 · {esc(item['label'])}**  `{item['method_id']}`")
    st.caption("独立附在标准结果之后，标签空间与分数语义各自独立，不改写上面的标准结论。")
    if item["status"] != "ok":
        st.info("该研究方法本次推理失败，未产生输出。", icon=":material/info:")
    elif out.get("thresholds"):
        st.dataframe(pd.DataFrame([{
            "阈值 ID": t["threshold_id"], "阈值": round(t["threshold"], 6), "本图读数": TENDENCY[t["label"]],
            "calibration FPR（真实图数）": f"{pct(t['calibration_fpr'])}（{t['calibration_real_n']}）",
            "evaluation FPR / TPR（样本数）": f"{pct(t['evaluation_fpr'])} / {pct(t['evaluation_tpr'])}（{t['evaluation_n']}）",
        } for t in out["thresholds"]]), hide_index=True, width="stretch")
    elif out.get("top_k"):
        st.markdown(f'<div class="wb-card"><div class="wb-muted">{esc(out["conditional_statement"])}</div>'
                    f'{ranked_rows(out["top_k"], 3)}</div>', unsafe_allow_html=True)
    if item.get("limitations"):
        st.markdown("\n".join(f"- {x}" for x in item["limitations"]))
