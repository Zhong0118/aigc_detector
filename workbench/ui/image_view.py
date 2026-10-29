"""Render an image_result (live result or restored snapshot); no recomputation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from workbench import export
from workbench.ui.common import esc, pct, ranked_rows, score_bar

STATUS = {"ok": "完成", "partial": "部分结果"}


def card(kicker: str, body: str) -> None:
    st.markdown(f'<div class="wb-card"><div class="wb-kicker">{esc(kicker)}</div>{body}</div>', unsafe_allow_html=True)


def render_image_result(r: dict, key: str) -> None:
    d, a, t = r["detection"], r["attribution"], r["timing"]
    chips = [STATUS.get(r["status"], r["status"]), "研究模式" if r.get("mode") == "research" else "默认模式",
             f"模型包 {r['model_bundle']['bundle_id']}", f"用时 {t.get('total_ms', 0) / 1000:.1f} 秒"]
    st.markdown("".join(f'<span class="wb-chip">{esc(c)}</span>' for c in chips), unsafe_allow_html=True)
    if r["status"] == "partial":
        st.warning("部分组件推理失败，未失败的结果仍可参考；失败部分显示为不可用，不填充默认值。")

    tp = d.get("threshold_provenance") or {}
    label = "达到阈值" if d["label"] == "at_or_above_threshold" else "低于阈值"
    card("AI 生成检测 · D-OpenCLIP", (
        f'<div class="wb-score">{d["score"]:.4f}</div>'
        f'<div class="wb-muted">模型检测分数（不是校准概率）</div>{score_bar(d["score"], d["threshold"])}'
        f'<div class="wb-line">固定阈值 {d["threshold"]} · <b>{label}</b></div>'
        f'<div class="wb-muted">阈值依据：evaluation {tp.get("n")} 张（{tp.get("negative_n")} 真实 / '
        f'{tp.get("positive_n")} AI），FPR {pct(tp.get("fpr"))}，TPR {pct(tp.get("tpr"))}；单 seed 开发协议。</div>'))

    if a["status"] == "ok":
        card("来源候选 · A0 ResNet50 · 22 类闭集",
             f'<div class="wb-muted">{esc(a["conditional_statement"])}</div>{ranked_rows(a["top_k"], 3)}')
        with st.expander("查看全部 22 类闭集分数"):
            st.dataframe(pd.DataFrame([dict(排名=x["rank"], 来源=x["label"], 闭集分数=round(x["score"], 6))
                                       for x in a["top_k"]]), hide_index=True, width="stretch")
    else:
        reason = {"detection_below_threshold": "检测分数低于阈值，主结论不给出来源候选。",
                  "inference_failed": "来源模型推理失败，来源候选不可用。"}.get(a.get("reason"), "来源候选不可用。")
        card("来源候选 · A0 ResNet50 · 22 类闭集", f'<div class="wb-line">{esc(reason)}</div>')
    card("未知来源", '<div class="wb-line">不可用</div><div class="wb-muted">主 22 类接口没有已验收的未知拒识，'
                     '不从 22 类分数推导“未知”。open20 拒识只在实验对照页展示登记结果。</div>')

    for item in r.get("research") or []:
        render_research_item(item)

    i = r["input"]
    with st.expander("输入、预处理与模型包"):
        st.markdown(
            f"- SHA-256：`{i['sha256']}`\n- 类型：{i['media_type']}，{i['width']}×{i['height']}，{i['bytes']:,} 字节，"
            f"像素模式 {i['mode']}\n- 透明通道：{'有，按研究预处理直接丢弃' if i['has_alpha'] else '无'}\n"
            f"- EXIF 方向：{i['exif_orientation'] or '无'}（不应用旋转，与研究预处理一致）\n"
            f"- 检测预处理：短边缩放到 224、中心裁剪、bicubic、CLIP 均值方差\n"
            f"- 来源预处理：Resize 256、CenterCrop 224、ImageNet 归一化\n- 设备：CPU；无外部上传；未检查 C2PA/水印")
        st.dataframe(pd.DataFrame([dict(组件=k, sha256=v) for k, v in r["model_bundle"]["components"].items()]),
                     hide_index=True, width="stretch")
    with st.expander("局限"):
        st.markdown("\n".join(f"- {x}" for x in r["limitations"]))
    cols = st.columns(2)
    cols[0].download_button("导出 JSON", export.to_json(r), file_name=f"image-{r['analysis_id']}.json",
                            mime="application/json", icon=":material/data_object:", key=f"{key}-json",
                            width="stretch")
    cols[1].download_button("导出 Markdown", export.to_markdown("image", r), file_name=f"image-{r['analysis_id']}.md",
                            mime="text/markdown", icon=":material/description:", key=f"{key}-md",
                            width="stretch")


def render_research_item(item: dict) -> None:
    if item["id"] == "d_openclip_calibrated_thresholds":
        st.markdown("**研究：校准阈值**（只用 calibration 真实图拟合；与固定 0.5 是不同的工作点）")
        st.dataframe(pd.DataFrame([{
            "目标 FPR": f"{row['target_fpr']:.0%}", "阈值": round(row["threshold"], 6), "本图达到": "是" if row["reached"] else "否",
            "calibration FPR（真实图数）": f"{pct(row['calibration_fpr'])}（{row['calibration_negative_n']}）",
            "evaluation FPR / TPR（样本数）": f"{pct(row['evaluation_fpr'])} / {pct(row['evaluation_tpr'])}（{row['evaluation_n']}）",
        } for row in item["rows"]]), hide_index=True, width="stretch")
        st.caption(item["note"])
    elif item["id"] == "a0_closed_set_research":
        st.markdown("**研究：A0 闭集分数（检测低于阈值，不进入主结论）**")
        st.markdown(ranked_rows(item["top_k"], 3), unsafe_allow_html=True)
        st.caption(item["note"])
    elif item["id"] == "ofa_pair_verification":
        st.markdown("**研究：OFA 配对验证**")
        if item["status"] == "ok":
            st.markdown(ranked_rows(item["scores"], 5), unsafe_allow_html=True)
        else:
            reason = {"cpu_reproduction_exceeds_registered_tolerance":
                      "CPU 复现与登记 GPU 预测误差约 5e-4，超过 OFA 登记容差 1e-4，预览中不展示分数。"}.get(
                item.get("reason"), f"不可用（{item.get('reason')}）")
            st.info(reason)
        st.caption(f"{item['note']} 参考库 {str(item.get('reference_bank_sha256') or '—')[:16]}…，"
                   f"参考清单 {str(item.get('reference_manifest_sha256') or '—')[:16]}…")
    elif item["id"] == "registered_only":
        st.caption("固定 50/50 融合、校准融合与 open20 未知拒识只展示登记结果（见“实验对照”），不在上传图片上推理。")
