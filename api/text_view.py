"""文本检测结果渲染（旧 Dashboard 与预览工作台共用）。

从 api/dashboard.py 原样抽出，保持文本流程展示不变；新增 run_with_elapsed，
等待期间只显示真实已等待时间，不再展示虚构的阶段百分比。
"""

import html
import threading
import time

import plotly.graph_objects as go
import streamlit as st

TEXT_CSS = """
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
    .segment-card {
        border-radius: 10px;
        padding: 0.75rem 0.95rem;
        margin-bottom: 0.55rem;
        border: 1px solid #e2e8f0;
    }
    .segment-ai { background: #fef2f2; border-color: #fecaca; }
    .segment-mid { background: #fffbeb; border-color: #fde68a; }
    .segment-human { background: #eff6ff; border-color: #bfdbfe; }
    .segment-title { font-weight: 650; margin-bottom: 0.2rem; }
    .segment-meta { color: #64748b; font-size: 0.86rem; margin-bottom: 0.35rem; }
    .segment-text { color: #0f172a; line-height: 1.55; font-size: 0.95rem; }
    </style>
    """


def run_with_elapsed(fn, label: str = "正在等待服务器返回结果"):
    """在后台线程执行 fn（不得调用 st.*），主线程每 0.25 秒刷新真实已等待时间。

    返回 (fn 的返回值, 用时秒数)。超时由 fn 内部的 HTTP timeout 决定，这里不自动重试。
    """
    box = st.empty()
    holder = {}

    def work():
        try:
            holder["value"] = fn()
        except BaseException as exc:  # 交回主线程处理
            holder["error"] = exc

    start = time.monotonic()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        box.info(f"{label}… 已等待 {time.monotonic() - start:.1f} 秒（后端不报告内部进度）")
        worker.join(0.25)
    box.empty()
    if "error" in holder:
        raise holder["error"]
    return holder["value"], time.monotonic() - start


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


def collect_segments(data: dict) -> list[dict]:
    """从检测结论或本地文本模型详情里取出分块结果。"""
    detection = data.get("detection") or {}
    decision = detection.get("decision") or {}
    segments = list(decision.get("segments") or [])
    if segments:
        return segments
    for provider in detection.get("providers") or []:
        details = provider.get("details") or {}
        raw = details.get("segments") or []
        if isinstance(raw, list) and raw:
            return raw
    return []


def segment_verdict(score: float, low_threshold: float, high_threshold: float) -> tuple[str, str]:
    """把单块分数映射成直观标签和卡片样式。"""
    if score > high_threshold:
        return "疑似 AI", "segment-ai"
    if score > low_threshold:
        return "不确定", "segment-mid"
    return "更像人工", "segment-human"


def summarize_segments(segments: list[dict], low_threshold: float, high_threshold: float) -> dict:
    """统计整篇：总块数、AI 块数、字符加权 AI 率。"""
    total = len(segments)
    ai_n = 0
    mid_n = 0
    human_n = 0
    weighted = 0.0
    weight = 0
    for item in segments:
        score = float(item.get("score") or 0.0)
        start = int(item.get("start") or 0)
        end = int(item.get("end") or start)
        chars = max(1, end - start)
        if "text" in item and item["text"]:
            chars = max(chars, len(str(item["text"])))
        weighted += score * chars
        weight += chars
        verdict, _ = segment_verdict(score, low_threshold, high_threshold)
        if verdict == "疑似 AI":
            ai_n += 1
        elif verdict == "不确定":
            mid_n += 1
        else:
            human_n += 1
    return {
        "total": total,
        "ai": ai_n,
        "uncertain": mid_n,
        "human": human_n,
        "ai_block_rate": (ai_n / total) if total else 0.0,
        "weighted_ai_rate": (weighted / weight) if weight else 0.0,
    }


def render_segment_panel(segments: list[dict], low_threshold: float, high_threshold: float) -> None:
    """展示整篇计数，以及每一块的 AI 率与原文摘要。"""
    stats = summarize_segments(segments, low_threshold, high_threshold)
    st.subheader("分块 AI 率")
    cols = st.columns(5)
    cols[0].metric("总块数", stats["total"])
    cols[1].metric("疑似 AI", stats["ai"])
    cols[2].metric("不确定", stats["uncertain"])
    cols[3].metric("更像人工", stats["human"])
    cols[4].metric("AI 块占比", f"{stats['ai_block_rate']:.1%}")
    st.caption(
        f"字符加权 AI 率 {stats['weighted_ai_rate']:.1%}。"
        f"单块超过高风险阈值 {high_threshold:.0%} 计为疑似 AI，"
        f"低于低风险阈值 {low_threshold:.0%} 计为更像人工。"
    )

    if segments:
        st.markdown("**每一块**")
        for idx, item in enumerate(segments):
            score = float(item.get("score") or 0.0)
            verdict, css = segment_verdict(score, low_threshold, high_threshold)
            start = item.get("start", 0)
            end = item.get("end", 0)
            text = str(item.get("text") or "").strip()
            preview = text if len(text) <= 220 else text[:220] + "…"
            block_no = int(item.get("index", idx)) + 1
            st.markdown(
                f"""
                <div class="segment-card {css}">
                  <div class="segment-title">第 {block_no} 块 · {verdict} · AI率 {score:.1%}</div>
                  <div class="segment-meta">字符 {start}-{end} · {max(0, int(end) - int(start))} 字</div>
                  <div class="segment-text">{html.escape(preview)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_score_gauge(score: float, low_threshold: float = 0.4, high_threshold: float = 0.8):
    """渲染 AI 特征风险分数仪表盘。

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
                    {"range": [0, low_threshold], "color": "#dbeafe"},
                    {"range": [low_threshold, high_threshold], "color": "#fde68a"},
                    {"range": [high_threshold, 1], "color": "#fecaca"},
                ],
                "threshold": {"line": {"color": "#991b1b", "width": 3}, "value": high_threshold},
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
    decision = detection.get("decision") or {}
    provenance = data["provenance"]
    report = data["report"]
    low_threshold = float(decision.get("low_threshold") or 0.4)
    high_threshold = float(decision.get("high_threshold") or detection.get("threshold") or 0.8)
    segments = collect_segments(data)
    segment_stats = summarize_segments(segments, low_threshold, high_threshold) if segments else None

    # 顶部 metric 用来快速回答：分数、结论、分块 AI 计数。
    label_names = {
        "low_risk": "未明显检出",
        "uncertain": "不确定",
        "high_risk_ai": "高风险疑似 AI",
        "insufficient_evidence": "证据不足",
        "human": "未明显检出",
        "ai": "疑似 AI",
    }
    domain_names = {
        "literature": "经典文学",
        "academic_technical": "论文/技术",
        "official_news": "公文/新闻",
        "general_qa": "普通写作/问答",
    }
    if segment_stats:
        top = st.columns([1.1, 1.1, 0.9, 0.9, 0.9])
        top[0].metric("整篇加权 AI 率", f"{segment_stats['weighted_ai_rate']:.1%}")
        top[1].metric("检测结论", label_names.get(detection["label"], detection["label"]))
        top[2].metric("总块数", segment_stats["total"])
        top[3].metric("疑似 AI 块", f"{segment_stats['ai']} / {segment_stats['total']}")
        top[4].metric("AI 块占比", f"{segment_stats['ai_block_rate']:.1%}")
        st.caption(
            f"文本类型：{domain_names.get(decision.get('domain'), data['modality'].upper())}。"
            f"单块超过 {high_threshold:.0%} 计为疑似 AI；"
            f"整篇加权 AI 率按字符长度加权，不是简单块数平均。"
        )
    else:
        top = st.columns([1.15, 1.15, 1, 0.8])
        top[0].metric("AI 特征风险分数", f"{detection['score']:.2%}")
        top[1].metric("检测结论", label_names.get(detection["label"], detection["label"]))
        top[2].metric("文本类型", domain_names.get(decision.get("domain"), data["modality"].upper()))
        top[3].metric("记录 ID", data["content_id"])
        st.caption(
            "该数值是检测模块输出并用于阈值判定的风险分数，"
            "不是内容由 AI 生成的概率。"
        )

    if detection["label"] == "uncertain":
        st.warning("当前分数位于不确定区间，不能直接判为 AI。")
    elif detection["label"] == "insufficient_evidence":
        st.info("当前没有足够的有效模型证据，系统没有把 0 分解释为人类文本。")

    left, right = st.columns([0.92, 1.08])
    with left:
        # 左侧主要展示检测层证据。
        render_score_gauge(detection["score"], low_threshold, high_threshold)
        if decision:
            threshold_cols = st.columns(3)
            threshold_cols[0].metric("低风险阈值", f"{low_threshold:.2%}")
            threshold_cols[1].metric("高风险阈值", f"{high_threshold:.2%}")
            threshold_cols[2].metric("检测覆盖", f"{float(decision.get('analyzed_coverage') or 0.0):.2%}")
            st.caption(
                f"阈值组：{decision.get('threshold_group', 'unknown')} | "
                f"校准版本：{decision.get('artifact_version', 'built-in')}"
            )
            validation_samples = int(decision.get("threshold_validation_samples") or 0)
            validation_fpr = decision.get("threshold_validation_fpr")
            ai_validation_samples = int(
                decision.get("threshold_ai_validation_samples") or 0
            )
            ai_recall = decision.get("threshold_ai_recall")
            validation_auroc = decision.get("threshold_validation_auroc")
            if (
                validation_samples
                or ai_validation_samples
                or validation_auroc is not None
            ):
                evidence_cols = st.columns(3)
                evidence_cols[0].metric(
                    "真人误报率",
                    "n/a"
                    if validation_fpr is None
                    else f"{float(validation_fpr):.2%}",
                    help=f"独立真人验证样本数：{validation_samples}",
                )
                evidence_cols[1].metric(
                    "AI 召回率",
                    "n/a" if ai_recall is None else f"{float(ai_recall):.2%}",
                    help=f"独立 AI 验证样本数：{ai_validation_samples}",
                )
                evidence_cols[2].metric(
                    "验证 AUROC",
                    "n/a"
                    if validation_auroc is None
                    else f"{float(validation_auroc):.3f}",
                )
                st.caption(
                    f"真人验证样本：{validation_samples} | "
                    f"AI 验证样本：{ai_validation_samples}"
                )
            elif str(decision.get("threshold_group") or "").startswith("built-in:"):
                st.caption("当前使用内置保守阈值，没有对应的独立验证集指标。")
            if decision.get("threshold_warnings"):
                warning_names = {
                    "calibrated_threshold_near_score_ceiling": "校准阈值接近模型分数上限",
                    "calibrated_high_threshold_replaced_by_floor": "运行高阈值已回落到有效的产品阈值",
                    "runtime_high_threshold_capped_to_floor": "阈值产物中的运行高阈值曾被调整",
                    "effective_high_threshold_not_evaluated": "当前运行阈值未完成独立验证，暂不报告误报率和召回率",
                    "holdout_fpr_above_target": "真人验证集误报率显著高于目标",
                    "insufficient_human_validation": "真人验证样本不足",
                    "insufficient_ai_validation": "AI 验证样本不足",
                    "ai_recall_below_minimum": "AI 验证召回率低于最低要求",
                    "auroc_unavailable": "当前无法计算验证集 AUROC",
                    "mixed_language_threshold_fallback": (
                        "混合语种未独立校准，已采用更严格的单语阈值"
                    ),
                }
                warnings = [
                    warning_names.get(str(item), str(item))
                    for item in decision.get("threshold_warnings") or []
                ]
                st.warning("阈值可靠性提示：" + "；".join(warnings))
            reasons = decision.get("uncertainty_reasons") or []
            if reasons:
                st.caption("判定说明：" + "、".join(str(reason) for reason in reasons))

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

        if segments:
            render_segment_panel(segments, low_threshold, high_threshold)
            st.subheader("片段风险分布")
            segment_scores = [float(item.get("score") or 0.0) for item in segments]
            colors = [
                "#b91c1c" if value > high_threshold else "#ca8a04" if value > low_threshold else "#2563eb"
                for value in segment_scores
            ]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[f"第 {int(item.get('index', idx)) + 1} 块" for idx, item in enumerate(segments)],
                        y=segment_scores,
                        marker_color=colors,
                        customdata=[
                            f"字符 {item.get('start', 0)}-{item.get('end', 0)}"
                            for item in segments
                        ],
                        hovertemplate="%{x}<br>AI率=%{y:.1%}<br>%{customdata}<extra></extra>",
                    )
                ]
            )
            fig.add_hline(y=low_threshold, line_dash="dot", line_color="#2563eb")
            fig.add_hline(y=high_threshold, line_dash="dash", line_color="#b91c1c")
            fig.update_layout(height=250, yaxis_range=[0, 1], margin={"l": 20, "r": 20, "t": 10, "b": 20})
            st.plotly_chart(fig, use_container_width=True)

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
        pcols[0].metric("Deep Check", "YES" if provenance.get("deep_triggered") else "NO")
        pcols[1].metric("C2PA", provenance["c2pa"].get("status", "unknown"))
        pcols[2].metric("Watermark", provenance["watermark"].get("status", "unknown"))
        registry = provenance.get("fingerprint_registry", {})
        pcols[3].metric("Fingerprint Hits", registry.get("match_count", 0))
        attribution = provenance.get("attribution", {})
        if provenance.get("attribution_triggered") and not provenance.get("deep_triggered"):
            st.caption("整篇 AI 率未过 0.6，但高分检测块已触发家族溯源。")
        family_status = attribution.get("family_status")
        verdict = attribution.get("verdict")
        if family_status == "family_only":
            st.success(verdict or "AI-家族明确")
        elif family_status == "unknown":
            st.warning(verdict or "AI-家族不明")
        st.caption(
            "家族溯源: "
            f"{family_status} | "
            f"confidence={float(attribution.get('confidence') or 0.0):.2%}"
            + (
                f" | {attribution.get('recipe')}"
                if attribution.get("recipe")
                else ""
            )
        )
        if attribution.get("status") == "data_mismatch":
            st.warning("LLMDet data cache is incomplete or inconsistent; local text attribution was skipped.")
            missing = attribution.get("missing_npz") or []
            bad = attribution.get("bad_npz") or []
            if missing:
                st.caption(f"Missing n-gram tables: {', '.join(missing)}")
            if bad:
                st.caption(f"Unreadable n-gram tables: {', '.join(bad)}")
        if attribution.get("error"):
            st.warning(f"Attribution failed: {attribution['error']}")
        if attribution.get("install_hint"):
            st.caption(attribution["install_hint"])
        if attribution.get("note") and attribution.get("status") in {"skipped", "not_configured", "dependency_missing"}:
            st.caption(attribution["note"])

        top_k = attribution.get("top_k") or []
        if top_k:
            st.subheader("总加权家族结论")
            if family_status == "unknown":
                st.caption("首位家族概率低于 20%，结论为家族不明；下面仍列出闭集参考 Top-K，不再另报未知概率。")
            for item in top_k:
                st.write(
                    f"- {item.get('model', 'unknown')}: "
                    f"{float(item.get('probability') or 0.0):.2%}"
                )
            st.caption("按进入溯源的检测块长度×检测分加权，不相邻块不粘贴。生产头是切窗 A0+B0+B1+B3（等权基线或经核验后显式配置的 validation 权重）。")

        blocks = attribution.get("blocks") or []
        if blocks:
            st.subheader("高分块家族 Top-K")
            for block in blocks:
                indices = block.get("block_indices") or []
                title = "、".join(f"第 {int(idx) + 1} 块" for idx in indices) or "检测块"
                score = float(block.get("detection_score") or 0.0)
                block_verdict = block.get("verdict") or block.get("family_status") or ""
                st.markdown(f"**{title}** · 检测 AI 率 {score:.1%} · {block_verdict}")
                preview = str(block.get("preview") or "").strip()
                if preview:
                    st.caption(preview)
                for item in block.get("top_k") or []:
                    st.write(
                        f"- {item.get('model', 'unknown')}: "
                        f"{float(item.get('probability') or 0.0):.2%}"
                    )

        provider_hints = provenance.get("provider_hints", {})
        provider_top_k = provider_hints.get("top_k") or []
        if provider_top_k:
            with st.expander("Provider Model Hints", expanded=False):
                st.caption("These are detection-side hints, not local attribution-model results.")
                for item in provider_top_k:
                    st.write(
                        f"- {item.get('model', 'unknown')}: "
                        f"{float(item.get('probability') or 0.0):.2%}"
                    )

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
        report_status = report.get("status", "unknown")
        report_provider = report.get("provider", "unknown")
        st.caption(f"报告来源：{report_provider} | 状态：{report_status}")
        if report.get("error"):
            st.warning(f"报告模型未正常返回，当前展示本地回退报告：{report['error']}")
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
