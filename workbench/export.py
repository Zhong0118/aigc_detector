"""JSON / Markdown export of results and stored snapshots (no image bytes, no server paths)."""
from __future__ import annotations

import json
from typing import Any

TENDENCY = {"ai": "AI 生成倾向", "real": "真实图片倾向"}
ATTRIBUTION_STATUS = {"skipped": "检测分数低于阈值，未运行来源模型（不是“已验证未知来源”）",
                      "error": "来源模型推理失败，整体为部分成功，检测结果仍有效",
                      "unavailable": "来源模型不可用"}
TEXT_LABELS = {"low_risk": "未明显检出", "uncertain": "不确定", "high_risk_ai": "高风险疑似 AI",
               "insufficient_evidence": "证据不足", "human": "未明显检出", "ai": "疑似 AI"}


def to_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _pct(value) -> str:
    return "不可用" if value is None else f"{value * 100:.2f}%"


def _ranked(lines: list[str], top_k: list[dict], limit: int) -> None:
    lines += ["| 排名 | 来源 | 22 类 softmax 分数 |", "|---:|---|---:|"]
    lines += [f"| {t['rank']} | {t['source']} | {t['score']:.4f} |" for t in top_k[:limit]]


def image_markdown(r: dict[str, Any]) -> str:
    d, a, i, rt = r["detection"], r["attribution"], r["input"], r["runtime"]
    persisted = r["persistence"]
    lines = [
        "# 图片检测报告", "",
        f"- 请求 ID：`{r['request_id']}`；时间（UTC）：{r['created_at']}",
        f"- 模式：{r['mode']}；状态：{r['status']}；报告保存："
        f"{'已保存（content_id ' + str(persisted['content_id']) + '）' if persisted['status'] == 'ok' else '未保存'}",
        f"- 模型包：`{r['bundle_id']}`；registry `{r['registry_sha256']}`；代码 `{r['inference_code_commit']}`", "",
        "## 输入", "",
        f"- SHA-256：`{i['sha256']}`；{i['media_type']}，{i['width']}×{i['height']}，{i['bytes']} 字节，{i['color_mode']}",
        f"- 解码策略：`{i['decode_policy']}`（模型随后各自缩放与中心裁剪）", "",
        f"## 真假检测（{d['model_id']}）", "",
        f"- 结论：**{TENDENCY[d['label']]}**",
        f"- AI 生成分数：{_pct(d['score'])}（原始值 {d['score']:.6f}，{d['score_kind']}）。这是 AI 类的分数，"
        "不是校准概率，低分也不是“真实图片”的置信度",
        f"- 固定阈值：{d['threshold']}（`{d['threshold_id']}`）",
    ]
    ev = d.get("threshold_evidence")
    if ev:
        lines.append(f"- 阈值在开发 evaluation 的表现：{ev['n']} 张（{ev['real_n']} 真实 / {ev['ai_n']} AI），"
                     f"FPR {_pct(ev['fpr'])}，TPR {_pct(ev['tpr'])}；不是新上传图片的性能保证")
    lines += ["", f"## 来源候选（{a['model_id']}，22 类）", ""]
    if a["status"] == "ok":
        lines += [a["conditional_statement"], ""]
        _ranked(lines, a["top_k"], 3)
        lines += ["", "Top3 保留原 22 类 softmax 值，未重新归一化。"]
    else:
        lines.append(ATTRIBUTION_STATUS.get(a["status"], a["status"]) + f"（reason_code：{a.get('reason_code')}）")
    lines += ["", "## 未知来源", "", f"不可用（score = null，{r['unknown']['reason_code']}）：尚无经验证的通用未知来源拒识能力。"]
    for item in r.get("research") or []:
        lines += ["", f"## 研究对照：{item['label']}（{item['method_id']}，{item['status']}）", "",
                  "独立附在标准结果之后，不改写标准结论。"]
        out = item.get("outputs") or {}
        if out.get("thresholds"):
            lines += ["", "| 阈值 ID | 阈值 | 本图读数 | calibration FPR（真实图） | evaluation FPR / TPR |",
                      "|---|---:|---|---:|---:|"]
            lines += [f"| {t['threshold_id']} | {t['threshold']:.4f} | {TENDENCY[t['label']]} | "
                      f"{_pct(t['calibration_fpr'])}（{t['calibration_real_n']}） | "
                      f"{_pct(t['evaluation_fpr'])} / {_pct(t['evaluation_tpr'])}（{t['evaluation_n']}） |"
                      for t in out["thresholds"]]
        if out.get("top_k"):
            lines += ["", out.get("conditional_statement", ""), ""]
            _ranked(lines, out["top_k"], 3)
        lines += [""] + [f"- {x}" for x in item.get("limitations", [])]
    t = rt["timing_ms"]
    lines += ["", "## 运行环境", "",
              f"- 设备：{rt['device']}，精度 {rt['precision']}；外部调用：{'有' if rt['external_calls'] else '无'}",
              f"- 耗时（毫秒）：解码 {t.get('decode_ms')}，排队 {t.get('queue_ms')}，检测 {t.get('detection_ms')}，"
              f"来源 {t.get('attribution_ms')}，研究 {t.get('research_ms')}，合计 {t.get('total_ms')}（不含网络传输）",
              "", "## 局限", ""] + [f"- {x}" for x in r["limitations"]] + [""]
    return "\n".join(lines)


def text_markdown(analysis: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
    detection, report = analysis.get("detection") or {}, analysis.get("report") or {}
    score = detection.get("score")
    lines = ["# 文本检测报告", ""]
    if meta:
        lines += [f"- 报告 ID：{meta.get('content_id')}；时间（UTC）：{meta.get('created_at')}"]
    lines += [f"- 名称：{analysis.get('filename')}",
              f"- 检测结论：{TEXT_LABELS.get(detection.get('label'), detection.get('label'))}",
              f"- AI 特征风险分数：{'不可用' if score is None else f'{score:.2%}'}（不是 AI 生成概率）", "",
              "## 报告", "", str(report.get("summary", ""))]
    lines += [f"- {x}" for x in report.get("evidence") or []]
    lines += ["", "## 局限", ""] + [f"- {x}" for x in report.get("limitations") or []] + [""]
    return "\n".join(lines)
