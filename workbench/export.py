"""JSON / Markdown export of stored snapshots (no image bytes, no server paths)."""
from __future__ import annotations

import json
from typing import Any

DETECTION_LABELS = {"at_or_above_threshold": "模型检测分数达到阈值", "below_threshold": "模型检测分数低于阈值"}
ATTRIBUTION_REASONS = {"detection_below_threshold": "检测分数低于阈值，主结论不给出来源",
                       "inference_failed": "来源模型推理失败（部分结果）"}
TEXT_LABELS = {"low_risk": "未明显检出", "uncertain": "不确定", "high_risk_ai": "高风险疑似 AI",
               "insufficient_evidence": "证据不足", "human": "未明显检出", "ai": "疑似 AI"}


def to_json(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)


def _pct(value) -> str:
    return "不可用" if value is None else f"{value * 100:.4f}%"


def image_markdown(r: dict[str, Any]) -> str:
    d, a, i = r["detection"], r["attribution"], r["input"]
    score = "不可用" if d["score"] is None else f"{d['score']:.6f}"
    lines = [
        "# 图片检测报告", "",
        f"- 分析 ID：`{r['analysis_id']}`", f"- 时间（UTC）：{r['created_at']}",
        f"- 模式：{'研究模式' if r.get('mode') == 'research' else '默认'}；状态：{r['status']}", "",
        "## 输入", "",
        f"- SHA-256：`{i['sha256']}`", f"- 类型：{i['media_type']}，{i['width']}×{i['height']}，{i['bytes']} 字节",
        f"- 透明通道：{'有（按研究预处理丢弃）' if i['has_alpha'] else '无'}；"
        f"EXIF 方向：{i['exif_orientation'] or '无'}（未应用旋转，与研究预处理一致）", "",
        "## 检测（D-OpenCLIP）", "",
        f"- 模型检测分数：{score}（不是校准概率）",
        f"- 阈值：{d['threshold']}；结论：{DETECTION_LABELS.get(d['label'], d['label'])}",
    ]
    tp = d.get("threshold_provenance") or {}
    if tp:
        lines.append(f"- 阈值依据：evaluation {tp['n']} 张（{tp['negative_n']} 真实 / {tp['positive_n']} AI），"
                     f"FPR {_pct(tp['fpr'])}，TPR {_pct(tp['tpr'])}")
    lines += ["", "## 来源候选（A0 ResNet50，22 类闭集）", ""]
    if a["status"] == "ok":
        lines += [a["conditional_statement"], "", "| 排名 | 来源 | 闭集分数 |", "|---:|---|---:|"]
        lines += [f"| {t['rank']} | {t['label']} | {t['score']:.4f} |" for t in a["top_k"][:3]]
    else:
        lines.append(ATTRIBUTION_REASONS.get(a.get("reason"), a.get("reason") or a["status"]))
    lines += ["", "## 未知来源", "", "不可用（score = null）：主 22 类接口没有已验收的未知拒识。"]
    for item in r.get("research") or []:
        if item["id"] == "d_openclip_calibrated_thresholds":
            lines += ["", "## 研究：校准阈值（只在 calibration 真实图拟合）", "",
                      "| 目标 FPR | 阈值 | 达到 | calibration FPR | evaluation FPR / TPR |", "|---:|---:|---|---:|---:|"]
            lines += [f"| {row['target_fpr']:.0%} | {row['threshold']:.4f} | {'是' if row['reached'] else '否'} | "
                      f"{_pct(row['calibration_fpr'])}（{row['calibration_negative_n']}） | "
                      f"{_pct(row['evaluation_fpr'])} / {_pct(row['evaluation_tpr'])}（{row['evaluation_n']}） |"
                      for row in item["rows"]]
        elif item["id"] == "ofa_pair_verification":
            lines += ["", f"## 研究：OFA 配对验证（{item['status']}）", "", item["note"]]
            if item.get("reason"):
                lines.append(f"原因：{item['reason']}")
    bundle = r["model_bundle"]
    lines += ["", "## 模型包", "", f"- bundle：`{bundle['bundle_id']}`"]
    lines += [f"- {name}：`{sha}`" for name, sha in bundle["components"].items()]
    lines += ["", "## 局限", ""] + [f"- {x}" for x in r["limitations"]]
    lines += ["", f"用时：{r['timing'].get('total_ms')} ms（不含网络传输）", ""]
    return "\n".join(lines)


def text_markdown(snapshot: dict[str, Any]) -> str:
    result = snapshot["result"]
    detection, report = result.get("detection") or {}, result.get("report") or {}
    score = detection.get("score")
    score = "不可用" if score is None else f"{score:.2%}"
    lines = ["# 文本检测报告", "", f"- 分析 ID：`{snapshot['analysis_id']}`", f"- 时间（UTC）：{snapshot['created_at']}",
             f"- 名称：{snapshot['display_name']}",
             f"- 检测结论：{TEXT_LABELS.get(detection.get('label'), detection.get('label'))}",
             f"- AI 特征风险分数：{score}（不是 AI 生成概率）", "",
             "## 报告", "", str(report.get("summary", ""))]
    lines += [f"- {x}" for x in report.get("evidence") or []]
    lines += ["", "## 局限", ""] + [f"- {x}" for x in report.get("limitations") or []] + [""]
    return "\n".join(lines)


def to_markdown(kind: str, snapshot: dict[str, Any]) -> str:
    return image_markdown(snapshot) if kind == "image" else text_markdown(snapshot)
