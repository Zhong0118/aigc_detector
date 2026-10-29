"""Read-only experiment catalog from hash-registered result files (no recomputation)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workbench.image_engine import sha256_file

BASE_LIMITATIONS = [
    "单 seed（42）；selection/calibration/evaluation 均为已查看过的开发角色，不是独立外部测试。",
    "IAB v7-200-q179：63,821 张有效图片，名义每格 200 张、实际 187–200 张；179 张隔离证据保留。",
    "原 holdout 未推理、未评分、未调参。",
    "不同分母的结果不能直接比较“提升”；只有同一批样本的配对比较给出差值与 95% 区间。",
]


def _load_sources(root: Path, sources: dict[str, dict]) -> tuple[dict, dict, list]:
    data, registry, errors = {}, {}, []
    for name, spec in sources.items():
        path = root / spec["path"]
        if not path.is_file():
            errors.append(dict(source=name, code="model_unavailable", message="登记结果文件缺失"))
            continue
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            errors.append(dict(source=name, code="artifact_mismatch", message="登记结果文件哈希不符"))
            continue
        data[name] = json.loads(path.read_text())
        registry[name] = dict(label=spec.get("label", name), sha256=actual)
    return data, registry, errors


def _count(rate: float, n: int) -> int:
    return int(round(rate * n))


def _detection_row(model, role, values, negatives, positives):
    return dict(model=model, role=role, n=negatives + positives, negative_n=negatives, positive_n=positives,
                auroc=values["auroc"], fpr=values["fpr_0_5"] if "fpr_0_5" in values else values["fpr"],
                tpr=values["tpr_0_5"] if "tpr_0_5" in values else values["tpr"],
                fp=_count(values["fpr_0_5"] if "fpr_0_5" in values else values["fpr"], negatives),
                tp=_count(values["tpr_0_5"] if "tpr_0_5" in values else values["tpr"], positives))


def _paired(name, comparison, metrics):
    ci = comparison["group_bootstrap"]["delta_95_ci"]
    return dict(comparison=name, n=comparison["n"], group_n=comparison["group_bootstrap"]["group_n"],
                repeats=comparison["group_bootstrap"]["repeats"],
                deltas={k: dict(delta=comparison["delta_after_minus_before"][k], ci95=ci[k]) for k in metrics})


def detection_view(d: dict) -> dict:
    s, closed, metrics = d["next_routes_summary"], d["closed_comparison"], d["dopenclip_metrics"]
    main = closed["openclip-detection-vs-d1-control"]
    ffhq = closed["openclip-ffhq-vs-d2-semantic"]
    det = s["detector"]
    neg, pos = det["D-OpenCLIP"]["negative_count"], det["D-OpenCLIP"]["positive_count"]
    fneg, fpos = det["D-OpenCLIP-FFHQ"]["negative_count"], det["D-OpenCLIP-FFHQ"]["positive_count"]
    thresholds = [dict(target_fpr=float(k), threshold=v["evaluation"]["threshold"],
                       calibration_negative_n=v["calibration_negative_n"], calibration_fpr=v["calibration_fpr"],
                       evaluation_n=v["evaluation"]["n"], evaluation_negative_n=v["evaluation"]["negative_count"],
                       evaluation_fpr=v["evaluation"]["fpr"], evaluation_tpr=v["evaluation"]["tpr"])
                  for k, v in sorted(metrics["calibrated"].items(), key=lambda kv: float(kv[0]))]
    return dict(
        id="detection", title="AI 生成检测（固定阈值 0.5）",
        groups=[
            dict(title="同一 2310 张 evaluation（979 真实 / 1331 AI）",
                 rows=[_detection_row("D-OpenCLIP（默认）", "default", main["after"], neg, pos),
                       _detection_row("ResNet50 D1-control", "comparison", main["before"], neg, pos),
                       _detection_row("OmniAID（冻结外部预训练）", "registered_only", det["OmniAID"],
                                      det["OmniAID"]["negative_count"], det["OmniAID"]["positive_count"])],
                 paired=[_paired("D-OpenCLIP − ResNet50 D1-control", main, ("auroc", "fpr_0_5", "tpr_0_5"))]),
            dict(title="held-FFHQ 跨语义（225 张：93 真实 / 132 AI；FFHQ 不进入训练、选择与校准）",
                 rows=[_detection_row("D-OpenCLIP-FFHQ", "comparison", ffhq["after"], fneg, fpos),
                       _detection_row("ResNet50 D2-semantic", "comparison", ffhq["before"], fneg, fpos)],
                 paired=[_paired("D-OpenCLIP − ResNet50 D2-semantic", ffhq, ("auroc", "fpr_0_5", "tpr_0_5"))]),
        ],
        research_thresholds=thresholds,
        notes=["分数是模型检测分数，不是校准概率。",
               "整体 FPR 差异区间到达 0，不能宣称已统计确认整体误报下降。",
               "held-FFHQ 上按非 FFHQ calibration 拟合的 1%/5% 阈值实际 FPR 为 7.53%/30.11%，目标 FPR 不跨域保证。"])


def attribution_view(d: dict) -> dict:
    f, s = d["fusion_summary"], d["next_routes_summary"]
    fm, boot = f["metrics"], f["paired_group_bootstrap"]

    def row(model, role, v):
        return dict(model=model, role=role, n=v["n"], top1=v["top1"], top3=v["top3"], macro_f1=v["macro_f1"],
                    top1_correct=_count(v["top1"], v["n"]), top3_correct=_count(v["top3"], v["n"]))

    paired = [dict(comparison=f"校准融合 − {label}", n=f["evaluation_n"], group_n=boot[key]["group_n"],
                   repeats=boot[key]["repeats"],
                   deltas={m: dict(delta=fm["Calibrated"][m] - fm[key][m], ci95=boot[key]["delta_95_ci"][m])
                           for m in ("top1", "top3", "macro_f1")})
              for key, label in (("ResNet50", "A0 ResNet50"), ("Uniform", "固定 50/50 融合"))]
    oc = d["closed_comparison"]["openclip-attribution22-vs-a2-clean"]
    paired.append(_paired("A-OpenCLIP22 − A0 ResNet50", oc, [k for k in ("top1", "top3", "macro_f1")
                                                            if k in oc["delta_after_minus_before"]]))
    return dict(
        id="attribution", title="22 来源闭集归因（同一 1331 张 AI evaluation）",
        groups=[dict(title="1331 张 AI evaluation，22 个已知来源",
                     rows=[row("A0 ResNet50（默认）", "default", fm["ResNet50"]),
                           row("固定 50/50 融合", "registered_only", fm["Uniform"]),
                           row(f"校准融合 α_resnet={f['alpha_resnet']}", "registered_only", fm["Calibrated"]),
                           row("A-Fusion（DINOv2+指纹）", "comparison", fm["FeatureFusion"]),
                           row("A-OpenCLIP22", "comparison", s["attribution"]["A-OpenCLIP22"]),
                           row("A-OpenCLIP-KNN22（k=1）", "comparison", s["attribution"]["A-OpenCLIP-KNN22"])],
                     paired=paired)],
        notes=[f"融合权重只在 {f['calibration_n']} 张 AI calibration 上选择；固定 50/50 与校准融合仅为登记结果，上传图片不做融合推理。",
               "来源标签是发布方目录，不是独立核实的后端版本。",
               "KNN 选 k=1，Top-3 部分依赖零票类别的并列排序。"])


def rejection_view(d: dict) -> dict:
    rej = d["next_routes_summary"]["rejection"]
    names = {"resnet_msp": "ResNet MSP", "openclip_msp": "OpenCLIP MSP", "openclip_prototype": "OpenCLIP 原型余弦"}
    rows = []
    for key, label in names.items():
        v = rej[key]
        rows.append(dict(model=label, role="registered_only", n=v["n"], known_n=v["known_n"], unknown_n=v["unknown_n"],
                         known_retention=v["known_retention"], known_retention_count=v["known_retention_count"],
                         known_correct_and_retained=v["known_correct_and_retained"],
                         known_correct_and_retained_count=v["known_correct_and_retained_count"],
                         unknown_false_acceptance=v["unknown_false_acceptance"],
                         unknown_false_acceptance_count=v["unknown_false_acceptance_count"],
                         unknown_rejection_count=v["unknown_rejection_count"], unknown_auroc=v["unknown_auroc"],
                         calibration_unknown_n=v["calibration_unknown_n"],
                         calibration_unknown_false_acceptance=v["calibration_unknown_false_acceptance"]))
    return dict(id="rejection", title="open20 未知拒识（20 已知来源 + 未知）",
                groups=[dict(title="evaluation：1212 张已知 + 63 张未知（grok3）；阈值只在 64 张 kling calibration 上确定",
                             rows=rows, paired=[])],
                notes=["拒识分数不是未知概率；20 类阈值不适用于 22 类主接口，主接口未知来源保持不可用。",
                       "不能只看未知拒绝率：须同时看已知保留与正确且保留。",
                       "未知迁移只测试了 kling→grok3，不代表通用未知检测。"])


def route_status(root: Path, routes: list[dict]) -> list[dict]:
    result = []
    for route in routes:
        status = route.get("status", "registered")
        if route.get("status_file"):
            path = root / route["status_file"]
            try:
                status = json.loads(path.read_text()).get("status", "unknown") if path.is_file() else "not_started"
            except (OSError, ValueError):
                status = "unreadable"
        result.append(dict(id=route["id"], run_status=status, acceptance=route.get("acceptance", "not_accepted"),
                           upload_inference=False, note=route.get("note", "")))
    return result


def build_catalog(settings: dict[str, Any]) -> dict[str, Any]:
    root, cat = Path(settings["research_root"]), settings.get("catalog") or {}
    data, registry, errors = _load_sources(root, cat.get("sources") or {})
    views = []
    for builder, needs in ((detection_view, ("next_routes_summary", "closed_comparison", "dopenclip_metrics")),
                           (attribution_view, ("fusion_summary", "next_routes_summary", "closed_comparison")),
                           (rejection_view, ("next_routes_summary",))):
        if all(n in data for n in needs):
            views.append(builder(data))
    summary = data.get("next_routes_summary", {})
    limitations = BASE_LIMITATIONS + list(summary.get("limitations", []))
    if "fusion_summary" in data:
        limitations.append(data["fusion_summary"]["limitation"])
    return dict(status="ok" if not errors else ("partial" if views else "unavailable"), errors=errors,
                sources=registry, dataset=dict(effective_images=summary.get("effective_images"),
                                               actual_generated_per_cell=summary.get("actual_generated_per_cell"),
                                               seed=summary.get("seed"), training_epochs=summary.get("training_epochs"),
                                               holdout_scored=summary.get("holdout_scored")),
                views=views, routes=route_status(root, cat.get("routes") or []), limitations=limitations)
