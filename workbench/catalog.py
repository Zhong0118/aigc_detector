"""Read-only experiment catalog from the sealed 2026-10-30 final overview (hash-checked).

Only aggregate JSON is read: no images, per-image predictions, re-scoring, threshold or
alpha refitting. Paired deltas are quoted from the registered paired reports; rows with
different denominators are grouped separately and never turned into "improvements".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workbench.image_engine import sha256_file

KEY_FACTS = [
    "held-FFHQ 同 93 张真实图：OpenCLIP 固定 0.5 误报 15 张（16.13%），ResNet 与 ConvNeXt 各 56 张（60.22%）。",
    "按非 FFHQ calibration 目标 1% 选阈值后，OpenCLIP 在 FFHQ 仍误报 7/93（7.53%，Wilson 95% 约 3.69–14.73%），不是跨域 1% 保证。",
    "OFA20+1 拒绝 63/63 未知图，但已知图只保留 58/1212（4.79%），正确且保留 28/1212（2.31%）；不能只宣传 100% 未知拒识。",
    "0/63 未知误接收的 Wilson 95% 上界仍约 5.75%，不能声称误接收恒为 0。",
    "A0 及部分旧模型的父 dev 参与选模；全部为单 seed 开发结果，不是独立测试。",
    "A3 完整家族映射未核验、完整 OFA 预训练未运行，均不提供入口。",
]
DET_PAIRS = (("convnext-comparison", "D-ConvNeXt-vs-openclip", "ConvNeXt − OpenCLIP"),
             ("convnext-comparison", "D-ConvNeXt-vs-resnet", "ConvNeXt − ResNet-control"))
FFHQ_PAIRS = (("convnext-comparison", "D-ConvNeXt-FFHQ-vs-openclip", "ConvNeXt-heldFFHQ − OpenCLIP-heldFFHQ"),
              ("convnext-comparison", "D-ConvNeXt-FFHQ-vs-resnet", "ConvNeXt-heldFFHQ − ResNet-heldFFHQ"))
SOURCE_PAIRS = (("dna-comparison", "A-DNA-Wild-LP22-vs-openclip", "DNA-Det 线性 − OpenCLIP 线性"),
                ("convnext-comparison", "A-ConvNeXt22-vs-openclip", "ConvNeXt 线性 − OpenCLIP 线性"),
                ("omnidfa-comparison", "A-OmniDFA-LP22-vs-openclip", "OmniDFA 线性 − OpenCLIP 线性"),
                ("omnidfa-comparison", "A-OmniDFA-prototype22-K10-vs-openclip", "OmniDFA K10 − OpenCLIP 线性"),
                ("dna-parent-dev-reference", "A-DNA-Wild-LP22-vs-resnet", "DNA-Det 线性 − A0 ResNet（父 dev 选模参考）"),
                ("convnext-parent-dev-reference", "A-ConvNeXt22-vs-resnet", "ConvNeXt 线性 − A0 ResNet（父 dev 选模参考）"),
                ("omnidfa-parent-dev-reference", "A-OmniDFA-LP22-vs-resnet", "OmniDFA 线性 − A0 ResNet（父 dev 选模参考）"))
DET_ROLES = {"OpenCLIP": "default", "ResNet-control": "comparison", "ResNet-augmented": "comparison",
             "OmniAID": "comparison", "ConvNeXt": "comparison", "DINOv2": "comparison"}
DET_LABELS = {"OpenCLIP": "D-OpenCLIP（默认 det-openclip-q179-v1）", "ResNet-control": "ResNet-control（D1-control）",
              "ResNet-augmented": "ResNet-augmented（D1 增强）", "OmniAID": "OmniAID（冻结外部模型）",
              "ConvNeXt": "ConvNeXt 冻结线性头", "DINOv2": "DINOv2 冻结线性头"}
SOURCE_LABELS = {
    "ResNet-A0-parent-dev-selected": ("A0 ResNet50（默认 attr-resnet50-q179-v1，父 dev 选模）", "default"),
    "A-RGB": ("A-RGB（DINOv2 CLS）", "comparison"), "A-FP": ("A-FP（OFA 指纹）", "comparison"),
    "A-Fusion": ("A-Fusion（RGB+FP 特征融合）", "comparison"), "OpenCLIP-linear": ("OpenCLIP 线性头", "comparison"),
    "OpenCLIP-KNN": ("OpenCLIP KNN（k=1）", "comparison"), "ConvNeXt-linear": ("ConvNeXt 线性头", "comparison"),
    "DNA-Det-linear": ("DNA-Det 线性头（研究上传可运行）", "research_online"),
    "OmniDFA-linear": ("OmniDFA 线性头", "comparison"), "OmniDFA-K10-prototype": ("OmniDFA K10 原型", "comparison"),
    "score-fusion-ResNet50": ("分数融合输入：ResNet50", "comparison"),
    "score-fusion-FeatureFusion": ("分数融合输入：A-Fusion", "comparison"),
    "score-fusion-Uniform": ("分数融合 固定 50/50", "registered_only"),
    "score-fusion-Calibrated": ("分数融合 calibration 75/25", "registered_only")}
UNKNOWN_LABELS = {"resnet_msp": "ResNet open20 MSP", "openclip_msp": "OpenCLIP open20 MSP",
                  "openclip_prototype": "OpenCLIP open20 原型余弦", "ofa_tplus1_known_mass": "OFA20+1（已知质量）"}


def _load(root: Path, spec: dict, name: str, errors: list) -> Any:
    path = root / spec["path"]
    if not path.is_file():
        errors.append(dict(source=name, code="model_unavailable", message="登记结果文件缺失"))
        return None
    if sha256_file(path) != spec["sha256"]:
        errors.append(dict(source=name, code="artifact_mismatch", message="登记结果文件哈希不符"))
        return None
    return json.loads(path.read_text())


def _det_row(label, role, v, scope, extra=None):
    neg, pos = v.get("negative_count"), v.get("positive_count")
    row = dict(method=label, role=role, scope=scope, n=v["n"], real_n=neg, ai_n=pos, auroc=v.get("auroc"),
               threshold=v.get("threshold", 0.5), fpr=v["fpr"], tpr=v["tpr"],
               fp=v.get("false_positives", round(v["fpr"] * neg) if neg else None),
               tp=v.get("true_positives", round(v["tpr"] * pos) if pos else None),
               fpr_wilson95=v.get("fpr_wilson95"))
    row.update(extra or {})
    return row


def _paired(label, comparison, metrics):
    boot, delta = comparison["group_bootstrap"], comparison["delta_after_minus_before"]
    return dict(comparison=label, n=comparison["n"], group_n=boot["group_n"], repeats=boot["repeats"],
                deltas={m: dict(delta=delta[m], ci95=boot["delta_95_ci"][m]) for m in metrics if m in boot["delta_95_ci"]})


def _pairs(o, spec, metrics):
    rc = o["registered_comparisons"]
    return [_paired(label, rc[report][key], metrics) for report, key, label in spec
            if key in rc.get(report, {})]


def detection_groups(o: dict) -> list[dict]:
    clean = o["clean_detection_2310"]
    rows = [_det_row(DET_LABELS[k], DET_ROLES[k], v, "clean") for k, v in clean.items()]
    ffhq = o["held_ffhq_225"]
    ffhq_rows, cal_rows = [], []
    for name, m in ffhq["methods"].items():
        ffhq_rows.append(_det_row(name, "comparison", m["fixed_0_5"], "held_ffhq", dict(threshold_id="fixed-0.5")))
        for target, c in sorted(m["calibrated"].items(), key=lambda kv: float(kv[0])):
            cal_rows.append(_det_row(name, "comparison", c["evaluation"], "held_ffhq", dict(
                threshold_id=f"cal-target-{float(target):.0%}", target_fpr=float(target), threshold=c["threshold"],
                threshold_fitted_role=c["threshold_fitted_role"], calibration_fpr=c["calibration"]["fpr"],
                calibration_real_n=c["calibration"]["negative_count"])))
    ffhq_pairs = [dict(comparison=k.replace("_minus_", " − "), n=ffhq["evaluation_n"], group_n=v["group_n"],
                       repeats=v["repeats"], deltas={m: dict(delta=None, ci95=ci) for m, ci in v["delta_95_ci"].items()})
                  for k, v in ffhq["fixed_0_5_paired_group_intervals"].items()]
    refit = []
    for target, methods in ffhq["threshold_refit_group_bootstrap"]["results"].items():
        for name, v in methods.items():
            refit.append(dict(method=name, target_fpr=float(target), threshold_ci95=v["threshold_95_ci"],
                              evaluation_fpr_ci95=v["evaluation_fpr_95_ci"], evaluation_tpr_ci95=v["evaluation_tpr_95_ci"]))
    ext = []
    for name in ("D1-control", "D1", "D2", "D2-semantic", "D3"):
        v = o["previous_extensions"][name]["conditions"]["clean"]["fixed_0_5"]
        ext.append(_det_row(name, "comparison", v, "previous_extension"))
    d0 = o["baseline_full_dev_separate_denominators"]["d0-resnet50"]
    return [
        dict(id="detection-clean", task="detection", kind="detection", title="真假检测 · 同 2310 张 evaluation（979 真实 / 1331 AI），固定阈值 0.5",
             rows=rows, paired=_pairs(o, DET_PAIRS, ("auroc", "fpr_0_5", "tpr_0_5")),
             notes=["分数是模型检测分数，不是校准概率。", "ResNet-control 与 D3 曾有完全相同的真实图概率向量，不能当独立复现。",
                    "D1 增强的 clean TPR 上升但 FPR 也上升；增强后的 JPEG/缩放鲁棒性未登记验证。"]),
        dict(id="detection-ffhq", task="detection", kind="detection",
             title=f"held-FFHQ 跨语义 · 同 {ffhq['evaluation_n']} 张 evaluation（{ffhq['evaluation_real_n']} 真实 / "
                   f"{ffhq['evaluation_ai_n']} AI），训练/选择/校准均排除 FFHQ",
             rows=ffhq_rows, calibrated_rows=cal_rows, paired=ffhq_pairs + _pairs(o, FFHQ_PAIRS, ("auroc", "fpr_0_5", "tpr_0_5")),
             threshold_refit=refit,
             notes=["held-FFHQ 是专门排除 FFHQ 训练的对照权重；默认 clean 模型不随“看起来是人脸”自动切换。",
                    f"校准阈值只用 {ffhq['calibration_real_n']} 张非 FFHQ calibration 真实图选择；FFHQ 只报告。",
                    ffhq["limitations"]]),
        dict(id="detection-extensions", task="detection", kind="detection", title="既有检测拓展（各自分母，不与上表直接比较提升）",
             rows=ext, paired=[],
             notes=["D2 排除 4o：59 AI + 979 真实，正例少且单一来源，不代表普遍未知检测。"]),
        dict(id="detection-full-dev", task="detection", kind="summary", title="D0 完整 7189 dev（另一分母，单列）",
             rows=[dict(method="D0 ResNet50", n=d0["n"], auroc=d0["auroc"], fpr=d0["fpr_0_5"], tpr=d0["tpr_0_5"],
                        selected_epoch=d0["selected_epoch"], train_n=d0["train_n"])], paired=[], notes=[]),
    ]


def source_groups(o: dict) -> list[dict]:
    rows = []
    for key, v in o["source_classification_1331"].items():
        label, role = SOURCE_LABELS.get(key, (key, "comparison"))
        rows.append(dict(method=label, role=role, n=v["n"], top1=v["top1"], top3=v["top3"], macro_f1=v["macro_f1"],
                         top1_correct=round(v["top1"] * v["n"]), top3_hit=round(v["top3"] * v["n"]),
                         training_epochs=v.get("training_epochs", 30 if "best_epoch" in v else None),
                         selected_epoch=v.get("epoch", v.get("best_epoch"))))
    sf = o["score_fusion_calibration_only"]
    fusion_pairs = [dict(comparison=f"分数融合 75/25 − {name}", n=sf["evaluation_n"], group_n=v["group_n"],
                         repeats=v["repeats"],
                         deltas={m: dict(delta=sf["metrics"]["Calibrated"][m] - sf["metrics"][name][m], ci95=ci)
                                 for m, ci in v["delta_95_ci"].items()})
                    for name, v in sf["paired_group_bootstrap"].items()]
    a2 = o["previous_extensions"]["A2"]["conditions"]
    robust = [dict(method=f"A0 · {cond}", n=v["n"], top1=v["top1"], top3=v["top3"], macro_f1=v["macro_f1"])
              for cond, v in a2.items()]
    sem = o["previous_extensions"]["A2-semantic"]["conditions"]["clean"]
    robust.append(dict(method="A2-semantic · FFHQ AI", n=sem["n"], top1=sem["top1"], top3=sem["top3"], macro_f1=sem["macro_f1"]))
    a0 = o["baseline_full_dev_separate_denominators"]["a0-resnet50"]
    return [
        dict(id="source-1331", task="source_classification", kind="source",
             title="22 来源闭集分类 · 同 1331 张已知 AI evaluation", rows=rows,
             paired=_pairs(o, SOURCE_PAIRS, ("top1", "top3", "macro_f1")) + fusion_pairs,
             notes=["闭集分数以“输入是已知 22 类 AI”为条件，不代表未知来源概率。",
                    f"分数融合 α_resnet={sf['alpha_resnet']} 只由 {sf['calibration_n']} 张 AI calibration 最小 NLL 确定；"
                    "固定 50/50 事前登记。虽 50/50 evaluation Top1 更高，不能据此改选权重。",
                    "A0 及其分数融合是父 dev 选模的探索参考，不是同等独立选模排名。",
                    "原型/KNN 的 epoch=null；KNN 选 k=1，Top3 部分依赖零票类别的并列排序。"]),
        dict(id="source-robustness", task="source_classification", kind="source",
             title="A0 来源鲁棒性（A2 六条件，同 1331 张）与 A2-semantic（132 张 FFHQ AI）", rows=robust, paired=[],
             notes=["A2-semantic 是来源分类，不是 93 张真实 FFHQ 的误报测试。"]),
        dict(id="source-full-dev", task="source_classification", kind="summary", title="A0 完整 4197 AI dev（另一分母，单列）",
             rows=[dict(method="A0 ResNet50", n=a0["n"], top1=a0["top1"], top3=a0["top3"], selected_epoch=a0["selected_epoch"],
                        train_n=a0["train_n"])], paired=[], notes=[]),
    ]


def unknown_groups(o: dict) -> list[dict]:
    u = o["unknown_rejection_1275"]
    rows = [dict(method=UNKNOWN_LABELS.get(k, k), n=v["n"], known_n=v["known_n"], unknown_n=v["unknown_n"],
                 threshold=v["threshold"], calibration_unknown_n=v["calibration_unknown_n"],
                 known_retention=v["known_retention"], known_retention_count=v["known_retention_count"],
                 known_correct_and_retained=v["known_correct_and_retained"],
                 known_correct_and_retained_count=v["known_correct_and_retained_count"],
                 unknown_false_acceptance=v["unknown_false_acceptance"],
                 unknown_false_acceptance_count=v["unknown_false_acceptance_count"],
                 unknown_false_acceptance_wilson95=v["unknown_false_acceptance_wilson95"],
                 unknown_rejection_count=v["unknown_rejection_count"], unknown_auroc=v["unknown_auroc"], oscr=v.get("oscr"))
            for k, v in u["methods"].items()]
    pu = u["paired_uncertainty"]
    paired = [dict(comparison=k.replace("_minus_", " − "), n=1275, group_n=pu["evaluation_groups"], repeats=pu["repeats"],
                   deltas={m: dict(delta=None, ci95=ci) for m, ci in v.items()}) for k, v in pu["delta_95_ci"].items()]
    ofa = o["baseline_full_dev_separate_denominators"]["ofa"]
    return [
        dict(id="unknown-1275", task="unknown_rejection", kind="unknown",
             title="未知拒识 · 20 已知类；calibration 64 张 kling，evaluation 63 张 grok3 + 1212 张已知", rows=rows,
             paired=paired, notes=["拒识分数不是未知来源概率；20 类阈值不移植到 22 类主接口。", u["note"],
                                   "OSCR 只报点值，没有区间。"]),
        dict(id="ofa-pair", task="source_verification", kind="summary", title="原 OFA 固定参考库配对验证 · 完整 4197 AI dev",
             rows=[dict(method="OFA 配对（每来源 100 个 train 参考）", n=ofa["query_n"], macro_auroc=ofa["macro_auroc"],
                        selected_epoch=ofa["epoch"])], paired=[],
             notes=["配对匹配分数不是来源概率，不与分类 softmax 混排。参考库固定，不能用上传图补库。"]),
    ]


def _methods(inventory: dict) -> list[dict]:
    out = []
    for m in inventory.get("defaults", []):
        out.append(dict(id=m["id"], task=m["task"], exposure=m["exposure"], offline_validation=m["offline_validation"],
                        architecture=m.get("architecture"), training_epochs=m.get("training_epochs"),
                        selected_epoch=m.get("selected_epoch")))
    for m in inventory.get("research_catalog", []):
        out.append(dict(id=m["id"], task=m["task"], exposure=m["exposure"], offline_validation=m["offline_validation"],
                        evidence_sha256=[e["sha256"] for e in m.get("evidence", [])]))
    return out


def build_catalog(settings: dict[str, Any]) -> dict[str, Any]:
    root, cat, errors = Path(settings["research_root"]), settings.get("catalog") or {}, []
    overview = _load(root, cat["overview"], "overview", errors) if cat.get("overview") else None
    inventory = _load(root, cat["inventory"], "inventory", errors) if cat.get("inventory") else None
    groups = []
    if overview:
        groups = detection_groups(overview) + source_groups(overview) + unknown_groups(overview)
    return dict(
        schema_version="experiments-v1", status="ok" if not errors else ("partial" if groups else "unavailable"),
        errors=errors, sources={k: dict(path=v["path"], sha256=v["sha256"]) for k, v in cat.items()},
        scope=overview["scope"] if overview else None, groups=groups,
        methods=_methods(inventory) if inventory else [],
        not_exposed=(inventory or {}).get("not_exposed", []),
        key_facts=KEY_FACTS, limitations=(overview or {}).get("limitations", []))
