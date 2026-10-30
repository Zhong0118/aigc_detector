"""Read-only experiment catalog from the sealed 2026-10-30 final overview (hash-checked).

Only aggregate JSON is read: no images, per-image predictions, re-scoring, threshold or
alpha refitting. Paired deltas are quoted from the registered paired reports; rows with
different denominators are grouped separately and never turned into "improvements".
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from workbench.image_engine import sha256_file

KEY_FACTS_PRIMARY = 4  # always visible above the tables; the rest sit one click away
KEY_FACTS = [
    "held-FFHQ 同 93 张真实图：OpenCLIP 固定 0.5 误报 15 张（16.13%），ResNet 与 ConvNeXt 各 56 张（60.22%）。",
    "按非 FFHQ calibration 目标 1% 选阈值后，OpenCLIP 在 FFHQ 仍误报 7/93（7.53%，Wilson 95% 约 3.69–14.73%），不是跨域 1% 保证。",
    "OFA20+1 拒绝 63/63 未知图，但已知图只保留 58/1212（4.79%），正确且保留 28/1212（2.31%）；不能只宣传 100% 未知拒识。",
    "A0 及部分旧模型的父 dev 参与选模；全部为单 seed 开发结果，不是独立测试。",
    "0/63 未知误接收的 Wilson 95% 上界仍约 5.75%，不能声称误接收恒为 0。",
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

# Result rows -> inventory method ids, whose hash-pinned evidence holds the training metadata.
DET_IDS = {"OpenCLIP": "det-openclip-q179-v1", "ResNet-control": "research-resnet-control",
           "OmniAID": "research-omniaid-v2", "ConvNeXt": "research-convnext-detection", "DINOv2": "research-dino-detection"}
FFHQ_IDS = {"ResNet-heldFFHQ": "research-resnet-ffhq", "OpenCLIP-heldFFHQ": "research-openclip-ffhq",
            "ConvNeXt-heldFFHQ": "research-convnext-ffhq"}
SOURCE_IDS = {"ResNet-A0-parent-dev-selected": "attr-resnet50-q179-v1", "A-RGB": "research-rgb22",
              "A-FP": "research-fingerprint22", "A-Fusion": "research-feature-fusion22",
              "OpenCLIP-linear": "research-openclip-source22", "OpenCLIP-KNN": "research-openclip-knn22",
              "ConvNeXt-linear": "research-convnext-source22", "DNA-Det-linear": "research-dna-lp22",
              "OmniDFA-linear": "research-omnidfa-lp22", "OmniDFA-K10-prototype": "research-omnidfa-k10",
              "score-fusion-ResNet50": "attr-resnet50-q179-v1", "score-fusion-FeatureFusion": "research-feature-fusion22",
              "score-fusion-Uniform": "research-score-fusion-uniform",
              "score-fusion-Calibrated": "research-score-fusion-calibrated"}
UNKNOWN_IDS = {"resnet_msp": "research-resnet-open20", "openclip_msp": "research-openclip-open20",
               "openclip_prototype": "research-openclip-open20", "ofa_tplus1_known_mass": "research-ofa-tplus1"}
# Protocol wording per method, and why epochs do not apply (registered as null by design).
PROTOCOLS = {
    "det-openclip-q179-v1": "检测 · clean", "research-convnext-detection": "检测 · clean",
    "research-dino-detection": "检测 · clean", "research-resnet-control": "检测 · clean（父 dev 选模）",
    "research-omniaid-v2": "检测 · 外部预训练模型（未在本数据训练）",
    "research-openclip-ffhq": "检测 · held-FFHQ（训练与选择排除 FFHQ）",
    "research-convnext-ffhq": "检测 · held-FFHQ（训练与选择排除 FFHQ）",
    "research-resnet-ffhq": "检测 · held-FFHQ（D2-semantic，父 dev 选模）",
    "attr-resnet50-q179-v1": "22 类来源（父 dev 参与选模）", "research-dna-lp22": "22 类来源",
    "research-convnext-source22": "22 类来源", "research-openclip-source22": "22 类来源",
    "research-omnidfa-lp22": "22 类来源", "research-rgb22": "22 类来源", "research-fingerprint22": "22 类来源",
    "research-feature-fusion22": "22 类来源", "research-openclip-knn22": "22 类来源 · KNN（k=1）",
    "research-omnidfa-k10": "22 类来源 · 原型（K=10）", "research-score-fusion-uniform": "22 类来源 · 分数融合 50/50",
    "research-score-fusion-calibrated": "22 类来源 · 分数融合（calibration 选 α）",
    "research-openclip-open20": "20 已知类 + 未知拒识", "research-ofa-tplus1": "20 已知类 + 未知拒识",
    "research-resnet-open20": "20 已知类 + 未知拒识（父 dev 选模）",
    "research-ofa-pair": "来源配对验证 · 完整 4197 AI dev",
}
NOT_APPLICABLE = {
    "research-omniaid-v2": "冻结外部模型，未在本数据训练", "research-openclip-knn22": "KNN 无训练轮次",
    "research-omnidfa-k10": "原型均值，无训练轮次", "research-score-fusion-uniform": "固定权重，无训练轮次",
    "research-score-fusion-calibrated": "只在 calibration 上选 α，无训练轮次",
}
BACKBONE_OVERRIDES = {"research-omniaid-v2": "冻结外部模型，不训练", "research-openclip-knn22": "冻结骨干，不训练头",
                      "research-omnidfa-k10": "冻结骨干，不训练头", "research-score-fusion-uniform": "不适用（只融合已有分数）",
                      "research-score-fusion-calibrated": "不适用（只融合已有分数）"}
ROW_NOTES = {"score-fusion-ResNet50": "融合输入，即 A0 本身。", "score-fusion-FeatureFusion": "融合输入，即 A-Fusion 本身。"}
# The open20 prototype row shares its run (research-openclip-open20) with the MSP row, but its rejection score is
# the cosine to train-feature class means: no gradient training and no selected epoch. The run's registered 30/29
# belong to the open20 linear head, which still supplies the known-class prediction behind "correct and retained".
PROTOTYPE_NOTE = "类原型为 train 特征 L2 归一化均值，无梯度训练轮次"
ROW_META = {"openclip_prototype": dict(
    backbone="冻结骨干，不训练；类原型 = train 特征均值",
    training_epochs=dict(value=None, status="not_applicable", note=PROTOTYPE_NOTE),
    selected_epoch=dict(value=None, status="not_applicable", note=PROTOTYPE_NOTE),
    support_head=dict(name="配套 open20 线性分类头", role="提供已知类预测（用于“正确且保留”）"))}
METHOD_NOTES = {"research-openclip-open20": "epoch 属于 open20 线性分类头（MSP 行及原型行的已知类预测）；原型拒识分数本身无训练轮次。"}
TRAINING_EPOCH_KEYS = ("training_epochs", "epochs", "training.epochs", "config.training.epochs",
                       "experiment.training_epochs", "config.protocol.epochs")
SELECTED_EPOCH_KEYS = ("selected_epoch", "epoch", "best_epoch")


def _load(root: Path, spec: dict, name: str, errors: list) -> Any:
    path = root / spec["path"]
    if not path.is_file():
        errors.append(dict(source=name, code="model_unavailable", message="登记结果文件缺失"))
        return None
    if sha256_file(path) != spec["sha256"]:
        errors.append(dict(source=name, code="artifact_mismatch", message="登记结果文件哈希不符"))
        return None
    return json.loads(path.read_text())


def _lookup(docs: list[dict], keys: tuple[str, ...], accept=lambda value: True) -> tuple[bool, Any]:
    """(present, value) for the first dotted key present in any evidence document; null stays null."""
    for doc in docs:
        for key in keys:
            node, present = doc, True
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    present = False
                    break
            if present and accept(node):
                return True, node
    return False, None


def _epoch_value(value: Any) -> bool:
    return value is None or type(value) is int  # per-epoch histories (lists/dicts) are not an epoch count


def _epoch(present: bool, value: Any, not_applicable: str | None) -> dict:
    if present and type(value) is int:
        return dict(value=value, status="registered")
    if not_applicable:
        return dict(value=None, status="not_applicable", note=not_applicable)
    return dict(value=None, status="not_registered")


UNREGISTERED = dict(protocol="未登记", train_n=None, selection_n=None, backbone="未登记",
                    training_epochs=dict(value=None, status="not_registered"),
                    selected_epoch=dict(value=None, status="not_registered"), notes=[])


def method_metadata(root: Path, inventory: dict, errors: list) -> dict[str, dict]:
    """Training scope and epochs from each method's hash-pinned evidence in the sealed inventory.
    Nothing is re-trained, re-scored or guessed: a value is shown only when a sealed file records it."""
    out = {}
    for m in inventory.get("defaults", []) + inventory.get("research_catalog", []):
        docs, broken = [], False
        for e in m.get("evidence", []):
            path = root / e["path"]
            if not path.is_file() or sha256_file(path) != e["sha256"]:
                broken = True
                continue
            if path.suffix == ".json":
                docs.append(json.loads(path.read_text()))
        if broken:
            errors.append(dict(source=m["id"], code="artifact_mismatch",
                               message="方法证据文件缺失或哈希不符，已不再使用该文件；仍有效的证据和已通过哈希校验的 "
                                       "inventory 中明确登记的字段照常显示，无法证实的字段显示未登记"))
        docs.append({k: m[k] for k in ("training_epochs", "selected_epoch") if k in m})  # inventory's own fields last
        frozen = _lookup(docs, ("backbone_frozen",))
        encoder = _lookup(docs, ("encoder_training",))
        notes = [n for n in (m.get("selection_bias"), _lookup(docs, ("evaluation_note",))[1]) if n]
        out[m["id"]] = dict(
            protocol=PROTOCOLS.get(m["id"], "未登记"),
            train_n=_lookup(docs, ("role_counts.train", "config.expected_role_counts.train", "train_n"))[1],
            selection_n=_lookup(docs, ("role_counts.selection", "config.expected_role_counts.selection"))[1],
            backbone=BACKBONE_OVERRIDES.get(m["id"]) or (
                "冻结骨干，只训练头" if frozen == (True, True) or encoder == (True, False) else "未登记"),
            training_epochs=_epoch(*_lookup(docs, TRAINING_EPOCH_KEYS, _epoch_value), NOT_APPLICABLE.get(m["id"])),
            selected_epoch=_epoch(*_lookup(docs, SELECTED_EPOCH_KEYS, _epoch_value), NOT_APPLICABLE.get(m["id"])),
            notes=notes)
    return out


def _with_meta(row: dict, meta: dict, method_id: str | None, note: str | None = None,
               override: dict | None = None) -> dict:
    # Deep copies: rows of one run (MSP and prototype) and the method table never share mutable metadata.
    info = copy.deepcopy((meta.get(method_id) if method_id else None) or UNREGISTERED)
    row.update(method_id=method_id, protocol=info["protocol"], train_n=row.get("train_n", info["train_n"]),
               selection_n=info["selection_n"], backbone=info["backbone"],
               training_epochs=info["training_epochs"], selected_epoch=info["selected_epoch"],
               meta_notes=info["notes"] + ([note] if note else []))
    override = copy.deepcopy(override or {})
    if "support_head" in override:  # the run's registered epochs move to the head they belong to
        override["support_head"].update(training_epochs=info["training_epochs"], selected_epoch=info["selected_epoch"])
    row.update(override)
    return row


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


def detection_groups(o: dict, meta: dict) -> list[dict]:
    clean = o["clean_detection_2310"]
    rows = [_with_meta(_det_row(DET_LABELS[k], DET_ROLES[k], v, "clean"), meta, DET_IDS.get(k)) for k, v in clean.items()]
    ffhq = o["held_ffhq_225"]
    ffhq_rows, cal_rows = [], []
    for name, m in ffhq["methods"].items():
        ffhq_rows.append(_with_meta(_det_row(name, "comparison", m["fixed_0_5"], "held_ffhq",
                                             dict(threshold_id="fixed-0.5")), meta, FFHQ_IDS.get(name)))
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
        ext.append(_with_meta(_det_row(name, "comparison", v, "previous_extension"), meta,
                              "research-resnet-control" if name == "D1-control" else None))
    d0 = o["baseline_full_dev_separate_denominators"]["d0-resnet50"]
    return [
        dict(id="detection-clean", task="detection", kind="detection", title="真假检测 · 同 2310 张 evaluation（979 真实 / 1331 AI），固定阈值 0.5",
             rows=rows, paired=_pairs(o, DET_PAIRS, ("auroc", "fpr_0_5", "tpr_0_5")),
             notes=["表中分数均为 AI 类分数，不是校准概率。", "ResNet-control 与 D3 曾有完全相同的真实图概率向量，不能当独立复现。",
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
                        protocol="检测 · 完整 dev 7189（另一分母）", train_n=d0["train_n"],
                        training_epochs=dict(value=None, status="not_registered"),
                        selected_epoch=_epoch(True, d0["selected_epoch"], None))], paired=[], notes=[]),
    ]


def source_groups(o: dict, meta: dict) -> list[dict]:
    rows = []
    for key, v in o["source_classification_1331"].items():
        label, role = SOURCE_LABELS.get(key, (key, "comparison"))
        rows.append(_with_meta(dict(method=label, role=role, n=v["n"], top1=v["top1"], top3=v["top3"],
                                    macro_f1=v["macro_f1"], top1_correct=round(v["top1"] * v["n"]),
                                    top3_hit=round(v["top3"] * v["n"])), meta, SOURCE_IDS.get(key), ROW_NOTES.get(key)))
    sf = o["score_fusion_calibration_only"]
    fusion_pairs = [dict(comparison=f"分数融合 75/25 − {name}", n=sf["evaluation_n"], group_n=v["group_n"],
                         repeats=v["repeats"],
                         deltas={m: dict(delta=sf["metrics"]["Calibrated"][m] - sf["metrics"][name][m], ci95=ci)
                                 for m, ci in v["delta_95_ci"].items()})
                    for name, v in sf["paired_group_bootstrap"].items()]
    a2 = o["previous_extensions"]["A2"]["conditions"]
    robust = [_with_meta(dict(method=f"A0 · {cond}", n=v["n"], top1=v["top1"], top3=v["top3"], macro_f1=v["macro_f1"]),
                         meta, None) for cond, v in a2.items()]
    sem = o["previous_extensions"]["A2-semantic"]["conditions"]["clean"]
    robust.append(_with_meta(dict(method="A2-semantic · FFHQ AI", n=sem["n"], top1=sem["top1"], top3=sem["top3"],
                                  macro_f1=sem["macro_f1"]), meta, None))
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
             rows=[dict(method="A0 ResNet50", n=a0["n"], top1=a0["top1"], top3=a0["top3"],
                        protocol="22 类来源 · 完整 AI dev 4197（另一分母，父 dev 选模）", train_n=a0["train_n"],
                        training_epochs=copy.deepcopy((meta.get("attr-resnet50-q179-v1") or UNREGISTERED)["training_epochs"]),
                        selected_epoch=_epoch(True, a0["selected_epoch"], None))], paired=[], notes=[]),
    ]


def unknown_groups(o: dict, meta: dict) -> list[dict]:
    u = o["unknown_rejection_1275"]
    rows = [_with_meta(dict(method=UNKNOWN_LABELS.get(k, k), n=v["n"], known_n=v["known_n"], unknown_n=v["unknown_n"],
                 threshold=v["threshold"], calibration_unknown_n=v["calibration_unknown_n"],
                 known_retention=v["known_retention"], known_retention_count=v["known_retention_count"],
                 known_correct_and_retained=v["known_correct_and_retained"],
                 known_correct_and_retained_count=v["known_correct_and_retained_count"],
                 unknown_false_acceptance=v["unknown_false_acceptance"],
                 unknown_false_acceptance_count=v["unknown_false_acceptance_count"],
                 unknown_false_acceptance_wilson95=v["unknown_false_acceptance_wilson95"],
                 unknown_rejection_count=v["unknown_rejection_count"], unknown_auroc=v["unknown_auroc"], oscr=v.get("oscr")),
                       meta, UNKNOWN_IDS.get(k), ROW_NOTES.get(k), ROW_META.get(k))
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
             rows=[_with_meta(dict(method="OFA 配对（每来源 100 个 train 参考）", n=ofa["query_n"],
                                   macro_auroc=ofa["macro_auroc"]), meta, "research-ofa-pair")], paired=[],
             notes=["配对匹配分数不是来源概率，不与分类 softmax 混排。参考库固定，不能用上传图补库。"]),
    ]


def _methods(inventory: dict, meta: dict) -> list[dict]:
    out = []
    for m in inventory.get("defaults", []) + inventory.get("research_catalog", []):
        info = copy.deepcopy(meta.get(m["id"]) or UNREGISTERED)
        if m["id"] in METHOD_NOTES:
            info["notes"].append(METHOD_NOTES[m["id"]])
        out.append(dict(id=m["id"], task=m["task"], exposure=m["exposure"], offline_validation=m["offline_validation"],
                        architecture=m.get("architecture"), evidence_sha256=[e["sha256"] for e in m.get("evidence", [])],
                        **info))
    return out


def build_catalog(settings: dict[str, Any]) -> dict[str, Any]:
    root, cat, errors = Path(settings["research_root"]), settings.get("catalog") or {}, []
    overview = _load(root, cat["overview"], "overview", errors) if cat.get("overview") else None
    inventory = _load(root, cat["inventory"], "inventory", errors) if cat.get("inventory") else None
    meta = method_metadata(root, inventory, errors) if inventory else {}
    groups = []
    if overview:
        groups = detection_groups(overview, meta) + source_groups(overview, meta) + unknown_groups(overview, meta)
    return dict(
        schema_version="experiments-v2", status="ok" if not errors else ("partial" if groups else "unavailable"),
        errors=errors, sources={k: dict(path=v["path"], sha256=v["sha256"]) for k, v in cat.items()},
        scope=overview["scope"] if overview else None, groups=groups,
        methods=_methods(inventory, meta) if inventory else [],
        not_exposed=(inventory or {}).get("not_exposed", []),
        key_facts=KEY_FACTS, key_facts_primary=KEY_FACTS_PRIMARY, limitations=(overview or {}).get("limitations", []))
