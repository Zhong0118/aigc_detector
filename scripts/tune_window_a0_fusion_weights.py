#!/usr/bin/env python3
"""Fit fusion weights on frozen validation; score frozen test once.

Protocol:
- Do not train A1–A4.
- Search the 4-simplex on validation only (step 0.05, non-negative, sum=1).
- Primary metric: macro-F1; tie-break accuracy then equal-weight distance.
- Test is scored once after the weights are locked. Do not retune after seeing test.
- A0 probabilities use the production window-mean path, not truncated-head eval.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from provenance.window_a0_fusion import (
    DEFAULT_BRANCH_WEIGHTS,
    FAMILY_LABELS,
    WindowA0EqualFusion,
)
from research.text_attribution.split_manifest import apply_split_manifest, load_split_manifest
from research.text_attribution.training_data import load_training_dataset


BRANCHES = ("A0", "B0", "B1", "B3")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _macro_f1(pred: Sequence[str], true: Sequence[str], labels: Sequence[str]) -> tuple[float, float]:
    f1s: list[float] = []
    for lab in labels:
        tp = sum(p == lab and t == lab for p, t in zip(pred, true))
        fp = sum(p == lab and t != lab for p, t in zip(pred, true))
        fn = sum(p != lab and t == lab for p, t in zip(pred, true))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    accuracy = sum(p == t for p, t in zip(pred, true)) / len(true)
    return accuracy, float(sum(f1s) / len(f1s))


def _mix(branch_rows: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    mixed = np.zeros_like(next(iter(branch_rows.values())))
    for name, weight in weights.items():
        mixed = mixed + float(weight) * branch_rows[name]
    totals = mixed.sum(axis=1, keepdims=True)
    return mixed / np.clip(totals, 1e-12, None)


def _argmax_labels(matrix: np.ndarray, labels: Sequence[str]) -> list[str]:
    return [labels[int(index)] for index in np.argmax(matrix, axis=1)]


def _simplex(step: float) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    weights: list[dict[str, float]] = []
    for a, b, c in product(range(units + 1), repeat=3):
        d = units - a - b - c
        if d < 0:
            continue
        weights.append(
            {
                "A0": a / units,
                "B0": b / units,
                "B1": c / units,
                "B3": d / units,
            }
        )
    return weights


def _score_weights(
    branch_rows: Mapping[str, np.ndarray],
    true: Sequence[str],
    labels: Sequence[str],
    weights: Mapping[str, float],
) -> dict[str, float]:
    mixed = _mix(branch_rows, weights)
    pred = _argmax_labels(mixed, labels)
    acc, f1 = _macro_f1(pred, true, labels)
    equal_distance = sum(abs(weights[name] - 0.25) for name in BRANCHES)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "equal_l1": equal_distance,
        **{f"w_{name}": float(weights[name]) for name in BRANCHES},
    }


def _load_or_score_branches(
    *,
    fusion: WindowA0EqualFusion,
    samples: Sequence[Any],
    cache_path: Path,
    split: str,
) -> dict[str, Any]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("n") != len(samples) or payload.get("split") != split:
            raise ValueError(f"stale branch cache: {cache_path}")
        return payload
    keys: list[str] = []
    true: list[str] = []
    texts = [sample.text for sample in samples]
    fusion._ensure_loaded()
    stores = {name: [] for name in BRANCHES}
    from research.text_attribution.classical_models import predict_family_probabilities

    for name in ("B0", "B1", "B3"):
        matrix = predict_family_probabilities(
            fusion._classical[name], texts, fusion.family_labels
        )
        stores[name] = matrix.tolist()
        print(f"{split} {name} scored {len(texts)}", flush=True)
    for index, sample in enumerate(samples):
        stores["A0"].append(fusion._predict_a0(sample.text).tolist())
        keys.append(sample.request_key)
        true.append(sample.family_id)
        if (index + 1) % 50 == 0 or index + 1 == len(samples):
            print(f"{split} A0 scored {index + 1}/{len(samples)}", flush=True)
    payload = {
        "schema_version": 1,
        "split": split,
        "n": len(samples),
        "family_labels": list(FAMILY_LABELS),
        "request_keys": keys,
        "true_family": true,
        "branches": stores,
        "inference": "production_window_mean_A0_plus_sklearn_B0_B1_B3",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _arrays(payload: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    labels = list(payload["family_labels"])
    branches = {
        name: np.asarray(payload["branches"][name], dtype=np.float64) for name in BRANCHES
    }
    return branches, list(payload["true_family"]), labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/attribution/family-window-split/fusion"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_training_dataset(
        PROJECT_ROOT / "data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl",
        PROJECT_ROOT / "configs/text_attribution/zhizengzeng_model_catalog.v5.json",
    )
    manifest = load_split_manifest(
        PROJECT_ROOT / "data/attribution/splits/family-attribution-v5-seed42.json"
    )
    splits = apply_split_manifest(dataset.samples, manifest)
    fusion = WindowA0EqualFusion({"device": "cuda:0" if _cuda() else "cpu"})

    val_cache = output_dir / "seed42-branch-probs-validation.json"
    test_cache = output_dir / "seed42-branch-probs-test.json"
    validation = _load_or_score_branches(
        fusion=fusion,
        samples=splits["validation"],
        cache_path=val_cache,
        split="validation",
    )
    val_branches, val_true, labels = _arrays(validation)

    candidates = []
    for weights in _simplex(args.step):
        scored = _score_weights(val_branches, val_true, labels, weights)
        scored["weights"] = weights
        candidates.append(scored)
    equal = _score_weights(val_branches, val_true, labels, DEFAULT_BRANCH_WEIGHTS)
    ranked = sorted(
        candidates,
        key=lambda row: (-row["macro_f1"], -row["accuracy"], row["equal_l1"]),
    )
    selected = ranked[0]

    # Lock weights before touching test.
    locked = {
        "schema_version": 1,
        "seed": 42,
        "fit_split": "validation",
        "eval_split": "test",
        "view": "closed",
        "step": args.step,
        "n_candidates": len(candidates),
        "weights": selected["weights"],
        "validation": {
            "selected": {k: selected[k] for k in ("accuracy", "macro_f1", "equal_l1")},
            "equal": {k: equal[k] for k in ("accuracy", "macro_f1")},
            "n": len(val_true),
        },
        "protocol": (
            "Weights fitted on frozen validation only. Test is scored once after lock. "
            "Do not retune after reading test metrics."
        ),
        "test_evaluated": False,
    }
    weights_path = output_dir / "seed42-weights.json"
    weights_path.write_text(json.dumps(locked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("locked validation weights", json.dumps(locked["weights"]), flush=True)
    print("validation selected", locked["validation"]["selected"], "equal", locked["validation"]["equal"], flush=True)

    test = _load_or_score_branches(
        fusion=fusion,
        samples=splits["test"],
        cache_path=test_cache,
        split="test",
    )
    test_branches, test_true, _ = _arrays(test)
    test_selected = _score_weights(test_branches, test_true, labels, selected["weights"])
    test_equal = _score_weights(test_branches, test_true, labels, DEFAULT_BRANCH_WEIGHTS)
    locked["test"] = {
        "selected": {k: test_selected[k] for k in ("accuracy", "macro_f1")},
        "equal": {k: test_equal[k] for k in ("accuracy", "macro_f1")},
        "n": len(test_true),
    }
    locked["test_evaluated"] = True
    weights_path.write_text(json.dumps(locked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# 切窗 A0 + B0/B1/B3 权重（seed42）",
        "",
        "协议：权重只在冻结 validation（912）上搜 simplex；**test（793）只评一次**。",
        "A0 用生产推理（非重叠 512 窗平均），不是训练时的截断 head eval。",
        "B0/B1/B3 是 sklearn 统计分类器，不是手写规则。",
        "",
        f"- 搜索步长：{args.step}，候选 {len(candidates)} 组，非负且和为 1。",
        f"- 选中权重：A0={selected['weights']['A0']:.2f}, B0={selected['weights']['B0']:.2f}, "
        f"B1={selected['weights']['B1']:.2f}, B3={selected['weights']['B3']:.2f}",
        "",
        "| 系统 | val Acc | val F1 | test Acc | test F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| 等权 0.25 | {equal['accuracy']:.3f} | {equal['macro_f1']:.3f} | {test_equal['accuracy']:.3f} | {test_equal['macro_f1']:.3f} |",
        f"| **val 选中权重** | **{selected['accuracy']:.3f}** | **{selected['macro_f1']:.3f}** | **{test_selected['accuracy']:.3f}** | **{test_selected['macro_f1']:.3f}** |",
        "",
        "看完 test 不再改权重。产物：`seed42-weights.json`。",
        "",
    ]
    (output_dir / "seed42-tuned-weights.md").write_text("\n".join(report), encoding="utf-8")
    print("".join(line + "\n" for line in report))
    return 0


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
