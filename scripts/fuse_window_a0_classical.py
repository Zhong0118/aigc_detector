#!/usr/bin/env python3
"""Equal-weight fusion of window-split A0 with classical branches on frozen val.

Weights are not tuned on this split. B3 predictions come from the Phase1 seed-42
artifact and are only comparable when request keys align.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_jsonl(path: Path, experiment: str | None = None, split: str | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if experiment and row.get("experiment_id") != experiment:
                continue
            if split and row.get("split") != split:
                continue
            rows[row["request_key"]] = row
    return rows


def macro_f1(pred: Sequence[str], true: Sequence[str], labels: Sequence[str]) -> tuple[float, float]:
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


def mix(
    keys: Sequence[str],
    stores: Mapping[str, Mapping[str, dict[str, Any]]],
    weights: Mapping[str, float],
) -> list[str]:
    pred: list[str] = []
    for key in keys:
        scores: dict[str, float] = defaultdict(float)
        for name, weight in weights.items():
            if weight == 0:
                continue
            for label, value in stores[name][key]["family_probabilities"].items():
                scores[label] += weight * float(value)
        pred.append(max(scores, key=scores.get))
    return pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    window = load_jsonl(
        root / f"outputs/attribution/family-window-split/A0-closed-seed{args.seed}/predictions.jsonl"
    )
    trunc = load_jsonl(
        root / f"outputs/attribution/family-phase3/A0/A0-closed-seed{args.seed}/predictions.jsonl"
    )
    b0 = load_jsonl(
        root
        / f"outputs/attribution/family-phase4/fast-track-v1/classical/closed-seed{args.seed}/B0/predictions.jsonl"
    )
    b1 = load_jsonl(
        root
        / f"outputs/attribution/family-phase4/fast-track-v1/classical/closed-seed{args.seed}/B1/predictions.jsonl"
    )
    b3 = load_jsonl(
        root / "outputs/attribution/family-phase1/final/predictions.jsonl",
        experiment="B3",
        split="validation",
    )
    e1_metrics = json.loads(
        (
            root
            / f"outputs/attribution/family-phase4/fast-track-v1/E1-fusion/closed-seed{args.seed}/metrics.json"
        ).read_text(encoding="utf-8")
    )
    keys = sorted(set(window) & set(trunc) & set(b0) & set(b1) & set(b3))
    labels = list(next(iter(window.values()))["family_probabilities"])
    true = [window[key]["true_family"] for key in keys]
    stores = {"A0w": window, "A0t": trunc, "B0": b0, "B1": b1, "B3": b3}

    systems = {
        "trunc_A0": [trunc[k]["predicted_family"] for k in keys],
        "window_A0": [window[k]["predicted_family"] for k in keys],
        "B0": [b0[k]["predicted_family"] for k in keys],
        "B1": [b1[k]["predicted_family"] for k in keys],
        "B3": [b3[k]["predicted_family"] for k in keys],
        "window_A0+B0+B1_equal": mix(keys, {"A0": window, "B0": b0, "B1": b1}, {"A0": 1 / 3, "B0": 1 / 3, "B1": 1 / 3}),
        "window_A0+B0+B1+B3_equal": mix(
            keys,
            {"A0": window, "B0": b0, "B1": b1, "B3": b3},
            {"A0": 0.25, "B0": 0.25, "B1": 0.25, "B3": 0.25},
        ),
        "window_A0+B3_equal": mix(keys, {"A0": window, "B3": b3}, {"A0": 0.5, "B3": 0.5}),
    }
    table = {}
    for name, pred in systems.items():
        acc, f1 = macro_f1(pred, true, labels)
        table[name] = {"accuracy": acc, "macro_f1": f1, "n": len(keys)}
    table["E1_A4+B0+B1_this_seed"] = {
        "accuracy": float(e1_metrics["overall"]["accuracy"]),
        "macro_f1": float(e1_metrics["overall"]["macro_f1"]),
        "n": int(e1_metrics["overall"]["support"]),
        "note": "Phase4 Fast Track on truncated A4, not window A0",
    }
    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "split": "validation",
        "view": "closed",
        "weights": "equal; not tuned on this split",
        "b3_trained_on_seed": 42,
        "n_aligned": len(keys),
        "systems": table,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
