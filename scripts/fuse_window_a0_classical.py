#!/usr/bin/env python3
"""Historical equal-weight validation comparison, NOT a verified fusion study.

The inputs are old prediction files with no complete dataset/model provenance.
The output is explicitly marked unverified and must not authorize production
weights, final test metrics, or a validated research claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_jsonl(path: Path, experiment: str | None = None, split: str | None = "validation") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if experiment and row.get("experiment_id") != experiment:
                continue
            if split and row.get("split") != split:
                continue
            if row["request_key"] in rows:
                raise ValueError(f"duplicate request key in {path}: {row['request_key']}")
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


def validate_alignment(stores: Mapping[str, Mapping[str, dict[str, Any]]]) -> tuple[list[str], list[str]]:
    if not stores or not all(stores.values()):
        raise ValueError("aligned prediction stores must be nonempty")
    reference = next(iter(stores.values()))
    keys = sorted(reference)
    labels = list(reference[keys[0]]["family_probabilities"])
    for name, rows in stores.items():
        if set(rows) != set(keys):
            raise ValueError(f"request keys do not align for {name}")
        for key in keys:
            row = rows[key]
            if row["request_key"] != key or row["true_family"] != reference[key]["true_family"]:
                raise ValueError(f"request key/true label mismatch for {name}: {key}")
            if list(row["family_probabilities"]) != labels or row["true_family"] not in labels:
                raise ValueError(f"family label order mismatch for {name}: {key}")
    return keys, labels


def mix(
    keys: Sequence[str],
    stores: Mapping[str, Mapping[str, dict[str, Any]]],
    weights: Mapping[str, float],
) -> list[str]:
    aligned_keys, _ = validate_alignment(stores)
    if len(keys) != len(aligned_keys) or set(keys) != set(aligned_keys):
        raise ValueError("fusion keys must exactly cover aligned predictions")
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
    parser.add_argument(
        "--acknowledge-unverified-inputs",
        action="store_true",
        help="Allow a historical comparison of prediction files without full provenance.",
    )
    args = parser.parse_args()
    if not args.acknowledge_unverified_inputs:
        parser.error("offline prediction files are unverified; pass --acknowledge-unverified-inputs")
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
    stores = {"A0w": window, "A0t": trunc, "B0": b0, "B1": b1, "B3": b3}
    keys, labels = validate_alignment(stores)
    true = [window[key]["true_family"] for key in keys]

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
    payload = {
        "schema_version": 2,
        "seed": args.seed,
        "split": "validation",
        "view": "closed",
        "provenance_verified": False,
        "purpose": "historical_comparison_only",
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "A0_window": root / f"outputs/attribution/family-window-split/A0-closed-seed{args.seed}/predictions.jsonl",
                "A0_truncated": root / f"outputs/attribution/family-phase3/A0/A0-closed-seed{args.seed}/predictions.jsonl",
                "B0": root / f"outputs/attribution/family-phase4/fast-track-v1/classical/closed-seed{args.seed}/B0/predictions.jsonl",
                "B1": root / f"outputs/attribution/family-phase4/fast-track-v1/classical/closed-seed{args.seed}/B1/predictions.jsonl",
                "B3": root / "outputs/attribution/family-phase1/final/predictions.jsonl",
            }.items()
        },
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
