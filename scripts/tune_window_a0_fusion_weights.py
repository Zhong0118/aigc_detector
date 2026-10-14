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
import hashlib
import math
import json
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from provenance.window_a0_fusion import (
    DEFAULT_BRANCH_WEIGHTS,
    DEFAULT_PATHS,
    load_branch_weights,
    _resolve_artifact,
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
    if not math.isfinite(step) or not 0 < step <= 1:
        raise ValueError("simplex step must be finite and in (0, 1]")
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("simplex step must divide one exactly")
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


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inference_identity(fusion: WindowA0EqualFusion) -> dict[str, Any]:
    artifacts = {}
    for name, default in DEFAULT_PATHS.items():
        path = _resolve_artifact(fusion.cfg.get(name), default)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"empty model artifact: {path}")
        artifacts[name] = {str(p.relative_to(path)) if path.is_dir() else p.name: _hash_file(p) for p in files}
    return {
        "artifacts": artifacts,
        "family_labels": list(fusion.family_labels),
        "max_length": fusion.max_length,
        "device": fusion.device_name,
        "window_policy": "nonoverlap_decode_retokenize_probability_mean_v1",
        "runtime_versions": {name: version(name) for name in ("numpy", "torch", "transformers", "scikit-learn")},
        "inference_sources": {name: _hash_file(PROJECT_ROOT / name) for name in (
            "provenance/window_a0_fusion.py",
            "research/text_attribution/window_split.py",
            "research/text_attribution/phase3_data.py",
            "research/text_attribution/hierarchical_model.py",
            "research/text_attribution/classical_models.py",
        )},
    }


def _load_or_score_branches(
    *,
    fusion: WindowA0EqualFusion,
    samples: Sequence[Any],
    cache_path: Path,
    split: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2 or not payload.get("provenance"):
            raise ValueError(f"stale branch cache without provenance: {cache_path}")
    identity = dict(provenance or {"inference": _inference_identity(fusion)})
    identity["ordered_samples_sha256"] = _hash_json([
        [sample.request_key, sample.text, sample.family_id] for sample in samples
    ])
    keys = [sample.request_key for sample in samples]
    true = [sample.family_id for sample in samples]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("branch samples must be nonempty with unique request keys")
    if cache_path.exists():
        if (payload.get("n") != len(samples) or payload.get("split") != split
                or payload.get("provenance") != identity
                or payload.get("request_keys") != keys or payload.get("true_family") != true
                or payload.get("family_labels") != list(fusion.family_labels)):
            raise ValueError(f"stale branch cache: {cache_path}")
        _arrays(payload)
        return payload
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
        if (index + 1) % 50 == 0 or index + 1 == len(samples):
            print(f"{split} A0 scored {index + 1}/{len(samples)}", flush=True)
    payload = {
        "schema_version": 2,
        "provenance": identity,
        "split": split,
        "n": len(samples),
        "family_labels": list(fusion.family_labels),
        "request_keys": keys,
        "true_family": true,
        "branches": stores,
        "inference": "production_window_mean_A0_plus_sklearn_B0_B1_B3",
    }
    _arrays(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False)
    return payload


def _arrays(payload: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    labels = list(payload["family_labels"])
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("branch cache family labels must be nonempty and unique")
    true = list(payload["true_family"])
    if not true or any(label not in labels for label in true):
        raise ValueError("branch cache true labels must be nonempty catalog labels")
    keys = payload.get("request_keys")
    if (not isinstance(keys, list) or len(keys) != len(true)
            or len(set(keys)) != len(keys) or len(true) != payload.get("n")):
        raise ValueError("branch cache request keys and labels are misaligned")
    branches = {
        name: np.asarray(payload["branches"][name], dtype=np.float64) for name in BRANCHES
    }
    for name, matrix in branches.items():
        if matrix.shape != (len(true), len(labels)):
            raise ValueError(f"branch cache {name} shape does not align with keys and labels")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError(f"branch cache {name} contains invalid probabilities")
        if len(true) and not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"branch cache {name} rows must sum to one")
    return branches, true, labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--mode", choices=("validation", "evaluate-test"), default="validation")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/attribution/family-window-split/fusion"),
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "seed42-weights.json"
    receipt_path = output_dir / "seed42-test-evaluation.json"
    if args.mode == "validation" and (weights_path.exists() or receipt_path.exists()):
        raise FileExistsError("fusion lock/evaluation already exists; refusing to retune or overwrite")
    if args.mode == "evaluate-test":
        if receipt_path.exists():
            raise FileExistsError("test evaluation already attempted; refusing duplicate evaluation")
        locked = json.loads(weights_path.read_text(encoding="utf-8"))
        if locked.get("schema_version") != 2 or locked.get("fit_split") != "validation" or not locked.get("provenance"):
            raise ValueError("unverified legacy fusion lock; explicit migration is required")
        weights = load_branch_weights(locked["weights"])

    data_path = PROJECT_ROOT / "data/attribution/raw/zhizengzeng-attribution-server-6k-v5.jsonl"
    catalog_path = PROJECT_ROOT / "configs/text_attribution/zhizengzeng_model_catalog.v5.json"
    split_path = PROJECT_ROOT / "data/attribution/splits/family-attribution-v5-seed42.json"
    manifest = load_split_manifest(split_path)
    split = "validation" if args.mode == "validation" else "test"
    # Full data is loaded solely to verify the frozen manifest fingerprint.
    # Validation mode never forwards, caches predictions for, or scores test rows.
    dataset = load_training_dataset(data_path, catalog_path)
    samples = apply_split_manifest(dataset.samples, manifest)[split]
    fusion = WindowA0EqualFusion({"device": "cuda:0" if _cuda() else "cpu"})
    provenance = {
        "dataset_sha256": _hash_file(data_path),
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "manifest_sha256": _hash_file(split_path),
        "catalog_sha256": _hash_file(catalog_path),
        "inference": _inference_identity(fusion),
    }
    if args.mode == "evaluate-test":
        if locked["provenance"] != provenance or locked.get("family_labels") != list(fusion.family_labels):
            raise ValueError("fusion lock provenance does not match current data/models/inference")
        # Reserve before touching test predictions. A failed attempt remains reserved:
        # the operator must investigate rather than silently evaluating test again.
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump({"status": "started", "lock_sha256": _hash_file(weights_path)}, handle)

    payload = _load_or_score_branches(
        fusion=fusion, samples=samples,
        cache_path=output_dir / f"seed42-branch-probs-{split}.json",
        split=split, provenance=provenance,
    )
    branches, true, labels = _arrays(payload)
    equal = _score_weights(branches, true, labels, DEFAULT_BRANCH_WEIGHTS)
    if args.mode == "evaluate-test":
        selected = _score_weights(branches, true, labels, weights)
        result = {
            "schema_version": 2, "status": "complete", "split": "test",
            "lock_sha256": _hash_file(weights_path), "n": len(true),
            "provenance": payload["provenance"], "selected": selected, "equal": equal,
        }
        receipt_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    candidates = []
    for weights in _simplex(args.step):
        scored = _score_weights(branches, true, labels, weights)
        scored["weights"] = weights
        candidates.append(scored)
    selected = min(candidates, key=lambda row: (-row["macro_f1"], -row["accuracy"], row["equal_l1"]))
    locked = {
        "schema_version": 2, "seed": 42, "fit_split": "validation", "view": "closed",
        "step": args.step, "n_candidates": len(candidates),
        "weights": selected["weights"], "family_labels": labels,
        "provenance": provenance, "validation_cache_provenance": payload["provenance"],
        "validation": {"selected": selected, "equal": equal, "n": len(true)},
        "protocol": "Validation-only lock. Explicit evaluate-test mode consumes this immutable lock once.",
    }
    with weights_path.open("x", encoding="utf-8") as handle:
        json.dump(locked, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print("locked validation weights", json.dumps(locked["weights"]), flush=True)
    return 0


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
