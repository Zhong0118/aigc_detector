"""Offline ablation of text scoring; writes evidence, never publishes thresholds.

Input: existing TextSample JSONL files. Run with python -m scripts.evaluate_text_detector --help.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from sklearn.metrics import roc_auc_score

from detection.text_resource_adapters import TextSample
from detection.text_resources import conformal_high_threshold
from scripts.bootstrap_text_resources import _build_detector, _load_config, _wilson_interval


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def select_samples(samples, *, limit: int, seed: int):
    """Group-split before capping; identical text is counted only once globally."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    samples = list(samples)
    unique = {}
    parents = {}
    text_groups = {}

    def component(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for sample in sorted(samples, key=lambda s: s.identity_key):
        fingerprint = _digest(" ".join(sample.text.split()))
        previous = unique.get(fingerprint)
        if previous is not None and previous.label != sample.label:
            raise ValueError("Identical text has conflicting ground-truth labels")
        unique.setdefault(fingerprint, sample)
        group = (sample.source, sample.partition_key)
        other = text_groups.setdefault(fingerprint, group)
        left, right = sorted((component(group), component(other)))
        parents[right] = left
    groups = defaultdict(list)
    for fingerprint, sample in unique.items():
        source, partition = component((sample.source, sample.partition_key))
        calibration = int(_digest(f"{seed}:{source}:{partition}"), 16) % 4 == 0
        if calibration and sample.label == "ai":
            continue
        role = "calibration" if calibration else "test"
        group = (role, sample.source, sample.language, sample.domain, sample.label)
        groups[group].append(sample)
    selected = []
    for group in sorted(groups):
        ranked = sorted(groups[group], key=lambda s: _digest(f"{seed}:{s.identity_key}"))
        selected.extend((group[0], s) for s in ranked[:limit])
    return selected


def binary_metrics(labels, scores, threshold: float):
    if len(labels) != len(scores) or any(label not in {"human", "ai"} for label in labels):
        raise ValueError("Scores and binary labels must match")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Scores must be finite probabilities")
    tp = sum(label == "ai" and s > threshold for label, s in zip(labels, scores))
    fp = sum(label == "human" and s > threshold for label, s in zip(labels, scores))
    ai = labels.count("ai")
    human = labels.count("human")
    return dict(
        samples=len(labels), ai=ai, human=human, tp=tp, fn=ai - tp, fp=fp, tn=human - fp,
        recall=tp / ai if ai else None, fpr=fp / human if human else None,
        recall_interval_95=_wilson_interval(tp, ai) if ai else None,
        fpr_interval_95=_wilson_interval(fp, human) if human else None,
        auroc=float(roc_auc_score([v == "ai" for v in labels], scores)) if ai and human else None,
    )


def score_variants(detector, legacy_run, text):
    """Reuse loaded weights and branch scores; only the named factor changes."""
    run = detector._run_model
    route = detector.select_model_keys
    strategy = detector.mixed_strategy
    cache = {}
    results = {}

    def cached(key, model_name, segment):
        cache_key = (legacy, key, segment)
        if cache_key not in cache:
            cache[cache_key] = (legacy_run(detector, key, model_name, segment)
                                if legacy else run(key, model_name, segment))
        return cache[cache_key]

    def dominant(segment):
        zh = sum("\u4e00" <= c <= "\u9fff" for c in segment)
        en = sum(c.isascii() and c.isalpha() for c in segment)
        return [f"{'zh' if zh >= en else 'en'}_{detector.length_bucket(segment)}"]

    try:
        detector._run_model = cached
        for name in ("legacy_mean", "complete_mean", "complete_max", "complete_dominant"):
            legacy = name == "legacy_mean"
            detector.mixed_strategy = "max" if name == "complete_max" else "mean"
            detector.select_model_keys = dominant if name == "complete_dominant" else route
            result = detector.detect(text)
            results[name] = {"score": result.score, "model_scores": result.details["model_scores"],
                             "segments": len(result.details["segments"])}
    finally:
        detector._run_model = run
        detector.select_model_keys = route
        detector.mixed_strategy = strategy
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--legacy-source", required=True, help="Unmodified baseline text_detector.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--per-group", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--regression", help="JSON with text; scored separately, excluded from calibration")
    args = parser.parse_args(argv)
    samples = []
    for filename in args.input:
        with Path(filename).open() as handle:
            for line in handle:
                sample = TextSample(**json.loads(line))
                if sample.language == args.language and len("".join(sample.text.split())) >= 120:
                    samples.append(sample)
    selected = select_samples(samples, limit=args.per_group, seed=args.seed)
    if not selected:
        parser.error("No eligible samples")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "selection.jsonl").open("w") as handle:
        for role, sample in selected:
            handle.write(json.dumps(dict(role=role, **asdict(sample)), ensure_ascii=False) + "\n")
    spec = importlib.util.spec_from_file_location("aigc_legacy_detector", args.legacy_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = _load_config(Path(args.config))
    config["detection"]["local_models"]["text_yuchuan"]["device"] = args.device
    detector = _build_detector(config)
    # Keep tokenizers and torch from using all host CPU threads during GPU scoring.
    torch.set_num_threads(2)
    rows = []
    with (output / "scores.jsonl").open("w") as handle:
        for i, (role, sample) in enumerate(selected):
            variants = score_variants(detector, module.RoutedTextAigcDetector._run_model, sample.text)
            row = dict(role=role, identity=sample.identity_key, source=sample.source,
                       language=sample.language, domain=sample.domain, label=sample.label,
                       generator=sample.generator, variants=variants)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if i % 25 == 0:
                print(f"scored {i + 1}/{len(selected)}", flush=True)
    summary = dict(arguments=vars(args), torch=torch.__version__, transformers=transformers.__version__,
                   input_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in args.input},
                   legacy_sha256=hashlib.sha256(Path(args.legacy_source).read_bytes()).hexdigest(),
                   model_config=config["detection"]["local_models"]["text_yuchuan"],
                   models={key: {"path": detector._local_snapshot_path(name)} for key, name in detector.models.items()},
                   notes=["Detector score metrics; excludes product controller gates.",
                          "Exploratory cached benchmark; detector training overlap is unknown.",
                          "Calibration uses human scores only; no thresholds are published.",
                          "Strategy selection here requires a fresh holdout before production claims."],
                   groups={})
    for name in rows[0]["variants"]:
        groups = {}
        for source, domain in sorted({(r["source"], r["domain"]) for r in rows if r["role"] == "test"}):
            test = [r for r in rows if r["role"] == "test" and r["source"] == source and r["domain"] == domain]
            calibration = [r["variants"][name]["score"] for r in rows
                           if r["role"] == "calibration" and r["domain"] == domain]
            cutoff = conformal_high_threshold(calibration, alpha=0.02)
            labels = [r["label"] for r in test]
            scores = [r["variants"][name]["score"] for r in test]
            groups[f"{source}:{domain}"] = dict(
                fixed_0_8=binary_metrics(labels, scores, 0.8),
                calibration_samples=len(calibration), calibrated_cutoff=cutoff if math.isfinite(cutoff) else None,
                calibrated=binary_metrics(labels, scores, cutoff) if math.isfinite(cutoff) else None,
            )
        summary["groups"][name] = groups
    if args.regression:
        text = json.loads(Path(args.regression).read_text())["text"]
        summary["regression"] = dict(sha256=_digest(text), chars=len(text),
            variants=score_variants(detector, module.RoutedTextAigcDetector._run_model, text))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Completed: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
