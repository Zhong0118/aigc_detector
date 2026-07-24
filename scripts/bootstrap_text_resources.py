"""Download text resources and calibrate false-positive thresholds once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

from detection.text_detector import RoutedTextAigcDetector
from detection.text_resources import DatasetSpec, TextResourceManager, conformal_high_threshold
from detection.text_segmentation import SentenceAwareSegmenter


def build_threshold_artifact(
    manager: TextResourceManager,
    detector: Any,
    sample_limit: int = 300,
    validation_limit: int = 100,
    validation_fraction: float = 0.2,
    minimum_group_samples: int = 50,
    minimum_validation_samples: int = 20,
    alpha: float = 0.02,
    short_text_chars: int = 200,
    long_target_chars: int = 480,
    long_max_chars: int = 640,
    sampling_seed: int = 20260722,
    max_source_items: int | None = None,
    max_windows_per_document: int = 1,
    low_threshold: float = 0.4,
    high_threshold_floors: dict[str, float] | None = None,
    low_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit thresholds on known-human texts and audit them on held-out texts."""
    if sample_limit <= 0 or validation_limit < 0:
        raise ValueError("sample limits must be positive for calibration and non-negative for validation")
    if validation_limit > 0 and not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    calibration_scores: dict[str, list[float]] = defaultdict(list)
    validation_scores: dict[str, list[float]] = defaultdict(list)
    source_counts: dict[str, int] = {}
    source_validation_counts: dict[str, int] = {}
    source_buckets: dict[str, dict[str, int]] = {}
    source_validation_buckets: dict[str, dict[str, int]] = {}
    source_licenses: dict[str, str] = {}
    for spec in manager.dataset_specs:
        selected: dict[str, list[str]] = defaultdict(list)
        for text, requested_bucket in _balanced_human_samples(
            manager,
            spec.name,
            sample_limit=sample_limit + validation_limit,
            short_text_chars=short_text_chars,
            long_target_chars=long_target_chars,
            long_max_chars=long_max_chars,
            allowed_buckets=spec.calibration_buckets,
            sampling_seed=sampling_seed,
            max_source_items=max_source_items,
            max_windows_per_document=max_windows_per_document,
        ):
            selected[requested_bucket].append(text)

        bucket_counts = {"short": 0, "long": 0}
        validation_bucket_counts = {"short": 0, "long": 0}
        for bucket in spec.calibration_buckets:
            texts = selected.get(bucket, [])
            calibration_count, held_out_count = _split_sample_counts(
                len(texts),
                calibration_limit=sample_limit,
                validation_limit=validation_limit,
                validation_fraction=validation_fraction,
                minimum_validation_samples=minimum_validation_samples,
            )
            calibration_texts = texts[:calibration_count]
            held_out_texts = texts[calibration_count : calibration_count + held_out_count]
            for text in calibration_texts:
                score = _detect_score(detector, text)
                if score is None:
                    continue
                _append_group_score(calibration_scores, spec, bucket, score)
                bucket_counts[bucket] += 1
            for text in held_out_texts:
                score = _detect_score(detector, text)
                if score is None:
                    continue
                _append_group_score(validation_scores, spec, bucket, score)
                validation_bucket_counts[bucket] += 1

        source_counts[spec.name] = sum(bucket_counts.values())
        source_validation_counts[spec.name] = sum(validation_bucket_counts.values())
        source_buckets[spec.name] = bucket_counts
        source_validation_buckets[spec.name] = validation_bucket_counts
        source_licenses[spec.name] = spec.license
        missing_buckets = [
            name
            for name in spec.calibration_buckets
            if bucket_counts[name] < minimum_group_samples
        ]
        if missing_buckets:
            raise RuntimeError(
                f"Calibration source '{spec.name}' has too few usable samples for "
                f"{', '.join(missing_buckets)}; at least {minimum_group_samples} per bucket are required."
            )
        missing_validation = [
            name
            for name in spec.calibration_buckets
            if validation_limit > 0 and validation_bucket_counts[name] < minimum_validation_samples
        ]
        if missing_validation:
            raise RuntimeError(
                f"Validation source '{spec.name}' has too few usable samples for "
                f"{', '.join(missing_validation)}; at least {minimum_validation_samples} per bucket are required."
            )

    groups: dict[str, dict[str, Any]] = {}
    high_threshold_floors = high_threshold_floors or {}
    low_thresholds = low_thresholds or {}
    for name, values in sorted(calibration_scores.items()):
        if len(values) < minimum_group_samples:
            continue
        parts = name.split(":")
        domain = parts[1] if len(parts) >= 2 else "default"
        threshold_floor = float(
            high_threshold_floors.get(domain, high_threshold_floors.get("default", 0.0))
        )
        calibrated = conformal_high_threshold(values, alpha=alpha)
        threshold = 1.0 if math.isinf(calibrated) else max(float(calibrated), threshold_floor)
        held_out = validation_scores.get(name, [])
        false_positives = sum(score > threshold for score in held_out)
        validation_fpr = false_positives / len(held_out) if held_out else None
        interval = _wilson_interval(false_positives, len(held_out)) if held_out else None
        warnings: list[str] = []
        if threshold >= 0.99:
            warnings.append("calibrated_threshold_near_score_ceiling")
        if interval is not None and interval[0] > alpha:
            warnings.append("holdout_fpr_above_target")
        groups[name] = {
            "low_threshold": round(float(low_thresholds.get(domain, low_threshold)), 6),
            "high_threshold": round(float(threshold), 6),
            "threshold_floor": round(threshold_floor, 6),
            "human_samples": len(values),
            "calibration_quantiles": _score_quantiles(values),
            "validation_samples": len(held_out),
            "validation_false_positives": false_positives,
            "validation_fpr": round(validation_fpr, 6) if validation_fpr is not None else None,
            "validation_fpr_interval_95": (
                [round(interval[0], 6), round(interval[1], 6)] if interval is not None else None
            ),
            "reliability_status": "warning" if warnings else "ok",
            "warnings": warnings,
            "target_fpr": alpha,
        }

    payload = {
        "version": datetime.now(timezone.utc).strftime("fp-%Y%m%dT%H%M%SZ"),
        "calibration_kind": "human_only_split_conformal",
        "decision_operator": "score > high_threshold",
        "target_fpr": alpha,
        "sample_limit_per_dataset": sample_limit,
        "validation_limit_per_dataset": validation_limit,
        "minimum_group_samples": minimum_group_samples,
        "minimum_validation_samples": minimum_validation_samples,
        "sources": source_counts,
        "validation_sources": source_validation_counts,
        "source_length_buckets": source_buckets,
        "validation_source_length_buckets": source_validation_buckets,
        "licenses": source_licenses,
        "sampling": {
            "method": "deterministic_reservoir_without_cross_document_concatenation",
            "seed": sampling_seed,
            "validation_fraction_when_limited": validation_fraction,
            "max_source_items": max_source_items,
            "max_windows_per_document": max_windows_per_document,
            "long_window_target_chars": long_target_chars,
            "long_window_max_chars": long_max_chars,
        },
        "groups": groups,
    }
    manager.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    manager.artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def _balanced_human_samples(
    manager: TextResourceManager,
    dataset_name: str,
    *,
    sample_limit: int,
    short_text_chars: int,
    long_target_chars: int,
    long_max_chars: int = 640,
    allowed_buckets: Sequence[str] = ("short", "long"),
    sampling_seed: int = 20260722,
    max_source_items: int | None = None,
    max_windows_per_document: int = 1,
) -> Iterator[tuple[str, str]]:
    """Reservoir-sample real documents or contiguous windows by length bucket."""
    long_target_chars = max(long_target_chars, short_text_chars + 1)
    long_max_chars = max(long_max_chars, long_target_chars)
    allowed = tuple(bucket for bucket in allowed_buckets if bucket in {"short", "long"})
    reservoirs: dict[str, list[str]] = {bucket: [] for bucket in allowed}
    seen_counts = {bucket: 0 for bucket in allowed}
    rngs = {
        bucket: random.Random(_stable_seed(sampling_seed, dataset_name, bucket))
        for bucket in allowed
    }
    scan_limit = max_source_items or max(sample_limit * 20, 5000)
    segmenter = SentenceAwareSegmenter(
        target_chars=long_target_chars,
        max_chars=long_max_chars,
        overlap_sentences=0,
    )
    for text in manager.iter_human_texts(dataset_name, limit=scan_limit):
        for candidate, bucket in _real_document_candidates(
            text,
            short_text_chars=short_text_chars,
            segmenter=segmenter,
            max_windows=max_windows_per_document,
        ):
            if bucket not in reservoirs:
                continue
            seen_counts[bucket] += 1
            bucket_samples = reservoirs[bucket]
            if len(bucket_samples) < sample_limit:
                bucket_samples.append(candidate)
                continue
            replacement = rngs[bucket].randrange(seen_counts[bucket])
            if replacement < sample_limit:
                bucket_samples[replacement] = candidate

    for bucket in ("short", "long"):
        if bucket not in reservoirs:
            continue
        rngs[bucket].shuffle(reservoirs[bucket])
        for text in reservoirs[bucket]:
            yield text, bucket


def _split_sample_counts(
    total: int,
    *,
    calibration_limit: int,
    validation_limit: int,
    validation_fraction: float,
    minimum_validation_samples: int,
) -> tuple[int, int]:
    if validation_limit == 0:
        return min(total, calibration_limit), 0
    if total >= calibration_limit + validation_limit:
        return calibration_limit, validation_limit
    held_out = min(
        validation_limit,
        max(minimum_validation_samples, math.floor(total * validation_fraction)),
    )
    held_out = min(held_out, total)
    return min(calibration_limit, total - held_out), held_out


def _real_document_candidates(
    text: str,
    *,
    short_text_chars: int,
    segmenter: SentenceAwareSegmenter,
    max_windows: int,
) -> Iterator[tuple[str, str]]:
    effective_chars = len("".join(text.split()))
    if effective_chars <= short_text_chars:
        yield text, "short"
        return

    windows = [
        segment.text.strip()
        for segment in segmenter.split(text)
        if len("".join(segment.text.split())) > short_text_chars
    ]
    for window in _evenly_spaced(windows, max_windows):
        yield window, "long"


def _evenly_spaced(values: Sequence[str], limit: int) -> list[str]:
    if limit <= 0 or not values:
        return []
    if len(values) <= limit:
        return list(values)
    if limit == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indices]


def _stable_seed(seed: int, dataset_name: str, bucket: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_name}:{bucket}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _detect_score(detector: Any, text: str) -> float | None:
    try:
        score = float(detector.detect(text).score)
    except Exception:  # noqa: BLE001 - one malformed sample must not discard calibration.
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _append_group_score(
    scores: dict[str, list[float]],
    spec: DatasetSpec,
    length_bucket: str,
    score: float,
) -> None:
    for name in (
        f"zh:{spec.domain}:{length_bucket}",
        f"zh:{spec.domain}",
        "zh",
        "global",
    ):
        scores[name].append(score)


def _score_quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    quantiles = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p98": 0.98, "p99": 0.99}
    result = {
        name: round(ordered[max(0, math.ceil(len(ordered) * quantile) - 1)], 6)
        for name, quantile in quantiles.items()
    }
    result["max"] = round(ordered[-1], 6)
    return result


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1 + (z * z / total)
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_detector(config: dict[str, Any]) -> RoutedTextAigcDetector:
    detection = config.get("detection", {})
    local = detection.get("local_models", {}).get("text_yuchuan", {})
    return RoutedTextAigcDetector(
        models=local.get("models", {}),
        cache_dir=local.get("cache_dir", "models/huggingface"),
        short_text_chars=int(local.get("short_text_chars", 200)),
        threshold=float(local.get("threshold", detection.get("threshold", 0.8))),
        device=str(local.get("device", "auto")),
        mixed_strategy=str(local.get("mixed_strategy", "mean")),
        max_length=int(local.get("max_length", 512)),
        segment_target_chars=int(local.get("segment_target_chars", 480)),
        segment_max_chars=int(local.get("segment_max_chars", 640)),
        segment_overlap_sentences=int(local.get("segment_overlap_sentences", 1)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--profile", choices=("status", "models", "data", "all"), default="all")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config = _load_config(project_root / args.config)
    manager = TextResourceManager(project_root=project_root)
    if args.profile == "status":
        print(json.dumps(manager.status(), ensure_ascii=False, indent=2))
        return 0

    try:
        detection = config.get("detection", {})
        local = detection.get("local_models", {}).get("text_yuchuan", {})
        fp = detection.get("text_false_positive_control", {})
        if args.profile in {"models", "all"}:
            model_ids = [str(value) for value in local.get("models", {}).values()]
            manager.ensure_models(model_ids)
        if args.profile in {"data", "all"}:
            manager.ensure_datasets()
        if args.profile == "all" and (args.force_calibration or not manager.artifact_path.exists()):
            detector = _build_detector(config)
            domain_thresholds = fp.get("domain_thresholds", {})
            build_threshold_artifact(
                manager=manager,
                detector=detector,
                sample_limit=int(args.sample_limit or fp.get("bootstrap_sample_limit", 300)),
                validation_limit=int(
                    args.validation_limit
                    if args.validation_limit is not None
                    else fp.get("bootstrap_validation_limit", 100)
                ),
                validation_fraction=float(fp.get("bootstrap_validation_fraction", 0.2)),
                minimum_group_samples=int(fp.get("min_group_calibration_humans", 50)),
                minimum_validation_samples=int(fp.get("min_group_validation_humans", 20)),
                alpha=float(fp.get("target_fpr", 0.02)),
                short_text_chars=int(local.get("short_text_chars", 200)),
                long_target_chars=int(local.get("segment_target_chars", 480)),
                long_max_chars=int(local.get("segment_max_chars", 640)),
                sampling_seed=int(fp.get("bootstrap_sampling_seed", 20260722)),
                max_source_items=int(fp.get("bootstrap_max_source_items", 100000)),
                max_windows_per_document=int(fp.get("bootstrap_max_windows_per_document", 1)),
                low_threshold=float(fp.get("default_low_threshold", 0.4)),
                high_threshold_floors={
                    "default": float(fp.get("default_high_threshold", 0.8)),
                    **{
                        name: float(values.get("high_threshold", fp.get("default_high_threshold", 0.8)))
                        for name, values in domain_thresholds.items()
                    },
                },
                low_thresholds={
                    name: float(values.get("low_threshold", fp.get("default_low_threshold", 0.4)))
                    for name, values in domain_thresholds.items()
                },
            )
        print(json.dumps(manager.status(), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - command reports a clear bootstrap failure.
        print(f"Text resource bootstrap failed: {exc}")
        if args.allow_fallback:
            print("Continuing with lazy model download and built-in conservative thresholds.")
            return 0
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
