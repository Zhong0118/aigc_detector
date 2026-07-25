"""Download text resources and calibrate false-positive thresholds once."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml
from sklearn.metrics import roc_auc_score

from detection.text_detector import RoutedTextAigcDetector
from detection.text_resource_adapters import TextSample
from detection.text_resources import TextResourceManager, conformal_high_threshold
from detection.text_segmentation import SentenceAwareSegmenter

ROLE_HUMAN_CALIBRATION = "human_calibration"
ROLE_HUMAN_VALIDATION = "human_validation"
ROLE_AI_VALIDATION = "ai_validation"
SELECTION_ROLES = (
    ROLE_HUMAN_CALIBRATION,
    ROLE_HUMAN_VALIDATION,
    ROLE_AI_VALIDATION,
)


@dataclass(frozen=True)
class _Candidate:
    sample: TextSample
    length_bucket: str
    role: str
    priority: int

    @property
    def exact_group(self) -> str:
        return self.sample.group_key(self.length_bucket)


@dataclass(frozen=True)
class _DescendingIdentity:
    """Reverse lexical ordering so heap root remains the worst identity."""

    value: str

    def __lt__(self, other: _DescendingIdentity) -> bool:
        return self.value > other.value


class _HashPriorityReservoir:
    """Keep the stable lowest-hash items without depending on input order."""

    def __init__(self, limit: int) -> None:
        self.limit = max(0, int(limit))
        self._heap: list[
            tuple[int, _DescendingIdentity, str, _Candidate]
        ] = []
        self._selected: set[str] = set()

    def add(self, candidate: _Candidate) -> None:
        if self.limit <= 0:
            return
        identity = candidate.sample.identity_key
        if identity in self._selected:
            return
        entry = (
            -candidate.priority,
            _DescendingIdentity(identity),
            identity,
            candidate,
        )
        if len(self._heap) < self.limit:
            heapq.heappush(self._heap, entry)
            self._selected.add(identity)
            return
        worst_priority = -self._heap[0][0]
        worst_identity = self._heap[0][2]
        if (candidate.priority, identity) >= (worst_priority, worst_identity):
            return
        removed = heapq.heapreplace(self._heap, entry)
        self._selected.discard(removed[2])
        self._selected.add(identity)

    def values(self) -> list[_Candidate]:
        return [
            entry[3]
            for entry in sorted(
                self._heap,
                key=lambda item: (-item[0], item[2]),
            )
        ]


def build_threshold_artifact(
    manager: TextResourceManager,
    detector: Any,
    sample_limit: int | None = None,
    validation_limit: int | None = None,
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
    *,
    human_calibration_limit: int | None = None,
    human_validation_limit: int | None = None,
    ai_validation_limit: int | None = None,
    minimum_ai_validation_samples: int | None = None,
    minimum_ai_recall: float = 0.5,
    min_effective_chars: int = 8,
) -> dict[str, Any]:
    """Build a v2 artifact from isolated human calibration and held-out evidence."""
    human_calibration_limit = _resolve_limit(
        human_calibration_limit,
        sample_limit,
        default=300,
    )
    human_validation_limit = _resolve_limit(
        human_validation_limit,
        validation_limit,
        default=100,
    )
    ai_validation_limit = _resolve_limit(
        ai_validation_limit,
        validation_limit,
        default=human_validation_limit,
    )
    minimum_ai_validation_samples = (
        minimum_validation_samples
        if minimum_ai_validation_samples is None
        else int(minimum_ai_validation_samples)
    )
    _validate_calibration_options(
        human_calibration_limit=human_calibration_limit,
        human_validation_limit=human_validation_limit,
        ai_validation_limit=ai_validation_limit,
        validation_fraction=validation_fraction,
        minimum_group_samples=minimum_group_samples,
        minimum_validation_samples=minimum_validation_samples,
        minimum_ai_validation_samples=minimum_ai_validation_samples,
        alpha=alpha,
        minimum_ai_recall=minimum_ai_recall,
        min_effective_chars=min_effective_chars,
        max_source_items=max_source_items,
    )

    selection, evidence = _select_multilingual_candidates(
        manager,
        human_calibration_limit=human_calibration_limit,
        human_validation_limit=human_validation_limit,
        ai_validation_limit=ai_validation_limit,
        validation_fraction=validation_fraction,
        short_text_chars=short_text_chars,
        long_target_chars=long_target_chars,
        long_max_chars=long_max_chars,
        min_effective_chars=min_effective_chars,
        sampling_seed=sampling_seed,
        max_source_items=max_source_items,
        max_windows_per_document=max_windows_per_document,
    )
    _check_required_source_evidence(manager, evidence)
    _persist_selected_samples(manager, selection)

    scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    detector_failures: Counter[str] = Counter()
    detector_failures_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    detector_failures_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    detector_failures_by_exact_group: dict[str, Counter[str]] = defaultdict(
        Counter
    )
    detector_failure_details: Counter[tuple[str, str, str, str]] = Counter()
    scored_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    exact_groups = sorted({key[0] for key in selection})
    total_selected = sum(len(values) for values in selection.values())
    print(f"Scoring {total_selected} selected text candidates...")
    for (exact_group, role), candidates in sorted(selection.items()):
        for candidate in candidates:
            score, error_name = _detect_score_with_error(
                detector,
                candidate.sample.text,
            )
            if error_name is not None:
                detector_failures[error_name] += 1
                detector_failures_by_role[role][error_name] += 1
                detector_failures_by_source[candidate.sample.source][
                    error_name
                ] += 1
                detector_failures_by_exact_group[exact_group][error_name] += 1
                detector_failure_details[
                    (
                        candidate.sample.source,
                        exact_group,
                        role,
                        error_name,
                    )
                ] += 1
                continue
            assert score is not None
            for group_name in _group_fallbacks(
                candidate.sample,
                candidate.length_bucket,
            ):
                scores[group_name][role].append(score)
            scored_by_source[candidate.sample.source][role] += 1
    if detector_failures:
        print(
            "Detector scoring failures: "
            + ", ".join(
                f"{name}={count}"
                for name, count in sorted(detector_failures.items())
            )
        )

    missing_calibration = [
        group_name
        for group_name in exact_groups
        if len(scores[group_name][ROLE_HUMAN_CALIBRATION])
        < minimum_group_samples
    ]
    if missing_calibration:
        failure_summary = dict(sorted(detector_failures.items()))
        raise RuntimeError(
            "Required human calibration evidence is insufficient for "
            f"{', '.join(missing_calibration)}; need at least "
            f"{minimum_group_samples} successful scores per exact group. "
            f"Detector failures: {failure_summary}"
        )

    high_threshold_floors = high_threshold_floors or {}
    low_thresholds = low_thresholds or {}
    groups: dict[str, dict[str, Any]] = {}
    for group_name in sorted(scores):
        human_calibration = scores[group_name][ROLE_HUMAN_CALIBRATION]
        if len(human_calibration) < minimum_group_samples:
            continue
        human_validation = scores[group_name][ROLE_HUMAN_VALIDATION]
        ai_validation = scores[group_name][ROLE_AI_VALIDATION]
        domain = _group_domain(group_name)
        threshold_floor = float(
            high_threshold_floors.get(
                domain,
                high_threshold_floors.get("default", 0.0),
            )
        )
        calibrated = conformal_high_threshold(human_calibration, alpha=alpha)
        threshold = (
            1.0
            if math.isinf(calibrated)
            else max(float(calibrated), threshold_floor)
        )
        groups[group_name] = _group_artifact(
            human_calibration=human_calibration,
            human_validation=human_validation,
            ai_validation=ai_validation,
            threshold=threshold,
            threshold_floor=threshold_floor,
            low_threshold=float(low_thresholds.get(domain, low_threshold)),
            alpha=alpha,
            minimum_validation_samples=minimum_validation_samples,
            minimum_ai_validation_samples=minimum_ai_validation_samples,
            minimum_ai_recall=minimum_ai_recall,
        )

    if not groups:
        raise RuntimeError("No threshold group has sufficient human calibration evidence")

    payload = _artifact_payload(
        manager=manager,
        groups=groups,
        selection=selection,
        evidence=evidence,
        scored_by_source=scored_by_source,
        detector_failures=detector_failures,
        detector_failures_by_role=detector_failures_by_role,
        detector_failures_by_source=detector_failures_by_source,
        detector_failures_by_exact_group=detector_failures_by_exact_group,
        detector_failure_details=detector_failure_details,
        alpha=alpha,
        human_calibration_limit=human_calibration_limit,
        human_validation_limit=human_validation_limit,
        ai_validation_limit=ai_validation_limit,
        validation_fraction=validation_fraction,
        minimum_group_samples=minimum_group_samples,
        minimum_validation_samples=minimum_validation_samples,
        minimum_ai_validation_samples=minimum_ai_validation_samples,
        min_effective_chars=min_effective_chars,
        short_text_chars=short_text_chars,
        long_target_chars=long_target_chars,
        long_max_chars=long_max_chars,
        sampling_seed=sampling_seed,
        max_source_items=max_source_items,
        max_windows_per_document=max_windows_per_document,
    )
    _publish_artifact_atomic(manager.artifact_path, payload)
    return payload


def _select_multilingual_candidates(
    manager: TextResourceManager,
    *,
    human_calibration_limit: int,
    human_validation_limit: int,
    ai_validation_limit: int,
    validation_fraction: float,
    short_text_chars: int,
    long_target_chars: int,
    long_max_chars: int,
    min_effective_chars: int,
    sampling_seed: int,
    max_source_items: int | None,
    max_windows_per_document: int,
) -> tuple[
    dict[tuple[str, str], list[_Candidate]],
    dict[str, dict[str, Any]],
]:
    limits = {
        ROLE_HUMAN_CALIBRATION: human_calibration_limit,
        ROLE_HUMAN_VALIDATION: human_validation_limit,
        ROLE_AI_VALIDATION: ai_validation_limit,
    }
    reservoirs: dict[tuple[str, str], _HashPriorityReservoir] = {}
    evidence: dict[str, dict[str, Any]] = {}
    segmenter = SentenceAwareSegmenter(
        target_chars=max(long_target_chars, short_text_chars + 1),
        max_chars=max(long_max_chars, long_target_chars, short_text_chars + 1),
        overlap_sentences=0,
    )
    for spec in manager.dataset_specs:
        print(f"Scanning text source: {spec.name}")
        source_evidence: dict[str, Any] = {
            "license": spec.license,
            "source": spec.source,
            "required": spec.required,
            "scanned_samples": 0,
            "usable_candidates": 0,
            "excluded_short": 0,
            "excluded_ai_calibration_partition": 0,
            "labels": Counter(),
            "usable_labels": Counter(),
            "length_buckets": Counter(),
            "selected_roles": Counter(),
        }
        evidence[spec.name] = source_evidence
        try:
            source_samples = manager.iter_source_samples(spec)
            for sample in source_samples:
                source_evidence["scanned_samples"] += 1
                source_evidence["labels"][sample.label] += 1
                candidates, excluded_short = _candidate_samples(
                    sample,
                    short_text_chars=short_text_chars,
                    min_effective_chars=min_effective_chars,
                    segmenter=segmenter,
                    max_windows=max_windows_per_document,
                )
                source_evidence["excluded_short"] += excluded_short
                for candidate_sample, length_bucket in candidates:
                    if length_bucket not in spec.calibration_buckets:
                        continue
                    partition = _partition_assignment(
                        candidate_sample.partition_key,
                        sampling_seed,
                        validation_fraction,
                    )
                    role = _selection_role(candidate_sample.label, partition)
                    if role is None:
                        source_evidence["excluded_ai_calibration_partition"] += 1
                        continue
                    source_evidence["usable_candidates"] += 1
                    source_evidence["usable_labels"][candidate_sample.label] += 1
                    source_evidence["length_buckets"][length_bucket] += 1
                    exact_group = candidate_sample.group_key(length_bucket)
                    reservoir_key = (exact_group, role)
                    reservoir = reservoirs.setdefault(
                        reservoir_key,
                        _HashPriorityReservoir(limits[role]),
                    )
                    reservoir.add(
                        _Candidate(
                            sample=candidate_sample,
                            length_bucket=length_bucket,
                            role=role,
                            priority=_selection_priority(
                                sampling_seed,
                                candidate_sample.identity_key,
                                role,
                            ),
                        )
                    )
        except (FileNotFoundError, RuntimeError) as exc:
            if (
                source_evidence["scanned_samples"] == 0
                and manager.is_normalized_ready(spec)
            ):
                source_evidence["input"] = "normalized_fallback"
                for sample in manager.iter_normalized_samples(spec):
                    effective = _effective_chars(sample.text)
                    if effective < min_effective_chars:
                        source_evidence["excluded_short"] += 1
                        continue
                    bucket = "short" if effective <= short_text_chars else "long"
                    if bucket not in spec.calibration_buckets:
                        continue
                    partition = _partition_assignment(
                        sample.partition_key,
                        sampling_seed,
                        validation_fraction,
                    )
                    role = _selection_role(sample.label, partition)
                    if role is None:
                        continue
                    source_evidence["scanned_samples"] += 1
                    source_evidence["labels"][sample.label] += 1
                    source_evidence["usable_candidates"] += 1
                    source_evidence["usable_labels"][sample.label] += 1
                    source_evidence["length_buckets"][bucket] += 1
                    exact_group = sample.group_key(bucket)
                    reservoirs.setdefault(
                        (exact_group, role),
                        _HashPriorityReservoir(limits[role]),
                    ).add(
                        _Candidate(
                            sample=sample,
                            length_bucket=bucket,
                            role=role,
                            priority=_selection_priority(
                                sampling_seed,
                                sample.identity_key,
                                role,
                            ),
                        )
                    )
            else:
                source_evidence["scan_error"] = f"{type(exc).__name__}: {exc}"

    selection = {
        key: reservoir.values()
        for key, reservoir in reservoirs.items()
        if reservoir.values()
    }
    for candidates in selection.values():
        for candidate in candidates:
            evidence[candidate.sample.source]["selected_roles"][candidate.role] += 1
    for source_name, values in evidence.items():
        for field in (
            "labels",
            "usable_labels",
            "length_buckets",
            "selected_roles",
        ):
            values[field] = dict(sorted(values[field].items()))
        print(
            f"Selected {source_name}: "
            f"{sum(values['selected_roles'].values())} candidates"
        )
    return selection, evidence


def _candidate_samples(
    sample: TextSample,
    *,
    short_text_chars: int,
    min_effective_chars: int,
    segmenter: SentenceAwareSegmenter,
    max_windows: int,
) -> tuple[list[tuple[TextSample, str]], int]:
    effective = _effective_chars(sample.text)
    if effective < min_effective_chars:
        return [], 1
    if effective <= short_text_chars:
        return [(sample, "short")], 0
    windows = [
        segment.text.strip()
        for segment in segmenter.split(sample.text)
        if _effective_chars(segment.text) > short_text_chars
    ]
    selected_windows = _evenly_spaced(windows, max_windows)
    results: list[tuple[TextSample, str]] = []
    for index, text in enumerate(selected_windows):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        results.append(
            (
                TextSample(
                    text=text,
                    label=sample.label,
                    language=sample.language,
                    domain=sample.domain,
                    source=sample.source,
                    source_id=f"{sample.source_id}:window:{index}:{digest}",
                    generator=sample.generator,
                    split=sample.split,
                    partition_id=sample.partition_key,
                ),
                "long",
            )
        )
    return results, 0


def _partition_assignment(
    partition_key: str,
    seed: int,
    validation_fraction: float,
) -> str:
    if not 0 <= validation_fraction <= 1:
        raise ValueError("validation_fraction must be between zero and one")
    digest = hashlib.sha256(f"{seed}:{partition_key}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "calibration"


def _selection_role(label: str, partition: str) -> str | None:
    if partition == "calibration":
        return ROLE_HUMAN_CALIBRATION if label == "human" else None
    return ROLE_HUMAN_VALIDATION if label == "human" else ROLE_AI_VALIDATION


def _selection_priority(seed: int, identity_key: str, role: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{role}:{identity_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big")


def _persist_selected_samples(
    manager: TextResourceManager,
    selection: dict[tuple[str, str], list[_Candidate]],
) -> None:
    by_source: dict[str, dict[str, TextSample]] = defaultdict(dict)
    for candidates in selection.values():
        for candidate in candidates:
            by_source[candidate.sample.source][
                candidate.sample.identity_key
            ] = candidate.sample
    for spec in manager.dataset_specs:
        samples = list(by_source.get(spec.name, {}).values())
        if not samples:
            continue
        samples.sort(key=lambda sample: sample.identity_key)
        manager.write_normalized_samples(spec, samples)


def _check_required_source_evidence(
    manager: TextResourceManager,
    evidence: dict[str, dict[str, Any]],
) -> None:
    failures = []
    for spec in manager.dataset_specs:
        item = evidence[spec.name]
        human_count = int(item.get("usable_labels", {}).get("human", 0))
        if spec.required and (item.get("scan_error") or human_count <= 0):
            failures.append(spec.name)
    if failures:
        raise RuntimeError(
            "Required text sources have no usable human evidence: "
            + ", ".join(failures)
        )


def _group_fallbacks(sample: TextSample, length_bucket: str) -> tuple[str, ...]:
    return (
        sample.group_key(length_bucket),
        f"{sample.language}:{sample.domain}",
        sample.language,
        "global",
    )


def _group_domain(group_name: str) -> str:
    parts = group_name.split(":")
    return parts[1] if len(parts) >= 2 else "default"


def _group_artifact(
    *,
    human_calibration: Sequence[float],
    human_validation: Sequence[float],
    ai_validation: Sequence[float],
    threshold: float,
    threshold_floor: float,
    low_threshold: float,
    alpha: float,
    minimum_validation_samples: int,
    minimum_ai_validation_samples: int,
    minimum_ai_recall: float,
) -> dict[str, Any]:
    false_positives = sum(score > threshold for score in human_validation)
    fpr = (
        false_positives / len(human_validation)
        if human_validation
        else None
    )
    fpr_interval = (
        _wilson_interval(false_positives, len(human_validation))
        if human_validation
        else None
    )
    true_positives = sum(score > threshold for score in ai_validation)
    recall = (
        true_positives / len(ai_validation)
        if ai_validation
        else None
    )
    recall_interval = (
        _wilson_interval(true_positives, len(ai_validation))
        if ai_validation
        else None
    )
    auroc = None
    if human_validation and ai_validation:
        labels = [0] * len(human_validation) + [1] * len(ai_validation)
        auroc = float(
            roc_auc_score(labels, [*human_validation, *ai_validation])
        )
    warnings: list[str] = []
    if threshold >= 0.99:
        warnings.append("calibrated_threshold_near_score_ceiling")
    if fpr_interval is not None and fpr_interval[0] > alpha:
        warnings.append("holdout_fpr_above_target")
    if len(human_validation) < minimum_validation_samples:
        warnings.append("insufficient_human_validation")
    if len(ai_validation) < minimum_ai_validation_samples:
        warnings.append("insufficient_ai_validation")
    if recall is not None and recall < minimum_ai_recall:
        warnings.append("ai_recall_below_minimum")
    if auroc is None:
        warnings.append("auroc_unavailable")
    return {
        "low_threshold": round(low_threshold, 6),
        "high_threshold": round(threshold, 6),
        "threshold_floor": round(threshold_floor, 6),
        "target_fpr": alpha,
        "human_calibration_samples": len(human_calibration),
        "human_validation_samples": len(human_validation),
        "ai_validation_samples": len(ai_validation),
        "validation_false_positives": false_positives,
        "validation_fpr": _round_optional(fpr),
        "validation_fpr_interval_95": _round_interval(fpr_interval),
        "ai_true_positives": true_positives,
        "ai_recall": _round_optional(recall),
        "ai_fnr": _round_optional(None if recall is None else 1 - recall),
        "ai_recall_interval_95": _round_interval(recall_interval),
        "validation_auroc": _round_optional(auroc),
        "calibration_quantiles": _score_quantiles(human_calibration),
        "reliability_status": "warning" if warnings else "ok",
        "warnings": warnings,
        # v1 readers
        "human_samples": len(human_calibration),
        "validation_samples": len(human_validation),
    }


def _artifact_payload(
    *,
    manager: TextResourceManager,
    groups: dict[str, dict[str, Any]],
    selection: dict[tuple[str, str], list[_Candidate]],
    evidence: dict[str, dict[str, Any]],
    scored_by_source: dict[str, Counter[str]],
    detector_failures: Counter[str],
    detector_failures_by_role: dict[str, Counter[str]],
    detector_failures_by_source: dict[str, Counter[str]],
    detector_failures_by_exact_group: dict[str, Counter[str]],
    detector_failure_details: Counter[tuple[str, str, str, str]],
    alpha: float,
    human_calibration_limit: int,
    human_validation_limit: int,
    ai_validation_limit: int,
    validation_fraction: float,
    minimum_group_samples: int,
    minimum_validation_samples: int,
    minimum_ai_validation_samples: int,
    min_effective_chars: int,
    short_text_chars: int,
    long_target_chars: int,
    long_max_chars: int,
    sampling_seed: int,
    max_source_items: int | None,
    max_windows_per_document: int,
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    validation_source_counts: Counter[str] = Counter()
    source_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    validation_source_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    selected_group_counts: dict[str, dict[str, int]] = defaultdict(dict)
    selected_identity_keys: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for (group_name, role), candidates in selection.items():
        selected_group_counts[group_name][role] = len(candidates)
        selected_identity_keys[group_name][role] = sorted(
            candidate.sample.identity_key for candidate in candidates
        )
        for candidate in candidates:
            source = candidate.sample.source
            if role == ROLE_HUMAN_CALIBRATION:
                source_counts[source] += 1
                source_buckets[source][candidate.length_bucket] += 1
            elif role == ROLE_HUMAN_VALIDATION:
                validation_source_counts[source] += 1
                validation_source_buckets[source][candidate.length_bucket] += 1
    source_evidence = {}
    for spec in manager.dataset_specs:
        item = dict(evidence[spec.name])
        item["scored_roles"] = dict(sorted(scored_by_source[spec.name].items()))
        source_evidence[spec.name] = item
    excluded_short = {
        name: int(item["excluded_short"])
        for name, item in evidence.items()
    }
    return {
        "artifact_schema_version": 2,
        "version": datetime.now(timezone.utc).strftime("fp-%Y%m%dT%H%M%SZ"),
        "calibration_kind": "human_only_split_conformal",
        "decision_operator": "score > high_threshold",
        "target_fpr": alpha,
        "dataset_profile": manager.dataset_profile,
        "minimum_group_samples": minimum_group_samples,
        "minimum_validation_samples": minimum_validation_samples,
        "minimum_ai_validation_samples": minimum_ai_validation_samples,
        "source_evidence": source_evidence,
        "licenses": {spec.name: spec.license for spec in manager.dataset_specs},
        "sampling": {
            "method": "sha256_hash_priority_top_k_without_cross_document_concatenation",
            "partition_method": "sha256(seed:partition_key) validation fraction",
            "partition_key": "TextSample.partition_key",
            "selection_identity": "TextSample.identity_key",
            "seed": sampling_seed,
            "validation_fraction": validation_fraction,
            "limits_per_exact_group": {
                ROLE_HUMAN_CALIBRATION: human_calibration_limit,
                ROLE_HUMAN_VALIDATION: human_validation_limit,
                ROLE_AI_VALIDATION: ai_validation_limit,
            },
            "selected_group_counts": {
                key: dict(sorted(value.items()))
                for key, value in sorted(selected_group_counts.items())
            },
            "selected_identity_keys": {
                key: dict(sorted(value.items()))
                for key, value in sorted(selected_identity_keys.items())
            },
            "max_source_items": max_source_items,
            "source_scan_mode": "complete",
            "max_source_items_applied": False,
            "max_source_items_semantics": (
                "compatibility setting retained for configuration audit; "
                "source scans are complete so input order and paired "
                "partition boundaries cannot change selection"
            ),
            "max_windows_per_document": max_windows_per_document,
            "min_effective_chars": min_effective_chars,
            "short_text_chars": short_text_chars,
            "long_window_target_chars": long_target_chars,
            "long_window_max_chars": long_max_chars,
            "excluded_short_by_source": excluded_short,
            "excluded_short_total": sum(excluded_short.values()),
        },
        "detector_failures": {
            "total": sum(detector_failures.values()),
            "by_exception_class": dict(sorted(detector_failures.items())),
            "by_role": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(detector_failures_by_role.items())
            },
            "by_source": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(detector_failures_by_source.items())
            },
            "by_exact_group": {
                group: dict(sorted(counts.items()))
                for group, counts in sorted(
                    detector_failures_by_exact_group.items()
                )
            },
            "details": [
                {
                    "source": source,
                    "exact_group": exact_group,
                    "role": role,
                    "exception_class": exception_class,
                    "count": count,
                }
                for (
                    source,
                    exact_group,
                    role,
                    exception_class,
                ), count in sorted(detector_failure_details.items())
            ],
        },
        "groups": groups,
        # v1 compatibility fields
        "sample_limit_per_dataset": human_calibration_limit,
        "validation_limit_per_dataset": human_validation_limit,
        "sources": dict(sorted(source_counts.items())),
        "validation_sources": dict(sorted(validation_source_counts.items())),
        "source_length_buckets": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_buckets.items())
        },
        "validation_source_length_buckets": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(validation_source_buckets.items())
        },
    }


def _publish_artifact_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_resource_stats(
    manager: TextResourceManager,
    *,
    short_text_chars: int = 200,
) -> dict[str, Any]:
    """Summarize already prepared resources without downloads or detector loading."""
    result: dict[str, Any] = {
        "dataset_profile": manager.dataset_profile,
        "sources": {},
    }
    for spec in manager.dataset_specs:
        normalized_ready = manager.is_normalized_ready(spec)
        source_ready = manager.is_source_ready(spec)
        counts: Counter[str] = Counter()
        sample_count = 0
        input_kind = "not_ready"
        samples: Iterable[TextSample] = ()
        if normalized_ready:
            samples = manager.iter_normalized_samples(spec)
            input_kind = "normalized"
        elif source_ready:
            samples = manager.iter_source_samples(spec)
            input_kind = "prepared_source"
        for sample in samples:
            bucket = (
                "short"
                if _effective_chars(sample.text) <= short_text_chars
                else "long"
            )
            counts[
                f"{sample.label}:{sample.language}:{sample.domain}:{bucket}"
            ] += 1
            sample_count += 1
        result["sources"][spec.name] = {
            "source_ready": source_ready,
            "normalized_ready": normalized_ready,
            "input": input_kind,
            "prepared_or_normalized_samples": sample_count,
            "normalized_samples": sample_count if normalized_ready else 0,
            "counts": dict(sorted(counts.items())),
        }
    return result


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
    """Compatibility helper for callers that still request legacy human texts."""
    long_target_chars = max(long_target_chars, short_text_chars + 1)
    long_max_chars = max(long_max_chars, long_target_chars)
    allowed = tuple(
        bucket for bucket in allowed_buckets if bucket in {"short", "long"}
    )
    reservoirs: dict[str, list[str]] = {bucket: [] for bucket in allowed}
    seen_counts = {bucket: 0 for bucket in allowed}
    rngs = {
        bucket: random.Random(_stable_seed(sampling_seed, dataset_name, bucket))
        for bucket in allowed
    }
    scan_limit = max_source_items
    segmenter = SentenceAwareSegmenter(
        target_chars=long_target_chars,
        max_chars=long_max_chars,
        overlap_sentences=0,
    )
    for index, text in enumerate(manager.iter_human_texts(dataset_name)):
        if scan_limit is not None and index >= scan_limit:
            break
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


def _real_document_candidates(
    text: str,
    *,
    short_text_chars: int,
    segmenter: SentenceAwareSegmenter,
    max_windows: int,
) -> Iterator[tuple[str, str]]:
    if _effective_chars(text) <= short_text_chars:
        yield text, "short"
        return
    windows = [
        segment.text.strip()
        for segment in segmenter.split(text)
        if _effective_chars(segment.text) > short_text_chars
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
    indices = [
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [values[index] for index in indices]


def _stable_seed(seed: int, dataset_name: str, bucket: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{dataset_name}:{bucket}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _detect_score_with_error(
    detector: Any,
    text: str,
) -> tuple[float | None, str | None]:
    try:
        score = float(detector.detect(text).score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("detector score must be finite and within [0, 1]")
        return score, None
    except Exception as exc:  # noqa: BLE001 - recorded and bounded by sample limits.
        return None, type(exc).__name__


def _detect_score(detector: Any, text: str) -> float | None:
    """Compatibility wrapper retained for external callers."""
    return _detect_score_with_error(detector, text)[0]


def _score_quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    quantiles = {
        "p50": 0.50,
        "p90": 0.90,
        "p95": 0.95,
        "p98": 0.98,
        "p99": 0.99,
    }
    result = {
        name: round(
            ordered[max(0, math.ceil(len(ordered) * quantile) - 1)],
            6,
        )
        for name, quantile in quantiles.items()
    }
    result["max"] = round(ordered[-1], 6)
    return result


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1 + (z * z / total)
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _round_interval(
    value: tuple[float, float] | None,
) -> list[float] | None:
    return (
        [round(value[0], 6), round(value[1], 6)]
        if value is not None
        else None
    )


def _effective_chars(text: str) -> int:
    return len("".join(text.split()))


def _resolve_limit(
    preferred: int | None,
    compatibility: int | None,
    *,
    default: int,
) -> int:
    if preferred is not None:
        return int(preferred)
    if compatibility is not None:
        return int(compatibility)
    return int(default)


def _validate_calibration_options(**options: Any) -> None:
    if options["human_calibration_limit"] <= 0:
        raise ValueError("human_calibration_limit must be positive")
    for name in ("human_validation_limit", "ai_validation_limit"):
        if options[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0 <= options["validation_fraction"] <= 1:
        raise ValueError("validation_fraction must be between zero and one")
    if (
        options["human_validation_limit"] > 0
        or options["ai_validation_limit"] > 0
    ) and not 0 < options["validation_fraction"] < 1:
        raise ValueError(
            "validation_fraction must be strictly between zero and one "
            "when validation samples are requested"
        )
    for name in (
        "minimum_group_samples",
        "minimum_validation_samples",
        "minimum_ai_validation_samples",
    ):
        if options[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0 < options["alpha"] < 1:
        raise ValueError("alpha must be between zero and one")
    if not 0 <= options["minimum_ai_recall"] <= 1:
        raise ValueError("minimum_ai_recall must be within [0, 1]")
    if options["min_effective_chars"] < 1:
        raise ValueError("min_effective_chars must be positive")
    if (
        options["max_source_items"] is not None
        and options["max_source_items"] <= 0
    ):
        raise ValueError("max_source_items must be positive when provided")


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
        segment_overlap_sentences=int(
            local.get("segment_overlap_sentences", 1)
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--profile",
        choices=("status", "stats", "models", "data", "all"),
        default="all",
    )
    parser.add_argument(
        "--dataset-profile",
        choices=("core", "multidomain", "all"),
        default=None,
    )
    parser.add_argument("--human-calibration-limit", type=int, default=None)
    parser.add_argument("--human-validation-limit", type=int, default=None)
    parser.add_argument("--ai-validation-limit", type=int, default=None)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Compatibility alias for --human-calibration-limit.",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=None,
        help="Compatibility alias for both held-out limits.",
    )
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    config = _load_config(project_root / args.config)
    detection = config.get("detection", {})
    local = detection.get("local_models", {}).get("text_yuchuan", {})
    fp = detection.get("text_false_positive_control", {})
    dataset_profile = str(
        args.dataset_profile or fp.get("dataset_profile", "multidomain")
    )
    manager = TextResourceManager(
        project_root=project_root,
        dataset_profile=dataset_profile,
    )
    if args.profile == "status":
        print(json.dumps(manager.status(), ensure_ascii=False, indent=2))
        return 0

    if args.profile == "stats":
        print(
            json.dumps(
                build_resource_stats(
                    manager,
                    short_text_chars=int(local.get("short_text_chars", 200)),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        if args.profile in {"models", "all"}:
            model_ids = [str(value) for value in local.get("models", {}).values()]
            print(f"Preparing {len(model_ids)} detector model snapshots...")
            manager.ensure_models(model_ids)
        if args.profile in {"data", "all"}:
            print(
                f"Preparing {len(manager.dataset_specs)} text sources "
                f"for profile {manager.dataset_profile}..."
            )
            manager.ensure_datasets()
        artifact_ready = manager.artifact_status()["artifact_ready"]
        if args.profile == "all" and (args.force_calibration or not artifact_ready):
            detector = _build_detector(config)
            domain_thresholds = fp.get("domain_thresholds", {})
            old_validation = args.validation_limit
            human_calibration_limit = _resolve_limit(
                args.human_calibration_limit,
                args.sample_limit,
                default=int(
                    fp.get(
                        "human_calibration_limit",
                        fp.get(
                            "bootstrap_human_calibration_limit",
                            fp.get("bootstrap_sample_limit", 300),
                        ),
                    )
                ),
            )
            human_validation_limit = _resolve_limit(
                args.human_validation_limit,
                old_validation,
                default=int(
                    fp.get(
                        "human_validation_limit",
                        fp.get(
                            "bootstrap_human_validation_limit",
                            fp.get("bootstrap_validation_limit", 100),
                        ),
                    )
                ),
            )
            ai_validation_limit = _resolve_limit(
                args.ai_validation_limit,
                old_validation,
                default=int(
                    fp.get(
                        "ai_validation_limit",
                        fp.get(
                            "bootstrap_ai_validation_limit",
                            human_validation_limit,
                        ),
                    )
                ),
            )
            max_source_items_value = fp.get("bootstrap_max_source_items")
            build_threshold_artifact(
                manager=manager,
                detector=detector,
                human_calibration_limit=human_calibration_limit,
                human_validation_limit=human_validation_limit,
                ai_validation_limit=ai_validation_limit,
                validation_fraction=float(
                    fp.get("bootstrap_validation_fraction", 0.2)
                ),
                minimum_group_samples=int(
                    fp.get("min_group_calibration_humans", 50)
                ),
                minimum_validation_samples=int(
                    fp.get(
                        "min_reliable_human_validation",
                        fp.get("min_group_validation_humans", 20),
                    )
                ),
                minimum_ai_validation_samples=int(
                    fp.get(
                        "min_reliable_ai_validation",
                        fp.get("min_group_validation_ai", 20),
                    )
                ),
                minimum_ai_recall=float(
                    fp.get(
                        "min_acceptable_ai_recall",
                        fp.get("minimum_ai_recall", 0.5),
                    )
                ),
                alpha=float(fp.get("target_fpr", 0.02)),
                short_text_chars=int(local.get("short_text_chars", 200)),
                long_target_chars=int(
                    local.get("segment_target_chars", 480)
                ),
                long_max_chars=int(local.get("segment_max_chars", 640)),
                min_effective_chars=int(
                    fp.get("bootstrap_min_effective_chars", 20)
                ),
                sampling_seed=int(
                    fp.get("bootstrap_sampling_seed", 20260722)
                ),
                max_source_items=(
                    int(max_source_items_value)
                    if max_source_items_value not in (None, "")
                    else None
                ),
                max_windows_per_document=int(
                    fp.get("bootstrap_max_windows_per_document", 1)
                ),
                low_threshold=float(fp.get("default_low_threshold", 0.4)),
                high_threshold_floors={
                    "default": float(
                        fp.get("default_high_threshold", 0.8)
                    ),
                    **{
                        name: float(
                            values.get(
                                "high_threshold",
                                fp.get("default_high_threshold", 0.8),
                            )
                        )
                        for name, values in domain_thresholds.items()
                    },
                },
                low_thresholds={
                    name: float(
                        values.get(
                            "low_threshold",
                            fp.get("default_low_threshold", 0.4),
                        )
                    )
                    for name, values in domain_thresholds.items()
                },
            )
        print(json.dumps(manager.status(), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI reports a clear failure.
        print(f"Text resource bootstrap failed: {exc}")
        if args.allow_fallback:
            print(
                "Continuing with lazy model download and built-in "
                "conservative thresholds."
            )
            return 0
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
