"""Frozen, leakage-safe group splits for text-attribution training data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

from research.text_attribution.training_data import AttributionSample


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    seed: int
    ratios: Mapping[str, float]
    dataset_fingerprint: str
    group_assignments: Mapping[str, str]


def dataset_fingerprint(samples: Sequence[AttributionSample]) -> str:
    """Return a stable identity for the split-relevant dataset fields."""

    rows = [
        [
            sample.request_key,
            sample.family_id,
            sample.model_id,
            sample.semantic_prompt_group,
        ]
        for sample in sorted(samples, key=lambda item: item.request_key)
    ]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_split_manifest(
    samples: Sequence[AttributionSample],
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitManifest:
    """Create and validate a deterministic semantic-group split manifest."""

    ratios = {
        "train": train_ratio,
        "validation": validation_ratio,
        "test": test_ratio,
    }
    if any(ratio <= 0 for ratio in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be positive and sum to 1")

    groups = sorted({sample.semantic_prompt_group for sample in samples})
    if len(groups) < len(SPLIT_NAMES):
        raise ValueError("At least three semantic prompt groups are required")

    shuffled_groups = list(groups)
    random.Random(seed).shuffle(shuffled_groups)
    group_counts = _allocate_group_counts(len(shuffled_groups), ratios)
    assignments: dict[str, str] = {}
    position = 0
    for split_name in SPLIT_NAMES:
        next_position = position + group_counts[split_name]
        for group in shuffled_groups[position:next_position]:
            assignments[group] = split_name
        position = next_position

    manifest = SplitManifest(
        schema_version=1,
        seed=seed,
        ratios=ratios,
        dataset_fingerprint=dataset_fingerprint(samples),
        group_assignments=assignments,
    )
    apply_split_manifest(samples, manifest)
    return manifest


def apply_split_manifest(
    samples: Sequence[AttributionSample], manifest: SplitManifest) -> dict[str, tuple[AttributionSample, ...]]:
    """Validate and apply a frozen manifest to exactly its source dataset."""

    if dataset_fingerprint(samples) != manifest.dataset_fingerprint:
        raise ValueError("Dataset fingerprint does not match split manifest")

    dataset_groups = {sample.semantic_prompt_group for sample in samples}
    assignment_groups = set(manifest.group_assignments)
    if dataset_groups != assignment_groups:
        raise ValueError("Split manifest group assignments do not match dataset groups")
    if any(split_name not in SPLIT_NAMES for split_name in manifest.group_assignments.values()):
        raise ValueError("Split manifest contains an invalid split name")

    split_samples: dict[str, list[AttributionSample]] = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        split_samples[manifest.group_assignments[sample.semantic_prompt_group]].append(sample)

    if any(not split_samples[name] for name in SPLIT_NAMES):
        raise ValueError("Split manifest produces an empty split")

    all_families = {sample.family_id for sample in samples}
    all_models = {sample.model_id for sample in samples}
    for split_name, assigned_samples in split_samples.items():
        if {sample.family_id for sample in assigned_samples} != all_families:
            raise ValueError(f"Split {split_name} does not contain every family")
        if {sample.model_id for sample in assigned_samples} != all_models:
            raise ValueError(f"Split {split_name} does not contain every model")

    return {name: tuple(split_samples[name]) for name in SPLIT_NAMES}


def save_split_manifest(path: str | Path, manifest: SplitManifest) -> None:
    """Persist a manifest as stable, human-readable UTF-8 JSON."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": manifest.schema_version,
        "seed": manifest.seed,
        "ratios": dict(manifest.ratios),
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "group_assignments": dict(manifest.group_assignments),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load a persisted split manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitManifest(
        schema_version=int(payload["schema_version"]),
        seed=int(payload["seed"]),
        ratios={name: float(value) for name, value in payload["ratios"].items()},
        dataset_fingerprint=str(payload["dataset_fingerprint"]),
        group_assignments={
            str(group): str(split_name)
            for group, split_name in payload["group_assignments"].items()
        },
    )


def _allocate_group_counts(group_total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Allocate every group, keeping every named split nonempty."""

    raw_counts = {name: group_total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(raw_counts[name]) for name in SPLIT_NAMES}
    unallocated = group_total - sum(counts.values())
    for name in sorted(
        SPLIT_NAMES,
        key=lambda item: (- (raw_counts[item] - counts[item]), SPLIT_NAMES.index(item)),
    )[:unallocated]:
        counts[name] += 1
    for name in SPLIT_NAMES:
        if counts[name] == 0:
            donor = max(SPLIT_NAMES, key=lambda item: counts[item])
            counts[donor] -= 1
            counts[name] += 1
    return counts
