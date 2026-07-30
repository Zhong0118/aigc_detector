"""Frozen, leakage-safe group splits for text-attribution training data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import random
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from research.text_attribution.training_data import AttributionSample


SPLIT_NAMES = ("train", "validation", "test")
_SCHEMA_VERSION = 1
_COVERAGE_TOLERANCE = 0.80
_MAX_ALLOCATION_ATTEMPTS = 1024
_MANIFEST_KEYS = {
    "schema_version",
    "seed",
    "ratios",
    "dataset_fingerprint",
    "group_assignments",
}
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    seed: int
    ratios: Mapping[str, float]
    dataset_fingerprint: str
    group_assignments: Mapping[str, str]

    def __post_init__(self) -> None:
        """Defensively freeze mapping fields without coercing their contents."""

        if isinstance(self.ratios, Mapping):
            object.__setattr__(self, "ratios", MappingProxyType(dict(self.ratios)))
        if isinstance(self.group_assignments, Mapping):
            object.__setattr__(
                self,
                "group_assignments",
                MappingProxyType(dict(self.group_assignments)),
            )


def dataset_fingerprint(samples: Sequence[AttributionSample]) -> str:
    """Return a stable identity over every field of every uniquely keyed sample."""

    _validate_unique_request_keys(samples)
    rows = [asdict(sample) for sample in sorted(samples, key=lambda item: item.request_key)]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_split_manifest(
    samples: Sequence[AttributionSample],
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitManifest:
    """Create a deterministic, coverage-safe semantic-group split manifest.

    Candidate allocations use ``seed + attempt`` for at most 1,024 deterministic
    shuffles. Each candidate must keep every label at least 80% of its expected
    count for that split, rounded up to an integer sample count.
    """

    ratios = {
        "train": train_ratio,
        "validation": validation_ratio,
        "test": test_ratio,
    }
    _validate_ratios(ratios)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Split manifest seed must be an integer")

    groups = sorted({sample.semantic_prompt_group for sample in samples})
    if len(groups) < len(SPLIT_NAMES):
        raise ValueError("At least three semantic prompt groups are required")

    fingerprint = dataset_fingerprint(samples)
    group_counts = _allocate_group_counts(len(groups), ratios)
    last_error: ValueError | None = None
    for attempt in range(_MAX_ALLOCATION_ATTEMPTS):
        shuffled_groups = list(groups)
        random.Random(seed + attempt).shuffle(shuffled_groups)
        manifest = SplitManifest(
            schema_version=_SCHEMA_VERSION,
            seed=seed,
            ratios=ratios,
            dataset_fingerprint=fingerprint,
            group_assignments=_assign_groups(shuffled_groups, group_counts),
        )
        try:
            _validate_split_manifest(manifest)
            apply_split_manifest(samples, manifest)
        except ValueError as exc:
            last_error = exc
        else:
            return manifest

    raise ValueError(
        "Unable to create a coverage-safe split manifest after "
        f"{_MAX_ALLOCATION_ATTEMPTS} deterministic allocations: {last_error}"
    )


def apply_split_manifest(
    samples: Sequence[AttributionSample],
    manifest: SplitManifest,
) -> dict[str, tuple[AttributionSample, ...]]:
    """Validate and apply a frozen manifest to exactly its source dataset."""

    _validate_split_manifest(manifest)
    if dataset_fingerprint(samples) != manifest.dataset_fingerprint:
        raise ValueError("Dataset fingerprint does not match split manifest")

    dataset_groups = {sample.semantic_prompt_group for sample in samples}
    assignment_groups = set(manifest.group_assignments)
    if dataset_groups != assignment_groups:
        raise ValueError("Split manifest group assignments do not match dataset groups")

    split_samples: dict[str, list[AttributionSample]] = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        split_samples[manifest.group_assignments[sample.semantic_prompt_group]].append(sample)

    if any(not split_samples[name] for name in SPLIT_NAMES):
        raise ValueError("Split manifest produces an empty split")

    _validate_label_coverage(split_samples, manifest.ratios)
    return {name: tuple(split_samples[name]) for name in SPLIT_NAMES}


def save_split_manifest(path: str | Path, manifest: SplitManifest) -> None:
    """Persist a validated manifest as stable, human-readable UTF-8 JSON."""

    _validate_split_manifest(manifest)
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
    """Load a strictly validated manifest without coercing JSON values."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
            raise ValueError("manifest must be an object with exactly the required fields")
        manifest = SplitManifest(
            schema_version=payload["schema_version"],
            seed=payload["seed"],
            ratios=payload["ratios"],
            dataset_fingerprint=payload["dataset_fingerprint"],
            group_assignments=payload["group_assignments"],
        )
        _validate_split_manifest(manifest)
        return manifest
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid split manifest at {manifest_path}: {exc}") from exc


def _validate_split_manifest(manifest: SplitManifest) -> None:
    if not isinstance(manifest, SplitManifest):
        raise ValueError("Split manifest must be a SplitManifest instance")
    if (
        isinstance(manifest.schema_version, bool)
        or not isinstance(manifest.schema_version, int)
        or manifest.schema_version != _SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported split manifest schema_version: {manifest.schema_version!r}")
    if isinstance(manifest.seed, bool) or not isinstance(manifest.seed, int):
        raise ValueError("Split manifest seed must be an integer")
    _validate_ratios(manifest.ratios)
    if not isinstance(manifest.dataset_fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(
        manifest.dataset_fingerprint
    ):
        raise ValueError("Split manifest dataset_fingerprint must be a 64-character hex SHA256")
    if not isinstance(manifest.group_assignments, Mapping):
        raise ValueError("Split manifest group_assignments must be an object")
    for group, split_name in manifest.group_assignments.items():
        if not isinstance(group, str) or not group:
            raise ValueError("Split manifest group assignment keys must be nonempty strings")
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Split manifest contains an invalid split name: {split_name!r}")


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    if not isinstance(ratios, Mapping) or set(ratios) != set(SPLIT_NAMES):
        raise ValueError("Split manifest ratios must contain exactly train, validation, and test")
    for name, ratio in ratios.items():
        if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Split manifest ratio {name!r} must be a finite positive number")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split manifest ratios must sum to 1")


def _validate_unique_request_keys(samples: Sequence[AttributionSample]) -> None:
    keys = [sample.request_key for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Dataset contains duplicate request_key values")


def _assign_groups(groups: Sequence[str], counts: Mapping[str, int]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    position = 0
    for split_name in SPLIT_NAMES:
        next_position = position + counts[split_name]
        assignments.update({group: split_name for group in groups[position:next_position]})
        position = next_position
    return assignments


def _validate_label_coverage(
    split_samples: Mapping[str, Sequence[AttributionSample]],
    ratios: Mapping[str, float],
) -> None:
    all_samples = tuple(sample for samples in split_samples.values() for sample in samples)
    for label_name, label_values in (
        ("family", sorted({sample.family_id for sample in all_samples})),
        ("model", sorted({sample.model_id for sample in all_samples})),
    ):
        for label_value in label_values:
            total = sum(getattr(sample, f"{label_name}_id") == label_value for sample in all_samples)
            for split_name in SPLIT_NAMES:
                expected = total * ratios[split_name]
                minimum = math.ceil(expected * _COVERAGE_TOLERANCE)
                actual = sum(
                    getattr(sample, f"{label_name}_id") == label_value
                    for sample in split_samples[split_name]
                )
                if actual < minimum:
                    raise ValueError(
                        "Split label underrepresented: "
                        f"split={split_name}, {label_name}_id={label_value!r}, "
                        f"actual={actual}, expected={expected:.3f}, minimum={minimum}"
                    )


def _allocate_group_counts(group_total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Allocate every group, keeping every named split nonempty."""

    raw_counts = {name: group_total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(raw_counts[name]) for name in SPLIT_NAMES}
    unallocated = group_total - sum(counts.values())
    for name in sorted(
        SPLIT_NAMES,
        key=lambda item: (-(raw_counts[item] - counts[item]), SPLIT_NAMES.index(item)),
    )[:unallocated]:
        counts[name] += 1
    for name in SPLIT_NAMES:
        if counts[name] == 0:
            donor = max(SPLIT_NAMES, key=lambda item: counts[item])
            counts[donor] -= 1
            counts[name] += 1
    return counts
