"""Project-local datasets, models and calibration artifacts for text detection."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import stat
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import requests

from detection.text_resource_adapters import (
    TextSample,
    adapt_hc3_rows,
    adapt_m4_rows,
    adapt_raid_rows,
    adapt_tnews_rows,
)

_PORTABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")
_UNSATISFIED_CONTENT_RANGE_PATTERN = re.compile(r"^bytes \*/(\d+)$")


def _safe_portable_component(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not _PORTABLE_NAME_PATTERN.fullmatch(value)
        or value in {".", ".."}
        or value.endswith((".", " "))
        or "/" in value
        or "\\" in value
        or ":" in value
        or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            f"{field_name} must be one safe portable path component"
        )
    return value


def _url_derived_filename(url: Any) -> str:
    if not isinstance(url, str):
        raise ValueError(
            "dataset filename must be one safe portable path component"
        )
    encoded_component = urlsplit(url).path.rsplit("/", 1)[-1]
    return _safe_portable_component(
        unquote(encoded_component),
        field_name="dataset filename",
    )


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    license: str
    domain: str
    source: str
    download_kind: str = "zip"
    filename: str | None = None
    calibration_buckets: tuple[str, ...] = ("short", "long")
    source_kind: str | None = None
    required: bool = True
    adapter: str = "legacy"
    dataset_id: str | None = None
    dataset_config: str | None = None
    dataset_builder: str | None = None
    dataset_data_files: tuple[tuple[str, str], ...] = ()
    splits: tuple[str, ...] = ("train",)
    language: str | None = "zh"
    attribution_url: str | None = None

    def __post_init__(self) -> None:
        _safe_portable_component(self.name, field_name="dataset name")
        source_kind = self.source_kind
        if source_kind is None:
            source_kind = {
                "zip": "legacy_zip",
                "file": "legacy_file",
            }.get(self.download_kind, self.download_kind)
            object.__setattr__(self, "source_kind", source_kind)
        if source_kind not in {
            "legacy_zip",
            "legacy_file",
            "hf_dataset",
            "hf_jsonl",
            "raid_csv",
        }:
            raise ValueError(f"Unsupported dataset source kind: {source_kind}")
        if source_kind in {"hf_dataset", "hf_jsonl"} and not self.dataset_id:
            raise ValueError("Hugging Face resources require dataset_id")
        if source_kind == "hf_jsonl" and (
            self.adapter != "hc3" or len(self.splits) != 1
        ):
            raise ValueError(
                "Hugging Face JSONL resources require the HC3 adapter "
                "and exactly one split"
            )
        if self.dataset_builder is not None and source_kind != "hf_dataset":
            raise ValueError("Dataset builders are only supported for Hugging Face resources")
        if self.dataset_data_files and not self.dataset_builder:
            raise ValueError("Dataset data_files require a generic dataset builder")
        data_file_splits: set[str] = set()
        for item in self.dataset_data_files:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(value, str) and value for value in item)
            ):
                raise ValueError("Dataset data_files must contain (split, URL) pairs")
            split, _ = item
            if split in data_file_splits:
                raise ValueError(f"Duplicate dataset data_files split: {split}")
            data_file_splits.add(split)
        if data_file_splits and data_file_splits != set(self.splits):
            raise ValueError("Dataset data_files splits must match required splits")
        if self.filename is not None:
            _safe_portable_component(
                self.filename,
                field_name="dataset filename",
            )
        elif source_kind in {"legacy_file", "hf_jsonl", "raid_csv"}:
            _url_derived_filename(self.url)
        if self.language not in {None, "zh", "en"}:
            raise ValueError(f"Unsupported dataset language: {self.language}")

    @property
    def fingerprint(self) -> str:
        payload = _stable_json_dumps(asdict(self))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_filename(spec: DatasetSpec) -> str:
    if spec.filename is not None:
        return _safe_portable_component(
            spec.filename,
            field_name="dataset filename",
        )
    return _url_derived_filename(spec.url)


CORE_DATASETS = (
    DatasetSpec(
        name="chinese_poetry",
        url="https://github.com/chinese-poetry/chinese-poetry/archive/refs/heads/master.zip",
        license="MIT",
        domain="literature",
        source="https://github.com/chinese-poetry/chinese-poetry",
        calibration_buckets=("short",),
    ),
    DatasetSpec(
        name="chinese_classical_corpus",
        url=(
            "https://huggingface.co/datasets/gujilab/chinese-classical-corpus/"
            "resolve/main/corpus.jsonl?download=true"
        ),
        license="CC0-1.0",
        domain="literature",
        source="https://huggingface.co/datasets/gujilab/chinese-classical-corpus",
        download_kind="file",
        filename="corpus.jsonl",
        calibration_buckets=("long",),
    ),
    DatasetSpec(
        name="csl",
        url="https://github.com/ydli-ai/CSL/archive/refs/heads/master.zip",
        license="Apache-2.0",
        domain="academic_technical",
        source="https://github.com/ydli-ai/CSL",
    ),
)

MULTIDOMAIN_DATASETS = CORE_DATASETS + (
    DatasetSpec(
        name="hc3_en",
        url=(
            "https://huggingface.co/datasets/Hello-SimpleAI/HC3/"
            "resolve/main/all.jsonl"
        ),
        license="CC-BY-SA-4.0",
        domain="general_qa",
        source="https://huggingface.co/datasets/Hello-SimpleAI/HC3",
        download_kind="file",
        source_kind="hf_jsonl",
        filename="all.jsonl",
        adapter="hc3",
        dataset_id="Hello-SimpleAI/HC3",
        language="en",
    ),
    DatasetSpec(
        name="hc3_zh",
        url=(
            "https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/"
            "resolve/main/all.jsonl"
        ),
        license="CC-BY-SA-4.0",
        domain="general_qa",
        source="https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese",
        download_kind="file",
        source_kind="hf_jsonl",
        filename="all.jsonl",
        adapter="hc3",
        dataset_id="Hello-SimpleAI/HC3-Chinese",
        language="zh",
    ),
    DatasetSpec(
        name="tnews",
        url="https://huggingface.co/datasets/clue/clue",
        license="unknown",
        domain="official_news",
        source="https://huggingface.co/datasets/clue/clue",
        download_kind="hf_dataset",
        source_kind="hf_dataset",
        adapter="tnews",
        dataset_id="clue",
        dataset_config="tnews",
        splits=("train", "validation"),
        language="zh",
    ),
    DatasetSpec(
        name="m4_multilingual",
        url="https://huggingface.co/datasets/d0rj/SemEval2024-task8",
        license="Apache-2.0",
        domain="general_qa",
        source="https://huggingface.co/datasets/d0rj/SemEval2024-task8",
        download_kind="hf_dataset",
        source_kind="hf_dataset",
        adapter="m4",
        dataset_id="d0rj/SemEval2024-task8",
        dataset_config="subtaskA_multilingual",
        language=None,
        splits=("train", "dev"),
        attribution_url="https://github.com/mbzuai-nlp/SemEval2024-task8",
    ),
    DatasetSpec(
        name="raid_no_adversarial",
        url="https://dataset.raid-bench.xyz/train_none.csv",
        license="MIT",
        domain="general_qa",
        source="https://github.com/liamdugan/raid",
        download_kind="raid_csv",
        source_kind="raid_csv",
        adapter="raid",
        filename="train_none.csv",
        language="en",
    ),
)

DEFAULT_DATASETS = CORE_DATASETS
RESOURCE_PROFILES = {
    "core": CORE_DATASETS,
    "multidomain": MULTIDOMAIN_DATASETS,
    "all": MULTIDOMAIN_DATASETS,
}

_LEGACY_MARKER_FIELDS = frozenset(
    {
        "name",
        "url",
        "license",
        "domain",
        "source",
        "download_kind",
        "filename",
        "calibration_buckets",
    }
)


def conformal_high_threshold(scores: Sequence[float], alpha: float = 0.02) -> float:
    """Return the finite-sample corrected human-score threshold."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    ordered = sorted(float(score) for score in scores)
    if not ordered:
        return float("inf")
    k = math.ceil((len(ordered) + 1) * (1 - alpha))
    if k > len(ordered):
        return float("inf")
    return ordered[k - 1]


def _finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _artifact_v2_validation_errors(
    payload: Mapping[str, Any] | None,
    *,
    expected_profile: str,
) -> list[str]:
    """Validate the artifact fields required for reliable threshold routing."""
    if payload is None:
        return ["artifact_json_invalid"]

    errors: list[str] = []
    if payload.get("artifact_schema_version") != 2:
        return ["artifact_schema_version_must_be_2"]
    if payload.get("dataset_profile") != expected_profile:
        return ["artifact_dataset_profile_mismatch"]

    root_target_fpr = payload.get("target_fpr")
    if (
        not _finite_number(root_target_fpr)
        or not 0 < float(root_target_fpr) <= 1
    ):
        errors.append("artifact_target_fpr_invalid")

    groups = payload.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        errors.append("artifact_groups_must_be_non_empty_mapping")
        return errors

    for group_name, group in groups.items():
        if not isinstance(group_name, str) or not group_name:
            errors.append("artifact_group_name_invalid")
            continue
        prefix = f"artifact_group_{group_name}"
        if not isinstance(group, Mapping):
            errors.append(f"{prefix}_must_be_mapping")
            continue

        low = group.get("low_threshold")
        high = group.get("high_threshold")
        floor = group.get("threshold_floor")
        group_target_fpr = group.get("target_fpr")
        calibration_count = group.get("human_calibration_samples")

        if not _finite_number(low) or not 0 <= float(low) <= 1:
            errors.append(f"{prefix}_low_threshold_invalid")
        if not _finite_number(high) or not 0 <= float(high) <= 1:
            errors.append(f"{prefix}_high_threshold_invalid")
        if not _finite_number(floor) or not 0 <= float(floor) <= 1:
            errors.append(f"{prefix}_threshold_floor_invalid")
        if (
            _finite_number(low)
            and _finite_number(high)
            and float(low) > float(high)
        ):
            errors.append(f"{prefix}_thresholds_inverted")
        if (
            _finite_number(floor)
            and _finite_number(high)
            and float(floor) > float(high)
        ):
            errors.append(f"{prefix}_high_threshold_below_floor")
        if (
            not _finite_number(group_target_fpr)
            or not 0 < float(group_target_fpr) <= 1
        ):
            errors.append(f"{prefix}_target_fpr_invalid")
        elif (
            _finite_number(root_target_fpr)
            and not math.isclose(
                float(group_target_fpr),
                float(root_target_fpr),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            errors.append(f"{prefix}_target_fpr_mismatch")
        if (
            not isinstance(calibration_count, int)
            or isinstance(calibration_count, bool)
            or calibration_count <= 0
        ):
            errors.append(f"{prefix}_human_calibration_samples_invalid")
        for field_name in ("human_validation_samples", "ai_validation_samples"):
            count = group.get(field_name)
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                errors.append(f"{prefix}_{field_name}_invalid")
    return errors


class TextResourceManager:
    """Download approved resources and keep every byte under the project root."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        dataset_specs: Sequence[DatasetSpec] | None = None,
        *,
        dataset_profile: str | None = None,
        profile: str | None = None,
        dataset_loader: Callable[..., Any] | None = None,
        download_retries: int = 3,
        request_timeout: tuple[float, float] = (10, 120),
        max_download_bytes: int = 20 * 1024**3,
        zip_max_members: int = 100_000,
        zip_max_total_uncompressed: int = 20 * 1024**3,
        zip_max_file_uncompressed: int = 2 * 1024**3,
        zip_max_compression_ratio: float = 1_000.0,
    ) -> None:
        if dataset_profile is not None and profile is not None and dataset_profile != profile:
            raise ValueError(
                "conflicting dataset_profile and profile values: "
                f"{dataset_profile!r} != {profile!r}"
            )
        selected_profile = dataset_profile or profile or "core"
        if selected_profile not in RESOURCE_PROFILES:
            raise ValueError(f"Unknown text resource profile: {selected_profile}")
        if download_retries < 1:
            raise ValueError("download_retries must be at least one")
        if max_download_bytes < 1:
            raise ValueError("max_download_bytes must be positive")
        if (
            zip_max_members < 1
            or zip_max_total_uncompressed < 1
            or zip_max_file_uncompressed < 1
            or zip_max_compression_ratio < 1
        ):
            raise ValueError("ZIP extraction limits must be positive")
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
        self.model_dir = self._contained_path(
            self.project_root,
            "models",
            "huggingface",
        )
        self.dataset_dir = self._contained_path(
            self.project_root,
            "data",
            "text_false_positive",
            "corpora",
        )
        self.normalized_dir = self._contained_path(
            self.project_root,
            "data",
            "text_false_positive",
            "normalized",
        )
        self.artifact_path = self._contained_path(
            self.project_root,
            "artifacts",
            "text_fp_controller",
            "thresholds.json",
        )
        self.dataset_specs = tuple(
            RESOURCE_PROFILES[selected_profile] if dataset_specs is None else dataset_specs
        )
        self.dataset_profile = selected_profile
        self.profile = selected_profile
        self._dataset_loader = dataset_loader
        self.download_retries = download_retries
        self.request_timeout = request_timeout
        self.max_download_bytes = max_download_bytes
        self.zip_max_members = zip_max_members
        self.zip_max_total_uncompressed = zip_max_total_uncompressed
        self.zip_max_file_uncompressed = zip_max_file_uncompressed
        self.zip_max_compression_ratio = float(zip_max_compression_ratio)

    def _contained_path(self, root: Path, *parts: str) -> Path:
        candidate = root.joinpath(*parts)
        try:
            resolved_root = root.resolve()
            resolved_candidate = candidate.resolve()
        except OSError as exc:
            raise ValueError(f"Could not resolve managed path: {candidate}") from exc
        if (
            resolved_candidate != resolved_root
            and resolved_root not in resolved_candidate.parents
        ):
            raise ValueError(f"Managed path escapes its root: {candidate}")
        return candidate

    def _project_path(self, path: Path) -> Path:
        try:
            relative = path.absolute().relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"Managed path escapes the project root: {path}") from exc
        return self._contained_path(self.project_root, *relative.parts)

    def _resource_root(self, spec: DatasetSpec) -> Path:
        return self._contained_path(self.dataset_dir, spec.name)

    def status(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "data_dir": str(self.dataset_dir),
            "normalized_dir": str(self.normalized_dir),
            "artifact_path": str(self.artifact_path),
            **self.artifact_status(),
            "datasets": [self._dataset_status(spec) for spec in self.dataset_specs],
        }

    def artifact_status(self) -> dict[str, Any]:
        exists = self.artifact_path.is_file()
        payload = self._read_json(self.artifact_path) if exists else None
        schema_version: int | None = None
        dataset_profile: str | None = None
        if isinstance(payload, Mapping):
            raw_schema = payload.get("artifact_schema_version", 1)
            if isinstance(raw_schema, int) and not isinstance(raw_schema, bool):
                schema_version = raw_schema
            raw_profile = payload.get("dataset_profile")
            if isinstance(raw_profile, str):
                dataset_profile = raw_profile
        profile_matches = dataset_profile == self.dataset_profile
        validation_errors = (
            _artifact_v2_validation_errors(
                payload,
                expected_profile=self.dataset_profile,
            )
            if exists
            else ["artifact_missing"]
        )
        return {
            "artifact_exists": exists,
            "artifact_schema_version": schema_version,
            "artifact_dataset_profile": dataset_profile,
            "artifact_profile_matches": profile_matches,
            "artifact_validation_errors": validation_errors,
            "artifact_ready": not validation_errors,
        }

    def _dataset_status(self, spec: DatasetSpec) -> dict[str, Any]:
        source_ready = self.is_source_ready(spec)
        return {
            **asdict(spec),
            "ready": source_ready,
            "source_ready": source_ready,
            "normalized_ready": self.is_normalized_ready(spec),
            "path": str(self._resource_root(spec)),
            "normalized_path": str(self.normalized_path(spec)),
        }

    def ensure_dataset(self, spec: DatasetSpec) -> Path:
        spec = self.resource_spec(spec)
        target = self._resource_root(spec)
        if self.is_source_ready(spec):
            marker_payload = self._read_json(self._source_marker_path(spec))
            if marker_payload and self._is_legacy_migration_marker(spec, marker_payload):
                if self._migrate_legacy_marker(spec):
                    return target
            else:
                return target
        if spec.source_kind == "legacy_zip":
            self._download_and_extract(spec, target)
            return target
        if spec.source_kind in {"legacy_file", "hf_jsonl", "raid_csv"}:
            self._download_and_publish_file(spec, target)
            return target
        if spec.source_kind != "hf_dataset":
            raise ValueError(f"Unsupported dataset source kind: {spec.source_kind}")
        target.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(
            self._source_marker_path(spec),
            self._prepare_hf_source(spec),
        )
        return target

    def ensure_datasets(self) -> list[Path]:
        return [self.ensure_dataset(spec) for spec in self.dataset_specs]

    def ensure_models(self, model_ids: Sequence[str]) -> list[str]:
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshots: list[str] = []
        for model_id in dict.fromkeys(model_ids):
            if not model_id:
                continue
            snapshots.append(snapshot_download(repo_id=model_id, cache_dir=str(self.model_dir)))
        return snapshots

    def iter_human_texts(self, dataset_name: str, limit: int | None = None) -> Iterator[str]:
        spec = next((item for item in self.dataset_specs if item.name == dataset_name), None)
        root = (
            self._resource_root(spec)
            if spec is not None
            else self._contained_path(self.dataset_dir, dataset_name)
        )
        if not root.exists():
            return
        if spec is not None and spec.source_kind == "legacy_file":
            filename = _dataset_filename(spec)
            paths = (root / filename,)
            strict = True
        elif spec is not None and spec.source_kind == "legacy_zip":
            marker_payload = self._read_json(self._source_marker_path(spec))
            corpus_files = (
                marker_payload.get("corpus_files")
                if isinstance(marker_payload, dict)
                and self._legacy_zip_manifest_matches(
                    spec,
                    marker_payload,
                    deep=False,
                )
                else None
            )
            if isinstance(corpus_files, list) and corpus_files:
                authenticated: list[Path] = []
                for relative in corpus_files:
                    if not isinstance(relative, str):
                        return
                    path = self._safe_evidence_path(root, relative)
                    if path is None:
                        return
                    authenticated.append(path)
                paths = tuple(authenticated)
                strict = True
            elif marker_payload is not None:
                return
            else:
                paths = sorted(root.rglob("*"))
                strict = False
        else:
            paths = sorted(root.rglob("*"))
            strict = False
        yielded = 0
        seen: set[str] = set()
        for path in paths:
            if not path.is_file() or path.name.startswith("."):
                continue
            for text in self._texts_from_file(
                path,
                dataset_name,
                strict=strict,
            ):
                normalized = text.strip()
                if len(normalized) < 8 or normalized in seen:
                    continue
                seen.add(normalized)
                yield normalized
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def resource_spec(self, resource: str | DatasetSpec) -> DatasetSpec:
        if isinstance(resource, DatasetSpec):
            return resource
        for spec in self.dataset_specs:
            if spec.name == resource:
                return spec
        raise KeyError(f"Unknown text resource: {resource}")

    def iter_samples(self, resource: str | DatasetSpec) -> Iterator[TextSample]:
        """Backward-compatible alias for iter_source_samples()."""
        yield from self.iter_source_samples(resource)

    def iter_source_samples(self, resource: str | DatasetSpec) -> Iterator[TextSample]:
        """Yield canonical source samples sequentially without a Python list copy."""
        spec = self.resource_spec(resource)
        if spec.source_kind == "hf_jsonl":
            path = self._contained_path(
                self._resource_root(spec),
                _dataset_filename(spec),
            )
            rows = self._iter_jsonl_rows(path, spec.name)
            yield from self._adapt_rows(spec, rows, spec.splits[0])
            return
        if spec.source_kind == "hf_dataset":
            for split in spec.splits:
                rows = self._load_hf_split(spec, split)
                yield from self._adapt_rows(spec, rows, split)
            return
        if spec.source_kind == "raid_csv":
            path = self._contained_path(
                self._resource_root(spec),
                _dataset_filename(spec),
            )
            with path.open(encoding="utf-8-sig", newline="") as handle:
                yield from self._adapt_rows(spec, csv.DictReader(handle), spec.splits[0])
            return

        language = spec.language or "zh"
        for text in self.iter_human_texts(spec.name):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            yield TextSample(
                text=text,
                label="human",
                language=language,
                domain=spec.domain,
                source=spec.name,
                source_id=f"{spec.name}:legacy:{digest}",
            )

    def write_normalized_samples(
        self,
        resource: str | DatasetSpec,
        samples: Iterable[TextSample],
    ) -> Path:
        """Atomically publish a selected canonical JSONL stream and its marker."""
        spec = self.resource_spec(resource)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        output = self.normalized_path(spec)
        partial = self._project_path(output.with_suffix(output.suffix + ".part"))
        marker = self.normalized_marker_path(spec)
        marker_partial = self._project_path(
            marker.with_suffix(marker.suffix + ".part")
        )
        sample_count = 0
        content_hasher = hashlib.sha256()
        try:
            with partial.open("wb") as handle:
                for sample in samples:
                    if not isinstance(sample, TextSample):
                        raise TypeError("normalized samples must be TextSample instances")
                    if sample.source != spec.name:
                        raise ValueError(
                            f"Sample source {sample.source!r} does not match {spec.name!r}"
                        )
                    line = f"{_stable_json_dumps(asdict(sample))}\n".encode("utf-8")
                    handle.write(line)
                    content_hasher.update(line)
                    sample_count += 1
                handle.flush()
                os.fsync(handle.fileno())
            if sample_count <= 0:
                raise ValueError("normalized output requires at least one sample")
            partial.replace(output)
            output_stat = output.stat()
            self._write_json_atomic(
                marker,
                {
                    "source_spec_fingerprint": spec.fingerprint,
                    "sample_count": sample_count,
                    "normalized_path": output.name,
                    "byte_size": output_stat.st_size,
                    "mtime_ns": output_stat.st_mtime_ns,
                    "content_sha256": content_hasher.hexdigest(),
                },
                partial_path=marker_partial,
            )
        finally:
            partial.unlink(missing_ok=True)
            marker_partial.unlink(missing_ok=True)
        return output

    def iter_normalized_samples(
        self,
        resource: str | DatasetSpec,
    ) -> Iterator[TextSample]:
        spec = self.resource_spec(resource)
        if not self.is_normalized_ready(spec):
            return
        with self.normalized_path(spec).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield TextSample(**json.loads(line))

    def normalized_path(self, resource: str | DatasetSpec) -> Path:
        spec = self.resource_spec(resource)
        return self._contained_path(self.normalized_dir, f"{spec.name}.jsonl")

    def normalized_marker_path(self, resource: str | DatasetSpec) -> Path:
        spec = self.resource_spec(resource)
        return self._contained_path(
            self.normalized_dir,
            f"{spec.name}.complete.json",
        )

    def is_source_ready(
        self,
        resource: str | DatasetSpec,
        *,
        deep: bool = False,
    ) -> bool:
        spec = self.resource_spec(resource)
        marker = self._source_marker_path(spec)
        payload = self._read_json(marker)
        if payload is None:
            return False
        if spec.source_kind == "hf_jsonl":
            return self._hf_jsonl_manifest_matches(spec, payload, deep=deep)
        if spec.source_kind == "hf_dataset":
            return self._hf_manifest_matches(spec, payload, deep=deep)
        elif spec.source_kind == "raid_csv":
            return self._raid_manifest_matches(spec, payload, deep=deep)
        if self._is_legacy_migration_marker(spec, payload):
            return self._can_build_legacy_manifest(spec)
        if spec.source_kind == "legacy_zip":
            return self._legacy_zip_manifest_matches(spec, payload, deep=deep)
        if spec.source_kind == "legacy_file":
            return self._legacy_file_manifest_matches(spec, payload, deep=deep)
        return False

    def verify_source(self, resource: str | DatasetSpec) -> bool:
        """Recompute source hashes for an explicit deep integrity check."""
        return self.is_source_ready(resource, deep=True)

    def is_normalized_ready(
        self,
        resource: str | DatasetSpec,
        *,
        deep: bool = False,
    ) -> bool:
        spec = self.resource_spec(resource)
        output = self.normalized_path(spec)
        if not output.is_file():
            return False
        payload = self._read_json(self.normalized_marker_path(spec))
        if not payload:
            return False
        expected_hash = payload.get("content_sha256")
        sample_count = payload.get("sample_count")
        byte_size = payload.get("byte_size")
        mtime_ns = payload.get("mtime_ns")
        if (
            not _is_sha256(expected_hash)
            or not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count <= 0
            or not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
            or byte_size <= 0
            or not isinstance(mtime_ns, int)
            or isinstance(mtime_ns, bool)
            or mtime_ns < 0
        ):
            return False
        try:
            stat = output.stat()
        except OSError:
            return False
        return bool(
            payload.get("source_spec_fingerprint") == spec.fingerprint
            and payload.get("normalized_path") == output.name
            and stat.st_size == byte_size
            and stat.st_mtime_ns == mtime_ns
            and (not deep or self._file_sha256(output) == expected_hash)
        )

    def verify_normalized(self, resource: str | DatasetSpec) -> bool:
        """Recompute the normalized JSONL hash for an explicit deep check."""
        return self.is_normalized_ready(resource, deep=True)

    def _marker_path(self, spec: DatasetSpec) -> Path:
        return self._contained_path(self._resource_root(spec), ".complete.json")

    def _source_marker_path(self, spec: DatasetSpec) -> Path:
        if spec.source_kind in {"legacy_zip", "legacy_file"}:
            return self._marker_path(spec)
        return self._contained_path(
            self._resource_root(spec),
            ".source.complete.json",
        )

    def _download_and_extract(self, spec: DatasetSpec, target: Path) -> None:
        target = self._project_path(target)
        parent = target.parent
        archive = self._contained_path(parent, f".{spec.name}.download.zip")
        staging = self._contained_path(parent, f".{spec.name}.staging")
        backup = self._contained_path(parent, f".{spec.name}.backup")
        parent.mkdir(parents=True, exist_ok=True)
        if backup.exists() and not target.exists():
            backup.replace(target)
        self._remove_path(staging)
        previous_tree_moved = False
        new_tree_published = False
        try:
            self._download_to_path(
                spec.url,
                archive,
                source_fingerprint=spec.fingerprint,
            )
            staging.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                self._safe_extract(bundle, staging)
            marker_payload = self._legacy_zip_manifest(spec, staging)
            self._write_json_atomic(
                self._contained_path(staging, ".complete.json"),
                marker_payload,
            )
            if not self._legacy_zip_manifest_matches(
                spec,
                marker_payload,
                deep=False,
                root=staging,
            ):
                raise RuntimeError("Staged ZIP resource failed manifest validation")

            if target.exists():
                if backup.exists():
                    self._remove_path(backup)
                target.replace(backup)
                previous_tree_moved = True
            staging.replace(target)
            new_tree_published = True
            if not self.is_source_ready(spec):
                raise RuntimeError("Published ZIP resource failed manifest validation")
            if backup.exists():
                self._remove_path(backup)
        except Exception:
            if new_tree_published and target.exists():
                self._remove_path(target)
            if previous_tree_moved and backup.exists():
                backup.replace(target)
            raise
        finally:
            archive.unlink(missing_ok=True)
            self._remove_path(staging)

    def _download_and_publish_file(self, spec: DatasetSpec, target: Path) -> None:
        target = self._project_path(target)
        parent = target.parent
        staging = self._contained_path(parent, f".{spec.name}.staging")
        backup = self._contained_path(parent, f".{spec.name}.backup")
        parent.mkdir(parents=True, exist_ok=True)
        if backup.exists() and not target.exists():
            backup.replace(target)
        self._remove_path(staging)
        previous_tree_moved = False
        new_tree_published = False
        try:
            staging.mkdir()
            self._download_file(spec, staging)
            if spec.source_kind == "legacy_file":
                marker_payload = self._legacy_file_manifest(spec, staging)
                marker_name = ".complete.json"
                matches = self._legacy_file_manifest_matches(
                    spec,
                    marker_payload,
                    deep=False,
                    root=staging,
                )
            elif spec.source_kind == "hf_jsonl":
                marker_payload = self._hf_jsonl_manifest(spec, staging)
                marker_name = ".source.complete.json"
                matches = self._hf_jsonl_manifest_matches(
                    spec,
                    marker_payload,
                    deep=False,
                    root=staging,
                )
            elif spec.source_kind == "raid_csv":
                marker_payload = self._raid_source_manifest(spec, staging)
                marker_name = ".source.complete.json"
                matches = self._raid_manifest_matches(
                    spec,
                    marker_payload,
                    deep=False,
                    root=staging,
                )
            else:
                raise ValueError(
                    f"Unsupported transactional file source: {spec.source_kind}"
                )
            self._write_json_atomic(
                self._contained_path(staging, marker_name),
                marker_payload,
            )
            if not matches:
                raise RuntimeError("Staged file resource failed manifest validation")

            if target.exists():
                if backup.exists():
                    self._remove_path(backup)
                target.replace(backup)
                previous_tree_moved = True
            staging.replace(target)
            new_tree_published = True
            if not self.is_source_ready(spec):
                raise RuntimeError("Published file resource failed manifest validation")
            if backup.exists():
                self._remove_path(backup)
        except Exception:
            if new_tree_published and target.exists():
                self._remove_path(target)
            if previous_tree_moved and backup.exists():
                backup.replace(target)
            raise
        finally:
            self._remove_path(staging)

    def _download_file(self, spec: DatasetSpec, target: Path) -> None:
        filename = _dataset_filename(spec)
        destination = self._contained_path(target, filename)
        self._download_to_path(
            spec.url,
            destination,
            source_fingerprint=spec.fingerprint,
        )

    def _download_to_path(
        self,
        url: str,
        destination: Path,
        *,
        source_fingerprint: str | None = None,
    ) -> None:
        destination = self._project_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = self._contained_path(
            destination.parent,
            destination.name + ".part",
        )
        sidecar = self._contained_path(
            destination.parent,
            destination.name + ".part.json",
        )
        fingerprint = source_fingerprint or hashlib.sha256(
            url.encode("utf-8")
        ).hexdigest()
        last_error: Exception | None = None
        for attempt in range(self.download_retries):
            resume = self._load_resume_metadata(
                partial,
                sidecar,
                url=url,
                source_fingerprint=fingerprint,
            )
            offset = partial.stat().st_size if resume is not None else 0
            request_headers: dict[str, str] = {}
            if offset:
                request_headers["Range"] = f"bytes={offset}-"
                validator = resume.get("etag") or resume.get("last_modified")
                if validator:
                    request_headers["If-Range"] = validator
            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=self.request_timeout,
                    headers=request_headers,
                ) as response:
                    status_code = int(getattr(response, "status_code", 200))
                    headers = getattr(response, "headers", {}) or {}
                    if status_code == 416:
                        try:
                            total = self._http_unsatisfied_total(headers)
                        except RuntimeError:
                            self._discard_partial(partial, sidecar)
                            raise
                        if total > self.max_download_bytes:
                            self._discard_partial(partial, sidecar)
                            raise RuntimeError(
                                "HTTP download exceeds the configured size limit"
                            )
                        if (
                            resume is not None
                            and resume.get("expected_total") == total
                            and self._resume_validator_matches(resume, headers)
                            and partial.is_file()
                            and partial.stat().st_size == total
                        ):
                            partial.replace(destination)
                            sidecar.unlink(missing_ok=True)
                            return
                        self._discard_partial(partial, sidecar)
                        raise RuntimeError(
                            "HTTP 416 does not confirm a complete partial download"
                        )

                    response.raise_for_status()
                    content_length = self._http_content_length(headers)
                    expected_total: int | None = None

                    if status_code == 206:
                        content_range = self._http_header(headers, "Content-Range")
                        match = (
                            _CONTENT_RANGE_PATTERN.fullmatch(content_range)
                            if content_range
                            else None
                        )
                        if match is None or int(match.group(1)) != offset:
                            self._discard_partial(partial, sidecar)
                            raise RuntimeError("Invalid HTTP Content-Range for resume")
                        range_end = int(match.group(2))
                        if range_end < offset:
                            self._discard_partial(partial, sidecar)
                            raise RuntimeError("Invalid HTTP Content-Range bounds")
                        range_size = range_end - offset + 1
                        if content_length is not None and content_length != range_size:
                            raise RuntimeError(
                                "HTTP Content-Length does not match Content-Range"
                            )
                        if match.group(3) == "*":
                            self._discard_partial(partial, sidecar)
                            raise RuntimeError(
                                "HTTP Content-Range total must be known"
                            )
                        expected_total = int(match.group(3))
                        if range_end >= expected_total:
                            self._discard_partial(partial, sidecar)
                            raise RuntimeError("Invalid HTTP Content-Range total")
                        if (
                            resume is None
                            or resume.get("expected_total") != expected_total
                            or not self._resume_validator_matches(resume, headers)
                        ):
                            self._discard_partial(partial, sidecar)
                            raise RuntimeError(
                                "HTTP resume metadata changed during download"
                            )
                        mode = "ab"
                    elif status_code == 200:
                        mode = "wb"
                        offset = 0
                        expected_total = content_length
                    else:
                        raise RuntimeError(
                            f"Unexpected HTTP status for download: {status_code}"
                        )

                    if (
                        expected_total is not None
                        and expected_total > self.max_download_bytes
                    ):
                        self._discard_partial(partial, sidecar)
                        raise RuntimeError(
                            "HTTP download exceeds the configured size limit"
                        )
                    metadata = self._download_metadata(
                        url=url,
                        source_fingerprint=fingerprint,
                        headers=headers,
                        expected_total=expected_total,
                    )
                    if resume is not None and status_code == 206:
                        metadata["etag"] = metadata.get("etag") or resume.get("etag")
                        metadata["last_modified"] = (
                            metadata.get("last_modified")
                            or resume.get("last_modified")
                        )
                    if expected_total is not None:
                        self._write_json_atomic(sidecar, metadata)

                    received = 0
                    with partial.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                if offset + received + len(chunk) > self.max_download_bytes:
                                    raise RuntimeError(
                                        "HTTP download exceeds the configured size limit"
                                    )
                                handle.write(chunk)
                                received += len(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if content_length is not None and received != content_length:
                        raise RuntimeError(
                            "HTTP response body does not match Content-Length"
                        )
                    final_size = partial.stat().st_size
                    if expected_total is None:
                        self._discard_partial(partial, sidecar)
                        raise RuntimeError(
                            "HTTP download total is unknown and cannot be verified"
                        )
                    if final_size != expected_total:
                        raise RuntimeError("HTTP download is incomplete")
                    partial.replace(destination)
                    sidecar.unlink(missing_ok=True)
                    return
            except (OSError, requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if (
                    isinstance(exc, RuntimeError)
                    and "size limit" in str(exc)
                ):
                    self._discard_partial(partial, sidecar)
                if attempt + 1 < self.download_retries:
                    time.sleep(min(float(2**attempt), 8.0))
        if last_error is not None:
            raise last_error

    def _download_metadata(
        self,
        *,
        url: str,
        source_fingerprint: str,
        headers: Mapping[str, Any],
        expected_total: int | None,
    ) -> dict[str, Any]:
        etag = self._http_header(headers, "ETag")
        if etag and etag.upper().startswith("W/"):
            etag = None
        return {
            "resume_version": 1,
            "url": url,
            "source_spec_fingerprint": source_fingerprint,
            "etag": etag,
            "last_modified": self._http_header(headers, "Last-Modified"),
            "expected_total": expected_total,
        }

    def _load_resume_metadata(
        self,
        partial: Path,
        sidecar: Path,
        *,
        url: str,
        source_fingerprint: str,
    ) -> dict[str, Any] | None:
        if not partial.is_file():
            sidecar.unlink(missing_ok=True)
            return None
        payload = self._read_json(sidecar)
        expected_total = payload.get("expected_total") if payload else None
        validator = (
            payload.get("etag") or payload.get("last_modified")
            if payload
            else None
        )
        try:
            partial_size = partial.stat().st_size
        except OSError:
            self._discard_partial(partial, sidecar)
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("resume_version") != 1
            or payload.get("url") != url
            or payload.get("source_spec_fingerprint") != source_fingerprint
            or not isinstance(expected_total, int)
            or isinstance(expected_total, bool)
            or expected_total < 0
            or expected_total > self.max_download_bytes
            or partial_size <= 0
            or partial_size > expected_total
            or not isinstance(validator, str)
            or not validator
        ):
            self._discard_partial(partial, sidecar)
            return None
        return payload

    def _resume_validator_matches(
        self,
        resume: Mapping[str, Any],
        headers: Mapping[str, Any],
    ) -> bool:
        previous_etag = resume.get("etag")
        current_etag = self._http_header(headers, "ETag")
        if previous_etag and current_etag:
            return previous_etag == current_etag
        previous_modified = resume.get("last_modified")
        current_modified = self._http_header(headers, "Last-Modified")
        if previous_modified and current_modified:
            return previous_modified == current_modified
        return bool(previous_etag or previous_modified)

    def _discard_partial(self, partial: Path, sidecar: Path) -> None:
        partial.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)

    def _http_unsatisfied_total(self, headers: Mapping[str, Any]) -> int:
        content_range = self._http_header(headers, "Content-Range")
        match = (
            _UNSATISFIED_CONTENT_RANGE_PATTERN.fullmatch(content_range)
            if content_range
            else None
        )
        if match is None:
            raise RuntimeError("Invalid HTTP 416 Content-Range")
        return int(match.group(1))

    def _http_header(self, headers: Mapping[str, Any], name: str) -> str | None:
        for key, value in headers.items():
            if str(key).casefold() == name.casefold():
                return str(value).strip()
        return None

    def _http_content_length(self, headers: Mapping[str, Any]) -> int | None:
        raw = self._http_header(headers, "Content-Length")
        if raw is None:
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError("Invalid HTTP Content-Length") from exc
        if value < 0:
            raise RuntimeError("Invalid HTTP Content-Length")
        return value

    def _iter_jsonl_rows(
        self,
        path: Path,
        source_name: str,
    ) -> Iterator[Mapping[str, Any]]:
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Malformed {source_name} JSONL row {line_number}: {exc}"
                        ) from exc
                    if not isinstance(row, Mapping):
                        raise RuntimeError(
                            f"Malformed {source_name} JSONL row {line_number}: "
                            "expected an object"
                        )
                    yield row
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                f"Could not read authenticated {source_name} JSONL: {exc}"
            ) from exc

    def _load_hf_split(self, spec: DatasetSpec, split: str) -> Any:
        """Load one Arrow-backed split from the project cache.

        With keep_in_memory=False, Hugging Face Dataset stays disk-backed and
        memory-mapped; row iteration is sequential rather than a Python list load.
        """
        loader = self._dataset_loader
        if loader is None:
            from datasets import load_dataset

            loader = load_dataset
        cache_dir = self._contained_path(self._resource_root(spec), "hf_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        args = [spec.dataset_builder or spec.dataset_id]
        loader_kwargs: dict[str, Any] = {}
        if spec.dataset_builder is not None:
            loader_kwargs["data_files"] = dict(spec.dataset_data_files)
        elif spec.dataset_config is not None:
            args.append(spec.dataset_config)
        try:
            return loader(
                *args,
                split=split,
                cache_dir=str(cache_dir),
                keep_in_memory=False,
                **loader_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Dataset {spec.name!r} does not provide required split {split!r}: {exc}"
            ) from exc

    def _prepare_hf_source(self, spec: DatasetSpec) -> dict[str, Any]:
        split_evidence: dict[str, dict[str, Any]] = {}
        expected_labels = (
            ("human", "ai")
            if spec.adapter in {"hc3", "m4"}
            else ("human",)
        )

        for split in spec.splits:
            rows = self._load_hf_split(spec, split)
            raw_rows = 0
            adapted_samples = 0
            label_counts = {"human": 0, "ai": 0}
            content_hasher = hashlib.sha256()

            def observed_rows() -> Iterator[Mapping[str, Any]]:
                nonlocal raw_rows
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise TypeError(
                            f"Dataset {spec.name!r} split {split!r} yielded a non-mapping row"
                        )
                    encoded = _evidence_json_dumps(row).encode("utf-8")
                    content_hasher.update(len(encoded).to_bytes(8, "big"))
                    content_hasher.update(encoded)
                    raw_rows += 1
                    yield row

            for sample in self._adapt_rows(spec, observed_rows(), split):
                adapted_samples += 1
                label_counts[sample.label] += 1

            if raw_rows == 0:
                raise RuntimeError(
                    f"Dataset {spec.name!r} split {split!r} is empty or incomplete"
                )
            if adapted_samples <= 0:
                raise RuntimeError(
                    f"Dataset {spec.name!r} split {split!r} produced no adapted samples"
                )
            if any(label_counts[label] <= 0 for label in expected_labels):
                raise RuntimeError(
                    f"Dataset {spec.name!r} split {split!r} must provide "
                    f"expected labels: {', '.join(expected_labels)}"
                )
            split_evidence[split] = {
                "raw_rows": raw_rows,
                "adapted_samples": adapted_samples,
                "label_counts": label_counts,
                "content_sha256": content_hasher.hexdigest(),
            }

        scan_sha256 = hashlib.sha256(
            _stable_json_dumps(split_evidence).encode("utf-8")
        ).hexdigest()
        cache_dir = self._contained_path(self._resource_root(spec), "hf_cache")
        cache_files = self._file_inventory(cache_dir)
        if not cache_files or not any(item["byte_size"] > 0 for item in cache_files):
            raise RuntimeError(
                f"Dataset {spec.name!r} scan completed without stable project-local cache evidence"
            )
        inventory_sha256 = hashlib.sha256(
            _stable_json_dumps(cache_files).encode("utf-8")
        ).hexdigest()
        return {
            "manifest_version": 1,
            "preparation_kind": "hf_disk_cache_v1",
            "source_spec_fingerprint": spec.fingerprint,
            "dataset_id": spec.dataset_id,
            "dataset_config": spec.dataset_config,
            "dataset_builder": spec.dataset_builder,
            "dataset_data_files": dict(spec.dataset_data_files),
            "validated_splits": list(spec.splits),
            "expected_labels": list(expected_labels),
            "split_evidence": split_evidence,
            "scan_sha256": scan_sha256,
            "cache_evidence": {
                "kind": "file_inventory_sha256_v1",
                "cache_dir": "hf_cache",
                "scan_sha256": scan_sha256,
                "files": cache_files,
                "inventory_sha256": inventory_sha256,
            },
        }

    def _adapt_rows(
        self,
        spec: DatasetSpec,
        rows: Iterable[Mapping[str, Any]],
        split: str,
    ) -> Iterator[TextSample]:
        if spec.adapter == "hc3":
            yield from adapt_hc3_rows(
                self._rows_with_source_domain(rows),
                source_name=spec.name,
                language=spec.language or "en",
                split=split,
            )
            return
        if spec.adapter == "tnews":
            yield from adapt_tnews_rows(
                rows,
                source_name=spec.name,
                language=spec.language or "zh",
                split=split,
            )
            return
        if spec.adapter == "m4":
            yield from adapt_m4_rows(
                self._supported_m4_rows(rows),
                source_name=spec.name,
                language=spec.language,
                split=split,
            )
            return
        if spec.adapter == "raid":
            yield from adapt_raid_rows(
                rows,
                source_name=spec.name,
                language=spec.language or "en",
                split=split,
            )
            return
        raise ValueError(f"Unsupported text resource adapter: {spec.adapter}")

    def _supported_m4_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> Iterator[Mapping[str, Any]]:
        english_sources = {
            "arxiv",
            "peerread",
            "peer_read",
            "reddit",
            "wikipedia",
            "wikihow",
            "wiki_how",
        }
        for row in rows:
            raw_language = row.get("language")
            source = str(row.get("source") or "").strip().casefold().replace("-", "_")
            language = str(raw_language or "").strip().casefold().replace("-", "_")
            if language in {"en", "eng", "english"} or source in english_sources:
                normalized_language = "en"
            elif language in {"zh", "zh_cn", "chinese"} or source in {"zh", "chinese"}:
                normalized_language = "zh"
            else:
                continue
            enriched = dict(row)
            enriched["language"] = normalized_language
            yield enriched

    def _rows_with_source_domain(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> Iterator[Mapping[str, Any]]:
        for row in rows:
            if row.get("domain") or not row.get("source"):
                yield row
                continue
            enriched = dict(row)
            enriched["domain"] = row["source"]
            yield enriched

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _legacy_marker_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
    ) -> bool:
        if set(payload) != _LEGACY_MARKER_FIELDS:
            return False
        try:
            historical_values = dict(payload)
            buckets = historical_values.get("calibration_buckets")
            if not isinstance(buckets, (list, tuple)):
                return False
            historical_values["calibration_buckets"] = tuple(buckets)
            reconstructed = DatasetSpec(**historical_values)
        except (TypeError, ValueError):
            return False
        return _stable_json_dumps(asdict(reconstructed)) == _stable_json_dumps(asdict(spec))

    def _migrate_legacy_marker(self, spec: DatasetSpec) -> bool:
        if spec.source_kind not in {"legacy_zip", "legacy_file"}:
            return False
        marker = self._source_marker_path(spec)
        payload = self._read_json(marker)
        if not payload or not self._is_legacy_migration_marker(spec, payload):
            return False
        try:
            manifest = self._legacy_source_manifest(spec)
        except (OSError, RuntimeError):
            return False
        self._write_json_atomic(marker, manifest)
        return True

    def _is_legacy_migration_marker(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
    ) -> bool:
        if payload.get("source_spec_fingerprint") is None:
            return self._legacy_marker_matches(spec, payload)
        return bool(
            payload.get("manifest_version") == 1
            and payload.get("preparation_kind") == "standard_download_v1"
            and payload.get("source_spec_fingerprint") == spec.fingerprint
            and isinstance(payload.get("source"), dict)
            and _stable_json_dumps(payload["source"])
            == _stable_json_dumps(asdict(spec))
        )

    def _can_build_legacy_manifest(self, spec: DatasetSpec) -> bool:
        try:
            self._legacy_source_manifest(spec)
        except (OSError, RuntimeError):
            return False
        return True

    def _legacy_source_manifest(self, spec: DatasetSpec) -> dict[str, Any]:
        target = self._resource_root(spec)
        if spec.source_kind == "legacy_zip":
            return self._legacy_zip_manifest(spec, target)
        if spec.source_kind == "legacy_file":
            return self._legacy_file_manifest(spec, target)
        raise RuntimeError(f"Unsupported legacy source kind: {spec.source_kind}")

    def _legacy_file_manifest(
        self,
        spec: DatasetSpec,
        target: Path,
    ) -> dict[str, Any]:
        filename = _dataset_filename(spec)
        path = self._contained_path(target, filename)
        usable_text_count = self._validate_legacy_file(spec, path)
        evidence = self._file_evidence(target, path)
        return {
            "manifest_version": 1,
            "preparation_kind": "legacy_file_v1",
            "source_spec_fingerprint": spec.fingerprint,
            "file": evidence,
            "usable_text_count": usable_text_count,
        }

    def _legacy_file_manifest_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
        *,
        deep: bool,
        root: Path | None = None,
    ) -> bool:
        if (
            payload.get("manifest_version") != 1
            or payload.get("preparation_kind") != "legacy_file_v1"
            or payload.get("source_spec_fingerprint") != spec.fingerprint
            or not isinstance(payload.get("usable_text_count"), int)
            or isinstance(payload.get("usable_text_count"), bool)
            or payload.get("usable_text_count") <= 0
        ):
            return False
        target = root or self._resource_root(spec)
        evidence = payload.get("file")
        if not isinstance(evidence, dict):
            return False
        return self._file_evidence_matches(target, evidence, deep=deep)

    def _legacy_zip_manifest(
        self,
        spec: DatasetSpec,
        target: Path,
    ) -> dict[str, Any]:
        corpus_files, usable_text_count = self._validate_legacy_zip_corpus(
            spec,
            target,
        )
        files, tree_sha256 = self._tree_evidence(
            target,
            excluded_paths={".complete.json"},
        )
        if not files:
            raise RuntimeError(f"Extracted source tree is empty: {target}")
        return {
            "manifest_version": 1,
            "preparation_kind": "legacy_zip_tree_v1",
            "source_spec_fingerprint": spec.fingerprint,
            "files": files,
            "tree_sha256": tree_sha256,
            "corpus_files": corpus_files,
            "usable_text_count": usable_text_count,
        }

    def _legacy_zip_manifest_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
        *,
        deep: bool,
        root: Path | None = None,
    ) -> bool:
        if (
            payload.get("manifest_version") != 1
            or payload.get("preparation_kind") != "legacy_zip_tree_v1"
            or payload.get("source_spec_fingerprint") != spec.fingerprint
            or not isinstance(payload.get("files"), list)
            or not _is_sha256(payload.get("tree_sha256"))
            or not isinstance(payload.get("corpus_files"), list)
            or not payload.get("corpus_files")
            or not isinstance(payload.get("usable_text_count"), int)
            or isinstance(payload.get("usable_text_count"), bool)
            or payload.get("usable_text_count") <= 0
        ):
            return False
        files = payload["files"]
        authenticated_paths = {
            item.get("path") for item in files if isinstance(item, dict)
        }
        corpus_files = payload["corpus_files"]
        if (
            any(not isinstance(path, str) for path in corpus_files)
            or len(set(corpus_files)) != len(corpus_files)
            or not set(corpus_files).issubset(authenticated_paths)
        ):
            return False
        expected_tree = hashlib.sha256(
            _stable_json_dumps(files).encode("utf-8")
        ).hexdigest()
        return bool(
            payload["tree_sha256"] == expected_tree
            and self._inventory_evidence_matches(
                root or self._resource_root(spec),
                files,
                deep=deep,
            )
        )

    def _validate_legacy_zip_corpus(
        self,
        spec: DatasetSpec,
        target: Path,
    ) -> tuple[list[str], int]:
        corpus_files: list[str] = []
        seen: set[str] = set()
        usable_text_count = 0
        for path in sorted(target.rglob("*"), key=lambda item: item.as_posix()):
            if (
                not path.is_file()
                or path.name.startswith(".")
                or path.suffix.lower() not in {".json", ".jsonl", ".tsv"}
            ):
                continue
            relative = path.relative_to(target).as_posix()
            safe_path = self._safe_evidence_path(target, relative)
            if safe_path is None:
                raise RuntimeError(f"Corpus file escapes extracted source: {path}")
            corpus_files.append(relative)
            for text in self._texts_from_file(
                safe_path,
                spec.name,
                strict=True,
            ):
                normalized = text.strip()
                if len(normalized) < 8 or normalized in seen:
                    continue
                seen.add(normalized)
                usable_text_count += 1
        if not corpus_files:
            raise RuntimeError(
                f"Extracted source has no authenticated corpus files: {target}"
            )
        if usable_text_count <= 0:
            raise RuntimeError(
                f"Extracted source contains no usable authenticated texts: {target}"
            )
        return corpus_files, usable_text_count

    def _raid_source_manifest(
        self,
        spec: DatasetSpec,
        target: Path,
    ) -> dict[str, Any]:
        filename = _dataset_filename(spec)
        path = self._contained_path(target, filename)
        required_fields = {"generation", "model", "domain", "attack"}
        total_rows = 0
        row_count = 0
        label_counts = {"human": 0, "ai": 0}
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = {
                    field.strip() for field in (reader.fieldnames or []) if field
                }
                missing = sorted(required_fields - fieldnames)
                if missing:
                    raise RuntimeError(
                        f"RAID CSV is missing required fields: {', '.join(missing)}"
                    )
                for row in reader:
                    total_rows += 1
                    try:
                        samples = adapt_raid_rows(
                            (row,),
                            source_name=spec.name,
                            language=spec.language or "en",
                            split=spec.splits[0],
                        )
                        for sample in samples:
                            row_count += 1
                            label_counts[sample.label] += 1
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(f"RAID CSV contains an invalid row: {exc}") from exc
        except (OSError, UnicodeError, csv.Error) as exc:
            raise RuntimeError(f"RAID CSV could not be validated: {exc}") from exc
        if row_count == 0:
            raise RuntimeError("RAID CSV contains no usable data rows")
        if label_counts["human"] == 0 or label_counts["ai"] == 0:
            raise RuntimeError("RAID CSV must contain both human and AI samples")
        file_evidence = self._file_evidence(target, path)
        return {
            "manifest_version": 1,
            "preparation_kind": "raid_csv_v1",
            "source_spec_fingerprint": spec.fingerprint,
            "file": file_evidence,
            "filename": file_evidence["path"],
            "byte_size": file_evidence["byte_size"],
            "content_sha256": file_evidence["content_sha256"],
            "required_fields": sorted(required_fields),
            "total_rows": total_rows,
            "row_count": row_count,
            "label_counts": label_counts,
        }

    def _raid_manifest_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
        *,
        deep: bool,
        root: Path | None = None,
    ) -> bool:
        if (
            payload.get("manifest_version") != 1
            or payload.get("preparation_kind") != "raid_csv_v1"
            or payload.get("source_spec_fingerprint") != spec.fingerprint
            or payload.get("required_fields")
            != ["attack", "domain", "generation", "model"]
        ):
            return False
        total_rows = payload.get("total_rows")
        row_count = payload.get("row_count")
        label_counts = payload.get("label_counts")
        human_count = label_counts.get("human") if isinstance(label_counts, dict) else None
        ai_count = label_counts.get("ai") if isinstance(label_counts, dict) else None
        if (
            not isinstance(total_rows, int)
            or isinstance(total_rows, bool)
            or not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or not isinstance(label_counts, dict)
            or set(label_counts) != {"human", "ai"}
            or not isinstance(human_count, int)
            or isinstance(human_count, bool)
            or not isinstance(ai_count, int)
            or isinstance(ai_count, bool)
            or total_rows < row_count
            or row_count <= 0
            or human_count <= 0
            or ai_count <= 0
            or human_count + ai_count != row_count
        ):
            return False
        evidence = payload.get("file")
        if not isinstance(evidence, dict):
            return False
        return bool(
            payload.get("filename") == evidence.get("path")
            and payload.get("byte_size") == evidence.get("byte_size")
            and payload.get("content_sha256") == evidence.get("content_sha256")
            and self._file_evidence_matches(
                root or self._resource_root(spec),
                evidence,
                deep=deep,
            )
        )

    def _hf_jsonl_manifest(
        self,
        spec: DatasetSpec,
        target: Path,
    ) -> dict[str, Any]:
        filename = _dataset_filename(spec)
        path = self._contained_path(target, filename)
        raw_rows = 0
        adapted_samples = 0
        label_counts = {"human": 0, "ai": 0}
        scan_hasher = hashlib.sha256()

        for row in self._iter_jsonl_rows(path, spec.name):
            raw_rows += 1
            encoded = _evidence_json_dumps(row).encode("utf-8")
            scan_hasher.update(len(encoded).to_bytes(8, "big"))
            scan_hasher.update(encoded)
            try:
                for sample in self._adapt_rows(spec, (row,), spec.splits[0]):
                    adapted_samples += 1
                    label_counts[sample.label] += 1
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"HC3 JSONL contains an invalid row: {exc}"
                ) from exc

        if raw_rows <= 0:
            raise RuntimeError("HC3 JSONL contains no data rows")
        if adapted_samples <= 0:
            raise RuntimeError("HC3 JSONL produced no adapted samples")
        if label_counts["human"] <= 0 or label_counts["ai"] <= 0:
            raise RuntimeError("HC3 JSONL must contain both human and AI labels")

        file_evidence = self._file_evidence(target, path)
        return {
            "manifest_version": 1,
            "preparation_kind": "hf_jsonl_v1",
            "source_spec_fingerprint": spec.fingerprint,
            "dataset_id": spec.dataset_id,
            "split": spec.splits[0],
            "file": file_evidence,
            "filename": file_evidence["path"],
            "raw_rows": raw_rows,
            "adapted_samples": adapted_samples,
            "label_counts": label_counts,
            "scan_sha256": scan_hasher.hexdigest(),
        }

    def _hf_jsonl_manifest_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
        *,
        deep: bool,
        root: Path | None = None,
    ) -> bool:
        if (
            payload.get("manifest_version") != 1
            or payload.get("preparation_kind") != "hf_jsonl_v1"
            or payload.get("source_spec_fingerprint") != spec.fingerprint
            or payload.get("dataset_id") != spec.dataset_id
            or payload.get("split") != spec.splits[0]
            or not _is_sha256(payload.get("scan_sha256"))
        ):
            return False
        raw_rows = payload.get("raw_rows")
        adapted_samples = payload.get("adapted_samples")
        label_counts = payload.get("label_counts")
        human_count = (
            label_counts.get("human") if isinstance(label_counts, dict) else None
        )
        ai_count = label_counts.get("ai") if isinstance(label_counts, dict) else None
        if (
            not isinstance(raw_rows, int)
            or isinstance(raw_rows, bool)
            or raw_rows <= 0
            or not isinstance(adapted_samples, int)
            or isinstance(adapted_samples, bool)
            or adapted_samples <= 0
            or not isinstance(label_counts, dict)
            or set(label_counts) != {"human", "ai"}
            or not isinstance(human_count, int)
            or isinstance(human_count, bool)
            or not isinstance(ai_count, int)
            or isinstance(ai_count, bool)
            or human_count <= 0
            or ai_count <= 0
            or human_count + ai_count != adapted_samples
        ):
            return False
        evidence = payload.get("file")
        if not isinstance(evidence, dict):
            return False
        return bool(
            payload.get("filename") == evidence.get("path")
            and self._file_evidence_matches(
                root or self._resource_root(spec),
                evidence,
                deep=deep,
            )
        )

    def _hf_manifest_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
        *,
        deep: bool,
    ) -> bool:
        if (
            payload.get("manifest_version") != 1
            or payload.get("preparation_kind") != "hf_disk_cache_v1"
            or payload.get("source_spec_fingerprint") != spec.fingerprint
            or payload.get("dataset_id") != spec.dataset_id
            or payload.get("dataset_config") != spec.dataset_config
            or payload.get("validated_splits") != list(spec.splits)
        ):
            return False
        expected_labels = (
            ["human", "ai"]
            if spec.adapter in {"hc3", "m4"}
            else ["human"]
        )
        if payload.get("expected_labels") != expected_labels:
            return False
        split_evidence = payload.get("split_evidence")
        if not isinstance(split_evidence, dict) or set(split_evidence) != set(spec.splits):
            return False
        for split in spec.splits:
            evidence = split_evidence.get(split)
            if not isinstance(evidence, dict):
                return False
            raw_rows = evidence.get("raw_rows")
            adapted_samples = evidence.get("adapted_samples")
            label_counts = evidence.get("label_counts")
            human_count = (
                label_counts.get("human")
                if isinstance(label_counts, dict)
                else None
            )
            ai_count = (
                label_counts.get("ai")
                if isinstance(label_counts, dict)
                else None
            )
            if (
                not isinstance(raw_rows, int)
                or isinstance(raw_rows, bool)
                or raw_rows <= 0
                or not isinstance(adapted_samples, int)
                or isinstance(adapted_samples, bool)
                or adapted_samples <= 0
                or not isinstance(label_counts, dict)
                or set(label_counts) != {"human", "ai"}
                or not isinstance(human_count, int)
                or isinstance(human_count, bool)
                or not isinstance(ai_count, int)
                or isinstance(ai_count, bool)
                or human_count < 0
                or ai_count < 0
                or human_count + ai_count != adapted_samples
                or any(label_counts[label] <= 0 for label in expected_labels)
                or not _is_sha256(evidence.get("content_sha256"))
            ):
                return False
        expected_scan = hashlib.sha256(
            _stable_json_dumps(split_evidence).encode("utf-8")
        ).hexdigest()
        cache_evidence = payload.get("cache_evidence")
        return bool(
            payload.get("scan_sha256") == expected_scan
            and isinstance(cache_evidence, dict)
            and cache_evidence.get("kind") == "file_inventory_sha256_v1"
            and cache_evidence.get("cache_dir") == "hf_cache"
            and cache_evidence.get("scan_sha256") == expected_scan
            and self._cache_evidence_matches(spec, cache_evidence, deep=deep)
        )

    def _cache_evidence_matches(
        self,
        spec: DatasetSpec,
        evidence: Mapping[str, Any],
        *,
        deep: bool,
    ) -> bool:
        files = evidence.get("files")
        if not isinstance(files, list) or not files:
            return False
        expected_inventory = hashlib.sha256(
            _stable_json_dumps(files).encode("utf-8")
        ).hexdigest()
        if evidence.get("inventory_sha256") != expected_inventory:
            return False
        root = self._contained_path(self._resource_root(spec), "hf_cache")
        return self._inventory_evidence_matches(root, files, deep=deep)

    def _file_inventory(self, root: Path) -> list[dict[str, Any]] | None:
        if not root.is_dir():
            return None
        inventory: list[dict[str, Any]] = []
        try:
            for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                if _is_transient_evidence_path(relative):
                    continue
                safe_path = self._safe_evidence_path(root, relative.as_posix())
                if safe_path is None:
                    continue
                inventory.append(
                    self._file_evidence(root, safe_path, allow_empty=True)
                )
        except OSError:
            return None
        return inventory

    def _file_evidence(
        self,
        root: Path,
        path: Path,
        *,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise RuntimeError(f"Source file is unavailable: {path}")
        try:
            relative = path.resolve().relative_to(root.resolve())
            before = path.stat()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Source file is outside its resource root: {path}") from exc
        content_sha256 = self._file_sha256(path)
        try:
            after = path.stat()
        except OSError as exc:
            raise RuntimeError(f"Source file became unavailable: {path}") from exc
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(f"Source file changed while hashing: {path}")
        if (after.st_size <= 0 and not allow_empty) or content_sha256 is None:
            raise RuntimeError(f"Source file is empty or unreadable: {path}")
        return {
            "path": relative.as_posix(),
            "byte_size": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "content_sha256": content_sha256,
        }

    def _file_evidence_matches(
        self,
        root: Path,
        evidence: Mapping[str, Any],
        *,
        deep: bool,
        allow_empty: bool = False,
    ) -> bool:
        if set(evidence) != {"path", "byte_size", "mtime_ns", "content_sha256"}:
            return False
        relative = evidence.get("path")
        byte_size = evidence.get("byte_size")
        mtime_ns = evidence.get("mtime_ns")
        content_sha256 = evidence.get("content_sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
            or byte_size < 0
            or (byte_size == 0 and not allow_empty)
            or not isinstance(mtime_ns, int)
            or isinstance(mtime_ns, bool)
            or mtime_ns < 0
            or not _is_sha256(content_sha256)
            or _is_transient_evidence_path(Path(relative))
        ):
            return False
        path = self._safe_evidence_path(root, relative)
        if path is None or not path.is_file():
            return False
        try:
            stat = path.stat()
            if stat.st_size != byte_size or stat.st_mtime_ns != mtime_ns:
                return False
        except OSError:
            return False
        return not deep or self._file_sha256(path) == content_sha256

    def _tree_evidence(
        self,
        root: Path,
        *,
        excluded_paths: set[str],
    ) -> tuple[list[dict[str, Any]], str]:
        if not root.is_dir():
            raise OSError(f"Source tree is unavailable: {root}")
        files: list[dict[str, Any]] = []
        tree_hasher = hashlib.sha256()
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            relative_text = relative.as_posix()
            if (
                relative_text in excluded_paths
                or _is_transient_evidence_path(relative)
            ):
                continue
            safe_path = self._safe_evidence_path(root, relative_text)
            if safe_path is None:
                raise OSError(f"Source tree file escapes its root: {path}")
            byte_size = safe_path.stat().st_size
            content_sha256 = self._file_sha256(safe_path)
            if content_sha256 is None:
                raise OSError(f"Source tree file is unreadable: {path}")
            files.append(
                {
                    "path": relative_text,
                    "byte_size": byte_size,
                    "mtime_ns": safe_path.stat().st_mtime_ns,
                    "content_sha256": content_sha256,
                }
            )
        tree_hasher.update(_stable_json_dumps(files).encode("utf-8"))
        return files, tree_hasher.hexdigest()

    def _inventory_evidence_matches(
        self,
        root: Path,
        files: Any,
        *,
        deep: bool,
    ) -> bool:
        if not root.is_dir() or not isinstance(files, list) or not files:
            return False
        has_nonempty_file = False
        for evidence in files:
            if not isinstance(evidence, dict):
                return False
            byte_size = evidence.get("byte_size")
            if not isinstance(byte_size, int) or isinstance(byte_size, bool):
                return False
            if not self._file_evidence_matches(
                root,
                evidence,
                deep=deep,
                allow_empty=True,
            ):
                return False
            has_nonempty_file = has_nonempty_file or byte_size > 0
        return has_nonempty_file

    def _safe_evidence_path(self, root: Path, relative: str) -> Path | None:
        candidate = Path(relative)
        if candidate.is_absolute():
            return None
        try:
            resolved_root = root.resolve()
            resolved = (root / candidate).resolve()
        except OSError:
            return None
        if resolved != resolved_root and resolved_root not in resolved.parents:
            return None
        return resolved

    def _file_sha256(self, path: Path) -> str | None:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None
        return digest.hexdigest()

    def _remove_path(self, path: Path) -> None:
        path = self._project_path(path)
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path)

    def _write_json_atomic(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        partial_path: Path | None = None,
    ) -> None:
        path = self._project_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._project_path(
            partial_path or path.with_suffix(path.suffix + ".part")
        )
        try:
            partial.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)


    def _safe_extract(self, bundle: zipfile.ZipFile, target: Path) -> None:
        target = self._project_path(target)
        members = bundle.infolist()
        if len(members) > self.zip_max_members:
            raise ValueError("ZIP member count exceeds the configured limit")

        prepared: list[tuple[Any, tuple[str, ...], Path]] = []
        path_kinds: dict[str, bool] = {}
        total_uncompressed = 0
        for member in members:
            raw_name = str(member.filename).replace("\\", "/")
            is_directory = member.is_dir() or raw_name.endswith("/")
            raw_parts = raw_name.split("/")
            if is_directory and raw_parts and raw_parts[-1] == "":
                raw_parts = raw_parts[:-1]
            if (
                not raw_parts
                or raw_name.startswith("/")
                or any(part in {"", ".", ".."} for part in raw_parts)
            ):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            for part in raw_parts:
                base_name = part.split(".", 1)[0].upper()
                if (
                    ":" in part
                    or "\x00" in part
                    or part.endswith((".", " "))
                    or base_name in _WINDOWS_RESERVED_NAMES
                ):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            parts = tuple(raw_parts)
            normalized = "/".join(parts)
            normalized_key = normalized.casefold()
            if normalized_key in path_kinds:
                raise ValueError(f"Duplicate normalized ZIP path: {normalized}")
            ancestors = [
                "/".join(parts[:index]).casefold()
                for index in range(1, len(parts))
            ]
            if any(path_kinds.get(ancestor) is False for ancestor in ancestors):
                raise ValueError(
                    f"Windows ZIP path hierarchy collision: {normalized}"
                )
            if not is_directory and any(
                existing.startswith(normalized_key + "/")
                for existing in path_kinds
            ):
                raise ValueError(
                    f"Windows ZIP path hierarchy collision: {normalized}"
                )
            path_kinds[normalized_key] = is_directory
            if member.flag_bits & 0x1:
                raise ValueError(f"ZIP encrypted member is not allowed: {normalized}")
            unix_mode = (member.external_attr >> 16) & 0o170000
            if unix_mode == stat.S_IFLNK:
                raise ValueError(f"ZIP symbolic link is not allowed: {normalized}")

            destination = self._contained_path(target, *parts)
            if not is_directory:
                if member.file_size > self.zip_max_file_uncompressed:
                    raise ValueError(
                        f"ZIP member exceeds the per-file size limit: {normalized}"
                    )
                total_uncompressed += member.file_size
                if total_uncompressed > self.zip_max_total_uncompressed:
                    raise ValueError("ZIP total uncompressed size exceeds the limit")
                if member.file_size:
                    if member.compress_size <= 0:
                        raise ValueError(
                            f"ZIP member has an unsafe compression ratio: {normalized}"
                        )
                    ratio = member.file_size / member.compress_size
                    if ratio > self.zip_max_compression_ratio:
                        raise ValueError(
                            f"ZIP member exceeds the compression ratio limit: {normalized}"
                        )
            prepared.append((member, parts, destination))

        target.mkdir(parents=True, exist_ok=True)
        extracted_total = 0
        for member, _parts, destination in prepared:
            if member.is_dir() or str(member.filename).endswith(("/", "\\")):
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted_file = 0
            with bundle.open(member) as source, destination.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    extracted_file += len(chunk)
                    extracted_total += len(chunk)
                    if (
                        extracted_file > self.zip_max_file_uncompressed
                        or extracted_total > self.zip_max_total_uncompressed
                        or extracted_file > member.file_size
                    ):
                        raise ValueError("ZIP expanded beyond declared safety limits")
                    output.write(chunk)
            if extracted_file != member.file_size:
                raise ValueError(
                    f"ZIP member size differs from its declaration: {member.filename}"
                )

    def _validate_legacy_file(self, spec: DatasetSpec, path: Path) -> int:
        usable_text_count = 0
        seen: set[str] = set()
        for text in self._texts_from_file(path, spec.name, strict=True):
            normalized = text.strip()
            if len(normalized) < 8 or normalized in seen:
                continue
            seen.add(normalized)
            usable_text_count += 1
        if usable_text_count <= 0:
            raise RuntimeError(
                f"Authenticated legacy file contains no usable texts: {path}"
            )
        return usable_text_count

    def _texts_from_file(
        self,
        path: Path,
        dataset_name: str,
        *,
        strict: bool = False,
    ) -> Iterator[str]:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                yield from self._texts_from_payload(payload, dataset_name)
            elif suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            yield from self._texts_from_payload(json.loads(line), dataset_name)
            elif suffix == ".tsv":
                with path.open(encoding="utf-8", newline="") as handle:
                    yield from self._texts_from_tsv(handle, dataset_name)
            elif strict:
                raise RuntimeError(
                    f"Unsupported authenticated legacy file format: {path.suffix}"
                )
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
            if strict:
                raise RuntimeError(
                    f"Malformed authenticated legacy file {path}: {exc}"
                ) from exc
            return

    def _texts_from_payload(self, payload: Any, dataset_name: str) -> Iterator[str]:
        if isinstance(payload, list):
            for item in payload:
                yield from self._texts_from_payload(item, dataset_name)
            return
        if not isinstance(payload, dict):
            return

        if dataset_name == "chinese_poetry":
            paragraphs = payload.get("paragraphs") or payload.get("content")
            if isinstance(paragraphs, list):
                body = "\n".join(str(item).strip() for item in paragraphs if str(item).strip())
            else:
                body = str(paragraphs or "").strip()
            title = str(payload.get("title") or payload.get("rhythmic") or "").strip()
            if body:
                yield f"{title}。{body}" if title else body
                return

        text = self._first_text(payload, ("text_a", "abstract", "input", "text", "content"))
        if text:
            yield text
            return
        for value in payload.values():
            if isinstance(value, (list, dict)):
                yield from self._texts_from_payload(value, dataset_name)

    def _first_text(self, payload: dict[str, Any], keys: Sequence[str]) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _texts_from_tsv(self, handle: Any, dataset_name: str) -> Iterator[str]:
        reader = csv.reader(handle, delimiter="\t")
        first = next(reader, None)
        if not first:
            return
        if dataset_name == "csl" and first[0].strip().startswith("to "):
            for row in itertools.chain((first,), reader):
                if len(row) >= 2 and row[0].strip() == "to title" and row[1].strip():
                    yield row[1].strip()
            return

        headers = [item.strip() for item in first]
        for values in reader:
            row = dict(zip(headers, values))
            text = self._first_text(row, ("text_a", "abstract", "input", "text"))
            if text:
                yield text


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _evidence_json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: {
            "type": type(value).__qualname__,
            "value": str(value),
        },
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_transient_evidence_path(path: Path) -> bool:
    transient_directories = {"tmp", "temp", ".tmp", ".temp", ".locks", "locks"}
    transient_suffixes = (".part", ".tmp", ".temp", ".lock", ".incomplete")
    for part in path.parts:
        normalized = part.casefold()
        if normalized in transient_directories:
            return True
        if normalized.endswith(transient_suffixes) or normalized.startswith(".tmp-"):
            return True
    return False
