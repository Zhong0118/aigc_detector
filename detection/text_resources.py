"""Project-local datasets, models and calibration artifacts for text detection."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import requests

from detection.text_resource_adapters import (
    TextSample,
    adapt_hc3_rows,
    adapt_m4_rows,
    adapt_raid_rows,
    adapt_tnews_rows,
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
    splits: tuple[str, ...] = ("train",)
    language: str | None = "zh"
    attribution_url: str | None = None

    def __post_init__(self) -> None:
        source_kind = self.source_kind
        if source_kind is None:
            source_kind = {
                "zip": "legacy_zip",
                "file": "legacy_file",
            }.get(self.download_kind, self.download_kind)
            object.__setattr__(self, "source_kind", source_kind)
        if source_kind not in {"legacy_zip", "legacy_file", "hf_dataset", "raid_csv"}:
            raise ValueError(f"Unsupported dataset source kind: {source_kind}")
        if source_kind == "hf_dataset" and not self.dataset_id:
            raise ValueError("Hugging Face resources require dataset_id")
        if self.language not in {None, "zh", "en"}:
            raise ValueError(f"Unsupported dataset language: {self.language}")

    @property
    def fingerprint(self) -> str:
        payload = _stable_json_dumps(asdict(self))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        url="https://huggingface.co/datasets/Hello-SimpleAI/HC3",
        license="CC-BY-SA-4.0",
        domain="general_qa",
        source="https://huggingface.co/datasets/Hello-SimpleAI/HC3",
        download_kind="hf_dataset",
        source_kind="hf_dataset",
        adapter="hc3",
        dataset_id="Hello-SimpleAI/HC3",
        dataset_config="all",
        language="en",
    ),
    DatasetSpec(
        name="hc3_zh",
        url="https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese",
        license="CC-BY-SA-4.0",
        domain="general_qa",
        source="https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese",
        download_kind="hf_dataset",
        source_kind="hf_dataset",
        adapter="hc3",
        dataset_id="Hello-SimpleAI/HC3-Chinese",
        dataset_config="all",
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
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
        self.model_dir = self.project_root / "models" / "huggingface"
        self.dataset_dir = self.project_root / "data" / "text_false_positive" / "corpora"
        self.normalized_dir = self.project_root / "data" / "text_false_positive" / "normalized"
        self.artifact_path = self.project_root / "artifacts" / "text_fp_controller" / "thresholds.json"
        self.dataset_specs = tuple(
            RESOURCE_PROFILES[selected_profile] if dataset_specs is None else dataset_specs
        )
        self.dataset_profile = selected_profile
        self.profile = selected_profile
        self._dataset_loader = dataset_loader
        self.download_retries = download_retries
        self.request_timeout = request_timeout

    def status(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "data_dir": str(self.dataset_dir),
            "normalized_dir": str(self.normalized_dir),
            "artifact_path": str(self.artifact_path),
            "artifact_ready": self.artifact_path.exists(),
            "datasets": [self._dataset_status(spec) for spec in self.dataset_specs],
        }

    def _dataset_status(self, spec: DatasetSpec) -> dict[str, Any]:
        source_ready = self.is_source_ready(spec)
        return {
            **asdict(spec),
            "ready": source_ready,
            "source_ready": source_ready,
            "normalized_ready": self.is_normalized_ready(spec),
            "path": str(self.dataset_dir / spec.name),
            "normalized_path": str(self.normalized_path(spec)),
        }

    def ensure_dataset(self, spec: DatasetSpec) -> Path:
        spec = self.resource_spec(spec)
        target = self.dataset_dir / spec.name
        if self.is_source_ready(spec):
            marker_payload = self._read_json(self._source_marker_path(spec))
            if marker_payload and self._is_legacy_migration_marker(spec, marker_payload):
                if self._migrate_legacy_marker(spec):
                    return target
            else:
                return target
        target.mkdir(parents=True, exist_ok=True)
        marker_payload: dict[str, Any]
        if spec.source_kind == "legacy_zip":
            self._download_and_extract(spec, target)
            marker_payload = self._legacy_zip_manifest(spec, target)
        elif spec.source_kind == "legacy_file":
            self._download_file(spec, target)
            marker_payload = self._legacy_file_manifest(spec, target)
        elif spec.source_kind == "raid_csv":
            self._download_file(spec, target)
            marker_payload = self._raid_source_manifest(spec, target)
        elif spec.source_kind == "hf_dataset":
            marker_payload = self._prepare_hf_source(spec)
        else:
            raise ValueError(f"Unsupported dataset source kind: {spec.source_kind}")
        self._write_json_atomic(
            self._source_marker_path(spec),
            marker_payload,
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
        root = self.dataset_dir / dataset_name
        if not root.exists():
            return
        spec = next((item for item in self.dataset_specs if item.name == dataset_name), None)
        if spec is not None and spec.source_kind == "legacy_file":
            filename = spec.filename or Path(spec.url.split("?", 1)[0]).name
            if not filename or Path(filename).name != filename:
                return
            paths = (root / filename,)
        else:
            paths = sorted(root.rglob("*"))
        yielded = 0
        seen: set[str] = set()
        for path in paths:
            if not path.is_file() or path.name.startswith("."):
                continue
            for text in self._texts_from_file(path, dataset_name):
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
        if spec.source_kind == "hf_dataset":
            for split in spec.splits:
                rows = self._load_hf_split(spec, split)
                yield from self._adapt_rows(spec, rows, split)
            return
        if spec.source_kind == "raid_csv":
            path = self.dataset_dir / spec.name / (spec.filename or "train_none.csv")
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
        partial = output.with_suffix(output.suffix + ".part")
        marker = self.normalized_marker_path(spec)
        marker_partial = marker.with_suffix(marker.suffix + ".part")
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
            partial.replace(output)
            self._write_json_atomic(
                marker,
                {
                    "source_spec_fingerprint": spec.fingerprint,
                    "sample_count": sample_count,
                    "normalized_path": output.name,
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
        return self.normalized_dir / f"{spec.name}.jsonl"

    def normalized_marker_path(self, resource: str | DatasetSpec) -> Path:
        spec = self.resource_spec(resource)
        return self.normalized_dir / f"{spec.name}.complete.json"

    def is_source_ready(self, resource: str | DatasetSpec) -> bool:
        spec = self.resource_spec(resource)
        marker = self._source_marker_path(spec)
        payload = self._read_json(marker)
        if payload is None:
            return False
        if spec.source_kind == "hf_dataset":
            return self._hf_manifest_matches(spec, payload)
        elif spec.source_kind == "raid_csv":
            return self._raid_manifest_matches(spec, payload)
        if self._is_legacy_migration_marker(spec, payload):
            return self._can_build_legacy_manifest(spec)
        if spec.source_kind == "legacy_zip":
            return self._legacy_zip_manifest_matches(spec, payload)
        if spec.source_kind == "legacy_file":
            return self._legacy_file_manifest_matches(spec, payload)
        return False

    def is_normalized_ready(self, resource: str | DatasetSpec) -> bool:
        spec = self.resource_spec(resource)
        if not self.normalized_path(spec).is_file():
            return False
        payload = self._read_json(self.normalized_marker_path(spec))
        if not payload:
            return False
        expected_hash = payload.get("content_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            return False
        return bool(
            payload.get("source_spec_fingerprint") == spec.fingerprint
            and payload.get("normalized_path") == self.normalized_path(spec).name
            and self._file_sha256(self.normalized_path(spec)) == expected_hash
        )

    def _marker_path(self, spec: DatasetSpec) -> Path:
        return self.dataset_dir / spec.name / ".complete.json"

    def _source_marker_path(self, spec: DatasetSpec) -> Path:
        if spec.source_kind in {"legacy_zip", "legacy_file"}:
            return self._marker_path(spec)
        return self.dataset_dir / spec.name / ".source.complete.json"

    def _download_and_extract(self, spec: DatasetSpec, target: Path) -> None:
        parent = target.parent
        archive = parent / f".{spec.name}.download.zip"
        staging = parent / f".{spec.name}.staging"
        backup = parent / f".{spec.name}.backup"
        parent.mkdir(parents=True, exist_ok=True)
        self._remove_path(staging)
        self._remove_path(backup)
        try:
            self._download_to_path(spec.url, archive)
            staging.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                self._safe_extract(bundle, staging)
            if target.exists():
                target.replace(backup)
            try:
                staging.replace(target)
            except Exception:
                if backup.exists() and not target.exists():
                    backup.replace(target)
                raise
            self._remove_path(backup)
        finally:
            archive.unlink(missing_ok=True)
            self._remove_path(staging)
            if backup.exists() and not target.exists():
                backup.replace(target)

    def _download_file(self, spec: DatasetSpec, target: Path) -> None:
        filename = spec.filename or Path(spec.url.split("?", 1)[0]).name
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Unsafe dataset filename: {filename}")
        destination = target / filename
        self._download_to_path(spec.url, destination)

    def _download_to_path(self, url: str, destination: Path) -> None:
        partial = destination.with_suffix(destination.suffix + ".part")
        last_error: Exception | None = None
        try:
            for _attempt in range(self.download_retries):
                partial.unlink(missing_ok=True)
                try:
                    with requests.get(
                        url,
                        stream=True,
                        timeout=self.request_timeout,
                    ) as response:
                        response.raise_for_status()
                        with partial.open("wb") as handle:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                    partial.replace(destination)
                    return
                except (OSError, requests.RequestException) as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
        finally:
            partial.unlink(missing_ok=True)

    def _load_hf_split(self, spec: DatasetSpec, split: str) -> Any:
        """Load one Arrow-backed split from the project cache.

        With keep_in_memory=False, Hugging Face Dataset stays disk-backed and
        memory-mapped; row iteration is sequential rather than a Python list load.
        """
        loader = self._dataset_loader
        if loader is None:
            from datasets import load_dataset

            loader = load_dataset
        cache_dir = self.dataset_dir / spec.name / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        args = [spec.dataset_id]
        if spec.dataset_config is not None:
            args.append(spec.dataset_config)
        try:
            return loader(
                *args,
                split=split,
                cache_dir=str(cache_dir),
                keep_in_memory=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Dataset {spec.name!r} does not provide required split {split!r}: {exc}"
            ) from exc

    def _prepare_hf_source(self, spec: DatasetSpec) -> dict[str, Any]:
        split_evidence: dict[str, dict[str, Any]] = {}

        for split in spec.splits:
            rows = self._load_hf_split(spec, split)
            raw_rows = 0
            adapted_samples = 0
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

            for _sample in self._adapt_rows(spec, observed_rows(), split):
                adapted_samples += 1

            if raw_rows == 0:
                raise RuntimeError(
                    f"Dataset {spec.name!r} split {split!r} is empty or incomplete"
                )
            split_evidence[split] = {
                "raw_rows": raw_rows,
                "adapted_samples": adapted_samples,
                "content_sha256": content_hasher.hexdigest(),
            }

        scan_sha256 = hashlib.sha256(
            _stable_json_dumps(split_evidence).encode("utf-8")
        ).hexdigest()
        cache_dir = self.dataset_dir / spec.name / "hf_cache"
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
            "validated_splits": list(spec.splits),
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
        target = self.dataset_dir / spec.name
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
        filename = spec.filename or Path(spec.url.split("?", 1)[0]).name
        path = target / filename
        evidence = self._file_evidence(target, path)
        return {
            "manifest_version": 1,
            "preparation_kind": "legacy_file_v1",
            "source_spec_fingerprint": spec.fingerprint,
            "file": evidence,
        }

    def _legacy_file_manifest_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
    ) -> bool:
        if (
            payload.get("manifest_version") != 1
            or payload.get("preparation_kind") != "legacy_file_v1"
            or payload.get("source_spec_fingerprint") != spec.fingerprint
        ):
            return False
        target = self.dataset_dir / spec.name
        evidence = payload.get("file")
        if not isinstance(evidence, dict):
            return False
        return self._file_evidence_matches(target, evidence)

    def _legacy_zip_manifest(
        self,
        spec: DatasetSpec,
        target: Path,
    ) -> dict[str, Any]:
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
        }

    def _legacy_zip_manifest_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
    ) -> bool:
        if (
            payload.get("manifest_version") != 1
            or payload.get("preparation_kind") != "legacy_zip_tree_v1"
            or payload.get("source_spec_fingerprint") != spec.fingerprint
            or not isinstance(payload.get("files"), list)
            or not _is_sha256(payload.get("tree_sha256"))
        ):
            return False
        try:
            files, tree_sha256 = self._tree_evidence(
                self.dataset_dir / spec.name,
                excluded_paths={".complete.json"},
            )
        except OSError:
            return False
        return files == payload["files"] and tree_sha256 == payload["tree_sha256"]

    def _raid_source_manifest(
        self,
        spec: DatasetSpec,
        target: Path,
    ) -> dict[str, Any]:
        filename = spec.filename or "train_none.csv"
        path = target / filename
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
            and self._file_evidence_matches(self.dataset_dir / spec.name, evidence)
        )

    def _hf_manifest_matches(
        self,
        spec: DatasetSpec,
        payload: Mapping[str, Any],
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
        split_evidence = payload.get("split_evidence")
        if not isinstance(split_evidence, dict) or set(split_evidence) != set(spec.splits):
            return False
        for split in spec.splits:
            evidence = split_evidence.get(split)
            if not isinstance(evidence, dict):
                return False
            raw_rows = evidence.get("raw_rows")
            adapted_samples = evidence.get("adapted_samples")
            if (
                not isinstance(raw_rows, int)
                or isinstance(raw_rows, bool)
                or raw_rows <= 0
                or not isinstance(adapted_samples, int)
                or isinstance(adapted_samples, bool)
                or adapted_samples < 0
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
            and self._cache_evidence_matches(spec, cache_evidence)
        )

    def _cache_evidence_matches(
        self,
        spec: DatasetSpec,
        evidence: Mapping[str, Any],
    ) -> bool:
        files = evidence.get("files")
        if not isinstance(files, list) or not files:
            return False
        expected_inventory = hashlib.sha256(
            _stable_json_dumps(files).encode("utf-8")
        ).hexdigest()
        if evidence.get("inventory_sha256") != expected_inventory:
            return False
        root = self.dataset_dir / spec.name / "hf_cache"
        if not root.is_dir():
            return False
        has_nonempty_file = False
        for item in files:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "byte_size", "content_sha256"}
            ):
                return False
            relative = item.get("path")
            byte_size = item.get("byte_size")
            content_sha256 = item.get("content_sha256")
            if (
                not isinstance(relative, str)
                or not isinstance(byte_size, int)
                or isinstance(byte_size, bool)
                or byte_size < 0
                or not _is_sha256(content_sha256)
                or _is_transient_evidence_path(Path(relative))
            ):
                return False
            path = self._safe_evidence_path(root, relative)
            if path is None or not path.is_file():
                return False
            try:
                if path.stat().st_size != byte_size:
                    return False
            except OSError:
                return False
            if self._file_sha256(path) != content_sha256:
                return False
            has_nonempty_file = has_nonempty_file or byte_size > 0
        return has_nonempty_file

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
                content_sha256 = self._file_sha256(safe_path)
                if content_sha256 is None:
                    return None
                inventory.append(
                    {
                        "path": relative.as_posix(),
                        "byte_size": safe_path.stat().st_size,
                        "content_sha256": content_sha256,
                    }
                )
        except OSError:
            return None
        return inventory

    def _file_evidence(self, root: Path, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise RuntimeError(f"Source file is unavailable: {path}")
        try:
            relative = path.resolve().relative_to(root.resolve())
            byte_size = path.stat().st_size
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Source file is outside its resource root: {path}") from exc
        content_sha256 = self._file_sha256(path)
        if byte_size <= 0 or content_sha256 is None:
            raise RuntimeError(f"Source file is empty or unreadable: {path}")
        return {
            "path": relative.as_posix(),
            "byte_size": byte_size,
            "content_sha256": content_sha256,
        }

    def _file_evidence_matches(
        self,
        root: Path,
        evidence: Mapping[str, Any],
    ) -> bool:
        if set(evidence) != {"path", "byte_size", "content_sha256"}:
            return False
        relative = evidence.get("path")
        byte_size = evidence.get("byte_size")
        content_sha256 = evidence.get("content_sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
            or byte_size <= 0
            or not _is_sha256(content_sha256)
            or _is_transient_evidence_path(Path(relative))
        ):
            return False
        path = self._safe_evidence_path(root, relative)
        if path is None or not path.is_file():
            return False
        try:
            if path.stat().st_size != byte_size:
                return False
        except OSError:
            return False
        return self._file_sha256(path) == content_sha256

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
            files.append({"path": relative_text, "byte_size": byte_size})
            encoded_path = relative_text.encode("utf-8")
            tree_hasher.update(len(encoded_path).to_bytes(8, "big"))
            tree_hasher.update(encoded_path)
            tree_hasher.update(byte_size.to_bytes(8, "big"))
            tree_hasher.update(bytes.fromhex(content_sha256))
        return files, tree_hasher.hexdigest()

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
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = partial_path or path.with_suffix(path.suffix + ".part")
        try:
            partial.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)


    def _safe_extract(self, bundle: zipfile.ZipFile, target: Path) -> None:
        root = target.resolve()
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(target)

    def _texts_from_file(self, path: Path, dataset_name: str) -> Iterator[str]:
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
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
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
