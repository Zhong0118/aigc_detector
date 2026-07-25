"""Project-local datasets, models and calibration artifacts for text detection."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
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
        profile: str = "core",
        dataset_loader: Callable[..., Any] | None = None,
        download_retries: int = 3,
        request_timeout: tuple[float, float] = (10, 120),
    ) -> None:
        if profile not in RESOURCE_PROFILES:
            raise ValueError(f"Unknown text resource profile: {profile}")
        if download_retries < 1:
            raise ValueError("download_retries must be at least one")
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
        self.model_dir = self.project_root / "models" / "huggingface"
        self.dataset_dir = self.project_root / "data" / "text_false_positive" / "corpora"
        self.normalized_dir = self.project_root / "data" / "text_false_positive" / "normalized"
        self.artifact_path = self.project_root / "artifacts" / "text_fp_controller" / "thresholds.json"
        self.dataset_specs = tuple(
            RESOURCE_PROFILES[profile] if dataset_specs is None else dataset_specs
        )
        self.profile = profile
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
            "datasets": [
                {
                    **asdict(spec),
                    "ready": self.is_source_ready(spec),
                    "source_ready": self.is_source_ready(spec),
                    "normalized_ready": self.is_normalized_ready(spec),
                    "path": str(self.dataset_dir / spec.name),
                    "normalized_path": str(self.normalized_path(spec)),
                }
                for spec in self.dataset_specs
            ],
        }

    def ensure_dataset(self, spec: DatasetSpec) -> Path:
        spec = self.resource_spec(spec)
        target = self.dataset_dir / spec.name
        if self.is_source_ready(spec):
            return target
        target.mkdir(parents=True, exist_ok=True)
        if spec.source_kind == "legacy_zip":
            self._download_and_extract(spec, target)
        elif spec.source_kind in {"legacy_file", "raid_csv"}:
            self._download_file(spec, target)
        elif spec.source_kind == "hf_dataset":
            dataset = self._load_hf_dataset(spec)
            self._validate_hf_splits(spec, dataset)
        else:
            raise ValueError(f"Unsupported dataset source kind: {spec.source_kind}")
        self._write_json_atomic(
            self._source_marker_path(spec),
            {
                "source_spec_fingerprint": spec.fingerprint,
                "source": asdict(spec),
            },
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
        yielded = 0
        seen: set[str] = set()
        for path in sorted(root.rglob("*")):
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
        """Yield canonical samples without materializing the complete source in RAM."""
        spec = self.resource_spec(resource)
        if spec.source_kind == "hf_dataset":
            dataset = self._load_hf_dataset(spec)
            self._validate_hf_splits(spec, dataset)
            for split in spec.splits:
                rows = dataset[split]
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
        try:
            with partial.open("w", encoding="utf-8", newline="\n") as handle:
                for sample in samples:
                    if not isinstance(sample, TextSample):
                        raise TypeError("normalized samples must be TextSample instances")
                    if sample.source != spec.name:
                        raise ValueError(
                            f"Sample source {sample.source!r} does not match {spec.name!r}"
                        )
                    handle.write(_stable_json_dumps(asdict(sample)))
                    handle.write("\n")
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
            if not (self.dataset_dir / spec.name / "hf_cache").is_dir():
                return False
        elif spec.source_kind == "raid_csv":
            filename = spec.filename or "train_none.csv"
            if not (self.dataset_dir / spec.name / filename).is_file():
                return False
        fingerprint = payload.get("source_spec_fingerprint")
        if fingerprint is None and spec.source_kind in {"legacy_zip", "legacy_file"}:
            return True
        return fingerprint == spec.fingerprint

    def is_normalized_ready(self, resource: str | DatasetSpec) -> bool:
        spec = self.resource_spec(resource)
        if not self.normalized_path(spec).is_file():
            return False
        payload = self._read_json(self.normalized_marker_path(spec))
        return bool(
            payload
            and payload.get("source_spec_fingerprint") == spec.fingerprint
            and payload.get("normalized_path") == self.normalized_path(spec).name
        )

    def _marker_path(self, spec: DatasetSpec) -> Path:
        return self.dataset_dir / spec.name / ".complete.json"

    def _source_marker_path(self, spec: DatasetSpec) -> Path:
        if spec.source_kind in {"legacy_zip", "legacy_file"}:
            return self._marker_path(spec)
        return self.dataset_dir / spec.name / ".source.complete.json"

    def _download_and_extract(self, spec: DatasetSpec, target: Path) -> None:
        archive = target / ".download.zip"
        try:
            self._download_to_path(spec.url, archive)
            with zipfile.ZipFile(archive) as bundle:
                self._safe_extract(bundle, target)
        finally:
            archive.unlink(missing_ok=True)

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

    def _load_hf_dataset(self, spec: DatasetSpec) -> Any:
        loader = self._dataset_loader
        if loader is None:
            from datasets import load_dataset

            loader = load_dataset
        cache_dir = self.dataset_dir / spec.name / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        args = [spec.dataset_id]
        if spec.dataset_config is not None:
            args.append(spec.dataset_config)
        return loader(*args, cache_dir=str(cache_dir))

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

    def _validate_hf_splits(self, spec: DatasetSpec, dataset: Any) -> None:
        for split in spec.splits:
            try:
                dataset[split]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Dataset {spec.name!r} does not provide required split {split!r}"
                ) from exc

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
