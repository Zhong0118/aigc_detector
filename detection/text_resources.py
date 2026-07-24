"""Project-local datasets, models and calibration artifacts for text detection."""

from __future__ import annotations

import csv
import itertools
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import requests


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


DEFAULT_DATASETS = (
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
        dataset_specs: Sequence[DatasetSpec] = DEFAULT_DATASETS,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
        self.model_dir = self.project_root / "models" / "huggingface"
        self.dataset_dir = self.project_root / "data" / "text_false_positive" / "corpora"
        self.artifact_path = self.project_root / "artifacts" / "text_fp_controller" / "thresholds.json"
        self.dataset_specs = tuple(dataset_specs)

    def status(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "data_dir": str(self.dataset_dir),
            "artifact_path": str(self.artifact_path),
            "artifact_ready": self.artifact_path.exists(),
            "datasets": [
                {
                    **asdict(spec),
                    "ready": self._marker_path(spec).exists(),
                    "path": str(self.dataset_dir / spec.name),
                }
                for spec in self.dataset_specs
            ],
        }

    def ensure_dataset(self, spec: DatasetSpec) -> Path:
        target = self.dataset_dir / spec.name
        marker = target / ".complete.json"
        if marker.exists():
            return target
        target.mkdir(parents=True, exist_ok=True)
        if spec.download_kind == "zip":
            self._download_and_extract(spec, target)
        elif spec.download_kind == "file":
            self._download_file(spec, target)
        else:
            raise ValueError(f"Unsupported dataset download kind: {spec.download_kind}")
        marker.write_text(
            json.dumps(asdict(spec), ensure_ascii=False, indent=2),
            encoding="utf-8",
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

    def _marker_path(self, spec: DatasetSpec) -> Path:
        return self.dataset_dir / spec.name / ".complete.json"

    def _download_and_extract(self, spec: DatasetSpec, target: Path) -> None:
        archive = target / ".download.zip"
        try:
            with requests.get(spec.url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with archive.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            with zipfile.ZipFile(archive) as bundle:
                self._safe_extract(bundle, target)
        finally:
            archive.unlink(missing_ok=True)

    def _download_file(self, spec: DatasetSpec, target: Path) -> None:
        filename = spec.filename or Path(spec.url.split("?", 1)[0]).name
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Unsafe dataset filename: {filename}")
        destination = target / filename
        partial = target / f".{filename}.part"
        try:
            with requests.get(spec.url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(destination)
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
