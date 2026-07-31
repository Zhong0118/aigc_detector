"""Build a secret-free, uploadable Phase2 neural-attribution server bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Iterable, Sequence
import zipfile


MANIFEST_NAME = "PHASE2_SERVER_MANIFEST.json"
PACKAGE_KIND = "attribution-phase2-neural-server"
PACKAGE_VERSION = "phase2-neural-screening-v1"

_EXACT_FILES = (
    "requirements.txt",
    "requirements-attribution-training.txt",
    "requirements-attribution-phase2.txt",
    "research/__init__.py",
    "scripts/run_family_attribution_baselines.py",
    "scripts/run_family_attribution_neural.py",
    "scripts/select_family_attribution_neural.py",
    "scripts/run_attribution_phase2_screening.sh",
    "scripts/package_attribution_phase2_server.py",
    "configs/text_attribution/zhizengzeng_model_catalog.v5.json",
    "configs/text_attribution/training/family_baselines.v1.json",
    "configs/text_attribution/training/family_neural_screening.v1.json",
    "docs/ATTRIBUTION_PHASE2_NEURAL_RUNBOOK.md",
    "docs/superpowers/specs/"
    "2026-07-30-hierarchical-llm-attribution-training-design.md",
    "docs/superpowers/specs/"
    "2026-07-31-family-attribution-phase2-neural-screening-design.md",
    "docs/superpowers/plans/"
    "2026-07-31-family-attribution-phase2-neural-screening.md",
    "tests/__init__.py",
    "tests/text_attribution_test_utils.py",
    "tests/test_text_attribution_training_data.py",
    "tests/test_text_attribution_split_manifest.py",
    "tests/test_text_attribution_metrics_v2.py",
    "tests/test_text_attribution_reporting_v2.py",
    "tests/test_run_family_attribution_baselines.py",
    "tests/test_text_attribution_neural_config.py",
    "tests/test_text_attribution_neural_data.py",
    "tests/test_text_attribution_neural_training.py",
    "tests/test_run_family_attribution_neural.py",
    "tests/test_text_attribution_neural_selection.py",
    "tests/test_attribution_phase2_server_package.py",
)

_FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "outputs",
        "models",
        "checkpoints",
    }
)
_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}
)
_SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        rb"(?im)^\s*(?:export\s+)?[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|SECRET_KEY)"
        rb"\s*=\s*[\"']?[A-Za-z0-9_./+=-]{12,}"
    ),
    re.compile(
        rb"(?i)[\"'](?:api[_-]?key|access[_-]?token|secret[_-]?key)[\"']"
        rb"\s*:\s*[\"'][A-Za-z0-9_./+=-]{12,}[\"']"
    ),
    re.compile(rb"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9_.-]{12,}"),
)


@dataclass(frozen=True)
class PackageBuildResult:
    archive_path: Path
    file_count: int
    source_commit: str
    source_dirty: bool | None


@dataclass(frozen=True)
class PackageVerificationResult:
    archive_path: Path
    file_count: int
    source_commit: str
    source_dirty: bool | None


def _relative_posix(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _required_source_files(project_root: Path) -> tuple[Path, ...]:
    paths = [project_root / relative_path for relative_path in _EXACT_FILES]
    text_attribution = project_root / "research" / "text_attribution"
    paths.extend(sorted(text_attribution.glob("*.py")))
    missing = [
        _relative_posix(path, project_root)
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Phase2 package is missing required source files: "
            + ", ".join(sorted(missing))
        )
    unique = {_relative_posix(path, project_root): path for path in paths}
    return tuple(unique[name] for name in sorted(unique))


def _validate_package_path(relative_path: str) -> None:
    if "\\" in relative_path or "\x00" in relative_path:
        raise ValueError(f"unsafe package path: {relative_path}")
    pure = PurePosixPath(relative_path)
    lowered_parts = tuple(part.casefold() for part in pure.parts)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or not pure.parts
        or ":" in pure.parts[0]
    ):
        raise ValueError(f"unsafe package path: {relative_path}")
    if any(part in _FORBIDDEN_PARTS or part.startswith(".env") for part in lowered_parts):
        raise ValueError(f"forbidden package path: {relative_path}")
    if any("apikey" in part or "api_key" in part for part in lowered_parts):
        raise ValueError(f"secret-like package path: {relative_path}")
    if pure.suffix.casefold() in _WEIGHT_SUFFIXES:
        raise ValueError(f"model/checkpoint weights are forbidden: {relative_path}")
    if lowered_parts[:3] == ("data", "attribution", "raw"):
        raise ValueError(f"raw data are forbidden: {relative_path}")


def _validated_source(
    source_path: Path,
    project_root: Path,
) -> tuple[str, Path]:
    """Reject lexical escapes and every symlink from root to source."""

    candidate = source_path.absolute()
    try:
        relative = candidate.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            f"package source is outside project root: {source_path}"
        ) from error

    current = project_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                f"package source has a symbolic-link ancestor: {current}"
            )

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"package source does not exist: {candidate}") from error
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            f"resolved package source is outside project root: {resolved}"
        ) from error
    if not resolved.is_file():
        raise ValueError(f"package source must be a regular file: {resolved}")
    return relative.as_posix(), resolved


def _validate_no_embedded_secret(relative_path: str, payload: bytes) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(payload):
            raise ValueError(
                f"secret-like credential detected in package source: {relative_path}"
            )


def _source_state(project_root: Path) -> tuple[str, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    value = revision.stdout.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", value) is None:
        return "unknown", None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return value, None
    return value, bool(status.stdout.strip())


def _zip_info(relative_path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative_path)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    mode = 0o755 if relative_path.endswith(".sh") else 0o644
    info.external_attr = (mode & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _manifest(
    file_payloads: Iterable[tuple[str, bytes]],
    *,
    source_commit: str,
    source_dirty: bool | None,
) -> dict[str, object]:
    records = [
        {
            "path": relative_path,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative_path, payload in file_payloads
    ]
    return {
        "schema_version": 1,
        "package_kind": PACKAGE_KIND,
        "package_version": PACKAGE_VERSION,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "layout": "project-root-overlay",
        "contains_data": False,
        "contains_model_weights": False,
        "files": records,
    }


def _load_manifest(
    archive: zipfile.ZipFile,
    archive_path: Path,
) -> Mapping[str, object]:
    try:
        payload = json.loads(archive.read(MANIFEST_NAME))
    except KeyError as error:
        raise ValueError(
            f"archive is missing required manifest: {MANIFEST_NAME}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"archive manifest is invalid JSON: {archive_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("archive manifest must be a JSON object")
    expected_keys = {
        "schema_version",
        "package_kind",
        "package_version",
        "source_commit",
        "source_dirty",
        "layout",
        "contains_data",
        "contains_model_weights",
        "files",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            f"archive manifest keys must be exactly {sorted(expected_keys)}"
        )
    if payload["schema_version"] != 1:
        raise ValueError("archive manifest schema_version must be 1")
    if payload["package_kind"] != PACKAGE_KIND:
        raise ValueError(f"archive package_kind must be {PACKAGE_KIND!r}")
    if payload["package_version"] != PACKAGE_VERSION:
        raise ValueError(f"archive package_version must be {PACKAGE_VERSION!r}")
    if payload["layout"] != "project-root-overlay":
        raise ValueError("archive layout must be project-root-overlay")
    if payload["contains_data"] is not False:
        raise ValueError("archive manifest must declare contains_data=false")
    if payload["contains_model_weights"] is not False:
        raise ValueError(
            "archive manifest must declare contains_model_weights=false"
        )
    source_commit = payload["source_commit"]
    if source_commit != "unknown" and (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-fA-F]{40,64}", source_commit) is None
    ):
        raise ValueError("archive source_commit must be a git hash or unknown")
    if payload["source_dirty"] is not None and not isinstance(
        payload["source_dirty"], bool
    ):
        raise ValueError("archive source_dirty must be boolean or null")
    return payload


def verify_phase2_server_package(
    archive_path: str | Path,
) -> PackageVerificationResult:
    """Verify CRC, manifest schema, exact entries, sizes and SHA256 hashes."""

    path = Path(archive_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Phase2 server package does not exist: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            corrupt_entry = archive.testzip()
            if corrupt_entry is not None:
                raise ValueError(
                    f"archive CRC check failed for entry: {corrupt_entry}"
                )
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("archive contains duplicate entry names")
            manifest = _load_manifest(archive, path)
            raw_records = manifest["files"]
            if not isinstance(raw_records, list):
                raise ValueError("archive manifest files must be a list")

            records: dict[str, Mapping[str, object]] = {}
            listed_order: list[str] = []
            for index, raw_record in enumerate(raw_records):
                if not isinstance(raw_record, Mapping):
                    raise ValueError(f"archive manifest file #{index} must be an object")
                if set(raw_record) != {"path", "size", "sha256"}:
                    raise ValueError(
                        "archive manifest file records require path, size and sha256"
                    )
                relative_path = raw_record["path"]
                if not isinstance(relative_path, str):
                    raise ValueError("archive manifest file path must be a string")
                _validate_package_path(relative_path)
                if relative_path == MANIFEST_NAME:
                    raise ValueError("manifest must not list itself as a source file")
                if relative_path in records:
                    raise ValueError(
                        f"archive manifest has duplicate file: {relative_path}"
                    )
                size = raw_record["size"]
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValueError(
                        f"archive manifest size is invalid: {relative_path}"
                    )
                digest = raw_record["sha256"]
                if not isinstance(digest, str) or re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ) is None:
                    raise ValueError(
                        f"archive manifest sha256 is invalid: {relative_path}"
                    )
                records[relative_path] = raw_record
                listed_order.append(relative_path)

            if listed_order != sorted(listed_order):
                raise ValueError("archive manifest files must be sorted by path")
            expected_entries = set(records) | {MANIFEST_NAME}
            if set(names) != expected_entries:
                extra = sorted(set(names) - expected_entries)
                missing = sorted(expected_entries - set(names))
                raise ValueError(
                    "archive entry set does not match manifest: "
                    f"extra={extra}, missing={missing}"
                )

            for relative_path, record in records.items():
                info = archive.getinfo(relative_path)
                payload = archive.read(relative_path)
                _validate_no_embedded_secret(relative_path, payload)
                expected_size = int(record["size"])
                if info.file_size != expected_size or len(payload) != expected_size:
                    raise ValueError(
                        f"archive size mismatch for {relative_path}"
                    )
                actual_digest = hashlib.sha256(payload).hexdigest()
                if actual_digest != record["sha256"]:
                    raise ValueError(
                        f"archive sha256 mismatch for {relative_path}"
                    )
    except zipfile.BadZipFile as error:
        raise ValueError(f"invalid ZIP archive: {path}") from error

    return PackageVerificationResult(
        archive_path=path,
        file_count=len(records),
        source_commit=str(manifest["source_commit"]),
        source_dirty=manifest["source_dirty"],
    )


def build_phase2_server_package(
    project_root: str | Path,
    output_path: str | Path,
) -> PackageBuildResult:
    """Build an atomic project-root overlay ZIP and return its metadata."""

    root_argument = Path(project_root).absolute()
    if root_argument.is_symlink():
        raise ValueError(f"project root must not be a symbolic link: {root_argument}")
    root = root_argument.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"project root does not exist: {root}")
    destination = Path(output_path).resolve()
    if destination.suffix.casefold() != ".zip":
        raise ValueError("Phase2 server package output must end with .zip")
    destination.parent.mkdir(parents=True, exist_ok=True)

    payloads: list[tuple[str, bytes]] = []
    for source_path in _required_source_files(root):
        relative_path, resolved_source = _validated_source(source_path, root)
        _validate_package_path(relative_path)
        payload = resolved_source.read_bytes()
        _validate_no_embedded_secret(relative_path, payload)
        payloads.append((relative_path, payload))

    source_commit, source_dirty = _source_state(root)
    manifest_payload = (
        json.dumps(
            _manifest(
                payloads,
                source_commit=source_commit,
                source_dirty=source_dirty,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for relative_path, payload in payloads:
                archive.writestr(_zip_info(relative_path), payload)
            archive.writestr(_zip_info(MANIFEST_NAME), manifest_payload)
        verify_phase2_server_package(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return PackageBuildResult(
        archive_path=destination,
        file_count=len(payloads),
        source_commit=source_commit,
        source_dirty=source_dirty,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a secret-free Phase2 neural-attribution project overlay ZIP."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root to package (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/aigc-detector-attribution-phase2-server.zip"),
        help="Destination .zip path.",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify an existing Phase2 server ZIP instead of building one.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify is not None:
        result = verify_phase2_server_package(args.verify)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "archive": str(result.archive_path),
                    "files": result.file_count,
                    "source_commit": result.source_commit,
                    "source_dirty": result.source_dirty,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = build_phase2_server_package(args.project_root, args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "archive": str(result.archive_path),
                "files": result.file_count,
                "source_commit": result.source_commit,
                "source_dirty": result.source_dirty,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
