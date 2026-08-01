#!/usr/bin/env python3
"""Build and verify a deterministic, secret-free Phase3 server overlay ZIP."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile


MANIFEST_NAME = "PHASE3_SERVER_MANIFEST.json"
PACKAGE_KIND = "attribution-phase3-enhancement-server"
PACKAGE_VERSION = "phase3-enhancement-v1"

_EXACT_FILES = (
    "requirements-attribution-phase3.txt",
    "research/__init__.py",
    "research/text_attribution/__init__.py",
    "research/text_attribution/baseline_data.py",
    "research/text_attribution/training_data.py",
    "research/text_attribution/split_manifest.py",
    "research/text_attribution/attribution_types.py",
    "research/text_attribution/attribution_metrics.py",
    "research/text_attribution/attribution_reporting.py",
    "research/text_attribution/phase3_config.py",
    "research/text_attribution/phase3_data.py",
    "research/text_attribution/attribution_losses.py",
    "research/text_attribution/hierarchical_model.py",
    "research/text_attribution/phase3_training.py",
    "research/text_attribution/phase3_selection.py",
    "scripts/run_family_attribution_phase3.py",
    "scripts/select_family_attribution_phase3.py",
    "scripts/summarize_attribution_phase3_confirmation.py",
    "scripts/run_attribution_phase3.sh",
    "scripts/package_attribution_phase3_server.py",
    "configs/text_attribution/zhizengzeng_model_catalog.v5.json",
    "configs/text_attribution/training/family_phase3_enhancement.v1.json",
    "docs/ATTRIBUTION_PHASE3_RUNBOOK.md",
    "docs/superpowers/specs/2026-08-02-family-attribution-phase3-enhancement-design.md",
    "docs/superpowers/plans/2026-08-02-family-attribution-phase3-enhancement.md",
    "tests/__init__.py",
    "tests/test_text_attribution_phase3_config.py",
    "tests/test_text_attribution_phase3_data.py",
    "tests/test_text_attribution_attribution_losses.py",
    "tests/test_text_attribution_hierarchical_model.py",
    "tests/test_text_attribution_phase3_training.py",
    "tests/test_run_family_attribution_phase3.py",
    "tests/test_text_attribution_phase3_selection.py",
    "tests/test_run_attribution_phase3_scheduler.py",
    "tests/test_attribution_phase3_server_package.py",
)
_FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "data",
        "outputs",
        "logs",
        "models",
        "checkpoints",
        "cache",
        "huggingface",
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


def _required_source_files(project_root: Path) -> tuple[Path, ...]:
    paths = tuple(project_root / relative for relative in _EXACT_FILES)
    missing = [
        path.relative_to(project_root).as_posix()
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Phase3 package is missing required source files: "
            + ", ".join(sorted(missing))
        )
    return paths


def validate_package_path(relative_path: str) -> None:
    if not isinstance(relative_path, str) or "\\" in relative_path or "\x00" in relative_path:
        raise ValueError(f"unsafe package path: {relative_path!r}")
    pure = PurePosixPath(relative_path)
    lowered = tuple(part.casefold() for part in pure.parts)
    if (
        not pure.parts
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or ":" in pure.parts[0]
    ):
        raise ValueError(f"unsafe package path: {relative_path}")
    if any(part in _FORBIDDEN_PARTS or part.startswith(".env") for part in lowered):
        raise ValueError(f"forbidden package path: {relative_path}")
    if any("apikey" in part or "api_key" in part for part in lowered):
        raise ValueError(f"secret-like package path: {relative_path}")
    if pure.suffix.casefold() in _WEIGHT_SUFFIXES:
        raise ValueError(f"model/checkpoint weights are forbidden: {relative_path}")


def _validated_source(source: Path, project_root: Path) -> tuple[str, Path]:
    candidate = source.absolute()
    try:
        relative = candidate.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"package source is outside project root: {source}") from exc
    current = project_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"package source has a symbolic-link ancestor: {current}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"resolved package source escapes project root: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"package source must be a regular file: {resolved}")
    relative_path = relative.as_posix()
    validate_package_path(relative_path)
    return relative_path, resolved


def _validate_no_secret(relative_path: str, payload: bytes) -> None:
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
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown", None
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is None:
        return "unknown", None
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return revision, dirty


def _zip_info(relative_path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
    mode = 0o755 if relative_path.endswith(".sh") else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    return info


def _manifest(
    payloads: Sequence[tuple[str, bytes]],
    source_commit: str,
    source_dirty: bool | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "package_kind": PACKAGE_KIND,
        "package_version": PACKAGE_VERSION,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "layout": "project-root-overlay",
        "contains_data": False,
        "contains_model_weights": False,
        "files": [
            {
                "path": relative_path,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for relative_path, payload in payloads
        ],
    }


def build_phase3_server_package(
    project_root: str | Path,
    archive_path: str | Path,
) -> PackageBuildResult:
    root = Path(project_root).resolve(strict=True)
    destination = Path(archive_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payloads: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for source in _required_source_files(root):
        relative_path, validated = _validated_source(source, root)
        if relative_path in seen:
            raise ValueError(f"duplicate Phase3 package source: {relative_path}")
        seen.add(relative_path)
        payload = validated.read_bytes()
        _validate_no_secret(relative_path, payload)
        payloads.append((relative_path, payload))
    payloads.sort(key=lambda row: row[0])
    source_commit, source_dirty = _source_state(root)
    manifest_payload = json.dumps(
        _manifest(payloads, source_commit, source_dirty),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for relative_path, payload in payloads:
                archive.writestr(_zip_info(relative_path), payload)
            archive.writestr(_zip_info(MANIFEST_NAME), manifest_payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    verify_phase3_server_package(destination)
    return PackageBuildResult(
        archive_path=destination,
        file_count=len(payloads),
        source_commit=source_commit,
        source_dirty=source_dirty,
    )


def verify_phase3_server_package(
    archive_path: str | Path,
) -> PackageVerificationResult:
    path = Path(archive_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Phase3 server package does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"archive CRC check failed: {corrupt}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate entries")
        for name in names:
            validate_package_path(name)
        try:
            manifest = json.loads(archive.read(MANIFEST_NAME))
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("archive has no valid Phase3 manifest") from exc
        if not isinstance(manifest, Mapping) or set(manifest) != {
            "schema_version",
            "package_kind",
            "package_version",
            "source_commit",
            "source_dirty",
            "layout",
            "contains_data",
            "contains_model_weights",
            "files",
        }:
            raise ValueError("Phase3 archive manifest has unknown or missing fields")
        if (
            manifest["schema_version"] != 1
            or manifest["package_kind"] != PACKAGE_KIND
            or manifest["package_version"] != PACKAGE_VERSION
            or manifest["layout"] != "project-root-overlay"
            or manifest["contains_data"] is not False
            or manifest["contains_model_weights"] is not False
        ):
            raise ValueError("Phase3 archive manifest identity is invalid")
        records = manifest["files"]
        if not isinstance(records, list):
            raise ValueError("Phase3 archive manifest files must be a list")
        indexed: dict[str, Mapping[str, object]] = {}
        listed_order: list[str] = []
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "size",
                "sha256",
            }:
                raise ValueError("invalid Phase3 manifest file record")
            relative_path = record["path"]
            if not isinstance(relative_path, str):
                raise ValueError("manifest path must be a string")
            validate_package_path(relative_path)
            if relative_path in indexed or relative_path == MANIFEST_NAME:
                raise ValueError("duplicate/self-listed manifest file")
            indexed[relative_path] = record
            listed_order.append(relative_path)
        if listed_order != sorted(listed_order):
            raise ValueError("manifest file records must use stable path order")
        if set(names) != set(indexed) | {MANIFEST_NAME}:
            raise ValueError("archive entry set contains unmanifested or missing files")
        if set(indexed) != set(_EXACT_FILES):
            missing = sorted(set(_EXACT_FILES) - set(indexed))
            extra = sorted(set(indexed) - set(_EXACT_FILES))
            raise ValueError(
                "archive does not match the exact Phase3 server allowlist; "
                f"missing={missing!r}, extra={extra!r}"
            )
        for relative_path, record in indexed.items():
            payload = archive.read(relative_path)
            _validate_no_secret(relative_path, payload)
            if record["size"] != len(payload):
                raise ValueError(f"manifest size mismatch: {relative_path}")
            if record["sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"manifest sha256 mismatch: {relative_path}")
            mode = archive.getinfo(relative_path).external_attr >> 16
            if stat.S_IFMT(mode) != stat.S_IFREG:
                raise ValueError(f"archive entry must be a regular file: {relative_path}")
            if relative_path.endswith(".sh") and not mode & 0o111:
                raise ValueError(f"shell entry must be executable: {relative_path}")
    return PackageVerificationResult(
        archive_path=path,
        file_count=len(indexed),
        source_commit=str(manifest["source_commit"]),
        source_dirty=manifest["source_dirty"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the Phase3 attribution server overlay ZIP."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output")
    action.add_argument("--verify")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        result = verify_phase3_server_package(args.verify)
        action = "verified"
    else:
        result = build_phase3_server_package(args.project_root, args.output)
        action = "built"
    print(
        json.dumps(
            {
                "action": action,
                "archive": str(result.archive_path),
                "file_count": result.file_count,
                "source_commit": result.source_commit,
                "source_dirty": result.source_dirty,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
