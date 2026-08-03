#!/usr/bin/env python3
"""Build and verify a deterministic, secret-free Phase4 Fast Track overlay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Sequence
import zipfile


MANIFEST_NAME = "PHASE4_FAST_TRACK_SERVER_MANIFEST.json"
PACKAGE_KIND = "attribution-phase4-fast-track-server"
PACKAGE_VERSION = "phase4-fast-track-v1"
_FILES = (
    "requirements-attribution-phase4-fast-track.txt",
    "research/__init__.py",
    "research/text_attribution/__init__.py",
    "research/text_attribution/baseline_data.py",
    "research/text_attribution/training_data.py",
    "research/text_attribution/split_manifest.py",
    "research/text_attribution/phase3_data.py",
    "research/text_attribution/style_features.py",
    "research/text_attribution/classical_models.py",
    "research/text_attribution/attribution_types.py",
    "research/text_attribution/attribution_metrics.py",
    "research/text_attribution/phase4_config.py",
    "research/text_attribution/phase4_metrics.py",
    "research/text_attribution/diagnostic_analysis.py",
    "research/text_attribution/phase4_classical.py",
    "research/text_attribution/probability_fusion.py",
    "research/text_attribution/selective_policy.py",
    "research/text_attribution/phase4_reporting.py",
    "scripts/run_family_attribution_phase4_fast_track.py",
    "scripts/package_attribution_phase4_fast_track.py",
    "configs/text_attribution/training/family_phase4_fast_track.v1.json",
    "docs/ATTRIBUTION_PHASE4_FAST_TRACK_RUNBOOK.md",
    "docs/superpowers/specs/2026-08-03-family-attribution-phase4-reliability-design.md",
    "docs/superpowers/plans/2026-08-03-family-attribution-phase4-fast-track.md",
    "tests/__init__.py",
    "tests/text_attribution_test_utils.py",
    "tests/test_text_attribution_phase4_config.py",
    "tests/test_text_attribution_phase4_metrics.py",
    "tests/test_text_attribution_diagnostic_analysis.py",
    "tests/test_text_attribution_phase4_classical.py",
    "tests/test_text_attribution_probability_fusion.py",
    "tests/test_text_attribution_selective_policy.py",
    "tests/test_run_family_attribution_phase4_fast_track.py",
    "tests/test_attribution_phase4_fast_track_package.py",
)
_FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".env",
        "data",
        "outputs",
        "logs",
        "models",
        "checkpoints",
        "cache",
        "huggingface",
        "__pycache__",
    }
)
_FORBIDDEN_SUFFIXES = frozenset(
    {".joblib", ".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}
)
_SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9_.-]{12,}"),
)


@dataclass(frozen=True)
class PackageBuildResult:
    archive_path: Path
    file_count: int
    relative_paths: tuple[str, ...]


@dataclass(frozen=True)
class PackageVerificationResult:
    archive_path: Path
    file_count: int
    relative_paths: tuple[str, ...]


def build_phase4_fast_track_package(
    project_root: str | Path, archive_path: str | Path
) -> PackageBuildResult:
    root = Path(project_root).resolve()
    archive = Path(archive_path).resolve()
    entries = []
    for relative in sorted(_FILES):
        validate_package_path(relative)
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"required Phase4 package source missing: {relative}")
        payload = source.read_bytes()
        _validate_no_secret(relative, payload)
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "payload": payload,
            }
        )
    manifest = {
        "schema_version": 1,
        "package_kind": PACKAGE_KIND,
        "package_version": PACKAGE_VERSION,
        "files": [
            {"path": row["path"], "sha256": row["sha256"], "size": row["size"]}
            for row in entries
        ],
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for row in entries:
            handle.writestr(_zip_info(str(row["path"])), row["payload"])
        handle.writestr(_zip_info(MANIFEST_NAME), manifest_payload)
    return PackageBuildResult(
        archive_path=archive,
        file_count=len(entries),
        relative_paths=tuple(str(row["path"]) for row in entries),
    )


def verify_phase4_fast_track_package(
    archive_path: str | Path,
) -> PackageVerificationResult:
    archive = Path(archive_path).resolve()
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("package contains duplicate members")
        if MANIFEST_NAME not in names:
            raise ValueError("package manifest is missing")
        manifest = json.loads(handle.read(MANIFEST_NAME))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("package_kind") != PACKAGE_KIND
            or manifest.get("package_version") != PACKAGE_VERSION
            or not isinstance(manifest.get("files"), list)
        ):
            raise ValueError("invalid Phase4 package manifest")
        expected = {}
        for row in manifest["files"]:
            if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
                raise ValueError("invalid package manifest file record")
            validate_package_path(row["path"])
            expected[row["path"]] = row
        if set(names) != set(expected) | {MANIFEST_NAME}:
            raise ValueError("archive members do not match manifest")
        for name, row in expected.items():
            payload = handle.read(name)
            _validate_no_secret(name, payload)
            if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError(f"package member hash mismatch: {name}")
    return PackageVerificationResult(
        archive_path=archive,
        file_count=len(expected),
        relative_paths=tuple(sorted(expected)),
    )


def validate_package_path(relative_path: Any) -> None:
    if not isinstance(relative_path, str) or "\\" in relative_path or "\x00" in relative_path:
        raise ValueError(f"unsafe package path: {relative_path!r}")
    path = PurePosixPath(relative_path)
    lowered = tuple(part.casefold() for part in path.parts)
    if not path.parts or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe package path: {relative_path!r}")
    if any(part in _FORBIDDEN_PARTS or part.startswith(".env") for part in lowered):
        raise ValueError(f"forbidden package path: {relative_path}")
    if path.suffix.casefold() in _FORBIDDEN_SUFFIXES:
        raise ValueError(f"model weight path is forbidden: {relative_path}")


def _validate_no_secret(relative_path: str, payload: bytes) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(payload):
            raise ValueError(f"secret-like content found in {relative_path}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 8, 3, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Phase4 Fast Track server ZIP.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = (
        verify_phase4_fast_track_package(args.output)
        if args.verify_only
        else build_phase4_fast_track_package(args.project_root, args.output)
    )
    print(
        json.dumps(
            {
                "archive": str(result.archive_path),
                "file_count": result.file_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
