"""Build a secret-free, uploadable Phase2 neural-attribution server bundle."""

from __future__ import annotations

import argparse
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
    pure = PurePosixPath(relative_path)
    lowered_parts = tuple(part.casefold() for part in pure.parts)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe package path: {relative_path}")
    if any(part in _FORBIDDEN_PARTS or part.startswith(".env") for part in lowered_parts):
        raise ValueError(f"forbidden package path: {relative_path}")
    if any("apikey" in part or "api_key" in part for part in lowered_parts):
        raise ValueError(f"secret-like package path: {relative_path}")
    if pure.suffix.casefold() in _WEIGHT_SUFFIXES:
        raise ValueError(f"model/checkpoint weights are forbidden: {relative_path}")
    if lowered_parts[:3] == ("data", "attribution", "raw"):
        raise ValueError(f"raw data are forbidden: {relative_path}")


def _validate_no_embedded_secret(relative_path: str, payload: bytes) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(payload):
            raise ValueError(
                f"secret-like credential detected in package source: {relative_path}"
            )


def _source_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = completed.stdout.strip()
    return value if re.fullmatch(r"[0-9a-fA-F]{40,64}", value) else "unknown"


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
        "layout": "project-root-overlay",
        "contains_data": False,
        "contains_model_weights": False,
        "files": records,
    }


def build_phase2_server_package(
    project_root: str | Path,
    output_path: str | Path,
) -> PackageBuildResult:
    """Build an atomic project-root overlay ZIP and return its metadata."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"project root does not exist: {root}")
    destination = Path(output_path).resolve()
    if destination.suffix.casefold() != ".zip":
        raise ValueError("Phase2 server package output must end with .zip")
    destination.parent.mkdir(parents=True, exist_ok=True)

    payloads: list[tuple[str, bytes]] = []
    for source_path in _required_source_files(root):
        if source_path.is_symlink():
            raise ValueError(f"symbolic links are not packaged: {source_path}")
        relative_path = _relative_posix(source_path, root)
        _validate_package_path(relative_path)
        payload = source_path.read_bytes()
        _validate_no_embedded_secret(relative_path, payload)
        payloads.append((relative_path, payload))

    source_commit = _source_commit(root)
    manifest_payload = (
        json.dumps(
            _manifest(payloads, source_commit=source_commit),
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
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return PackageBuildResult(
        archive_path=destination,
        file_count=len(payloads),
        source_commit=source_commit,
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_phase2_server_package(args.project_root, args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "archive": str(result.archive_path),
                "files": result.file_count,
                "source_commit": result.source_commit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
