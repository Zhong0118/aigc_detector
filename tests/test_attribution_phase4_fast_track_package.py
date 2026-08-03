from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from scripts.package_attribution_phase4_fast_track import (
    build_phase4_fast_track_package,
    verify_phase4_fast_track_package,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_contains_fast_track_code_and_no_runtime_artifacts(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "phase4.zip"
    built = build_phase4_fast_track_package(PROJECT_ROOT, archive)
    verified = verify_phase4_fast_track_package(archive)
    assert verified.file_count == built.file_count
    names = set(verified.relative_paths)
    assert "research/text_attribution/probability_fusion.py" in names
    assert "scripts/run_family_attribution_phase4_fast_track.py" in names
    assert "docs/ATTRIBUTION_PHASE4_FAST_TRACK_RUNBOOK.md" in names
    assert not any(name.startswith("data/") for name in names)
    assert not any("outputs/" in name for name in names)
    assert not any(
        name.endswith((".joblib", ".safetensors", ".pt")) for name in names
    )


def test_package_bytes_are_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    build_phase4_fast_track_package(PROJECT_ROOT, first)
    build_phase4_fast_track_package(PROJECT_ROOT, second)
    assert first.read_bytes() == second.read_bytes()


def test_verify_rejects_unmanifested_member(tmp_path: Path) -> None:
    archive = tmp_path / "phase4.zip"
    build_phase4_fast_track_package(PROJECT_ROOT, archive)
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("data/secret.jsonl", "secret")
    with pytest.raises(ValueError, match="manifest|forbidden"):
        verify_phase4_fast_track_package(archive)
