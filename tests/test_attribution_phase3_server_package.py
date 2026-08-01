from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


PROJECT_ROOT = Path(__file__).parents[1]


def test_package_contains_phase3_contract_and_no_runtime_artifacts(tmp_path) -> None:
    from scripts.package_attribution_phase3_server import (
        MANIFEST_NAME,
        build_phase3_server_package,
    )

    archive_path = tmp_path / "phase3.zip"
    result = build_phase3_server_package(PROJECT_ROOT, archive_path)
    assert result.archive_path == archive_path
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        required = {
            MANIFEST_NAME,
            "requirements-attribution-phase3.txt",
            "configs/text_attribution/training/family_phase3_enhancement.v1.json",
            "configs/text_attribution/zhizengzeng_model_catalog.v5.json",
            "research/text_attribution/phase3_config.py",
            "research/text_attribution/phase3_data.py",
            "research/text_attribution/attribution_losses.py",
            "research/text_attribution/hierarchical_model.py",
            "research/text_attribution/phase3_training.py",
            "research/text_attribution/phase3_selection.py",
            "scripts/run_family_attribution_phase3.py",
            "scripts/select_family_attribution_phase3.py",
            "scripts/run_attribution_phase3.sh",
            "scripts/package_attribution_phase3_server.py",
            "docs/ATTRIBUTION_PHASE3_RUNBOOK.md",
            "docs/superpowers/specs/2026-08-02-family-attribution-phase3-enhancement-design.md",
            "docs/superpowers/plans/2026-08-02-family-attribution-phase3-enhancement.md",
            "tests/test_text_attribution_phase3_config.py",
            "tests/test_text_attribution_phase3_selection.py",
            "tests/test_attribution_phase3_server_package.py",
        }
        assert required <= names
        forbidden = (
            ".git",
            ".env",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "data/",
            "outputs/",
            "logs/",
            "checkpoints/",
            "huggingface/",
        )
        assert not any(part in name.casefold() for name in names for part in forbidden)
        assert not any(
            name.casefold().endswith(
                (".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx")
            )
            for name in names
        )
        manifest = json.loads(archive.read(MANIFEST_NAME))
        assert manifest["package_kind"] == "attribution-phase3-enhancement-server"
        assert manifest["layout"] == "project-root-overlay"
        assert manifest["contains_data"] is False
        assert manifest["contains_model_weights"] is False
        records = {row["path"]: row for row in manifest["files"]}
        assert set(records) == names - {MANIFEST_NAME}
        for name, record in records.items():
            payload = archive.read(name)
            assert record["size"] == len(payload)
            assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        shell_mode = archive.getinfo("scripts/run_attribution_phase3.sh").external_attr >> 16
        assert shell_mode & 0o111


def test_runbook_covers_full_validation_only_lifecycle() -> None:
    text = (PROJECT_ROOT / "docs/ATTRIBUTION_PHASE3_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "--verify",
        "HF_ENDPOINT",
        "pip freeze",
        "nvidia-smi",
        "--dry-run",
        "--smoke-test",
        "--preflight-only",
        "A1-alpha01",
        "A1-alpha03",
        "--parent-decision",
        "--resume-from-checkpoint",
        "12,22,32,72,82",
        "COMPLETED.json",
        "test_evaluated",
        "promoted_recipe.json",
    ):
        assert required in text
    assert "--evaluate-test" not in text


def test_package_bytes_are_reproducible(tmp_path) -> None:
    from scripts.package_attribution_phase3_server import build_phase3_server_package

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    build_phase3_server_package(PROJECT_ROOT, first)
    build_phase3_server_package(PROJECT_ROOT, second)
    assert first.read_bytes() == second.read_bytes()


def test_verify_accepts_valid_and_rejects_unmanifested_entry(tmp_path) -> None:
    from scripts.package_attribution_phase3_server import (
        build_phase3_server_package,
        verify_phase3_server_package,
    )

    archive = tmp_path / "phase3.zip"
    build_phase3_server_package(PROJECT_ROOT, archive)
    result = verify_phase3_server_package(archive)
    assert result.file_count > 10
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("unexpected.txt", "bad")
    with pytest.raises(ValueError, match="entry set|unmanifested"):
        verify_phase3_server_package(archive)


def test_package_rejects_symlink_secret_weight_and_traversal(tmp_path, monkeypatch) -> None:
    import scripts.package_attribution_phase3_server as module

    safe = tmp_path / "safe.txt"
    safe.write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    monkeypatch.setattr(module, "_required_source_files", lambda root: (link,))
    with pytest.raises(ValueError, match="symbolic|symlink"):
        module.build_phase3_server_package(tmp_path, tmp_path / "bad.zip")

    secret = tmp_path / "safe-name.txt"
    secret.write_text("OPENAI_API_KEY=sk-" + "x" * 30, encoding="utf-8")
    monkeypatch.setattr(module, "_required_source_files", lambda root: (secret,))
    with pytest.raises(ValueError, match="secret"):
        module.build_phase3_server_package(tmp_path, tmp_path / "secret.zip")

    for unsafe in ("../escape.txt", "weights/model.safetensors", ".env.local"):
        with pytest.raises(ValueError):
            module.validate_package_path(unsafe)


def test_requirement_contract_excludes_pytorch_and_pins_numpy_compatibility() -> None:
    lines = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements-attribution-phase3.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.lower().startswith(("torch", "pytorch")) for line in lines)
    assert any(line.startswith("transformers>=4.50") for line in lines)
    assert "numpy>=1.24,<2.0" in lines


def test_extracted_package_supports_offline_help(tmp_path) -> None:
    from scripts.package_attribution_phase3_server import build_phase3_server_package

    archive = tmp_path / "phase3.zip"
    build_phase3_server_package(PROJECT_ROOT, archive)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(extracted)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(extracted)
    for module in (
        "scripts.run_family_attribution_phase3",
        "scripts.select_family_attribution_phase3",
        "scripts.package_attribution_phase3_server",
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=extracted,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
