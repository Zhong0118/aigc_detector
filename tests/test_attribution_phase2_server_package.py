from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

import scripts.package_attribution_phase2_server as package_module
from scripts.package_attribution_phase2_server import (
    MANIFEST_NAME,
    build_phase2_server_package,
)
from research.text_attribution.split_manifest import (
    create_split_manifest,
    save_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_neural_config(path: Path, catalog_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "phase2-package-test",
                "seed": 42,
                "model_catalog": os.path.relpath(catalog_path, path.parent),
                "common": {
                    "learning_rate": 2e-5,
                    "epochs": 1,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1,
                    "label_smoothing_factor": 0.05,
                    "gradient_accumulation_steps": 1,
                    "early_stopping_patience": 1,
                    "save_total_limit": 1,
                    "bf16": False,
                },
                "experiments": {
                    experiment_id: {
                        "model_id": f"local/{experiment_id.lower()}",
                        "max_length": 64,
                        "per_device_train_batch_size": 2,
                    }
                    for experiment_id in ("N0", "N1", "N2")
                },
            }
        ),
        encoding="utf-8",
    )


def _prepare_dry_run_inputs(root: Path) -> tuple[Path, Path, Path]:
    catalog_path = root / "catalog.json"
    write_model_catalog(catalog_path)
    data_path = root / "data.jsonl"
    write_dataset(data_path, groups=20)
    dataset = load_training_dataset(data_path, catalog_path)
    split_path = root / "split.json"
    save_split_manifest(
        split_path,
        create_split_manifest(dataset.samples, seed=42),
    )
    config_path = root / "neural.json"
    _write_neural_config(config_path, catalog_path)
    return config_path, data_path, split_path


def test_server_runner_is_two_gpu_and_validation_only() -> None:
    text = (
        PROJECT_ROOT / "scripts" / "run_attribution_phase2_screening.sh"
    ).read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert '--experiment "$experiment"' in text
    assert 'run_experiment "N0"' in text
    assert 'run_experiment "N1"' in text
    assert 'run_experiment "N2"' in text
    assert 'CUDA_VISIBLE_DEVICES="0"' in text
    assert 'CUDA_VISIBLE_DEVICES="1"' in text
    assert '--device "cuda:0"' in text
    assert "mkdir -p" in text
    assert "wait " in text
    assert "--evaluate-test" not in text


def test_package_contains_required_code_and_no_runtime_or_secret_files(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "phase2.zip"
    result = build_phase2_server_package(PROJECT_ROOT, archive_path)

    assert result.archive_path == archive_path
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert {
            MANIFEST_NAME,
            "requirements-attribution-phase2.txt",
            "research/__init__.py",
            "research/text_attribution/neural_training.py",
            "research/text_attribution/neural_selection.py",
            "scripts/run_family_attribution_neural.py",
            "scripts/select_family_attribution_neural.py",
            "scripts/run_attribution_phase2_screening.sh",
            "scripts/package_attribution_phase2_server.py",
            "configs/text_attribution/training/"
            "family_neural_screening.v1.json",
            "configs/text_attribution/zhizengzeng_model_catalog.v5.json",
            "docs/ATTRIBUTION_PHASE2_NEURAL_RUNBOOK.md",
            "tests/test_text_attribution_neural_training.py",
            "tests/test_attribution_phase2_server_package.py",
        } <= names

        lowered = [name.casefold() for name in names]
        forbidden_fragments = (
            ".git",
            ".env",
            "apikey",
            "api_key",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "data/attribution/raw",
            "outputs/",
            "models/",
            "checkpoints/",
        )
        assert not any(
            fragment in name
            for name in lowered
            for fragment in forbidden_fragments
        )
        assert not any(
            name.endswith((".bin", ".pt", ".pth", ".safetensors", ".ckpt"))
            for name in lowered
        )

        manifest = json.loads(archive.read(MANIFEST_NAME))
        assert manifest["schema_version"] == 1
        assert manifest["package_kind"] == "attribution-phase2-neural-server"
        records = {row["path"]: row for row in manifest["files"]}
        assert set(records) == names - {MANIFEST_NAME}
        for relative_path, record in records.items():
            payload = archive.read(relative_path)
            assert record["size"] == len(payload)
            assert record["sha256"] == hashlib.sha256(payload).hexdigest()


def test_package_rejects_dotenv_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env.local"
    dotenv_path.write_text("VALUE=placeholder\n", encoding="utf-8")
    monkeypatch.setattr(
        package_module,
        "_required_source_files",
        lambda _root: (dotenv_path,),
    )

    with pytest.raises(ValueError, match="forbidden package path"):
        build_phase2_server_package(tmp_path, tmp_path / "package.zip")


def test_package_rejects_secret_looking_file_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "safe-name.txt"
    source_path.write_bytes(b"OPENAI_API_KEY=" + b"sk-" + b"x" * 24 + b"\n")
    monkeypatch.setattr(
        package_module,
        "_required_source_files",
        lambda _root: (source_path,),
    )

    with pytest.raises(ValueError, match="secret-like credential"):
        build_phase2_server_package(tmp_path, tmp_path / "package.zip")


def test_extracted_package_supports_import_help_and_offline_dry_run(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "phase2.zip"
    build_phase2_server_package(PROJECT_ROOT, archive_path)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(extracted)
    for module_name in (
        "scripts.run_family_attribution_neural",
        "scripts.select_family_attribution_neural",
        "scripts.package_attribution_phase2_server",
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            cwd=extracted,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.casefold()

    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    config_path, data_path, split_path = _prepare_dry_run_inputs(fixture_root)
    output_dir = fixture_root / "must-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_family_attribution_neural",
            "--config",
            str(config_path),
            "--experiment",
            "N0",
            "--input",
            str(data_path),
            "--split-manifest",
            str(split_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=extracted,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["experiment"] == "N0"
    assert summary["evaluate_test"] is False
    assert not output_dir.exists()
