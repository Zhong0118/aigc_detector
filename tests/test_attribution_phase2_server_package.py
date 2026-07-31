from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import pytest

import scripts.package_attribution_phase2_server as package_module
from scripts.package_attribution_phase2_server import (
    MANIFEST_NAME,
    build_phase2_server_package,
    main as package_main,
    verify_phase2_server_package,
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
    assert 'start_experiment "N0" "0"' in text
    assert 'start_experiment "N1" "1"' in text
    assert 'start_experiment "N2" "0"' in text
    assert 'export CUDA_VISIBLE_DEVICES="$physical_gpu"' in text
    assert '--device "cuda:0"' in text
    assert "mkdir -p" in text
    assert "wait " in text
    assert "setsid" in text
    assert 'kill -TERM -- "-$pid"' in text
    assert "--evaluate-test" not in text


def test_runbook_uses_real_completion_marker_and_audits_environment() -> None:
    text = (
        PROJECT_ROOT / "docs" / "ATTRIBUTION_PHASE2_NEURAL_RUNBOOK.md"
    ).read_text(encoding="utf-8")

    assert "COMPLETED.json" in text
    assert "/COMPLETED\n" not in text
    assert "pip freeze" in text
    assert "nvidia-smi" in text
    assert "--verify" in text


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
        assert isinstance(manifest["source_dirty"], bool)
        records = {row["path"]: row for row in manifest["files"]}
        assert set(records) == names - {MANIFEST_NAME}
        for relative_path, record in records.items():
            payload = archive.read(relative_path)
            assert record["size"] == len(payload)
            assert record["sha256"] == hashlib.sha256(payload).hexdigest()


def test_verify_rejects_unmanifested_entry(tmp_path: Path) -> None:
    archive_path = tmp_path / "phase2.zip"
    build_phase2_server_package(PROJECT_ROOT, archive_path)
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("unexpected.txt", b"not declared")

    with pytest.raises(ValueError, match="entry set"):
        verify_phase2_server_package(archive_path)


def test_verify_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "phase2.zip"
    build_phase2_server_package(PROJECT_ROOT, archive_path)
    corrupted_path = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(archive_path) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    target_name = "research/__init__.py"
    original = entries[target_name]
    entries[target_name] = (
        bytes([original[0] ^ 1]) + original[1:] if original else b"x"
    )
    with zipfile.ZipFile(corrupted_path, "w") as target:
        for name, payload in entries.items():
            target.writestr(name, payload)

    with pytest.raises(ValueError, match="sha256"):
        verify_phase2_server_package(corrupted_path)


def test_verify_cli_accepts_valid_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "phase2.zip"
    build_phase2_server_package(PROJECT_ROOT, archive_path)

    assert package_main(["--verify", str(archive_path)]) == 0


def test_manifest_records_clean_and_dirty_git_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    root = tmp_path / "repository"
    root.mkdir()
    source_path = root / "safe.txt"
    source_path.write_text("clean\n", encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "phase2@example.invalid"],
        ["git", "config", "user.name", "Phase2 Test"],
        ["git", "add", "safe.txt"],
        ["git", "commit", "-q", "-m", "initial"],
    ):
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"temporary git repository unavailable: {completed.stderr}")
    monkeypatch.setattr(
        package_module,
        "_required_source_files",
        lambda _root: (source_path,),
    )

    clean_archive = tmp_path / "clean.zip"
    build_phase2_server_package(root, clean_archive)
    with zipfile.ZipFile(clean_archive) as archive:
        clean_manifest = json.loads(archive.read(MANIFEST_NAME))
    assert clean_manifest["source_dirty"] is False
    assert clean_manifest["source_commit"] != "unknown"

    source_path.write_text("dirty\n", encoding="utf-8")
    dirty_archive = tmp_path / "dirty.zip"
    build_phase2_server_package(root, dirty_archive)
    with zipfile.ZipFile(dirty_archive) as archive:
        dirty_manifest = json.loads(archive.read(MANIFEST_NAME))
    assert dirty_manifest["source_dirty"] is True


def test_manifest_uses_unknown_state_outside_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "safe.txt"
    source_path.write_text("not a repository\n", encoding="utf-8")
    monkeypatch.setattr(
        package_module,
        "_required_source_files",
        lambda _root: (source_path,),
    )

    archive_path = tmp_path / "package.zip"
    build_phase2_server_package(tmp_path, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME))
    assert manifest["source_commit"] == "unknown"
    assert manifest["source_dirty"] is None


def test_package_rejects_symlinked_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "safe.txt").write_text("outside root\n", encoding="utf-8")
    linked_parent = root / "linked"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    escaped_file = linked_parent / "safe.txt"
    monkeypatch.setattr(
        package_module,
        "_required_source_files",
        lambda _root: (escaped_file,),
    )

    with pytest.raises(ValueError, match="symbolic-link ancestor|outside project root"):
        build_phase2_server_package(root, tmp_path / "package.zip")


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


def _usable_bash() -> str:
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash is unavailable")
    completed = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or "bash" not in completed.stdout.casefold():
        pytest.skip("a runnable GNU Bash is unavailable")
    if shutil.which("setsid") is None:
        pytest.skip("setsid is unavailable")
    return executable


def _write_python_stub(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
experiment=""
previous=""
for argument in "$@"; do
  if [[ "$previous" == "--experiment" ]]; then
    experiment="$argument"
  fi
  previous="$argument"
done
printf 'START|%s|%s|%s\\n' "$experiment" "$CUDA_VISIBLE_DEVICES" "$$" >> "$STUB_RECORD"
trap 'printf "TERM|%s|%s|%s\\n" "$experiment" "$CUDA_VISIBLE_DEVICES" "$$" >> "$STUB_RECORD"; exit 143' TERM INT
if [[ "${STUB_FAIL_EXPERIMENT:-}" == "$experiment" ]]; then
  exit 17
fi
if [[ "${STUB_SLEEP:-0}" == "1" ]]; then
  sleep 30 &
  wait $!
fi
exit 0
""",
        encoding="utf-8",
        newline="\n",
    )
    path.chmod(0o755)


def _runner_command(tmp_path: Path) -> tuple[list[str], dict[str, str], Path]:
    bash = _usable_bash()
    data_path = tmp_path / "data.jsonl"
    split_path = tmp_path / "split.json"
    config_path = tmp_path / "config.json"
    for path in (data_path, split_path, config_path):
        path.write_text("{}\n", encoding="utf-8")
    stub_path = tmp_path / "python-stub.sh"
    _write_python_stub(stub_path)
    record_path = tmp_path / "calls.log"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": str(stub_path),
            "PHASE2_CONFIG": str(config_path),
            "PHASE2_LOG_DIR": str(tmp_path / "logs"),
            "STUB_RECORD": str(record_path),
        }
    )
    command = [
        bash,
        str(PROJECT_ROOT / "scripts" / "run_attribution_phase2_screening.sh"),
        str(data_path),
        str(split_path),
        str(tmp_path / "cache"),
        str(tmp_path / "outputs"),
        "42",
    ]
    return command, environment, record_path


def _start_records(record_path: Path) -> list[tuple[str, str, int]]:
    if not record_path.is_file():
        return []
    records: list[tuple[str, str, int]] = []
    for line in record_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) == 4 and parts[0] == "START":
            records.append((parts[1], parts[2], int(parts[3])))
    return records


def test_runner_maps_gpus_and_starts_n2_only_after_parallel_success(
    tmp_path: Path,
) -> None:
    command, environment, record_path = _runner_command(tmp_path)
    completed = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    records = _start_records(record_path)
    assert {(experiment, gpu) for experiment, gpu, _pid in records} == {
        ("N0", "0"),
        ("N1", "1"),
        ("N2", "0"),
    }
    assert [experiment for experiment, _gpu, _pid in records][-1] == "N2"


def test_runner_propagates_parallel_failure_and_does_not_start_n2(
    tmp_path: Path,
) -> None:
    command, environment, record_path = _runner_command(tmp_path)
    environment["STUB_FAIL_EXPERIMENT"] = "N0"
    completed = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 17
    records = _start_records(record_path)
    assert {(experiment, gpu) for experiment, gpu, _pid in records} == {
        ("N0", "0"),
        ("N1", "1"),
    }


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process signals required")
def test_runner_terminates_python_and_tee_process_groups(
    tmp_path: Path,
) -> None:
    command, environment, record_path = _runner_command(tmp_path)
    environment["STUB_SLEEP"] = "1"
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    records: list[tuple[str, str, int]] = []
    while time.monotonic() < deadline:
        records = _start_records(record_path)
        if len(records) >= 2:
            break
        time.sleep(0.05)
    assert len(records) >= 2

    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=10)
    assert process.returncode == 143
    for _experiment, _gpu, pid in records:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
