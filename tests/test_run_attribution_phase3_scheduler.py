from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parents[1] / "scripts/run_attribution_phase3.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _fake_environment(tmp_path: Path, *, fail_closed: bool = False):
    calls = tmp_path / "calls.txt"
    fake_python = tmp_path / "fake-python"
    failure_logic = ""
    if fail_closed:
        failure_logic = """
case "$*" in
  *A0-closed-seed42*) exit 9 ;;
  *) sleep 1 ;;
esac
"""
    _write_executable(
        fake_python,
        f"""#!/bin/bash
printf '%s|%s\\n' "$CUDA_VISIBLE_DEVICES" "$*" >> "$CALLS_FILE"
{failure_logic}
exit 0
""",
    )
    fake_setsid = tmp_path / "fake-setsid"
    _write_executable(fake_setsid, "#!/bin/bash\nexec \"$@\"\n")
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "PHASE3_SETSID_BIN": str(fake_setsid),
            "CALLS_FILE": str(calls),
        }
    )
    return calls, env


def _run(tmp_path: Path, seeds: str, *, fail_closed: bool = False):
    calls, env = _fake_environment(tmp_path, fail_closed=fail_closed)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "A0",
            str(tmp_path / "data.jsonl"),
            str(tmp_path / "split.json"),
            str(tmp_path / "cache"),
            str(tmp_path / "outputs"),
            seeds,
        ],
        cwd=SCRIPT.parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    return result, lines


def test_scheduler_runs_exact_seed_view_cartesian_product(tmp_path) -> None:
    result, lines = _run(tmp_path, "42,52")
    assert result.returncode == 0, result.stderr
    assert len(lines) == 6
    for seed in (42, 52):
        for view in ("closed", "fast-to-balanced", "balanced-to-fast"):
            matching = [
                line
                for line in lines
                if f"--seed {seed}" in line and f"--view {view}" in line
            ]
            assert len(matching) == 1
    assert {line.split("|", 1)[0] for line in lines} == {"0", "1"}
    assert all("--device cuda:0" in line for line in lines)
    assert all("--evaluate-test" not in line for line in lines)
    assert all("--test-split" not in line for line in lines)


def test_scheduler_uses_recipe_mode_for_recipe_file(tmp_path) -> None:
    recipe = tmp_path / "promoted_recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    calls, env = _fake_environment(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(recipe),
            "data.jsonl",
            "split.json",
            "cache",
            str(tmp_path / "outputs"),
            "42",
        ],
        cwd=SCRIPT.parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert all(f"--recipe {recipe}" in line for line in lines)
    assert all("--stage" not in line for line in lines)


def test_scheduler_stops_dispatch_after_failure_and_waits_inflight(tmp_path) -> None:
    result, lines = _run(tmp_path, "42,52", fail_closed=True)
    assert result.returncode != 0
    assert len(lines) == 2
    assert any("A0-closed-seed42" in line for line in lines)
    assert "failure" in (result.stdout + result.stderr).lower()


def test_scheduler_contains_process_group_interrupt_cleanup() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "setsid" in source.lower()
    assert "kill -TERM -- -" in source
    assert "trap" in source
    assert "tee" in source
