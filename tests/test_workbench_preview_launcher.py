"""Preview launcher liveness, startup failure and stop, with controlled child processes only."""
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="launcher identity uses /proc (Linux)")

_spec = importlib.util.spec_from_file_location(
    "workbench_preview", Path(__file__).resolve().parents[1] / "scripts/workbench_preview.py")
wp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp)

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def listener(port):
    return [sys.executable, "-c", "import socket, time; s = socket.socket(); "
            f"s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', {port})); "
            "s.listen(); time.sleep(60)"]


@pytest.fixture
def children():
    procs = []

    def spawn(command=SLEEPER):
        proc = subprocess.Popen(command, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(proc)
        return proc
    yield spawn
    for proc in procs:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def wait_dead(pid, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stat = wp.proc_stat(pid)
        if stat is None or stat[0] in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("record", [None, {}, dict(pid=2147483647, start_ticks=None),
                                    dict(pid=2147483647, start_ticks="123"), dict(pid=None, start_ticks=None),
                                    dict(pid="1", start_ticks="1"), dict(pid=-1, start_ticks="1"),
                                    dict(pid=True, start_ticks="1")])
def test_missing_or_invalid_records_are_never_alive(record):
    # The first-version check returned True for {pid: <gone>, start_ticks: None} (None == None).
    assert wp.alive(record) is False


def test_live_child_is_alive_only_with_matching_identity(children):
    proc = children()
    record = dict(pid=proc.pid, start_ticks=wp.start_ticks(proc.pid))
    assert record["start_ticks"] and wp.alive(record) is True
    assert wp.alive(dict(record, start_ticks=str(int(record["start_ticks"]) + 1))) is False  # PID reuse guard
    assert wp.alive(dict(record, start_ticks=None)) is False


def test_child_outside_its_own_session_is_not_ours():
    proc = subprocess.Popen(SLEEPER)  # same process group as pytest, i.e. not launcher-created
    try:
        assert wp.alive(dict(pid=proc.pid, start_ticks=wp.start_ticks(proc.pid))) is False
    finally:
        proc.kill()
        proc.wait()


def test_exited_child_is_dead_before_and_after_reaping(children):
    proc = children([sys.executable, "-c", "pass"])
    ticks = None
    while proc.poll() is None:
        ticks = wp.start_ticks(proc.pid) or ticks
    record = dict(pid=proc.pid, start_ticks=ticks)
    assert wp.alive(record) is False


def test_zombie_is_not_alive(children):
    proc = children()
    record = dict(pid=proc.pid, start_ticks=wp.start_ticks(proc.pid))
    os.killpg(proc.pid, signal.SIGKILL)
    assert wait_dead(proc.pid) and wp.proc_stat(proc.pid)[0] == "Z"  # not yet reaped
    assert wp.alive(record) is False


def test_quick_exit_fails_start_stops_siblings_and_keeps_logs(tmp_path):
    api_port, ui_port = free_port(), free_port()
    commands = dict(api=listener(api_port), ui=[sys.executable, "-c", "print('ui boom'); raise SystemExit(3)"])
    with pytest.raises(wp.StartupFailed, match="ui exited with code 3"):
        wp.launch(commands, dict(api=api_port, ui=ui_port), dict(os.environ), tmp_path, cwd=tmp_path, timeout=10)
    assert "ui boom" in (tmp_path / "ui.log").read_text()
    time.sleep(0.2)
    assert not wp.listening(api_port)  # the already-started sibling was stopped


def test_never_listening_child_times_out(tmp_path, children):
    port = free_port()
    with pytest.raises(wp.StartupFailed, match="did not listen"):
        wp.launch(dict(api=SLEEPER), dict(api=port), dict(os.environ), tmp_path, cwd=tmp_path, timeout=1)


def test_start_records_only_after_listening_and_stop_signals_only_verified(tmp_path):
    api_port, ui_port = free_port(), free_port()
    records = wp.launch(dict(api=listener(api_port), ui=listener(ui_port)), dict(api=api_port, ui=ui_port),
                        dict(os.environ), tmp_path, cwd=tmp_path, timeout=10)
    assert all(wp.alive(r) for r in records.values())
    pids_file = tmp_path / "pids.json"
    pids_file.write_text(json.dumps(dict(records, ghost=dict(pid=2147483647, start_ticks=None, port=1))))
    assert set(wp.running_records(pids_file)) == {"api", "ui"}
    wp.stop(SimpleNamespace(preview_dir=tmp_path))  # must not raise for the ghost record
    assert not pids_file.exists()
    assert all(wait_dead(r["pid"]) for r in records.values())
    assert not wp.listening(api_port) and not wp.listening(ui_port)


def test_stop_never_signals_a_reused_or_missing_pid(tmp_path, children):
    proc = children()
    mismatched = dict(pid=proc.pid, start_ticks=str(int(wp.start_ticks(proc.pid)) + 1), port=1)
    (tmp_path / "pids.json").write_text(json.dumps(dict(api=mismatched, ui=dict(pid=2147483647, start_ticks=None))))
    wp.stop(SimpleNamespace(preview_dir=tmp_path))
    assert proc.poll() is None  # identity mismatch: left alone
    assert not (tmp_path / "pids.json").exists()


def test_stale_record_file_is_cleared_so_start_can_proceed(tmp_path):
    pids_file = tmp_path / "pids.json"
    pids_file.write_text(json.dumps(dict(api=dict(pid=2147483647, start_ticks=None, port=1))))
    assert wp.running_records(pids_file) == {} and not pids_file.exists()


def pid_listener(port, pid_file, delay=0.0, then=""):
    """Writes its own PID, listens on port after `delay` seconds, then runs `then` (default: stay up)."""
    return [sys.executable, "-c", "import os, socket, sys, time; "
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep({delay}); s = socket.socket(); "
            f"s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', {port})); s.listen(); "
            f"{then or 'time.sleep(60)'}"]


def recorded_pid(pid_file, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid_file.exists() and pid_file.read_text().strip():
            return int(pid_file.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{pid_file} was never written")


@pytest.mark.parametrize("then, reason", [
    ("time.sleep(0.8); sys.exit(7)", "api exited with code 7"),  # exits after passing its own check
    ("time.sleep(0.8); s.close(); time.sleep(60)", "api is no longer listening"),  # alive but no longer ready
])
def test_earlier_child_that_fails_while_a_later_one_starts_fails_the_whole_start(tmp_path, then, reason):
    # The first fix only checked each child right after it started: the API passed, then died while the UI
    # was still starting, and launch() still returned both records (api_exit_code=7, cleanup_calls=0).
    api_port, ui_port = free_port(), free_port()
    api_pid, ui_pid = tmp_path / "api.pid", tmp_path / "ui.pid"
    commands = dict(api=pid_listener(api_port, api_pid, then=then), ui=pid_listener(ui_port, ui_pid, delay=2.0))
    with pytest.raises(wp.StartupFailed, match=reason):
        wp.launch(commands, dict(api=api_port, ui=ui_port), dict(os.environ), tmp_path, cwd=tmp_path, timeout=10)
    assert wait_dead(recorded_pid(ui_pid)) and wait_dead(recorded_pid(api_pid))  # the live UI was cleaned up too
    assert not wp.listening(ui_port) and not wp.listening(api_port)
    assert (tmp_path / "api.log").exists() and (tmp_path / "ui.log").exists()


def test_failed_ui_start_stops_the_already_running_api(tmp_path):
    api_port, ui_port = free_port(), free_port()
    api_pid = tmp_path / "api.pid"
    commands = dict(api=pid_listener(api_port, api_pid), ui=[sys.executable, "-c", "raise SystemExit(5)"])
    with pytest.raises(wp.StartupFailed, match="ui exited with code 5"):
        wp.launch(commands, dict(api=api_port, ui=ui_port), dict(os.environ), tmp_path, cwd=tmp_path, timeout=10)
    assert wait_dead(recorded_pid(api_pid)) and not wp.listening(api_port)


def test_start_writes_no_record_when_launch_fails(tmp_path, monkeypatch):
    def failing_launch(*args, **kwargs):
        raise wp.StartupFailed("api exited with code 7 during the final check")
    monkeypatch.setattr(wp, "prepare", lambda args: tmp_path / "config.yaml")
    monkeypatch.setattr(wp, "port_free", lambda port: True)
    monkeypatch.setattr(wp, "launch", failing_launch)
    args = SimpleNamespace(preview_dir=tmp_path, api_port=free_port(), ui_port=free_port())
    (tmp_path / "logs").mkdir()
    with pytest.raises(SystemExit, match="preview start failed: api exited with code 7"):
        wp.start(args)
    assert not (tmp_path / "pids.json").exists()
