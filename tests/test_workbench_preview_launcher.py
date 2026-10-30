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
