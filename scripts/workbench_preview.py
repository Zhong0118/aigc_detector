#!/usr/bin/env python3
"""Isolated preview launcher: prepare | start | status | stop.

Builds <preview_dir>/config.yaml (config.yaml + overlay), verifies the ports are free,
starts uvicorn and Streamlit bound to 127.0.0.1 with a clean environment (no API keys,
GPUs hidden, Hugging Face offline, private TMPDIR) and records PID + start time so
`stop` only ever signals the processes it started. Never touches online services.

A start is recorded only after every child is still running and listening on its port;
a child that exits during startup fails the start, its siblings are stopped and the
logs are kept. Liveness requires a valid recorded start time equal to the live one.
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAY = ROOT / "configs/workbench/preview.overlay.example.yaml"


def merge(base, overlay):
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for key, value in overlay.items():
            out[key] = merge(base.get(key), value)
        return out
    return overlay


def prepare(args):
    preview = args.preview_dir.resolve()
    (preview / "tmp").mkdir(parents=True, exist_ok=True)
    (preview / "logs").mkdir(exist_ok=True)
    values = dict(RESEARCH_ROOT=str(args.research_root.resolve()), MAIN_ROOT=str(args.main_root.resolve()),
                  PREVIEW_DIR=str(preview))
    text = re.sub(r"\$\{(\w+)\}", lambda m: values[m.group(1)], args.overlay.read_text())
    config = merge(yaml.safe_load((ROOT / "config.yaml").read_text()), yaml.safe_load(text))
    config["api"]["port"], config["dashboard"]["port"] = args.api_port, args.ui_port
    target = preview / "config.yaml"
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return target


def port_free(port):
    with socket.socket() as sock:
        # Same option uvicorn/Streamlit use: TIME_WAIT leftovers pass, a live listener still fails.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    listening = subprocess.run(["ss", "-ltnH"], capture_output=True, text=True).stdout
    return not re.search(rf"[:.]{port}\s", listening + " ")


STARTUP_TIMEOUT = 60.0


class StartupFailed(RuntimeError):
    pass


def proc_stat(pid):
    """(state, start_ticks) of a live PID from /proc, or None if it does not exist."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except (OSError, IndexError, TypeError, ValueError):
        return None


def start_ticks(pid):
    stat = proc_stat(pid)
    return stat[1] if stat else None


def alive(record):
    """Identity-checked liveness. Both the recorded and the live start time must be valid and equal,
    the process must not be a zombie, and it must still lead the session the launcher created."""
    if not isinstance(record, dict):
        return False
    pid, ticks = record.get("pid"), record.get("start_ticks")
    if type(pid) is not int or pid <= 0 or not isinstance(ticks, str) or not ticks.isdigit():
        return False
    stat = proc_stat(pid)
    if stat is None or stat[0] in ("Z", "X") or stat[1] != ticks:
        return False
    try:
        return os.getpgid(pid) == pid
    except OSError:
        return False


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def terminate(proc, timeout=10.0):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout)


def launch(commands, ports, env, logs, cwd=None, timeout=STARTUP_TIMEOUT):
    """Start each command in its own session and wait until it listens on its port.

    Identity (PID + start ticks) is read right after spawning. A child that exits or never
    listens raises StartupFailed after every child started here has been stopped; nothing
    is returned, so nothing can be recorded as started. Logs are always kept.
    """
    procs, records = [], {}
    try:
        for name, command in commands.items():
            with open(logs / f"{name}.log", "ab") as log:
                proc = subprocess.Popen(command, cwd=cwd or ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
            procs.append(proc)
            records[name] = dict(pid=proc.pid, start_ticks=start_ticks(proc.pid), port=ports[name])
            deadline = time.monotonic() + timeout
            while True:
                code = proc.poll()
                if code is not None:
                    raise StartupFailed(f"{name} exited with code {code} during startup; see {logs / (name + '.log')}")
                if listening(ports[name]):
                    break
                if time.monotonic() > deadline:
                    raise StartupFailed(f"{name} did not listen on 127.0.0.1:{ports[name]} within {timeout:.0f}s; "
                                        f"see {logs / (name + '.log')}")
                time.sleep(0.2)
            if not alive(records[name]):
                raise StartupFailed(f"{name} failed the identity check after startup")
        return records
    except BaseException:
        for proc in reversed(procs):
            terminate(proc)
        raise


def running_records(pids_file):
    """Records of a previous start that are still alive; a fully stale record file is removed."""
    if not pids_file.exists():
        return {}
    records = json.loads(pids_file.read_text())
    live = {name: r for name, r in records.items() if alive(r)}
    if not live:
        pids_file.unlink()
    return live


def clean_env(config, api_port):
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG", "LC_ALL") if k in os.environ}
    env.update(AIGC_CONFIG=str(config), AIGC_API_BASE=f"http://127.0.0.1:{api_port}/api/v1",
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TMPDIR=str(config.parent / "tmp"), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    return env


def start(args):
    pids_file = args.preview_dir / "pids.json"
    if running_records(pids_file):
        sys.exit("preview already running; use status/stop")
    for port in (args.api_port, args.ui_port):
        if not port_free(port):
            sys.exit(f"port {port} is in use; choose another")
    config = prepare(args)
    env, logs = clean_env(config, args.api_port), args.preview_dir.resolve() / "logs"
    commands = {
        "api": [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
                "--port", str(args.api_port), "--workers", "1"],
        "ui": [sys.executable, "-m", "streamlit", "run", "workbench/ui/app.py", "--server.address", "127.0.0.1",
               "--server.port", str(args.ui_port), "--server.headless", "true", "--server.maxUploadSize", "20",
               "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer",
               "--client.showErrorDetails", "none",  # never show tracebacks/server paths to users
               "--theme.base", "light", "--theme.primaryColor", "#1d4ed8", "--theme.backgroundColor", "#ffffff",
               "--theme.secondaryBackgroundColor", "#f4f6fa", "--theme.textColor", "#111827"],
    }
    try:
        records = launch(commands, dict(api=args.api_port, ui=args.ui_port), env, logs)
    except StartupFailed as exc:
        sys.exit(f"preview start failed: {exc}")
    pids_file.write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(config=str(config), **records), indent=2))


def status(args):
    pids_file = args.preview_dir / "pids.json"
    records = json.loads(pids_file.read_text()) if pids_file.exists() else {}
    out = {name: dict(r, alive=bool(alive(r))) for name, r in records.items()}
    if "api" in records:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{records['api']['port']}/api/v1/capabilities", timeout=10) as r:
                out["image_state"] = json.load(r)["image"]["state"]
        except Exception as exc:
            out["image_state"] = f"unreachable: {type(exc).__name__}"
    print(json.dumps(out, indent=2, ensure_ascii=False))


def stop(args):
    pids_file = args.preview_dir / "pids.json"
    if not pids_file.exists():
        return print("no preview recorded")
    remaining = {}
    for name, record in json.loads(pids_file.read_text()).items():
        if not alive(record):
            print(f"{name} not running (identity check failed or process gone; nothing signalled)")
            continue
        try:
            os.killpg(record["pid"], signal.SIGTERM)
        except ProcessLookupError:
            print(f"{name} exited before it was signalled")
            continue
        deadline = time.monotonic() + 15
        while alive(record) and time.monotonic() < deadline:
            time.sleep(0.2)
        if alive(record):
            remaining[name] = record
            print(f"{name} pid={record['pid']} still running after SIGTERM; record kept")
        else:
            print(f"stopped {name} pid={record['pid']}")
    if remaining:
        pids_file.write_text(json.dumps(remaining, indent=2))
    else:
        pids_file.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "start", "status", "stop"])
    parser.add_argument("--preview-dir", type=Path, default=ROOT / ".preview")
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--research-root", type=Path, default=ROOT.parent / "attribution-pilot-baseline")
    parser.add_argument("--main-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--api-port", type=int, default=8611)
    parser.add_argument("--ui-port", type=int, default=8612)
    args = parser.parse_args()
    dict(prepare=lambda a: print(prepare(a)), start=start, status=status, stop=stop)[args.action](args)


if __name__ == "__main__":
    main()
