#!/usr/bin/env python3
"""Isolated preview launcher: prepare | start | status | stop.

Builds <preview_dir>/config.yaml (config.yaml + overlay), verifies the ports are free,
starts uvicorn and Streamlit bound to 127.0.0.1 with a clean environment (no API keys,
GPUs hidden, Hugging Face offline, private TMPDIR) and records PID + start time so
`stop` only ever signals the processes it started. Never touches online services.
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


def start_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def alive(record):
    return record and start_ticks(record["pid"]) == record["start_ticks"]


def clean_env(config, api_port):
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG", "LC_ALL") if k in os.environ}
    env.update(AIGC_CONFIG=str(config), AIGC_API_BASE=f"http://127.0.0.1:{api_port}/api/v1",
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TMPDIR=str(config.parent / "tmp"), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    return env


def start(args):
    pids_file = args.preview_dir / "pids.json"
    if pids_file.exists() and any(alive(r) for r in json.loads(pids_file.read_text()).values()):
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
    records = {}
    for name, command in commands.items():
        with open(logs / f"{name}.log", "ab") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        time.sleep(0.5)
        records[name] = dict(pid=proc.pid, start_ticks=start_ticks(proc.pid), port=args.api_port if name == "api" else args.ui_port)
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
    for name, record in json.loads(pids_file.read_text()).items():
        if alive(record):
            os.killpg(record["pid"], signal.SIGTERM)
            print(f"stopped {name} pid={record['pid']}")
        else:
            print(f"{name} not running (pid reuse guarded)")
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
