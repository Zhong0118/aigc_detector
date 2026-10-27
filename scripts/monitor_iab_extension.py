#!/usr/bin/env python3
"""Registered extension supervisor; actual retained downloads validate transport."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import monitor_iab_v2 as old
from scripts.prepare_iab_extension_manifest import selected_receipts
from research.image_attribution.protocol import sha256

_old_command = old.command
RUNTIME_CODE = (
    "scripts/monitor_iab_extension.py", "scripts/prepare_iab_extension_manifest.py", "scripts/run_iab_extension_m2.sh",
    "scripts/monitor_iab_v2.py", "scripts/prepare_iab_manifest.py", "scripts/download_iab_subset.py",
    "scripts/run_image_experiment.py", "scripts/summarize_iab_m2.py", "scripts/run_image_research_queue.py",
    "research/image_attribution/extension_split.py", "research/image_attribution/iab_protocol.py",
    "research/image_attribution/protocol.py")
BASELINE_ASSETS = tuple("outputs/image_experiments/assets/" + name for name in
    ("resnet50-0676ba61.pth", "ofa_fingerprint.pth", "model_best.th"))


def read_state(path):
    state = old.read(path)
    if path.exists() and not isinstance(state, dict):
        raise ValueError("unreadable state; preserve outputs: " + str(path))
    return state


def settings(root):
    cfg = old.settings(root)
    required = dict(route="direct", workers=4, per_cell=200, epochs=30)
    if any(cfg.get(k) != value for k, value in required.items()) or cfg.get("gpu_index") not in (0, 1) or not cfg.get("parent_manifest") or not cfg.get("research_config"):
        raise ValueError("extension requires direct route, 4 workers, 200/cell, 30 epochs, registered physical GPU 0/1 and parent/research config")
    return cfg


def verify_registration(root):
    cfg = settings(root)
    folder = old.monitor_directory(root)
    registry = old.read(folder / "registration.json", {}).get("files_sha256", {})
    research = old.read(root / cfg["research_config"], {})
    assets, implementation = research.get("assets", {}), research.get("implementation_sha256", {})
    registered_assets = set(BASELINE_ASSETS) | {asset["path"] for asset in assets.values()}
    non_assets = set(RUNTIME_CODE) | set(implementation) | {
        str((folder / "config.json").relative_to(root)), cfg["research_config"], cfg["parent_manifest"],
        str(Path(cfg["dataset"]) / "selection.json"), str((folder / "iab-public-v3.2-metadata.json").relative_to(root))}
    if not registry or not (non_assets | registered_assets).issubset(registry):
        raise ValueError("missing extension file/hash registration")
    for name, digest in registry.items():
        path = root / name
        relative = Path(name)
        shared_asset = name in registered_assets and name not in non_assets and relative.is_relative_to("outputs/image_experiments/assets")
        if (relative.is_absolute() or ".." in relative.parts or
            (not shared_asset and not path.resolve().is_relative_to(root.resolve())) or
            not path.is_file() or sha256(path) != digest):
            raise ValueError("registered file changed: " + name)
    if not assets or not implementation or any(registry.get(name) != digest for name, digest in implementation.items()) or any(
        registry.get(asset["path"]) != asset["sha256"] for asset in assets.values()):
        raise ValueError("research asset/implementation registration mismatch")
    dataset, parent = research.get("dataset", {}), research.get("parent_manifest", {})
    if (dataset.get("path") != cfg["dataset"] or
        dataset.get("selection_sha256") != registry[str(Path(cfg["dataset"]) / "selection.json")] or
        parent.get("path") != cfg["parent_manifest"] or parent.get("sha256") != registry[cfg["parent_manifest"]]):
        raise ValueError("research dataset/parent registration mismatch")
    if research.get("gpu_index") != cfg["gpu_index"]:
        raise ValueError("research GPU index must match registered baseline GPU index")
    return cfg


def command(root, kind):
    cfg = settings(root)
    if kind == "download":
        return _old_command(root, kind)
    return ["bash", "scripts/run_iab_extension_m2.sh", str(root / cfg["dataset"]),
            cfg["suffix"], cfg["summary"], str(root / cfg["parent_manifest"]), "30", str(cfg["gpu_index"])]


def verify_download(root):
    data = root / settings(root)["dataset"]
    status = old.read(data / "download_status.json", {})
    receipts = [old.read(path) for path in sorted((data / "receipts").glob("*.json"))]
    if status.get("status") != "completed" or len(receipts) != 230 or any(not r for r in receipts):
        raise ValueError("acquisition not complete with 230 receipts")
    selected_receipts(old.read(data / "selection.json"), status, receipts)


def certify_baseline(root, seal=False):
    cfg = settings(root)
    folder = old.monitor_directory(root)
    completion = read_state(folder / "completion.json")
    if completion is None:
        return False
    state = read_state(folder / "pipeline.json") or {}
    if state.get("status") != "completed" or state.get("exit_code") != 0:
        raise ValueError("baseline certificate lacks a completed exit-0 pipeline")
    summary_path, validated_path = root / cfg["summary"], folder / "validated-summary.json"
    summary, validated = old.read(summary_path, {}), old.read(validated_path, {})
    manifest = sha256(root / cfg["dataset"] / "manifest.jsonl")
    if completion.get("summary") != cfg["summary"] or any(
        item.get("status") != "completed" or item.get("manifest_sha256") != manifest
        for item in (completion, summary, validated)):
        raise ValueError("baseline completion/validated-summary/summary/manifest mismatch")
    if summary != validated:
        raise ValueError("baseline summary content mismatch")
    digests = dict(summary_sha256=sha256(summary_path), validated_summary_sha256=sha256(validated_path))
    if seal:
        old.write(folder / "completion.json", dict(completion, **digests,
            completion_scope="extension_baseline_only", final_project_completed=False))
    elif any(completion.get(key) != digest for key, digest in digests.items()):
        raise ValueError("baseline summary certificate hash mismatch")
    return True


def blocked(state, label, report):
    if state is None:
        return False
    if state.get("status") == "failed" or (state.get("status") == "completed" and state.get("exit_code") != 0):
        report["needs_attention"] = label + " failed or lacks successful exit; preserve outputs"
    elif old.alive(state):
        report["waiting"] = label + " worker active"
    elif state.get("status") != "completed":
        report["needs_attention"] = label + " stopped or launch ambiguous; preserve outputs"
    else:
        return False
    return True


def gpu_idle(report, index):
    card = old.gpu(index)
    report["gpu"] = card
    if card.get("unavailable") or card.get("used_mib", 999999) > 1024:
        report["waiting"] = f"physical GPU {index} unavailable or busy"
        return False
    return True


def launch(root, kind):
    """Caller holds supervisor.lock; child waits for its matching PID registration."""
    cfg = verify_registration(root)
    folder = old.monitor_directory(root)
    if (folder / (kind + ".json")).exists():
        raise FileExistsError("existing worker state retained")
    with old.lock(folder / (kind + ".lock")) as held:
        if held is None:
            raise RuntimeError("existing worker lock is busy")
        for proc in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                args = proc.read_bytes().split(b"\0")
                args = [b"scripts/run_iab_m2.sh" if Path(os.fsdecode(a)).name == "run_iab_extension_m2.sh" else a for a in args]
            except OSError:
                continue
            if old.matches_worker(args, root / cfg["dataset"], kind, proc.parent / "cwd"):
                raise RuntimeError("unregistered existing worker detected; preserve outputs")
        state = dict(status="launch_intent", time=time.time(), verified_proxy=False)
        old.write(folder / (kind + ".json"), state)
        with (folder / (kind + ".log")).open("ab") as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                "--state-dir", str(folder), "--worker", kind], cwd=root, env=old.worker_environment(),
                stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        old.write(folder / (kind + ".json"), dict(state, status="running", pid=proc.pid, identity=old.identity(proc.pid)))


def advance(root, report):
    cfg = verify_registration(root)
    folder = old.monitor_directory(root)
    download = read_state(folder / "download.json")
    if blocked(download, "download", report):
        return
    if download is None:
        if old.read(root / cfg["dataset"] / "download_status.json", {}).get("status") == "completed":
            report["needs_attention"] = "completed acquisition lacks successful worker exit record"
            return
        if not (folder / "iab-public-v3.2-metadata.json").is_file():
            raise ValueError("public dataset metadata missing")
        free = shutil.disk_usage(root).free
        if old.remaining_selected_bytes(root / cfg["dataset"]) > free * .8:
            report["waiting"] = "insufficient free disk for remaining selected data and reserve"
            return
        launch(root, "download")
        report["launched"] = "download"
        return
    verify_download(root)
    pipeline = read_state(folder / "pipeline.json")
    if blocked(pipeline, "pipeline", report):
        return
    if not certify_baseline(root):
        if pipeline:
            report["needs_attention"] = "completed pipeline lacks baseline certificate"
            return
        if not gpu_idle(report, cfg["gpu_index"]):
            return
        if old.required_assets(root):
            raise ValueError("required image model assets missing")
        launch(root, "pipeline")
        report["launched"] = "pipeline"
        return
    research = old.read(root / cfg["research_config"])
    base = root / research["output"]
    state, launched = read_state(base / "state.json"), read_state(base / "launch.json")
    if blocked(state, "feature", report):
        return
    if state and state.get("status") == "completed":
        if state.get("config_sha256") != sha256(root / cfg["research_config"]):
            raise ValueError("feature completion config hash mismatch")
        summary = old.read(base / "summary.json", {})
        if summary.get("status") != "completed" or summary.get("config_sha256") != state["config_sha256"]:
            raise ValueError("feature completion lacks matching summary")
        report["status"] = "awaiting_analysis"
        report["final_project_completed"] = False
        return
    if launched is not None:
        if old.alive(launched):
            report["waiting"] = "feature worker active"
        else:
            report["needs_attention"] = "feature launch ambiguous; preserve outputs"
        return
    if not gpu_idle(report, cfg["gpu_index"]):
        return
    result = subprocess.run([str(root / ".venv-image/bin/python"), "scripts/run_image_research_queue.py",
        "--config", cfg["research_config"], "--advance"], cwd=root, env=old.worker_environment(),
        capture_output=True, text=True)
    report["feature_advance_exit_code"] = result.returncode
    if result.returncode:
        report["needs_attention"] = "feature queue advance failed; inspect queue state"
    else:
        report["waiting"] = "feature queue advanced; awaiting worker evidence"


def worker(root, kind):
    folder = old.monitor_directory(root)
    with (folder / "supervisor.lock").open("a") as supervisor:
        fcntl.flock(supervisor, fcntl.LOCK_EX)
    state = read_state(folder / (kind + ".json")) or {}
    if state.get("pid") != os.getpid() or state.get("identity") != old.identity(os.getpid()) or state.get("status") != "running":
        raise PermissionError("worker lacks matching supervisor PID/boot/start registration")
    try:
        verify_registration(root)
        if kind == "pipeline":
            verify_download(root)
            if (folder / "validated-summary.json").exists() or (folder / "completion.json").exists():
                raise FileExistsError("existing pipeline certificate retained")
        old.command = command
        code = old.worker(root, kind)
        if code == 0:
            if kind == "download":
                verify_download(root)
            else:
                certify_baseline(root, seal=True)
        return code
    except Exception as error:
        old.write(folder / (kind + ".json"), dict(state, status="failed", exit_code=1,
            finished=time.time(), error_type=type(error).__name__))
        raise


def snapshot(root):
    cfg = settings(root)
    folder, data = old.monitor_directory(root), root / cfg["dataset"]
    workers = {kind: read_state(folder / (kind + ".json")) for kind in ("download", "pipeline")}
    return dict(time=time.time(), dataset=cfg["dataset"], gpu=old.gpu(cfg["gpu_index"]),
        disk_free_bytes=shutil.disk_usage(root).free, download=old.read(data / "download_status.json"),
        receipts=len(list((data / "receipts").glob("*.json"))),
        workers={k: dict(v, alive=old.alive(v)) if v else None for k,v in workers.items()},
        baseline_completion=old.read(folder / "completion.json"), final_project_completed=False,
        transport_validation="direct official downloader retained-image I/O; no probes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state-dir", type=Path, default=Path("outputs/iab-v7-200-e30-monitor"))
    parser.add_argument("--advance", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--worker", choices=("download", "pipeline"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ["IAB_MONITOR_STATE_DIR"] = str(args.state_dir)
    if args.worker:
        return worker(root, args.worker)
    if args.interval < 1:
        parser.error("interval must be positive")
    folder = old.monitor_directory(root)
    loop_handle = None
    if args.loop:
        folder.mkdir(parents=True, exist_ok=True)
        loop_handle = (folder / "loop.lock").open("a")
        try:
            fcntl.flock(loop_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 3
        previous = read_state(folder / "loop.json")
        if previous is not None:
            return 3  # Prior loop identity remains evidence until explicitly reviewed.
        old.write(folder / "loop.json", dict(status="running", pid=os.getpid(),
            identity=old.identity(os.getpid()), interval=args.interval))
    while True:
        report = {}
        try:
            verify_registration(root)
            report = snapshot(root)
            if args.advance:
                with old.lock(folder / "supervisor.lock") as held:
                    if held is None:
                        report["waiting"] = "supervisor lock busy"
                    else:
                        advance(root, report)
                        old.write(folder / "snapshot.json", report)
        except Exception as error:
            report.update(time=time.time(), needs_attention=type(error).__name__ + ": " + str(error))
            if args.advance:
                old.write(folder / "snapshot.json", report)
        print(json.dumps(report), flush=True)
        terminal = report.get("needs_attention") or report.get("status") == "awaiting_analysis"
        if not args.loop or terminal:
            if args.loop:
                state = old.read(folder / "loop.json", {})
                old.write(folder / "loop.json", dict(state, status="needs_attention" if report.get("needs_attention") else "awaiting_analysis", finished=time.time()))
            return 1 if report.get("needs_attention") else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
