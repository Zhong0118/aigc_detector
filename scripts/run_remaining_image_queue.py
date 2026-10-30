#!/usr/bin/env python3
"""One-shot serial continuation; requires explicit execution and never retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.wait_convnext_q179 import alive, identity, sha, write

SELF = "scripts/run_remaining_image_queue.py"
HELPER = "scripts/wait_convnext_q179.py"
BASE = "outputs/image-remaining-routes-20261029"
STATE = BASE + "/queue-continuation-v1"
COMPARISON_CODE = (
    "scripts/compare_image_next_routes.py",
    "scripts/compare_image_scale_epochs.py",
    "scripts/compare_iab_v6_100_vs_v7_200_q179.py",
    "scripts/compare_ofa_tplus1.py",
    "scripts/compare_image_rejection_routes.py",
)
CONVNEXT = dict(name="convnext", script="scripts/run_image_convnext.py",
    config="configs/image_experiments/iab_convnext_q179_20261029.json",
    output="outputs/image_experiments/convnext-q179-20261029")
GPU_STAGES = (
    dict(name="dna", script="scripts/run_image_dna.py",
         config="configs/image_experiments/iab_dna_q179_20261029.json",
         output="outputs/image_experiments/dna-wild-q179-20261029"),
    dict(name="omnidfa", script="scripts/run_image_omnidfa.py",
         config="configs/image_experiments/iab_omnidfa_q179_20261029.json",
         output="outputs/image_experiments/omnidfa-part1-q179-20261029"),
    dict(name="ofa-tplus1", script="scripts/run_image_ofa_tplus1.py",
         config="configs/image_experiments/iab_ofa_tplus1_q179_20261029.json",
         output="outputs/image_experiments/ofa-tplus1-q179-20261029"),
)
PREDECESSORS = (
    dict(launch=CONVNEXT["output"]+"/launch.json",
         status=CONVNEXT["output"]+"/status.json", terminal="completed",
         identity=dict(pid=1002134, boot_id="c57302cb-f225-4cb9-8f13-2424628e7687",
                       process_start_ticks="28907415")),
    dict(launch=BASE+"/queue-convnext-v1/launch.json",
         status=BASE+"/queue-convnext-v1/status.json", terminal="awaiting_analysis",
         identity=dict(pid=1002130, boot_id="c57302cb-f225-4cb9-8f13-2424628e7687",
                       process_start_ticks="28907410")),
)


def stages():
    result = [dict(stage, gpu=True, completion="status.json") for stage in GPU_STAGES]
    for route in ("convnext", "dna", "omnidfa"):
        for suffix, label in (("comparison", "comparison"), ("parent_dev_reference", "parent-dev-reference")):
            result.append(dict(name=f"{route}-{label}", gpu=False,
                script="scripts/compare_image_next_routes.py",
                config=f"configs/image_experiments/iab_{route}_{suffix}_20261029.json",
                output=f"{BASE}/{route}-{label}", completion="completion.json"))
    result.append(dict(name="ofa-tplus1-comparison", gpu=False,
        script="scripts/compare_ofa_tplus1.py", output=BASE+"/ofa-tplus1-comparison",
        completion="completion.json"))
    return result


def read(relative):
    return json.loads((ROOT/relative).read_text())


def check_binding(binding):
    for relative, expected in binding.items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f"registered input changed: {relative}")


def unused(plan):
    for stage in plan:
        if (ROOT/stage["output"]).exists():
            raise FileExistsError(f"preserve existing output: {stage['output']}")


def predecessors_ready():
    ready = True
    for pred in PREDECESSORS:
        launch = read(pred["launch"])
        if any(launch.get(k) != v for k,v in pred["identity"].items()):
            raise ValueError(f"predecessor identity changed: {pred['launch']}")
        try:
            status = read(pred["status"])
        except FileNotFoundError:
            status = {}
        embedded = status.get("launch")
        if embedded and any(embedded.get(k) != v for k,v in pred["identity"].items()):
            raise ValueError(f"predecessor status identity changed: {pred['status']}")
        if status.get("status") == pred["terminal"] and status.get("exit_code") == 0:
            continue
        if (status.get("status") in ("failed", "completed", "awaiting_analysis")
                or status.get("exit_code") is not None or not alive(pred["identity"])):
            raise RuntimeError(f"predecessor failed or died without successful terminal: {pred['status']}")
        if status.get("status") not in (None, "running", "waiting_gpu0"):
            raise RuntimeError(f"unknown predecessor state: {pred['status']}")
        ready = False
    return ready


def prepare():
    plan, binding = stages(), {}
    def bind(relative, expected=None):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"input must be workspace-relative: {relative}")
        actual = sha(ROOT/path)
        if expected is not None and actual != expected:
            raise ValueError(f"registered implementation changed: {relative}")
        if relative in binding and binding[relative] != actual:
            raise ValueError(f"input changed during registration: {relative}")
        binding[relative] = actual
    for relative in (SELF, HELPER, *COMPARISON_CODE):
        bind(relative)
    for stage in (CONVNEXT, *GPU_STAGES):
        bind(stage["config"])
        config = read(stage["config"])
        if config.get("execution_enabled") is not True:
            raise PermissionError(f"worker execution disabled: {stage['config']}")
        if config["output_root"] != stage["output"]:
            raise ValueError(f"unregistered worker output: {stage['config']}")
        implementation = config["implementation_sha256"]
        if not implementation or stage["script"] not in implementation:
            raise ValueError(f"worker entry missing implementation binding: {stage['script']}")
        for relative, expected in implementation.items():
            bind(relative, expected)
    for stage in plan:
        bind(stage["script"])
        if not stage["gpu"] and stage.get("config"):
            bind(stage["config"])
            read(stage["config"])
    for pred in PREDECESSORS:
        bind(pred["launch"])
    predecessors_ready()
    unused(plan)
    check_binding(binding)
    return plan, binding


def gpu_gate():
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError("disk below 20 GiB")
    used = int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    available = False
    with (ROOT/"outputs/image_experiments/gpu0-research.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            available = True
        except BlockingIOError:
            pass
    # The worker reacquires this lock and rechecks physical GPU0 before loading.
    return dict(ready=used <= 1024 and available, used_mib=used,
                research_lock_available=available)


def wait_predecessors(binding, plan, save):
    while True:
        check_binding(binding)
        unused(plan)
        if predecessors_ready():
            return
        save(status="waiting_convnext", predecessors=[p["status"] for p in PREDECESSORS])
        time.sleep(60)


def wait_gpu(binding, remaining, save):
    while True:
        check_binding(binding)
        unused(remaining)
        health = gpu_gate()
        if health["ready"]:
            return
        save(status="waiting_gpu0", stage=remaining[0]["name"], **health)
        time.sleep(60)


def run_child(stage, binding, state, save):
    check_binding(binding)
    unused([stage])
    folder = state/stage["name"]
    folder.mkdir()
    command = [str(ROOT/".venv-image-next/bin/python"), stage["script"]]
    if stage.get("config"):
        command += ["--config", stage["config"]]
    command += ["--execute"] if stage["gpu"] else ["--output", stage["output"]]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0" if stage["gpu"] else "",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    child_identity = None
    started = time.time()
    def status(**kwargs):
        value = dict(stage=stage["name"], child=child_identity, updated_unix=time.time(), **kwargs)
        write(folder/"status.json", value)
        save(**value)
    try:
        status(status="starting")
        with (folder/"worker.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            child_identity = identity(child.pid)
            write(folder/"launch.json", dict(child_identity, command=command, started_unix=started))
            status(status="running")
            while True:
                try:
                    code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    check_binding(binding)
                    status(status="running", elapsed_seconds=time.time()-started)
        write(folder/"exit.json", dict(exit_code=code, finished_unix=time.time()))
        if code != 0:
            raise RuntimeError(f"{stage['name']} exit {code}; no automatic retry")
        completion = read(stage["output"]+"/"+stage["completion"])
        if completion.get("status") != "completed" or completion.get("exit_code") != 0:
            raise RuntimeError(f"{stage['name']} successful completion missing")
        check_binding(binding)
        status(status="completed", exit_code=0)
    except BaseException as error:
        # Never kill an unrelated process or retry a child whose exit is unknown.
        status(status="failed", error=repr(error))
        raise


def execute(plan, binding, state, save):
    wait_predecessors(binding, plan, save)
    for i, stage in enumerate(plan):
        check_binding(binding)
        unused(plan[i:])
        if stage["gpu"]:
            wait_gpu(binding, plan[i:], save)
        run_child(stage, binding, state, save)
    save(status="awaiting_git_archive", exit_code=0, completed_stages=[s["name"] for s in plan])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("explicit --execute required")
    with (ROOT/BASE/"continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = ROOT/STATE
        if state.exists():
            raise FileExistsError("preserve existing continuation state; no restart")
        plan, binding = prepare()
        state.mkdir()
        launch = dict(identity(os.getpid()), started_unix=time.time())
        write(state/"launch.json", launch)
        write(state/"registration.json", dict(inputs=binding, stages=plan, predecessors=PREDECESSORS))
        def save(**kwargs):
            kwargs.setdefault("updated_unix", time.time())
            write(state/"status.json", dict(launch=launch, **kwargs))
        try:
            execute(plan, binding, state, save)
        except BaseException as error:
            save(status="failed", exit_code=1, error=repr(error))
            raise


if __name__ == "__main__":
    main()
