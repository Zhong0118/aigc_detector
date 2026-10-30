"""No GPU/process execution: exercise queue boundaries with temporary files."""
import json
import sys
from types import SimpleNamespace

import pytest


def queue():
    from scripts import run_remaining_image_queue
    return run_remaining_image_queue


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    q = queue()
    monkeypatch.setattr(q, "ROOT", tmp_path)
    for name in (q.SELF, q.HELPER, *q.COMPARISON_CODE):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# fixed input\n")
    common = tmp_path/"shared.py"
    common.write_text("# implementation\n")
    specs = [q.CONVNEXT, *q.GPU_STAGES]
    for spec in specs:
        p = tmp_path/spec["script"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# worker\n")
        put(tmp_path/spec["config"], dict(execution_enabled=True, output_root=spec["output"],
            implementation_sha256={spec["script"]:q.sha(p), "shared.py":q.sha(common)}))
    for stage in q.stages():
        p = tmp_path/stage["script"]
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("# comparison\n")
        if not stage["gpu"] and stage.get("config"):
            put(tmp_path/stage["config"], {"automatic_execution":False})
    for pred in q.PREDECESSORS:
        put(tmp_path/pred["launch"], dict(pred["identity"], command=["registered"]))
        put(tmp_path/pred["status"], dict(status=pred["terminal"], exit_code=0,
                                          launch=pred["identity"]))
    (tmp_path/"outputs/image_experiments").mkdir(parents=True, exist_ok=True)
    return q


def test_prepare_fails_before_any_output_for_missing_or_disabled_worker(workspace):
    q = workspace
    cfg_path = q.ROOT/q.GPU_STAGES[1]["config"]
    cfg = json.loads(cfg_path.read_text())
    cfg_path.unlink()
    with pytest.raises(FileNotFoundError):
        q.prepare()
    put(cfg_path, dict(cfg, execution_enabled=False))
    with pytest.raises(PermissionError, match="disabled"):
        q.prepare()
    assert not (q.ROOT/q.STATE).exists()


def test_registration_pins_worker_and_comparison_inputs(workspace):
    q = workspace
    stages, binding = q.prepare()
    assert len(stages) == 10
    for relative in ["shared.py", q.SELF, q.HELPER,
                     "scripts/compare_ofa_tplus1.py",
                     "configs/image_experiments/iab_omnidfa_comparison_20261029.json"]:
        assert relative in binding
    changed = q.ROOT/"shared.py"
    changed.write_text("# edited\n")
    with pytest.raises(ValueError, match="changed"):
        q.check_binding(binding)


def test_declared_implementation_digest_must_match(workspace):
    q = workspace
    path = q.ROOT/q.GPU_STAGES[0]["config"]
    cfg = json.loads(path.read_text())
    cfg["implementation_sha256"]["shared.py"] = "0"*64
    put(path, cfg)
    with pytest.raises(ValueError, match="changed"):
        q.prepare()


def test_registered_output_is_exact_and_existing_output_is_never_reused(workspace):
    q = workspace
    path = q.ROOT/q.GPU_STAGES[0]["config"]
    cfg = json.loads(path.read_text())
    put(path, dict(cfg, output_root="other"))
    with pytest.raises(ValueError, match="output"):
        q.prepare()
    put(path, cfg)
    output = q.ROOT/q.GPU_STAGES[0]["output"]
    put(output/"status.json", {"status":"completed", "exit_code":0})
    with pytest.raises(FileExistsError):
        q.prepare()
    assert json.loads((output/"status.json").read_text())["status"] == "completed"


def test_predecessor_requires_both_success_terminals_and_original_identity(workspace, monkeypatch):
    q = workspace
    monkeypatch.setattr(q, "alive", lambda record: True)
    assert q.predecessors_ready()
    pred = q.PREDECESSORS[1]
    put(q.ROOT/pred["status"], {"status":"running", "launch":pred["identity"]})
    assert not q.predecessors_ready()
    record = dict(pred["identity"], process_start_ticks="reused")
    put(q.ROOT/pred["launch"], record)
    with pytest.raises(ValueError, match="identity"):
        q.predecessors_ready()


@pytest.mark.parametrize("status", [
    {"status":"failed","exit_code":1},
    {"status":"completed"},
    {"status":"running"},
])
def test_dead_or_failed_predecessor_never_waits_forever(workspace, monkeypatch, status):
    q = workspace
    monkeypatch.setattr(q, "alive", lambda record: False)
    pred = q.PREDECESSORS[0]
    put(q.ROOT/pred["status"], status)
    with pytest.raises(RuntimeError):
        q.predecessors_ready()


def test_pid_reuse_and_boot_change_are_not_alive():
    import os
    q = queue()
    record = q.identity(os.getpid())
    assert q.alive(record)
    assert not q.alive(dict(record, process_start_ticks="0"))
    assert not q.alive(dict(record, boot_id="different"))


def test_gpu_gate_requires_memory_disk_and_shared_lock(workspace, monkeypatch):
    import fcntl
    q = workspace
    monkeypatch.setattr(q.shutil, "disk_usage", lambda path:SimpleNamespace(free=20*1024**3))
    monkeypatch.setattr(q.subprocess, "check_output", lambda *a,**kw:"1024\n")
    assert q.gpu_gate()["ready"]
    with (q.ROOT/"outputs/image_experiments/gpu0-research.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not q.gpu_gate()["ready"]
    monkeypatch.setattr(q.subprocess, "check_output", lambda *a,**kw:"1025\n")
    assert not q.gpu_gate()["ready"]
    monkeypatch.setattr(q.shutil, "disk_usage", lambda path:SimpleNamespace(free=20*1024**3-1))
    with pytest.raises(RuntimeError, match="disk"):
        q.gpu_gate()


def test_wait_checks_binding_and_outputs_on_each_poll(workspace, monkeypatch):
    q = workspace
    stages, binding = q.prepare()
    monkeypatch.setattr(q, "gpu_gate", lambda:dict(ready=False, used_mib=1287))
    messages = []
    def changed_after_wait(seconds):
        assert seconds == 60
        (q.ROOT/"shared.py").write_text("changed")
    monkeypatch.setattr(q.time, "sleep", changed_after_wait)
    with pytest.raises(ValueError, match="changed"):
        q.wait_gpu(binding, stages, lambda **v:messages.append(v))
    assert messages[0]["status"] == "waiting_gpu0"


def test_child_failure_preserves_exit_log_and_blocks_all_later_stages(workspace, monkeypatch):
    q = workspace
    stages, binding = q.prepare()
    state = q.ROOT/q.STATE
    state.mkdir(parents=True)
    monkeypatch.setattr(q, "identity", lambda pid:dict(pid=pid, boot_id="fixture", process_start_ticks="9"))
    calls = []
    class Failed:
        pid = 1234
        def __init__(self, command, **kwargs):
            calls.append(command)
            kwargs["stdout"].write(b"intentional worker failure\n")
        def wait(self, timeout):
            return 7
    monkeypatch.setattr(q.subprocess, "Popen", Failed)
    monkeypatch.setattr(q, "wait_predecessors", lambda *args:None)
    monkeypatch.setattr(q, "wait_gpu", lambda *args:None)
    with pytest.raises(RuntimeError, match="exit"):
        q.execute(stages, binding, state, lambda **v:None)
    assert len(calls) == 1
    folder = state/"dna"
    assert json.loads((folder/"exit.json").read_text())["exit_code"] == 7
    assert (folder/"worker.log").read_text() == "intentional worker failure\n"
    assert json.loads((folder/"launch.json").read_text())["process_start_ticks"] == "9"
    assert not (state/"omnidfa").exists()


def test_zero_child_exit_without_completion_is_failure(workspace, monkeypatch):
    q = workspace
    stages, binding = q.prepare()
    state = q.ROOT/q.STATE
    state.mkdir(parents=True)
    monkeypatch.setattr(q, "identity", lambda pid:dict(pid=pid,boot_id="fixture",process_start_ticks="9"))
    class Empty:
        pid = 1234
        def __init__(self,*args,**kwargs):pass
        def wait(self,timeout):return 0
    monkeypatch.setattr(q.subprocess,"Popen",Empty)
    with pytest.raises((RuntimeError,FileNotFoundError)):
        q.run_child(stages[0],binding,state,lambda **v:None)
    assert json.loads((state/"dna/exit.json").read_text())["exit_code"] == 0
    assert json.loads((state/"dna/status.json").read_text())["status"] == "failed"


def test_serial_order_commands_logs_and_comparisons_skip_gpu_gate(workspace, monkeypatch):
    q = workspace
    stages, binding = q.prepare()
    state = q.ROOT/q.STATE
    state.mkdir(parents=True)
    seen, gates, final = [], [], []
    monkeypatch.setattr(q, "identity", lambda pid:dict(pid=pid,boot_id="fixture",process_start_ticks="9"))
    monkeypatch.setattr(q, "wait_gpu", lambda binding,remaining,save:gates.append(remaining[0]["name"]))
    class Completed:
        pid = 1234
        def __init__(self, command, **kwargs):
            stage = stages[len(seen)]
            assert command[:2] == [str(q.ROOT/".venv-image-next/bin/python"),stage["script"]]
            if stage["gpu"]:
                assert command[2:] == ["--config",stage["config"],"--execute"]
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            else:
                assert command[-2:] == ["--output",stage["output"]]
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            seen.append(stage["name"])
            kwargs["stdout"].write(b"fixture completed\n")
            put(q.ROOT/stage["output"]/stage["completion"],dict(status="completed",exit_code=0))
        def wait(self, timeout):
            return 0
    monkeypatch.setattr(q.subprocess,"Popen",Completed)
    q.execute(stages,binding,state,lambda **v:final.append(v))
    assert seen == ["dna","omnidfa","ofa-tplus1",
                    "convnext-comparison","convnext-parent-dev-reference",
                    "dna-comparison","dna-parent-dev-reference",
                    "omnidfa-comparison","omnidfa-parent-dev-reference","ofa-tplus1-comparison"]
    assert gates == ["dna","omnidfa","ofa-tplus1"]
    assert final[-1]["status"] == "awaiting_git_archive"
    assert final[-1]["exit_code"] == 0
    for name in seen:
        assert json.loads((state/name/"status.json").read_text())["status"] == "completed"
        assert json.loads((state/name/"exit.json").read_text())["exit_code"] == 0
        assert (state/name/"worker.log").read_text() == "fixture completed\n"


def test_private_lock_blocks_another_queue_before_state_creation(workspace, monkeypatch):
    import fcntl
    q = workspace
    monkeypatch.setattr(sys,"argv",["queue","--execute"])
    with (q.ROOT/q.BASE/"continuation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            q.main()
    assert not (q.ROOT/q.STATE).exists()


def test_binding_change_during_child_preserves_unknown_exit_and_stops(workspace,monkeypatch):
    q=workspace
    stages,binding=q.prepare()
    state=q.ROOT/q.STATE
    state.mkdir(parents=True)
    monkeypatch.setattr(q,"identity",lambda pid:dict(pid=pid,boot_id="fixture",process_start_ticks="9"))
    class Running:
        pid=1234
        def __init__(self,*args,**kwargs):pass
        def wait(self,timeout):
            (q.ROOT/"shared.py").write_text("changed during execution")
            raise q.subprocess.TimeoutExpired("fixture",timeout)
    monkeypatch.setattr(q.subprocess,"Popen",Running)
    with pytest.raises(ValueError,match="changed"):
        q.run_child(stages[0],binding,state,lambda **v:None)
    assert not (state/"dna/exit.json").exists()
    record=json.loads((state/"dna/status.json").read_text())
    assert record["status"]=="failed" and record["child"]["pid"]==1234


def test_main_requires_explicit_execute_and_refuses_reusing_queue_state(workspace, monkeypatch):
    q = workspace
    monkeypatch.setattr(sys,"argv",["queue"])
    with pytest.raises(SystemExit):
        q.main()
    assert not (q.ROOT/q.STATE).exists()
    state=q.ROOT/q.STATE
    state.mkdir(parents=True)
    put(state/"status.json",dict(status="failed",exit_code=1))
    monkeypatch.setattr(sys,"argv",["queue","--execute"])
    with pytest.raises(FileExistsError):
        q.main()
    assert json.loads((state/"status.json").read_text())["status"] == "failed"


def test_rejection_statistical_helper_is_immutable(workspace):
    q=workspace
    _,binding=q.prepare()
    relative="scripts/compare_image_rejection_routes.py"
    assert relative in binding
    (q.ROOT/relative).write_text("# changed bootstrap or calibration logic\n")
    with pytest.raises(ValueError,match="changed"):
        q.check_binding(binding)
