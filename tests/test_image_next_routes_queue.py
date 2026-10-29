import importlib,json
from pathlib import Path
import pytest

def mod():return importlib.import_module("scripts.run_image_next_routes")
def test_failed_import_is_not_retried(tmp_path):
    p=tmp_path/"import.json";p.write_text(json.dumps(dict(status="failed",http_code=403)))
    with pytest.raises(RuntimeError,match="failed"):mod().import_ready(p)
def test_completed_import_requires_exit_evidence_and_file(tmp_path):
    p=tmp_path/"import.json";p.write_text(json.dumps(dict(status="completed")))
    with pytest.raises(ValueError,match="receipt"):mod().import_ready(p)
def test_stale_running_import_is_not_treated_as_healthy(tmp_path,monkeypatch):
    p=tmp_path/"import.json";p.write_text(json.dumps(dict(status="running",launch=dict(pid=99999999,boot_id="old",process_start_ticks="1"))))
    monkeypatch.setattr(mod(),"alive",lambda _:False)
    with pytest.raises(RuntimeError,match="exited"):mod().import_ready(p)
def test_absent_import_waits_without_spawning(tmp_path):
    assert mod().import_ready(tmp_path/"absent.json") is False
def test_identity_must_match_boot_and_start():
    identity=mod().identity()
    assert mod().alive(identity)
    assert not mod().alive(dict(identity,boot_id="wrong"))


def queue_fixture(tmp_path, monkeypatch, fail=False):
    m=mod(); monkeypatch.setattr(m,'ROOT',tmp_path)
    cfg=tmp_path/m.OMNI_CONFIG;cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({'assets':{'model_config':{},'clip_config':{},'checkpoint':{},'clip_weights':{'derived_from':{}}}}))
    monkeypatch.setattr(m,'SOURCES',[])
    monkeypatch.setattr(m,'import_ready',lambda *a:True)
    monkeypatch.setattr(m,'gpu_used',lambda:18)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:type('Usage',(),{'free':21*1024**3})())
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'testcommit\n')
    monkeypatch.setattr(m,'stages',lambda:[('first',['stub'],True,Path('first.json'),True),('second',['stub'],False,Path('second.json'),False)])
    calls=[]
    class Child:
        pid=99999999
        def wait(self):return 1 if fail else 0
    def spawn(*a,**kw):
        name=['first','second'][len(calls)];calls.append(name)
        if not fail:
            (tmp_path/(name+'.json')).write_text(json.dumps({'status':'completed','exit_code':0}))
            if name=='first':cfg.write_text(json.dumps({'converted':True}))
        return Child()
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    return m,calls

def test_queue_stops_on_child_failure_preserves_state_and_refuses_restart(tmp_path,monkeypatch):
    m,calls=queue_fixture(tmp_path,monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='exited 1'):m.execute()
    state=json.loads((tmp_path/m.STATE/'status.json').read_text())
    assert state['status']=='failed' and state['child_exit_code']==1 and calls==['first']
    with pytest.raises(FileExistsError):m.execute()
    assert calls==['first']

def test_queue_serial_completion_accepts_only_owned_config_transition(tmp_path,monkeypatch):
    m,calls=queue_fixture(tmp_path,monkeypatch)
    m.execute()
    state=json.loads((tmp_path/m.STATE/'status.json').read_text())
    assert state['status']=='awaiting_git_archive' and calls==['first','second']
    assert [s['name'] for s in state['completed_stages']]==calls
    assert all(s['child']['pid']==99999999 for s in state['completed_stages'])
    assert (tmp_path/m.STATE/'first-launch.json').exists()
    assert (tmp_path/m.STATE/'second-launch.json').exists()
    assert (tmp_path/m.STATE/'initial_omniaid_config.json').exists()
    assert (tmp_path/m.STATE/'first-config.json').exists()

def test_queue_disk_failure_prevents_child_launch(tmp_path,monkeypatch):
    m,calls=queue_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:type('Usage',(),{'free':19*1024**3})())
    with pytest.raises(RuntimeError,match='20 GiB'):m.execute()
    assert calls==[]

def test_model_completion_requires_explicit_zero_exit(tmp_path):
    p=tmp_path/'status.json';p.write_text(json.dumps({'status':'completed'}))
    with pytest.raises(ValueError,match='successfully'):mod().completed(p)
