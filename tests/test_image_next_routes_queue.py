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

@pytest.mark.parametrize('change',[None,'live','config','evidence','output'])
def test_namespace_recovery_gate_preserves_failure_and_recipe(tmp_path,monkeypatch,change):
    m=mod();monkeypatch.setattr(m,'ROOT',tmp_path)
    def put(name,data):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(data));return p
    cfg={'verification_receipt':{'path':'old.json','sha256':None},'epoch':None}
    converted=put(m.STATE/'omniaid-conversion-config.json',cfg)
    receipt=put(m.ASSETS/'preparation_imported_q179_20261029.json',{'status':'completed'})
    state=put(m.STATE/'status.json',{'status':'failed','stage':'omniaid-verify','child_exit_code':1,'launch':{'pid':1},'child':{'pid':2},'completed_stages':[{'name':'omniaid-conversion','exit_code':0,'receipt':{'path':str(receipt.relative_to(tmp_path)),'sha256':m.sha256(receipt)},'config_sha256':m.sha256(converted)}]})
    failed=put(m.ASSETS/'verification_status_q179_20261029.json',{'status':'failed','error_type':'UnpicklingError','error':'argparse.Namespace'})
    registration=put(m.STATE/'registration.json',{})
    original_hash=m.sha256
    monkeypatch.setattr(m,'sha256',lambda p:'dd3b6c313cad34b2cefa284e9de45e73c258e8d9ce0d790b7e741681b8437262' if Path(p)==registration else original_hash(p))
    put(m.RECOVERY_SNAPSHOT,{'files':{str(state.relative_to(tmp_path)):m.sha256(state),str(failed.relative_to(tmp_path)):m.sha256(failed)}})
    cfg['metadata_load']={'weights_only':True,'allowed_globals':['argparse.Namespace']}
    cfg['verification_receipt']['path']=str(m.ASSETS/'full_reload_q179_20261029_namespace_v1.json')
    if change=='config':cfg['epoch']=30
    put(m.OMNI_CONFIG,cfg)
    monkeypatch.setattr(m,'alive',lambda _:change=='live')
    if change=='evidence':failed.write_text('{}')
    if change=='output':(tmp_path/'outputs/image_experiments/openclip-q179-20261029').mkdir(parents=True)
    if change is None:assert m.namespace_recovery_evidence()['scope'].startswith('skip completed conversion')
    else:
        with pytest.raises((RuntimeError,ValueError,FileExistsError)):m.namespace_recovery_evidence()
