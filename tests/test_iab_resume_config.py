import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from types import SimpleNamespace

from scripts import monitor_iab_v2 as m


def test_pipeline_passes_registered_epochs_and_physical_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv('IAB_MONITOR_STATE_DIR', 'outputs/new-monitor')
    m.write(tmp_path/'outputs/new-monitor/config.json', dict(
        dataset='outputs/data', suffix='new-e20', epochs=20, gpu_index=1,
        parent_manifest='outputs/parent/manifest.jsonl'))
    args=m.command(tmp_path, 'pipeline')
    assert args[5] == str(tmp_path/'outputs/parent/manifest.jsonl')
    assert args[6:] == ['20', '1']


def test_gpu_probe_uses_selected_physical_card(monkeypatch):
    seen=[]
    def run(args, **kwargs):
        seen.append(args)
        return SimpleNamespace(stdout='400, 24564, 0')
    monkeypatch.setattr(m.subprocess, 'run', run)
    result=m.gpu(1)
    assert '--id=1' in seen[0]
    assert result['index']==1 and result['used_mib']==400


def test_gpu1_snapshot_does_not_mislabel_ollama_as_research_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv('IAB_MONITOR_STATE_DIR','outputs/new-monitor')
    m.write(tmp_path/'outputs/new-monitor/config.json',dict(gpu_index=1,epochs=20))
    monkeypatch.setattr(m,'gpu',lambda index=0:dict(index=index,used_mib=400))
    report=m.snapshot(tmp_path)
    assert report['gpu']['index']==1
    assert 'gpu0' not in report


@pytest.mark.parametrize('existing_epochs',[None,12])
def test_pipeline_shell_applies_epochs_and_gpu_without_training(tmp_path,existing_epochs):
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'scripts').mkdir()
    shutil.copyfile(root/'scripts/run_iab_m2.sh',tmp_path/'scripts/run_iab_m2.sh')
    data=tmp_path/'data';data.mkdir()
    (data/'manifest.jsonl').write_text('')
    (data/'selection.json').write_text(json.dumps(dict(entries={'a/cat':[{}]},audit=dict(selected_by_cell={'a/cat':1}))))
    (data/'download_status.json').write_text(json.dumps(dict(status='completed',selected=1,selected_by_cell={'a/cat':1})))
    runtime=tmp_path/'.venv-image/bin';runtime.mkdir(parents=True)
    py=runtime/'python'
    py.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
        'if sys.argv[1]=="-c": os.execv(sys.executable,[sys.executable]+sys.argv[1:])\n'
        'with open("calls.jsonl","a") as f: f.write(json.dumps(dict(args=sys.argv[1:],gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))+"\\n")\n')
    py.chmod(0o755)
    binpath=tmp_path/'bin';binpath.mkdir()
    smi=binpath/'nvidia-smi';smi.write_text('#!/bin/sh\n[ "$1" = "--id=1" ] || exit 7\necho 400\n');smi.chmod(0o755)
    if existing_epochs is not None:
        previous=tmp_path/'outputs/image_experiments/d0-resnet50-iab-test-e20';previous.mkdir(parents=True)
        (previous/'status.json').write_text(json.dumps(dict(status='completed')))
        (previous/'resolved_config.json').write_text(json.dumps(dict(epochs=existing_epochs)))
    env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'])
    result=subprocess.run(['bash','scripts/run_iab_m2.sh',str(data),'test-e20','outputs/summary.json','','20','1'],
        cwd=tmp_path,env=env,text=True,capture_output=True)
    if existing_epochs is not None:
        assert result.returncode!=0 and 'epoch count differs' in result.stderr
        assert not (tmp_path/'calls.jsonl').exists()
        return
    assert result.returncode==0,result.stdout+result.stderr
    calls=[json.loads(x) for x in (tmp_path/'calls.jsonl').read_text().splitlines()]
    neural=[x for x in calls if '--epochs' in x['args']]
    assert len(neural)==2
    assert all(x['args'][x['args'].index('--epochs')+1]=='20' and x['gpu']=='1' for x in neural)
    ofa=next(x for x in calls if '--task' in x['args'] and x['args'][x['args'].index('--task')+1]=='ofa')
    assert '--epochs' not in ofa['args'] and ofa['gpu']=='1'
