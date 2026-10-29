"""CPU-only guard tests for the new frozen-detector route."""
import hashlib
import json
from pathlib import Path
import pytest
import torch

def test_asset_destination_cannot_escape_owned_directory(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    with pytest.raises(ValueError, match='asset directory'):
        prep.asset_path(tmp_path, '../outside')

def test_conversion_is_tensor_exact_and_registers_hash(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    source = tmp_path / 'input.bin'
    torch.save({'weight': torch.arange(6).reshape(2,3)}, source)
    spec = {'bytes': source.stat().st_size, 'sha256': prep.sha256(source)}
    result = prep.convert_clip(source, tmp_path / 'model.safetensors', spec)
    from safetensors.torch import load_file
    assert torch.equal(load_file(str(tmp_path / 'model.safetensors'))['weight'], torch.arange(6).reshape(2,3))
    assert result['tensor_equality_verified'] is True
    assert result['sha256'] == prep.sha256(tmp_path / 'model.safetensors')

def test_conversion_refuses_bad_hash_without_loading(tmp_path, monkeypatch):
    from scripts import prepare_omniaid_q179 as prep
    source = tmp_path / 'input.bin'
    source.write_bytes(b'untrusted')
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('must not load'))
    with pytest.raises(ValueError, match='hash/size'):
        prep.convert_clip(source, tmp_path / 'model.safetensors', {'bytes':9,'sha256':'0'*64})

def test_unsafe_pickle_is_never_retried(tmp_path, monkeypatch):
    from scripts import prepare_omniaid_q179 as prep
    source = tmp_path / 'input.bin'
    source.write_bytes(b'x')
    calls = []
    def blocked(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('unsupported pickle global')
    monkeypatch.setattr(torch, 'load', blocked)
    with pytest.raises(RuntimeError, match='unsupported pickle'):
        prep.convert_clip(source, tmp_path / 'model.safetensors', {'bytes':1,'sha256':prep.sha256(source)})
    assert len(calls) == 1 and calls[0]['weights_only'] is True
    assert not (tmp_path / 'model.safetensors').exists()

def test_non_tensor_checkpoint_is_refused(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    source = tmp_path / 'input.bin'
    torch.save({'metadata': 'not a tensor'}, source)
    with pytest.raises(ValueError, match='tensor state'):
        prep.convert_clip(source, tmp_path / 'model.safetensors', {'bytes':source.stat().st_size,'sha256':prep.sha256(source)})

def test_wrapper_rejects_synthetic_receipt(tmp_path):
    from scripts import evaluate_omniaid_q179 as wrapper
    with pytest.raises(ValueError, match='actual selection'):
        wrapper.validate_actual_receipt(tmp_path, {}, {'status':'verified','reload_verified':True})

def test_wrapper_requires_physical_gpu0_idle(monkeypatch):
    from scripts import evaluate_omniaid_q179 as wrapper
    monkeypatch.setattr(wrapper.subprocess, 'check_output', lambda *a,**kw: '1025\n')
    with pytest.raises(RuntimeError, match='GPU 0'):
        wrapper.gpu_guard({'gpu_index':0}, 'cuda')
    monkeypatch.setattr(wrapper.subprocess, 'check_output', lambda *a,**kw: '1024\n')
    assert wrapper.gpu_guard({'gpu_index':0}, 'cuda') == 1024
    with pytest.raises(ValueError, match='GPU 0'):
        wrapper.gpu_guard({'gpu_index':1}, 'cuda')

def test_wrapper_rejects_escaping_output(tmp_path):
    from scripts import evaluate_omniaid_q179 as wrapper
    with pytest.raises(ValueError, match='output'):
        wrapper.validate_route(tmp_path, {'output':'../escape'})

def test_actual_probe_rejects_other_roles(tmp_path):
    from scripts import evaluate_omniaid_q179 as wrapper
    for role in ['calibration','evaluation','holdout','train']:
        with pytest.raises(ValueError, match='actual selection'):
            wrapper.validate_actual_receipt(tmp_path, {}, {
                'actual_images_verified':True, 'probe_role':role, 'sample_ids':['a','b']})

def test_preparation_lock_prevents_second_downloader(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    with prep.preparation_lock(tmp_path):
        with pytest.raises(RuntimeError, match='asset preparation'):
            with prep.preparation_lock(tmp_path):
                pytest.fail('second downloader entered')

def test_actual_image_receipt_requires_equal_finite_cache(tmp_path):
    from scripts import evaluate_omniaid_q179 as wrapper
    import numpy as np
    path=tmp_path / 'outputs/image_experiments/assets/omniaid_v2/probe.npy'
    path.parent.mkdir(parents=True)
    np.save(path, np.array([[.8,.2],[.1,.9]],dtype=np.float32),allow_pickle=False)
    scores=np.load(path,allow_pickle=False).tolist()
    config={'dataset':{'manifest_sha256':'m','selection_sha256':'s'}}
    receipt=dict(actual_images_verified=True,probe_role='selection',sample_ids=['a','b'],
        manifest_sha256='m',selection_sha256='s',cache_max_error=0.,reload_max_error=0.,
        single_batch_max_error=0.,cache={'path':str(path.relative_to(tmp_path)), 'sha256':wrapper.core.sha256(path)},scores=scores)
    wrapper.validate_actual_receipt(tmp_path,config,receipt)
    receipt['reload_max_error']=float('nan')
    with pytest.raises(ValueError,match='tolerance'):
        wrapper.validate_actual_receipt(tmp_path,config,receipt)
    receipt['reload_max_error']=0.
    receipt['scores'][0][0]=0.2
    with pytest.raises(ValueError,match='contents'):
        wrapper.validate_actual_receipt(tmp_path,config,receipt)

def test_probe_opens_only_two_selection_images(tmp_path, monkeypatch):
    from scripts import evaluate_omniaid_q179 as wrapper
    from PIL import Image
    rows=[dict(sample_id=str(i),group_id=str(i),split='dev',is_ai=i%2,
        source_label='generator' if i%2 else None,semantic_class='fixture',
        relative_path=str(i)+'.png') for i in range(120)]
    parts=wrapper.core.partition_rows(rows,dict(task='detection',mode='calibration',seed=42))
    selected=[next(row for row in parts['selection'] if row['is_ai'] == truth) for truth in (False,True)]
    for index,row in enumerate(selected):
        Image.new('RGB',(11,13),(index*200,50,100)).save(tmp_path/row['relative_path'])
        next(item for item in rows if item['sample_id']==row['sample_id'])['sha256']=wrapper.core.sha256(tmp_path/row['relative_path'])
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    (tmp_path/'selection.json').write_text('{}')
    monkeypatch.setattr(wrapper,'MANIFEST_SHA',wrapper.core.sha256(manifest))
    monkeypatch.setattr(wrapper,'SELECTION_SHA',wrapper.core.sha256(tmp_path/'selection.json'))
    monkeypatch.setattr(wrapper.core,'validate_manifest',lambda *args,**kwargs:None)
    actual, images=wrapper.probe_images(tmp_path,{'dataset':{'path':str(tmp_path)}})
    assert [row['sample_id'] for row in actual]==[row['sample_id'] for row in selected]
    assert images.shape==(2,3,336,336)
    assert not torch.equal(images[0],images[1])

def test_http_403_is_not_retried_and_partial_is_preserved(tmp_path, monkeypatch):
    from scripts import prepare_omniaid_q179 as prep
    import urllib.error
    calls=[]
    def denied(*args,**kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://official.example/pinned',403,'Forbidden',{},None)
    monkeypatch.setattr(prep.urllib.request,'urlopen',denied)
    spec=dict(path=str(prep.ASSET_ROOT/'denied.bin'),source_url='https://official.example/pinned',bytes=1,sha256='0'*64)
    with pytest.raises(urllib.error.HTTPError):
        prep.download(tmp_path,spec)
    assert len(calls)==1
    assert (tmp_path/spec['path']).with_suffix('.bin.partial').exists()
    assert not (tmp_path/spec['path']).exists()
    with pytest.raises(FileExistsError):
        prep.download(tmp_path,spec)
    assert len(calls)==1

def test_bad_download_hash_preserves_partial_without_publishing(tmp_path, monkeypatch):
    from scripts import prepare_omniaid_q179 as prep
    import io
    class Response(io.BytesIO):
        status=200
    monkeypatch.setattr(prep.urllib.request,'urlopen',lambda *a,**kw:Response(b'wrong'))
    spec=dict(path=str(prep.ASSET_ROOT/'bad.bin'),source_url='https://official.example/pinned',bytes=5,sha256='0'*64)
    with pytest.raises(ValueError,match='hash/size'):
        prep.download(tmp_path,spec)
    assert (tmp_path/spec['path']).with_suffix('.bin.partial').read_bytes()==b'wrong'
    assert not (tmp_path/spec['path']).exists()

def test_verification_failure_preserves_process_identity(tmp_path):
    from scripts import evaluate_omniaid_q179 as wrapper
    path=tmp_path/'status.json'
    with pytest.raises(RuntimeError,match='model failed'):
        with wrapper.stage_status(path, 'verify-load', {'seed':42}):
            running=json.loads(path.read_text())
            assert running['status']=='running'
            raise RuntimeError('model failed')
    failed=json.loads(path.read_text())
    assert failed['status']=='failed' and failed['exit_code']==1
    assert failed['launch']==running['launch']
    assert failed['launch']['pid']>0 and failed['launch']['boot_id']
    assert int(failed['launch']['process_start_ticks'])>0
    assert failed['launch']['commandline']
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        with wrapper.stage_status(path,'verify-load',{'seed':42}):
            pytest.fail('existing evidence must not be overwritten')
    assert path.read_bytes()==before

def test_explicit_stream_import_registers_original_identity(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    import io
    payload=b'exact pinned asset'
    spec=dict(path=str(prep.ASSET_ROOT/'imported.bin'),bytes=len(payload),
              sha256=hashlib.sha256(payload).hexdigest(),source_url='https://official.example/pinned')
    receipt=prep.import_stream(tmp_path,spec,io.BytesIO(payload),'fixture')
    assert (tmp_path/spec['path']).read_bytes()==payload
    assert receipt['source_url']==spec['source_url'] and receipt['sha256']==spec['sha256']
    assert receipt['transport']=='explicit official HTTPS stream over SSH'
    assert receipt['status']=='completed'
    assert receipt['launch']['boot_id'] and receipt['launch']['process_start_ticks']

def test_import_mismatch_preserves_bytes_and_failure_receipt(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    import io
    spec=dict(path=str(prep.ASSET_ROOT/'imported.bin'),bytes=4,
              sha256='0'*64,source_url='https://official.example/pinned')
    with pytest.raises(ValueError,match='hash/size'):
        prep.import_stream(tmp_path,spec,io.BytesIO(b'bad!'),'fixture')
    path=tmp_path/spec['path']
    assert not path.exists() and path.with_suffix('.bin.partial').read_bytes()==b'bad!'
    receipt=json.loads((tmp_path/prep.ASSET_ROOT/'import_fixture_q179_20261029.json').read_text())
    assert receipt['status']=='failed'
    before=path.with_suffix('.bin.partial').read_bytes()
    with pytest.raises(FileExistsError):
        prep.import_stream(tmp_path,spec,io.BytesIO(b'new!'),'fixture')
    assert path.with_suffix('.bin.partial').read_bytes()==before

def test_offline_import_receipt_must_match_existing_asset(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    import io
    payload=b'exact'
    spec=dict(path=str(prep.ASSET_ROOT/'imported.bin'),bytes=len(payload),
              sha256=hashlib.sha256(payload).hexdigest(),source_url='https://official.example/pinned')
    prep.import_stream(tmp_path,spec,io.BytesIO(payload),'fixture')
    prep.verify_import(tmp_path,spec,'fixture')
    (tmp_path/spec['path']).write_bytes(b'other')
    with pytest.raises(ValueError,match='hash/size'):
        prep.verify_import(tmp_path,spec,'fixture')

def test_import_worker_cannot_auto_start_queue_owned_conversion(tmp_path, monkeypatch):
    from scripts import prepare_omniaid_q179 as prep
    monkeypatch.setattr(prep,'ROOT',tmp_path)
    monkeypatch.setattr(prep.sys,'argv',['prepare_omniaid_q179.py','--convert-imported'])
    with pytest.raises(PermissionError,match='serial queue'):
        prep._main()
    assert not list(tmp_path.iterdir())

def test_unmarked_conversion_cannot_race_for_import_lock(monkeypatch):
    from scripts import prepare_omniaid_q179 as prep
    monkeypatch.setattr(prep.sys,'argv',['prepare_omniaid_q179.py','--convert-imported'])
    monkeypatch.setattr(prep,'preparation_lock',lambda *a:pytest.fail('unmarked tail must not race queue for lock'))
    with pytest.raises(PermissionError,match='serial queue'):
        prep.main()

@pytest.mark.parametrize('arguments',[['--import-stdin=checkpoint'],['--import-stdin','checkpoint']])
def test_both_import_cli_spellings_acquire_lock(monkeypatch,arguments):
    from scripts import prepare_omniaid_q179 as prep
    from contextlib import contextmanager
    entered=[]
    @contextmanager
    def locked(root):
        entered.append(True)
        yield
    monkeypatch.setattr(prep.sys,'argv',['prepare_omniaid_q179.py']+arguments)
    monkeypatch.setattr(prep,'preparation_lock',locked)
    monkeypatch.setattr(prep,'_prepare',lambda args:args.import_stdin)
    assert prep._main()=='checkpoint'
    assert entered==[True]

def test_controlled_resume_preserves_archived_prefix_and_checks_full_sha(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    import io
    payload=b'complete pinned payload'
    spec=dict(path=str(prep.ASSET_ROOT/'resume.bin'),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
        source_url='https://official.example/pinned')
    archive=tmp_path/prep.ASSET_ROOT/'failed-import-checkpoint-20261029'
    archive.mkdir(parents=True)
    prefix=archive/'resume.bin.partial'
    prefix.write_bytes(payload[:8])
    old=dict(status='failed',received_bytes=8,launch={'pid':99999999},**spec)
    (archive/'import_checkpoint_q179_20261029.json').write_text(json.dumps(old))
    archived=prefix.read_bytes()
    receipt=prep.resume_import(tmp_path,spec,io.BytesIO(payload[8:]),'checkpoint',archive)
    assert receipt['status']=='completed'
    assert receipt['resume']['prefix_bytes']==8
    assert receipt['sha256']==spec['sha256']
    assert prefix.read_bytes()==archived
    assert (tmp_path/spec['path']).read_bytes()==payload

def test_controlled_resume_rejects_changed_prior_identity(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    import io
    spec=dict(path=str(prep.ASSET_ROOT/'resume.bin'),bytes=5,sha256='0'*64,source_url='https://official.example/pinned')
    archive=tmp_path/prep.ASSET_ROOT/'failed-import-checkpoint-20261029'
    archive.mkdir(parents=True)
    (archive/'resume.bin.partial').write_bytes(b'pre')
    (archive/'import_checkpoint_q179_20261029.json').write_text(json.dumps(dict(spec,status='failed',received_bytes=3,
        sha256='1'*64,launch={'pid':99999999})))
    with pytest.raises(ValueError,match='prior import'):
        prep.resume_import(tmp_path,spec,io.BytesIO(b'fix'),'checkpoint',archive)
    assert not (tmp_path/spec['path']).exists()

def test_resume_handles_existing_shared_asset_directory_symlink(tmp_path):
    from scripts import prepare_omniaid_q179 as prep
    import io
    root=tmp_path/'worktree'
    target=tmp_path/'shared-assets'
    target.mkdir()
    (root/prep.ASSET_ROOT).parent.mkdir(parents=True)
    (root/prep.ASSET_ROOT).symlink_to(target,target_is_directory=True)
    payload=b'complete pinned payload'
    spec=dict(path=str(prep.ASSET_ROOT/'resume.bin'),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
        source_url='https://official.example/pinned')
    archive=root/prep.ASSET_ROOT/'failed-import-checkpoint-20261029'
    archive.mkdir()
    (archive/'resume.bin.partial').write_bytes(payload[:8])
    (archive/'import_checkpoint_q179_20261029.json').write_text(json.dumps(
        dict(status='failed',received_bytes=8,launch={'pid':99999999},**spec)))
    receipt=prep.resume_import(root,spec,io.BytesIO(payload[8:]),'checkpoint',archive)
    assert receipt['status']=='completed'
    assert receipt['resume']['prefix_path']==str(prep.ASSET_ROOT/'failed-import-checkpoint-20261029/resume.bin.partial')
    assert prep.asset_reference(root,target/'probe.npy')==str(prep.ASSET_ROOT/'probe.npy')
