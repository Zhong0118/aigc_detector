import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn


def lib():
    return importlib.import_module('research.image_attribution.feature_ablation')


def fixture_rows():
    rows=[]
    for role in range(3):
        group=next('dev-'+str(n) for n in range(1000)
            if int(hashlib.sha256(('42:dev-'+str(n)).encode()).hexdigest(),16)%3==role)
        for label in ['a','b']:
            rows.append(dict(sample_id=f'{label}-{role}',group_id=group,split='dev',is_ai=1,
                source_label=label,semantic_class='cat',relative_path=f'{label}-{role}.png'))
    for label in ['a','b']:
        rows.append(dict(sample_id=label+'-train',group_id=label+'-train',split='train',is_ai=1,
            source_label=label,semantic_class='cat',relative_path=label+'-train.png'))
    rows += [dict(sample_id='sealed',group_id='sealed',split='holdout',is_ai=1,
        source_label='a',semantic_class='cat',relative_path='sealed.png'),
        dict(sample_id='real',group_id='real',split='train',is_ai=0,
        source_label=None,semantic_class='cat',relative_path='real.png')]
    return rows


def test_fusion_has_unit_norm_and_equal_branch_scale():
    m=lib();rgb=torch.zeros(3,2048);rgb[:,0]=3
    fp=torch.zeros(3,128);fp[:,0]=4
    result=m.combine_features('fusion',rgb=rgb,fingerprint=fp)
    assert result.shape==(3,2176)
    assert torch.allclose(result.norm(dim=1),torch.ones(3))
    assert torch.allclose(result[:,:2048].norm(dim=1),result[:,2048:].norm(dim=1))
    assert torch.allclose(result,m.combine_features('fusion',rgb=100*rgb,fingerprint=fp/10))
    assert m.combine_features('rgb',rgb=rgb).shape==(3,2048)
    assert m.combine_features('fingerprint',fingerprint=fp).shape==(3,128)


def test_invalid_or_misaligned_features_are_rejected():
    m=lib()
    with pytest.raises(ValueError):m.combine_features('fusion',rgb=torch.ones(2,2048),fingerprint=torch.ones(3,128))
    with pytest.raises(ValueError):m.combine_features('rgb',rgb=torch.full((2,2048),float('nan')))
    with pytest.raises(ValueError):m.combine_features('fingerprint',fingerprint=torch.ones(2,64))


def test_existing_group_roles_and_holdout_are_preserved():
    m=lib();rows=fixture_rows();original=copy.deepcopy(rows)
    parts,classes=m.source_partitions(rows,seed=42,expected_classes=['a','b'])
    assert classes==['a','b'] and rows==original
    assert {k:len(v) for k,v in parts.items()}==dict(train=2,selection=2,calibration=2,evaluation=2)
    chosen=[r for v in parts.values() for r in v]
    assert all(r['is_ai'] and r['split']!='holdout' for r in chosen)
    assert 'sealed' not in {r['sample_id'] for r in chosen}
    assert len({r['sample_id'] for r in chosen})==len(chosen)


def test_frozen_encoder_keeps_batchnorm_and_dropout_in_eval_mode():
    m=lib();model=nn.Sequential(nn.BatchNorm1d(4),nn.Dropout(.8),nn.Linear(4,2))
    model.train();before=model[0].running_mean.clone()
    m.freeze_encoder(model)
    x=torch.randn(8,4)
    with torch.no_grad():a=model(x);b=model(x)
    assert torch.equal(a,b) and torch.equal(before,model[0].running_mean)
    assert not any(p.requires_grad for p in model.parameters())


def test_cache_refuses_same_length_but_different_sample_order(tmp_path):
    m=lib();p=tmp_path/'features.pt'
    torch.save(dict(identity={'sample_ids':['b','a']},features=torch.ones(2,2048)),p)
    with pytest.raises(ValueError):m.load_bound_features(p,{'sample_ids':['a','b']},2048)


def test_disabled_execution_rejects_even_explicit_execute():
    m=lib()
    with pytest.raises(PermissionError,match='execution disabled'):
        m.require_execution({'execution_enabled':False},execute=True)


def test_metadata_preparation_never_opens_images(tmp_path,monkeypatch):
    m=lib();data=tmp_path/'data';data.mkdir();rows=fixture_rows()
    (data/'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (data/'selection.json').write_text('{}')
    (data/'download_status.json').write_text(json.dumps(dict(status='completed',selected=len(rows))))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=dict(seed=42,classes=['a','b'],execution_enabled=False,prerequisites=[],datasets={'fixture':dict(
        path='data',manifest_sha256=sha(data/'manifest.jsonl'),selection_sha256=sha(data/'selection.json'))})
    import PIL.Image
    monkeypatch.setattr(PIL.Image,'open',lambda *a,**k:(_ for _ in ()).throw(AssertionError('image opened')))
    plan,parts,classes=m.prepare(tmp_path,cfg,'fixture')
    assert plan['manifest_bound'] and plan['role_counts']['train']==2
    assert plan['images_opened'] is False and plan['training_started'] is False
    assert not list(tmp_path.rglob('*.pt')) and not list(tmp_path.rglob('*.pth'))


def test_unbound_future_manifest_is_not_guessed(tmp_path):
    m=lib();data=tmp_path/'future';data.mkdir();(data/'selection.json').write_text('{}')
    sha=hashlib.sha256((data/'selection.json').read_bytes()).hexdigest()
    cfg=dict(seed=42,classes=['a','b'],prerequisites=[],datasets={'future':dict(path='future',selection_sha256=sha,manifest_sha256=None)})
    plan,parts,classes=m.prepare(tmp_path,cfg,'future')
    assert plan['status']=='awaiting_manifest' and parts is None and not plan['manifest_bound']


def test_cli_cannot_start_registered_experiment(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/run_image_feature_ablation.py','--dataset','v5-50',
        '--variant','fusion','--action','run','--execute'],cwd=root,text=True,capture_output=True)
    assert result.returncode!=0 and 'execution disabled' in result.stderr
    assert not (root/'outputs/image_experiments/feature-ablation-v5-50/fusion').exists()


def test_toy_head_save_reload_and_evaluation_role_only(tmp_path, monkeypatch):
    m=lib();parts,classes=m.source_partitions(fixture_rows(),42,['a','b'])
    x=torch.zeros(6,2048)
    x[:,0]=torch.tensor([1.,-1.,1.,-1.,1.,-1.])
    calls=[];original=m.evaluate_epoch
    def record(model,loader,*a,**k):
        calls.append(len(loader.dataset))
        return original(model,loader,*a,**k)
    monkeypatch.setattr(m,'evaluate_epoch',record)
    import PIL.Image
    monkeypatch.setattr(PIL.Image,'open',lambda *a,**k:(_ for _ in ()).throw(AssertionError('image opened')))
    cfg=dict(seed=42,training=dict(epochs=2,batch_size=2,learning_rate=.001,scheduler_step=1,scheduler_gamma=.9))
    result=m.fit_head(x,parts,classes,cfg,tmp_path,torch.device('cpu'),'rgb',{'test':'synthetic-only'})
    saved=torch.load(tmp_path/'head.pth',map_location='cpu',weights_only=True)
    assert saved['schema']=='frozen-source-linear-v1' and saved['classes']==classes
    assert result['reload_max_error']==0 and result['single_batch_max_error']<1e-5
    assert len(calls)==4  # train/selection for two epochs, no per-epoch evaluation
    preds=[json.loads(line) for line in (tmp_path/'predictions.jsonl').read_text().splitlines()]
    assert [p['sample_id'] for p in preds]==[p['sample_id'] for p in parts['evaluation']]
    assert not result['calibration_used'] and not result['holdout_scored']
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['status']=='completed' and status['exit_code']==0
    history=[json.loads(line) for line in (tmp_path/'history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history]==[1,2]
    assert all('selection' in row and 'evaluation' not in row for row in history)


def test_same_count_cache_with_wrong_dimension_is_rejected(tmp_path):
    p=tmp_path/'cache.pt';identity={'sample_ids':['a','b']}
    torch.save(dict(identity=identity,features=torch.ones(2,128)),p)
    with pytest.raises(ValueError,match='shape mismatch'):lib().load_bound_features(p,identity,2048)


def test_missing_execute_never_grants_permission():
    with pytest.raises(PermissionError,match='explicit --execute'):lib().require_execution({'execution_enabled':True},False)


def test_unbound_manifest_remains_unbound_even_after_it_appears(tmp_path):
    data=tmp_path/'data';data.mkdir()
    (data/'selection.json').write_text('{}');(data/'manifest.jsonl').write_text('not read before binding')
    cfg=dict(datasets={'future':dict(path='data',selection_sha256=hashlib.sha256(b'{}').hexdigest(),manifest_sha256=None)})
    plan,parts,classes=lib().prepare(tmp_path,cfg,'future')
    assert plan['status']=='manifest_binding_required' and parts is None and classes is None
    assert plan['manifest_sha256']==hashlib.sha256(b'not read before binding').hexdigest()
    assert not plan['manifest_bound']


def test_runtime_refuses_unbound_plan_before_touching_assets(tmp_path):
    with pytest.raises(ValueError,match='explicitly bound'):lib().verify_runtime(tmp_path,{},dict(manifest_bound=False))
