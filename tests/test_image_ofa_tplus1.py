"""Isolation and rejection semantics for the separately registered OFA adaptation."""
import importlib.util
import random
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
import torch


def route():
    spec = importlib.util.find_spec('research.image_attribution.ofa_tplus1')
    assert spec is not None, 'OFA T+1 isolation module must be implemented'
    from research.image_attribution import ofa_tplus1
    return ofa_tplus1


def row(i, split='train', source=None, role=None):
    result = dict(sample_id=f'r{i}', group_id=f'g{i}', sha256=f'h{i}', pixel_sha256=f'p{i}',
        split=split, source_label=source, is_ai=source is not None, semantic_class=f's{i%10}', relative_path=f'{i}.png')
    if role: result['role'] = role
    return result


def test_native_rng_isolation_trace_and_reload_without_json_replay():
    m = route()
    class Decoder(torch.nn.Module):
        def forward(self, x):
            choice = random.randrange(1000000)
            return x + choice / 1000000, [{'index': 1, 'op': choice}]
    x = torch.ones(2, 3, 4, 4)
    random.seed(99);before = random.getstate()
    y,trace = m.reconstruct(torch.nn.Identity(), Decoder(), x, 42)
    assert random.getstate() == before
    single = [m.reconstruct(torch.nn.Identity(), Decoder(), v[None], 42, trace)[0] for v in x]
    assert torch.equal(y, torch.cat(single))
    assert torch.equal(y, m.reconstruct(torch.nn.Identity(), Decoder(), x, 42, trace)[0])
    with pytest.raises(ValueError, match='architecture'):
        m.reconstruct(torch.nn.Identity(), Decoder(), x, 43, trace)
    assert random.getstate() == before


def test_1024_distinct_real_train_groups_balanced_semantics_and_architectures():
    m=route();rows=[row(i) for i in range(1400)]+[row(2000,'holdout'),row(2001,'dev')]
    rows += [dict(rows[0],sample_id='sibling')]
    selected=m.select_reconstructions(rows,42)
    assert selected == m.select_reconstructions(list(reversed(rows)),42)
    assert len(selected)==len({r['group_id'] for r in selected})==1024
    assert len({r['sample_id'] for r in selected})==1024
    assert set(Counter(r['architecture_index'] for r in selected).values())=={128}
    assert all(len({r['semantic_class'] for r in selected if r['architecture_index']==a})==10 for a in range(8))
    counts=Counter(r['semantic_class'] for r in selected)
    assert max(counts.values())-min(counts.values())<=1
    assert all(r['parent']['split']=='train' and not r['parent']['is_ai'] for r in selected)
    assert not ({r['sample_id'] for r in selected}&{r['sample_id'] for r in rows})


@pytest.mark.parametrize('field',['group_id','sha256','pixel_sha256'])
def test_raw_parent_identity_cannot_reuse_any_nontrain_group_or_hash(field):
    m=route();rows=[row(i) for i in range(1100)]
    forbidden=row(5000,'holdout');forbidden[field]=rows[0][field]
    with pytest.raises(ValueError,match='protected'):
        m.select_reconstructions(rows+[forbidden],42)


def test_selection_uses_known_only_and_unknown_argmax_is_wrong():
    m=route();rows=[row(1,'dev','a','selection'),row(2,'dev','b','selection')]
    p=np.array([[.1,.1,.8],[.1,.8,.1]])
    assert m.selection_f1(rows,['a','b'],p)==.5
    with pytest.raises(ValueError,match='selection'):
        m.selection_f1([dict(rows[0],role='calibration'),rows[1]],['a','b'],p)
    with pytest.raises(ValueError,match='selection'):
        m.selection_f1([dict(rows[0],source_label='kling'),rows[1]],['a','b'],p)


def test_threshold_is_only_calibration_kling_not_eval_or_known_confidence():
    m=route();rows=[row(1,'dev','a','calibration'),row(2,'dev','kling','calibration')]
    p=np.array([[.8,.1,.1],[.1,.1,.8]])
    threshold=m.calibrate(rows,['a','b'],p)
    p[0]=[.01,.01,.98]
    assert m.calibrate(rows,['a','b'],p)==threshold
    for replacement in [dict(rows[1],role='evaluation'),dict(rows[1],source_label='grok3'),dict(rows[1],split='holdout')]:
        with pytest.raises(ValueError,match='calibration'):
            m.calibrate([rows[0],replacement],['a','b'],p)
    report=m.report([row(3,'dev','a','evaluation'),row(4,'dev','grok3','evaluation')],['a','b'],np.array([[.8,.1,.1],[.05,.05,.9]]),threshold)
    assert report['counts']['unknown_rejection']==dict(successes=1,total=1)
    assert report['counts']['known_correct_and_retained']==dict(successes=1,total=1)
    assert report['unknown_auroc']==1


def test_safe_checkpoint_rejects_unknown_globals_without_allowlisting(tmp_path):
    m=route();path=tmp_path/'weights.pt'
    torch.save({'x':torch.ones(1)},path)
    assert torch.equal(m.safe_load(path)['x'],torch.ones(1))
    torch.save({'unsafe':Path('/no-execution')},path)
    with pytest.raises(ValueError,match='unsafe'):
        m.safe_load(path)


def test_training_runs_30_epochs_with_no_calibration_or_evaluation_argument(tmp_path):
    m=route();torch.set_num_threads(1)
    x=torch.nn.functional.normalize(torch.randn(12,128,generator=torch.Generator().manual_seed(1)),dim=1)
    rows=[row(1,'dev','a','selection'),row(2,'dev','b','selection')]
    head,result=m.fit_head(x,torch.tensor([0,1,2]*4),x[:2],rows,['a','b'],tmp_path,'cpu')
    import json
    history=[json.loads(x) for x in (tmp_path/'history.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in history]==list(range(1,31))
    best=max(r['selection_macro_f1'] for r in history)
    assert result['best_epoch']==next(r['epoch'] for r in history if r['selection_macro_f1']==best)
    assert head.out_features==3


def runner():
    assert importlib.util.find_spec('scripts.run_image_ofa_tplus1') is not None, 'separate runner is required'
    from scripts import run_image_ofa_tplus1
    return run_image_ofa_tplus1


def test_output_one_shot_and_identity_includes_roles_and_source_assets(tmp_path):
    r=runner();out=tmp_path/'failed';out.mkdir();(out/'status.json').write_text('failed')
    with pytest.raises(FileExistsError):r.new_output(out)
    assert (out/'status.json').read_text()=='failed'
    cfg=dict(assets={'simulator':{'sha256':'fixed'}},implementation_sha256={'code':'fixed'},preprocessing={'crop':256})
    rows=[row(1,'dev','a','selection')]
    assert r.feature_identity(rows,cfg)!=r.feature_identity([dict(rows[0],role='evaluation')],cfg)
    with pytest.raises(ValueError,match='holdout'):
        r.feature_identity([row(1,'holdout','a','evaluation')],cfg)


def test_actual_selection_reload_must_match_saved_features(tmp_path,monkeypatch):
    r=runner();import hashlib
    from PIL import Image
    class Encoder(torch.nn.Module):
        def forward(self,x):return torch.cat([x.mean((2,3))]*43,1)[:,:128]
    rows=[]
    for i,color in enumerate([(30,90,180),(180,90,30)]):
        path=tmp_path/f'{i}.png';Image.new('RGB',(256,256),color).save(path)
        rows.append(dict(row(i,'dev','a','selection'),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    monkeypatch.setattr(r,'load_fingerprint',lambda *args:(Encoder(),torch.nn.Identity()))
    cached=r.encode_known(Encoder(),torch.nn.Identity(),tmp_path,rows,'cpu',2)
    head=torch.nn.Linear(128,3).eval()
    assert r.verify_selection({},tmp_path,rows,cached,head,'cpu')['images_opened']==2
    with pytest.raises(ValueError,match='reproducibility'):
        r.verify_selection({},tmp_path,rows,cached.flip(0),head,'cpu')


def test_saved_prediction_keeps_actual_source_and_float64_threshold_boundary():
    r=runner()
    assert hasattr(r,'prediction_record'), 'saved prediction contract must be explicit'
    p=np.array([.4,.4,.2],dtype=np.float32)
    threshold=np.nextafter(1-float(p[-1]),np.inf)
    item=r.prediction_record(row(1,'dev','kling','calibration'),['a','b'],p,threshold)
    assert item['source_label']=='kling'
    assert item['classes']==['a','b',route().UNKNOWN]
    assert not item['accepted']
    assert item['acceptance_confidence']==1-item['simulated_unknown_score']


def test_cpu_pipeline_keeps_original_roles_lineage_and_reload_before_calibration(tmp_path,monkeypatch):
    r=runner();m=route();import json
    torch.set_num_threads(1)
    class Encoder(torch.nn.Module):
        def forward(self,x):return torch.cat([x.mean((2,3))]*43,1)[:,:128]
    class Decoder(torch.nn.Module):
        def forward(self,x):
            n=random.randrange(1000000)
            return x+n/1000000,[{'index':1,'op':n}]
    monkeypatch.setattr(r,'load_fingerprint',lambda *args:(Encoder(),torch.nn.Identity()))
    monkeypatch.setattr(r,'load_simulator',lambda *args:(torch.nn.Identity(),Decoder()))
    monkeypatch.setattr(r,'image_tensor',lambda data,row,transform:torch.tensor([.2,.5,.8])[:,None,None].expand(3,4,4))
    classes=[f'c{i:02d}' for i in range(20)]
    parts={};index=0
    for role in ('train','selection','calibration','evaluation'):
        values=[]
        for source in classes+(['kling'] if role=='calibration' else ['grok3'] if role=='evaluation' else []):
            values.append(row(index,'train' if role=='train' else 'dev',source,role));index+=1
        parts[role]=values
    synthetic=[dict(sample_id=f'sim{i}',group_id=f'parent{i}',split='train',role='train',source_label=m.UNKNOWN,
        is_ai=True,semantic_class='s0',parent=row(i+5000),architecture_index=i%8) for i in range(1024)]
    cfg=dict(assets={},implementation_sha256={},preprocessing={},architecture_seeds=list(range(8)),limitations=[])
    original=r.calibrate
    def checked_calibrate(*args):
        assert (tmp_path/'actual_selection_verification.json').is_file()
        return original(*args)
    monkeypatch.setattr(r,'calibrate',checked_calibrate)
    r.run(cfg,tmp_path,parts,classes,synthetic,tmp_path,'cpu',lambda **kwargs:None)
    assert json.loads((tmp_path/'roles.json').read_text())=={k:[v['sample_id'] for v in values] for k,values in parts.items()}
    assert len((tmp_path/'derived_training.jsonl').read_text().splitlines())==1024
    predictions=[json.loads(line) for line in (tmp_path/'predictions.jsonl').read_text().splitlines()]
    assert len(predictions)==42 and all(v['classes']==classes+[m.UNKNOWN] for v in predictions)
    assert {v['source_label'] for v in predictions}==set(classes)|{'kling','grok3'}
    result=json.loads((tmp_path/'metrics.json').read_text())
    assert result['training_epochs']==30
    assert result['rejection']['evaluation']['unknown_n']==1
    assert result['rejection']['calibration']['threshold_unknown_source']=='kling'
