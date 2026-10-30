import copy
import numpy as np
import pytest
from scripts.compare_ofa_tplus1 import extract, oscr, validate_saved_tplus, FIELDS


def fixture():
    parts={r:[] for r in ('train','selection','calibration','evaluation')}
    rows=[]
    for i,(role,source) in enumerate([('calibration','a'),('calibration','kling'),('evaluation','a'),('evaluation','grok3')]):
        x=dict(sample_id=str(i),group_id=str(i),source_label=source,is_ai=1,semantic_class='faces',split='dev')
        parts[role].append(x)
        rows.append(dict(x,role=role,condition='clean',classes=['a','b','simulated_unknown'],scores=[.2,.1,.7]))
    return parts,rows


def test_tplus_acceptance_is_known_mass_and_predictions_keep_known_space():
    parts,rows=fixture();result=extract(rows,parts,['a','b'],'tplus1')
    assert np.allclose(result['calibration']['confidence'],[.3,.3])
    assert np.array_equal(result['evaluation']['pred'],[0,0])
    assert np.array_equal(result['evaluation']['full_pred'],[2,2])


def test_complete_roles_labels_and_vocab_required():
    parts,rows=fixture()
    with pytest.raises(ValueError):extract(rows[:-1],parts,['a','b'],'tplus1')
    with pytest.raises(ValueError):extract(rows+[rows[0]],parts,['a','b'],'tplus1')
    for field,value in [('role','train'),('group_id','other'),('source_label','wrong'),('classes',['simulated_unknown','a','b'])]:
        bad=copy.deepcopy(rows);bad[0][field]=value
        with pytest.raises(ValueError):extract(bad,parts,['a','b'],'tplus1')
    parts['calibration'][0]['split']='holdout'
    with pytest.raises(ValueError):extract(rows,parts,['a','b'],'tplus1')


def test_oscr_accounts_for_known_misclassification_and_ties():
    rows=[dict(source_label=x) for x in ['a','b','grok3']]
    assert oscr(rows,['a','b'],[.9,.8,.1],[0,0,0])==.5
    assert oscr(rows,['a','b'],[.5,.5,.5],[0,1,0])==.5
    assert oscr(rows,['a','b'],[.9,.8,.1],[0,1,0])==1.
    with pytest.raises(ValueError):oscr(rows,['a','b'],[.9,float('nan'),.1],[0,1,0])


def test_tplus_saved_metrics_and_calibration_threshold_must_match():
    report=dict(threshold=.7,**{f:.5 for f in FIELDS})
    saved=dict(holdout_scored=False,training_epochs=30,rejection=dict(
        threshold_fitted_role='calibration',acceptance_threshold=.7,evaluation={f:.5 for f in FIELDS}))
    validate_saved_tplus(report,saved)
    saved['rejection']['acceptance_threshold']=.8
    with pytest.raises(ValueError):validate_saved_tplus(report,saved)


def test_main_file_fixture_completes_and_checks_model_hash(tmp_path,monkeypatch):
    import json
    import sys
    from pathlib import Path
    from scripts import compare_ofa_tplus1 as m
    from research.image_attribution.openclip_route import digest
    def put(path,value,lines=False):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(json.dumps(x)+'\n' for x in value) if lines else json.dumps(value))
        return p
    known=[f's{i:02}' for i in range(20)];parts={r:[] for r in ('train','selection','calibration','evaluation')}
    for role,n,unknown_n,unknown_source in [('train',20,0,''),('selection',2,0,''),('calibration',1482,64,'kling'),('evaluation',1275,63,'grok3')]:
        for i in range(n):
            parts[role].append(dict(sample_id=f'{role}-{i}',group_id=f'{role}-{i}',is_ai=True,
                source_label=unknown_source if i<unknown_n else known[i%20],semantic_class='faces',split='train' if role=='train' else 'dev'))
    manifest=put('manifest.jsonl',[r for v in parts.values() for r in v],True)
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'MANIFEST','manifest.jsonl')
    monkeypatch.setattr(m,'MANIFEST_SHA',m.sha256(manifest));monkeypatch.setattr(m,'partition_rows',lambda rows,recipe:parts)
    monkeypatch.setattr(m,'bootstrap',lambda *a,**k:{'fixture_only':True})
    script=put('scripts/compare_ofa_tplus1.py',{});monkeypatch.setattr(m,'__file__',str(script))
    for helper in ['scripts/compare_image_rejection_routes.py','research/image_attribution/followups.py','scripts/compare_image_scale_epochs.py','research/image_attribution/openclip_route.py']:put(helper,{})
    folders={'resnet':'outputs/image_experiments/followups-v7-200-q179-e30/A1/evaluate',
        'openclip':'outputs/image_experiments/openclip-q179-20261029/A-OpenCLIP-open20',
        'tplus1':'outputs/image_experiments/ofa-tplus1-q179-20261029'}
    for kind,folder in folders.items():
        p=Path(folder);records=[]
        for role in ('calibration','evaluation'):
            for row in parts[role]:
                prob=np.full(20,.1/19);prob[known.index(row['source_label']) if row['source_label'] in known else 0]=.9
                if kind=='tplus1':
                    mass=.99 if row['source_label'] in known else .2
                    prob=np.r_[prob*mass,1-mass]
                records.append(dict(row,role=role,condition='clean',classes=known+(['simulated_unknown'] if kind=='tplus1' else []),
                    scores=prob.tolist(),prototype_cosine=.8 if row['source_label'] in known else .1))
        route=m.extract(records,parts,known,kind)
        rep=m.report(parts['calibration'],parts['evaluation'],known,route['calibration']['confidence'],route['evaluation']['confidence'],route['evaluation']['pred'])
        if kind=='resnet':
            metric={'conditions':{'clean':{'acceptance_threshold':rep['threshold'],'known_acceptance':rep['known_retention'],
                'known_correct_and_accepted_rate':rep['known_correct_and_retained'],'unknown_false_acceptance':rep['unknown_false_acceptance'],'unknown_auroc':rep['unknown_auroc']}}}
        elif kind=='openclip':
            proto=m.report(parts['calibration'],parts['evaluation'],known,route['calibration']['prototype'],route['evaluation']['prototype'],route['evaluation']['pred'])
            metric={'rejection':{'msp':{'evaluation':rep},'prototype_cosine':{'evaluation':proto}}}
        else:
            metric=dict(holdout_scored=False,training_epochs=30,rejection=dict(threshold_fitted_role='calibration',acceptance_threshold=rep['threshold'],evaluation=rep))
        put(p/'status.json',dict(status='completed',exit_code=0));put(p/'metrics.json',metric)
        put(p/'roles.json',{k:[r['sample_id'] for r in v] for k,v in parts.items()});put(p/'predictions.jsonl',records,True)
    p=Path(folders['tplus1']);cfg={'manifest_sha256':m.MANIFEST_SHA}
    put(p/'registration.json',cfg)
    prov=dict(config=cfg,config_sha256=digest(cfg),holdout_read_for_inference=False,
        role_ids_sha256={k:digest([r['sample_id'] for r in v]) for k,v in parts.items()})
    for filename,key in [('head.pth','head_sha256'),('architecture_traces.json','architecture_traces_sha256'),('derived_training.jsonl','synthetic_manifest_sha256')]:
        f=put(p/filename,{});prov[key]=m.sha256(f)
    put(p/'provenance.json',prov);put(p/'history.jsonl',[dict(epoch=i) for i in range(1,31)],True)
    put(p/'actual_selection_verification.json',dict(images_opened=2,full_offline_backbone_reload=True,
        selection_sample_ids=[r['sample_id'] for r in parts['selection']],head_reload_max_error=0.))
    output=tmp_path/'report';monkeypatch.setattr(sys,'argv',['compare','--output',str(output)])
    m.main()
    assert json.loads((output/'completion.json').read_text())['exit_code']==0
    assert json.loads((output/'summary.json').read_text())['methods']['ofa_tplus1_known_mass']['oscr']==1.
    prov['head_sha256']='0'*64;put(p/'provenance.json',prov)
    monkeypatch.setattr(sys,'argv',['compare','--output',str(tmp_path/'bad-report')])
    with pytest.raises(ValueError,match='identity changed'):m.main()
    assert not (tmp_path/'bad-report').exists()
