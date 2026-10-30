"""Final read-only acceptance for OFA20+1 and the seven registered saved-score reports."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from research.image_attribution.protocol import sha256
from research.image_attribution.openclip_route import digest
from research.image_attribution.ofa_tplus1 import calibrate,report,safe_load,UNKNOWN
from research.image_attribution.followups import validate_probabilities
from scripts.run_image_ofa_tplus1 import prepare,feature_identity
from scripts.run_remaining_image_queue import stages,check_binding
from scripts.compare_image_scale_epochs import indexed,stats
from scripts.compare_image_next_routes import selected,normalize_legacy,validate_saved_roles
from scripts.compare_ofa_tplus1 import extract,report as rejection_report,oscr,FIELDS
OUT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/image-remaining-routes-20261029'
sources={};copied=[];acceptance={}
def read(path,lines=False):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    sources[str(path.relative_to(ROOT))]=sha256(path)
    return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())
def copy(path,relative):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    dest=OUT/'artifacts'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,dest);h=sha256(path);assert sha256(dest)==h
    sources[str(path.relative_to(ROOT))]=h
    copied.append(dict(source=str(path.relative_to(ROOT)),path=str(dest.relative_to(OUT)),sha256=h,bytes=path.stat().st_size))
def equal(a,b):
    if isinstance(a,dict):
        for k,v in a.items():equal(v,b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for v,w in zip(a,b):equal(v,w)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert np.isclose(a,b,rtol=0,atol=1e-10),(a,b)
    else:assert a==b,(a,b)
def completed(path,terminal='completed'):
    d=read(path);assert d['status']==terminal and d['exit_code']==0;return d

assert not (OUT/'snapshot.json').exists()
queue=BASE/'queue-continuation-v1'
registration=read(queue/'registration.json');check_binding(registration['inputs'])
completed(queue/'status.json','awaiting_git_archive')
assert read(queue/'status.json')['completed_stages']==[s['name'] for s in stages()]
for stage in stages():
    folder=queue/stage['name'];completed(folder/'status.json')
    assert read(folder/'exit.json')['exit_code']==0
    completed(ROOT/stage['output']/stage['completion'])
    for f in ('status.json','exit.json','launch.json'):copy(folder/f,Path('queue')/stage['name']/f)
for f in ('status.json','registration.json','launch.json'):copy(queue/f,Path('queue')/f)

cp=ROOT/'configs/image_experiments/iab_ofa_tplus1_q179_20261029.json';c=read(cp)
data,parts,known,synthetic=prepare(c)
run=ROOT/c['output_root'];completed(run/'status.json')
assert read(run/'registration.json')==c
prov=read(run/'provenance.json');m=read(run/'metrics.json')
assert prov['config']==c and prov['config_sha256']==digest(c) and m['holdout_scored'] is False and prov['holdout_read_for_inference'] is False
assert read(run/'roles.json')=={k:[r['sample_id'] for r in v] for k,v in parts.items()}
assert prov['role_ids_sha256']=={k:digest([r['sample_id'] for r in v]) for k,v in parts.items()}
derived=read(run/'derived_training.jsonl',True);assert derived==synthetic and len(derived)==1024
assert len({r['parent']['group_id'] for r in derived})==1024
assert all(r['parent']['split']=='train' and not r['parent']['is_ai'] for r in derived)
assert {i:sum(r['architecture_index']==i for r in derived) for i in range(8)}=={i:128 for i in range(8)}
traces=read(run/'architecture_traces.json')
cpu=read('outputs/image_experiments/assets/ofa-simulator-20261029/cpu-verification-tplus1.json')
assert len({v['sha256'] for v in traces.values()})==8
assert [traces[str(i)]['sha256'] for i in range(8)]==cpu['trace_hashes']
assert all(v['sha256']==digest(v['trace']) and v['seed']==c['architecture_seeds'][int(i)] for i,v in traces.items())
for f,key in [('known_features.pt','known_cache_sha256'),('unknown_train_features.pt','unknown_cache_sha256'),('head.pth','head_sha256'),('architecture_traces.json','architecture_traces_sha256'),('derived_training.jsonl','synthetic_manifest_sha256')]:
    h=sha256(run/f);assert h==prov[key];sources[str((run/f).relative_to(ROOT))]=h
history=read(run/'history.jsonl',True);assert [h['epoch'] for h in history]==list(range(1,31))
best=max(history,key=lambda h:h['selection_macro_f1'])
assert m['training_epochs']==30 and m['best_epoch']==best['epoch']==18 and m['selection_macro_f1']==best['selection_macro_f1']
head=safe_load(run/'head.pth')
assert head['classes']==known+[UNKNOWN] and head['input_dimensions']==128 and head['best_epoch']==m['best_epoch']
linear=torch.nn.Linear(128,21);linear.load_state_dict(head['state_dict'],strict=True)
checks=read(run/'actual_selection_verification.json')
assert checks['images_opened']==2 and checks['full_offline_backbone_reload'] is True
assert checks['selection_sample_ids']==[r['sample_id'] for r in parts['selection'][:2]]
assert all(np.isfinite(v) and v<=1e-5 for k,v in checks.items() if k.endswith('max_error'))
records=read(run/'predictions.jsonl',True)
route=extract(records,parts,known,'tplus1')
values={}
for role in ('calibration','evaluation'):
    rmap=indexed([r for r in records if r['role']==role])
    values[role]=validate_probabilities([rmap[r['sample_id']]['scores'] for r in parts[role]],len(parts[role]),21)
threshold=calibrate(parts['calibration'],known,values['calibration'])
assert threshold==m['rejection']['acceptance_threshold']
for role in values:
    equal(report(parts[role],known,values[role],threshold),m['rejection'][role])
for r in records:
    assert r['split']=='dev'
    assert r['simulated_unknown_score']==r['scores'][-1] and r['acceptance_confidence']==1-r['scores'][-1]
    assert r['accepted']==(r['acceptance_confidence']>=threshold)
    assert r['known_prediction']==known[int(np.argmax(r['scores'][:-1]))]
acceptance['ofa_tplus1']=dict(status='completed',epochs=30,selected_epoch=m['best_epoch'],synthetic_train_n=1024,
    role_counts={k:len(v) for k,v in parts.items()},head_sha256=prov['head_sha256'],
    actual_selection_verification=checks,rejection=m['rejection'],holdout_scored=False)
for f in ('metrics.json','history.jsonl','provenance.json','registration.json','status.json','launch.json','actual_selection_verification.json','architecture_traces.json'):
    copy(run/f,Path('ofa-tplus1')/f)
copy(cp,Path('config')/cp.name)

manifest=indexed(read(data/'manifest.jsonl',True))
comparisons={}
for name in ['convnext-comparison','convnext-parent-dev-reference','dna-comparison','dna-parent-dev-reference','omnidfa-comparison','omnidfa-parent-dev-reference','ofa-tplus1-comparison']:
    p=BASE/name;completion=completed(p/'completion.json')
    assert sha256(p/'summary.json')==completion['summary_sha256'] and sha256(p/'provenance.json')==completion['provenance_sha256']
    report_prov=read(p/'provenance.json')
    for f,h in report_prov['inputs'].items():assert sha256(ROOT/f)==h;sources[f]=h
    summary=read(p/'summary.json')
    if name!='ofa-tplus1-comparison':
        config_paths=[f for f in report_prov['inputs'] if f.startswith('configs/') and f.endswith('.json')]
        assert len(config_paths)==1;cc=read(config_paths[0]);copy(ROOT/config_paths[0],Path('config')/Path(config_paths[0]).name)
        for pair in cc['pairs']:
            arrays=[]
            for spec in (pair['before'],pair['after']):
                folder=ROOT/spec['run'];completed(folder/'status.json')
                rr=read(folder/spec.get('predictions','predictions.jsonl'),True)
                saved_roles=read(spec['roles']) if spec.get('roles') else None
                if saved_roles is not None:validate_saved_roles(manifest,spec['recipe'],saved_roles)
                if all('role' not in r for r in rr):
                    rr=normalize_legacy(rr,manifest,spec['recipe'],saved_roles,spec.get('legacy_scope','evaluation'))
                arrays.append(selected(rr,manifest,spec['recipe']))
            a,b=arrays;ids=sorted(b);assert set(b)<=set(a)
            cls=b[ids[0]]['classes'];y=[cls.index(('ai' if manifest[s]['is_ai'] else 'real') if pair['task']=='detection' else manifest[s]['source_label']) for s in ids]
            for label,arr in [('before',a),('after',b)]:
                equal(stats(np.array(y),np.array([arr[s]['scores'] for s in ids]),'D0' if pair['task']=='detection' else 'A0',cls),summary[pair['id']][label])
            assert summary[pair['id']]['same_ids_labels_groups_verified'] is True
            boot=summary[pair['id']]['group_bootstrap'];assert boot['repeats']==1000 and boot['group_n']==len({manifest[s]['group_id'] for s in ids})
    else:
        for key,directory,kind in [('resnet_msp','outputs/image_experiments/followups-v7-200-q179-e30/A1/evaluate','resnet'),('openclip_msp','outputs/image_experiments/openclip-q179-20261029/A-OpenCLIP-open20','openclip'),('ofa_tplus1_known_mass',str(run.relative_to(ROOT)),'tplus1')]:
            rr=extract(read(Path(directory)/'predictions.jsonl',True),parts,known,kind)
            computed=rejection_report(parts['calibration'],parts['evaluation'],known,rr['calibration']['confidence'],rr['evaluation']['confidence'],rr['evaluation']['pred'])
            equal(computed,{k:summary['methods'][key][k] for k in computed})
            assert oscr(parts['evaluation'],known,rr['evaluation']['confidence'],rr['evaluation']['pred'])==summary['methods'][key]['oscr']
        assert summary['paired_uncertainty']['repeats']==summary['paired_uncertainty']['valid_repeats']==1000
    comparisons[name]=summary
    for f in ('summary.json','provenance.json','completion.json'):copy(p/f,Path(name)/f)
for f,h in registration['inputs'].items():assert sha256(ROOT/f)==h;sources[f]=h
(OUT/'acceptance.json').write_text(json.dumps(dict(verification=acceptance,all_ten_stages_completed=True,all_seven_reports_point_metrics_recomputed=True,holdout_scored=False,images_read=False),indent=2)+'\n')
(OUT/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
(OUT/'snapshot.json').write_text(json.dumps(dict(sources=sources,copied=copied),indent=2)+'\n')
print(json.dumps(dict(status='verified',copied=len(copied),sources=len(sources),ofa_selected_epoch=m['best_epoch'],all_seven_reports_verified=True)))
