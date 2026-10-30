"""Recheck the completed ConvNeXt snapshot using saved nonholdout scores only."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
from research.image_attribution.protocol import sha256
from research.image_attribution.openclip_route import digest,job_parts
from research.image_attribution.followups import binary_report,choose_threshold,validate_probabilities
from scripts.run_image_followup import class_report
from scripts.run_image_convnext import prepare
from scripts.compare_image_scale_epochs import indexed
from sklearn.metrics import confusion_matrix

OUT=ROOT/'docs/image_research_results/2026-10-30-convnext-completed'
RUN=ROOT/'outputs/image_experiments/convnext-q179-20261029'
sources={}
def read(path,lines=False):
    path=Path(path)
    sources[str(path.relative_to(ROOT))]=sha256(path)
    return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())
def equal(a,b):
    if isinstance(a,dict):
        for k,v in a.items():equal(v,b[k])
    elif isinstance(a,list):assert a==b
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert np.isclose(a,b,rtol=0,atol=1e-10),(a,b)
    else:assert a==b

configpath=ROOT/'configs/image_experiments/iab_convnext_q179_20261029.json'
config=read(configpath)
data,rows,cache_rows,identity,jobs=prepare(config)
status=read(RUN/'status.json');assert status['status']=='completed' and status['exit_code']==0
registration=read(RUN/'registration.json');assert registration['config']==config and registration['config_sha256']==digest(config)
cache=read(RUN/'cache.json');assert cache['identity']==identity and cache['n']==44527 and cache['holdout_read'] is False
assert cache['sha256']==sha256(RUN/'features.pt')
sources[str((RUN/'features.pt').relative_to(ROOT))]=cache['sha256']
summary={};matrices={};copy_paths=[configpath,RUN/'status.json',RUN/'launch.json',RUN/'registration.json']
for name,recipe in config['experiments'].items():
    folder=RUN/name
    parts,classes=job_parts(rows,recipe,42)
    roles=read(folder/'roles.json');assert roles=={k:[r['sample_id'] for r in v] for k,v in parts.items()}
    for role,rs in parts.items():
        assert all(r['split']==('train' if role=='train' else 'dev') for r in rs)
    group_sets={k:{r['group_id'] for r in v} for k,v in parts.items()}
    for a in group_sets:
        for b in group_sets:
            if a!=b:assert not group_sets[a]&group_sets[b]
    if name=='D-ConvNeXt-FFHQ':
        for role in ('train','selection','calibration'):assert all(r['semantic_class']!='FFHQ' for r in parts[role])
    status=read(folder/'status.json');assert status['status']=='completed' and status['exit_code']==0
    linear_status=read(folder/'linear/status.json');assert linear_status['status']=='completed' and linear_status['exit_code']==0
    metrics=read(folder/'metrics.json');provenance=read(folder/'provenance.json')
    assert metrics['holdout_scored'] is False and metrics['failed_samples']==[] and metrics['training_epochs']==30
    assert provenance['manifest_sha256']==config['manifest_sha256'] and provenance['code']==config['implementation_sha256']
    assert provenance['feature_cache_sha256']==cache['sha256'] and provenance['classes']==classes
    assert provenance['role_ids_sha256']=={k:digest(v) for k,v in roles.items()}
    h=read(folder/'linear/history.jsonl',True);assert [r['epoch'] for r in h]==list(range(1,31))
    field='auroc' if recipe['task']=='detection' else 'macro_f1'
    best=max(h,key=lambda r:r['selection'][field]);assert metrics['best_epoch']==best['epoch']
    assert np.isclose(metrics['selection_'+field],best['selection'][field],atol=1e-12)
    hp=folder/'linear/head.pth';assert sha256(hp)==metrics['head_asset_sha256']==provenance['head_asset_sha256']
    sources[str(hp.relative_to(ROOT))]=sha256(hp)
    head=torch.load(hp,map_location='cpu',weights_only=True)
    assert head['classes']==classes and head['input_dimensions']==768 and head['best_epoch']==metrics['best_epoch']
    checks=metrics['actual_image_verification']
    assert checks['images_opened']==2 and checks['full_offline_backbone_reload'] is True
    assert checks['selection_sample_ids']==roles['selection'][:2]
    assert all(np.isfinite(v) and v<=1e-5 for k,v in checks.items() if k.endswith('max_error'))
    assert metrics['reload_max_error']<=1e-5 and metrics['single_batch_max_error']<=1e-5
    predictions=read(folder/'predictions.jsonl',True)
    assert set(r['role'] for r in predictions)=={'calibration','evaluation'}
    values={}
    for role in ('calibration','evaluation'):
        selected=indexed([r for r in predictions if r['role']==role])
        assert set(selected)==set(roles[role])
        scores=[]
        for row in parts[role]:
            record=selected[row['sample_id']]
            assert record['classes']==classes and record['condition']=='clean'
            assert all(record[k]==row[k] for k in ('group_id','source_label','is_ai','semantic_class'))
            scores.append(record['scores'])
        values[role]=validate_probabilities(scores,len(parts[role]),len(classes))
    p=values['evaluation']
    if recipe['task']=='detection':
        y=np.array([int(r['is_ai']) for r in parts['evaluation']]);recomputed=binary_report(y,p[:,1],.5)
        equal(recomputed,metrics['fixed_0_5'])
        # fit_head persisted the exact batch32 calibration/eval scores used for these metrics.
        original={};export_drift={}
        for role,path in [('calibration',folder/'linear/calibration/predictions.jsonl'),('evaluation',folder/'linear/predictions.jsonl')]:
            saved_predictions=indexed(read(path,True));assert set(saved_predictions)==set(roles[role])
            for row in parts[role]:
                record=saved_predictions[row['sample_id']]
                assert record['classes']==classes and record['split']=='dev' and record['status']=='ok'
                assert record['true_label']==('ai' if row['is_ai'] else 'real')
            original[role]=validate_probabilities([saved_predictions[r['sample_id']]['scores'] for r in parts[role]],len(parts[role]),2)
            export_drift[role]=float(np.abs(original[role]-values[role]).max())
            assert export_drift[role]<=1e-5
            assert np.array_equal(original[role][:,1]>=.5,values[role][:,1]>=.5)
        for target in ('.01','.05'):
            key=str(float(target));cal=original['calibration'][:,1]
            real=np.array([not r['is_ai'] for r in parts['calibration']])
            threshold=choose_threshold(cal[real],float(target));saved=metrics['calibrated'][key]
            assert saved['calibration_negative_n']==int(real.sum())
            assert np.isclose(saved['calibration_fpr'],(cal[real]>=threshold).mean())
            equal(binary_report(y,original['evaluation'][:,1],threshold),saved['evaluation'])
        pred=(p[:,1]>=.5).astype(int)
        summary[name]={k:metrics[k] for k in ('n','auroc','fpr_0_5','tpr_0_5','best_epoch','head_asset_sha256')}
    else:
        report=class_report(parts['evaluation'],classes,p)
        for k in ('n','accuracy','macro_f1','top1','top3'):equal(report[k],metrics[k])
        y=np.array([classes.index(r['source_label']) for r in parts['evaluation']]);pred=p.argmax(1)
        summary[name]={k:metrics[k] for k in ('n','top1','top3','macro_f1','best_epoch','head_asset_sha256')}
    matrices[name]=dict(classes=classes,counts=confusion_matrix(y,pred,labels=list(range(len(classes)))).tolist())
    summary[name].update(export_score_max_error=export_drift if recipe['task']=='detection' else None,status='completed',epochs_verified=30,role_counts={k:len(v) for k,v in parts.items()},all_prediction_ids_metadata_verified=True,actual_image_verification=checks)
    copy_paths += [folder/'status.json',folder/'metrics.json',folder/'provenance.json',folder/'linear/history.jsonl',folder/'linear/status.json']
# Hash metadata only for large/model inputs, without copying them into Git.
for key,h in config['implementation_sha256'].items():assert sha256(ROOT/key)==h;sources[key]=h
for key in ('asset','model_config'):
    spec=config[key];assert sha256(ROOT/spec['path'])==spec['sha256'];sources[spec['path']]=spec['sha256']
for key in ('manifest','selection'):
    path=data/(key+'.jsonl' if key=='manifest' else key+'.json');assert sha256(path)==config[key+'_sha256'];sources[str(path.relative_to(ROOT))]=sha256(path)
assert not (OUT/'snapshot.json').exists()
items=[]
for path in copy_paths:
    relative=path.relative_to(RUN) if path.is_relative_to(RUN) else Path('config')/path.name
    dest=OUT/'artifacts'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,dest);h=sha256(path);assert sha256(dest)==h
    sources[str(path.relative_to(ROOT))]=h;items.append(dict(source=str(path.relative_to(ROOT)),path=str(dest.relative_to(OUT)),sha256=h,bytes=path.stat().st_size))
(OUT/'summary.json').write_text(json.dumps(dict(experiments=summary,confusion_matrices=matrices,holdout_scored=False,paired_intervals_pending=True),indent=2)+'\n')
(OUT/'snapshot.json').write_text(json.dumps(dict(sources=sources,copied=items,images_read=False,model_data_copied=False),indent=2)+'\n')
(OUT/'README.md').write_text('ConvNeXt三项真实完成30轮并通过保存分数、完整角色、源/模型/缓存SHA和selection重载验收。clean2310 AUROC99.6180%、FPR3.5751%、TPR97.6709%；FFHQ225仍56/93真实图误报60.2151%；来源1331 Top1 61.9083%、Top3 84.8234%。单seed开发结果，不能称已解决FFHQ。配对group区间由串行队列后续独立比较产生，此快照未声称已有。阈值逐项使用linear/calibration保存的原batch32分数精确复算；根目录另导出的batch128分数有float32数值微差，逐ID校验最大误差<=1e-5且固定.5决策一致，漂移值另存summary。只读保存scores，未推理holdout。模型、缓存、每图预测均未复制。\n')
print(json.dumps(summary,indent=2))
