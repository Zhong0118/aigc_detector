"""Score-only FFHQ threshold transfer audit. No model or image inference."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from research.image_attribution.protocol import sha256
from research.image_attribution.openclip_route import digest
from research.image_attribution.followups import partition_rows,validate_probabilities,binary_report,choose_threshold
from scripts.compare_image_scale_epochs import indexed
from scripts.compare_image_next_routes import selected
from scripts.compare_iab_v6_100_vs_v7_200_q179 import detection_bootstrap
OUT=Path(__file__).resolve().parent
recipe=dict(task='detection',mode='cross_semantic',held_semantics=['FFHQ'],seed=42)
MP=ROOT/'outputs/image_experiments/iab-balanced-v7-200-q179/manifest.jsonl'
assert sha256(MP)=='9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3'
inputs={}
def read(path,lines=False):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    inputs[str(p.relative_to(ROOT))]=sha256(p)
    return [json.loads(x) for x in p.read_text().splitlines()] if lines else json.loads(p.read_text())
manifest=indexed(read(MP,True));parts=partition_rows(list(manifest.values()),recipe)
assert len(parts['evaluation'])==225 and len(parts['calibration'])==2298
assert all(r['semantic_class']!='FFHQ' for role in ('train','selection','calibration') for r in parts[role])
assert all(r['semantic_class']=='FFHQ' and r['split']=='dev' for r in parts['evaluation'])
negative=np.array([not r['is_ai'] for r in parts['calibration']])
truth=np.array([bool(r['is_ai']) for r in parts['evaluation']])
assert negative.sum()==914 and (~truth).sum()==93 and truth.sum()==132
specs={
 'ResNet-heldFFHQ':('outputs/image_experiments/followups-v7-200-q179-e30/D2-semantic/evaluate',False),
 'OpenCLIP-heldFFHQ':('outputs/image_experiments/openclip-q179-20261029/D-OpenCLIP-FFHQ',True),
 'ConvNeXt-heldFFHQ':('outputs/image_experiments/convnext-q179-20261029/D-ConvNeXt-FFHQ',True)}
scores={};results={}
for name,(directory,canonical) in specs.items():
    folder=ROOT/directory;s=read(folder/'status.json');assert s['status']=='completed' and s['exit_code']==0
    exported=read(folder/'predictions.jsonl',True)
    selected(exported,manifest,recipe) # Requires complete calibration/evaluation IDs and metadata.
    values={}
    for role in ('calibration','evaluation'):
        if canonical:
            path=folder/'linear'/('calibration/predictions.jsonl' if role=='calibration' else 'predictions.jsonl')
            records=read(path,True)
        else:records=[r for r in exported if r['role']==role and r['condition']=='clean']
        mapping=indexed(records);ids=[r['sample_id'] for r in parts[role]]
        assert set(mapping)==set(ids)
        for r in parts[role]:
            pred=mapping[r['sample_id']];assert pred['classes']==['real','ai']
            if canonical:assert pred['status']=='ok' and pred['split']=='dev' and pred['true_label']==('ai' if r['is_ai'] else 'real')
        values[role]=validate_probabilities([mapping[s]['scores'] for s in ids],len(ids),2)[:,1]
    m=read(folder/'metrics.json');saved=m['fixed_0_5'] if canonical else m['conditions']['clean']['fixed_0_5']
    fixed=binary_report(truth,values['evaluation'],.5)
    for key in ('n','auroc','accuracy','fpr','tpr'):assert np.isclose(fixed[key],saved[key],rtol=0,atol=1e-10)
    result=dict(fixed_0_5=dict(fixed,false_positives=int(((values['evaluation']>=.5)&~truth).sum()),true_positives=int(((values['evaluation']>=.5)&truth).sum())),calibrated={})
    for target in (.01,.05):
        threshold=choose_threshold(values['calibration'][negative],target)
        cal=binary_report([r['is_ai'] for r in parts['calibration']],values['calibration'],threshold)
        ev=binary_report(truth,values['evaluation'],threshold)
        assert int(((values['calibration'][negative]>=threshold)).sum())<=int(np.floor(target*negative.sum()))
        result['calibrated'][str(target)]=dict(threshold=threshold,threshold_fitted_role='nonFFHQ_calibration_real_only',
            calibration=cal,evaluation=dict(ev,false_positives=int(((values['evaluation']>=threshold)&~truth).sum()),true_positives=int(((values['evaluation']>=threshold)&truth).sum())))
        if canonical:
            saved=m['calibrated'][str(target)]
            assert threshold==saved['evaluation']['threshold']
            for key in ('fpr','tpr','auroc'):assert ev[key]==saved['evaluation'][key]
    scores[name]=values;results[name]=result
def group_index(rows):
    groups=sorted({r['group_id'] for r in rows});lookup={s:i for i,s in enumerate(groups)}
    return groups,np.array([lookup[r['group_id']] for r in rows])
calneg=[r for r in parts['calibration'] if not r['is_ai']]
cg,ci=group_index(calneg);eg,ei=group_index(parts['evaluation'])
assert not set(cg)&set(eg)
# Refit threshold only on resampled calibration; all models share original-group draws.
rng=np.random.default_rng(42);samples={str(t):{name:[] for name in scores} for t in (.01,.05)}
valid=0
for _ in range(1000):
    cw=np.bincount(rng.integers(0,len(cg),len(cg)),minlength=len(cg))[ci]
    ew=np.bincount(rng.integers(0,len(eg),len(eg)),minlength=len(eg))[ei]
    if ew[truth].sum()==0 or ew[~truth].sum()==0:continue
    valid+=1
    for target in (.01,.05):
        for name,values in scores.items():
            threshold=choose_threshold(np.repeat(values['calibration'][negative],cw),target)
            accepted=values['evaluation']>=threshold
            samples[str(target)][name].append([threshold,ew[~truth&accepted].sum()/ew[~truth].sum(),ew[truth&accepted].sum()/ew[truth].sum()])
assert valid>=950
uncertainty={}
for target,methods in samples.items():
    base=np.array(methods['ResNet-heldFFHQ']);summary={}
    for name,values in methods.items():
        arr=np.array(values);bounds=np.quantile(arr,[.025,.975],axis=0).T
        summary[name]=dict(threshold_95_ci=bounds[0].tolist(),evaluation_fpr_95_ci=bounds[1].tolist(),evaluation_tpr_95_ci=bounds[2].tolist())
        if name!='ResNet-heldFFHQ':
            delta=arr[:,1:]-base[:,1:];ci95=np.quantile(delta,[.025,.975],axis=0).T
            summary[name]['minus_resnet_delta_95_ci']=dict(fpr=ci95[0].tolist(),tpr=ci95[1].tolist())
    uncertainty[target]=summary
fixed_pairs={}
for name in ('OpenCLIP-heldFFHQ','ConvNeXt-heldFFHQ'):
    p=scores['ResNet-heldFFHQ']['evaluation'];q=scores[name]['evaluation']
    fixed_pairs[name+'_minus_ResNet-heldFFHQ']=detection_bootstrap(truth.astype(int),np.c_[1-p,p],np.c_[1-q,q],[r['group_id'] for r in parts['evaluation']],['real','ai'],repeats=1000)
result=dict(recipe=recipe,evaluation_n=225,calibration_n=2298,calibration_real_n=914,evaluation_real_n=93,evaluation_ai_n=132,
    methods=results,fixed_0_5_paired_group_intervals=fixed_pairs,
    threshold_refit_group_bootstrap=dict(repeats=1000,valid_repeats=valid,seed=42,calibration_real_groups=len(cg),evaluation_groups=len(eg),results=uncertainty,
        method='Paired original-group percentile bootstrap; shared independent calibration-real/evaluation group draws; threshold refitted only in calibration; no retraining or checkpoint selection'),
    images_read=False,holdout_scored=False,limitations='Previously viewed single-seed developer evaluation, small93real FFHQ cohort; threshold stability and generalization remain limited; no causal backbone-only claim.')
for path,h in inputs.items():assert sha256(ROOT/path)==h
inputs[str(Path(__file__).relative_to(ROOT))]=sha256(__file__)
(OUT/'ffhq-threshold-transfer.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'ffhq-analysis-inputs.json').write_text(json.dumps(dict(inputs=inputs,role_ids_sha256={r:digest([x['sample_id'] for x in parts[r]]) for r in ('calibration','evaluation')},threshold_targets=[.01,.05],bootstrap_repeats=1000,images_read=False,holdout_scored=False),indent=2)+'\n')
print(json.dumps({k:{'fixed':v['fixed_0_5'],'targets':{t:r['evaluation'] for t,r in v['calibrated'].items()}} for k,v in results.items()},indent=2))
