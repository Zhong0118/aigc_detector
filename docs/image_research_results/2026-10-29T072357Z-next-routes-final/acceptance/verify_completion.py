"""Read-only saved-artifact acceptance; run from the research repository root."""
import argparse, json
from pathlib import Path
import numpy as np
from scripts.run_image_next_routes import sha256, alive
from research.image_attribution.openclip_route import digest as canonical_digest
from scripts.compare_image_next_routes import indexed, selected, normalize_legacy, validate_saved_roles
from research.image_attribution.followups import partition_rows, validate_probabilities, choose_threshold
from scripts.run_image_followup import class_report
from research.image_attribution.followups import binary_report

ROOT=Path.cwd()
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
assert not args.output.exists()
inputs={}
def read(name,lines=False):
    p=Path(name);inputs[str(p)]=sha256(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if lines else json.loads(p.read_text())
def close(actual,expected):assert np.isclose(actual,expected,rtol=0,atol=1e-9),(actual,expected)
manifest_path='outputs/image_experiments/iab-balanced-v7-200-q179/manifest.jsonl'
rows=read(manifest_path,True);manifest=indexed(rows)
assert inputs[manifest_path]=='9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3'
queue=read('outputs/image-next-routes-20261029/queue-namespace-v1/status.json')
assert queue['status']=='awaiting_git_archive' and queue['exit_code']==0 and not alive(queue['launch'])
for stage in queue['completed_stages']:
    assert stage['exit_code']==0 and not alive(stage['child'])
    assert sha256(stage['receipt']['path'])==stage['receipt']['sha256']
reg=read('outputs/image-next-routes-20261029/queue-namespace-v1/registration.json')
for path,digest in reg['sources'].items():assert sha256(path)==digest
oc=read('configs/image_experiments/iab_openclip_q179_20261029.json')
ob=Path(oc['output_root']);registered=read(ob/'registration.json')
assert registered['config']==oc and registered['config_sha256']==canonical_digest(oc)
asset=oc['asset'];assert sha256(asset['path'])==asset['sha256']
cache=read(ob/'cache.json')
assert cache['holdout_read'] is False and cache['n']==44527 and cache['dimensions']==768
assert set(cache['identity']['ids'])=={r['sample_id'] for r in rows if r['split']!='holdout'}
assert sha256(ob/'features.pt')==cache['sha256']
checks={}
for name,recipe in [('OmniAID',dict(task='detection',mode='calibration',seed=42))]+[(n,dict(v,seed=42)) for n,v in oc['experiments'].items()]:
    folder=Path('outputs/image_experiments/omniaid-v2-q179-20261029') if name=='OmniAID' else ob/name
    status=read(folder/'status.json');metric=read(folder/'metrics.json');provenance=read(folder/'provenance.json')
    assert status['status']=='completed' and status['exit_code']==0 and metric['holdout_scored'] is False
    parts=partition_rows(rows,recipe)
    if name=='OmniAID':
        assert provenance['role_ids']=={k:[r['sample_id'] for r in v] for k,v in parts.items() if k in ['calibration','evaluation']}
        config=read(folder/'config.json');assert sha256(config['verification_receipt']['path'])==config['verification_receipt']['sha256']
        for a in config['assets'].values():assert sha256(a['path'])==a['sha256']
        records=[]
        for role in ['calibration','evaluation']:
            legacy=read(folder/role/'predictions.jsonl',True)
            assert set(indexed(legacy))=={r['sample_id'] for r in parts[role]}
            for rec in legacy:
                row=manifest[rec['sample_id']]
                assert rec['status']=='ok' and rec['split']=='dev' and rec['task']=='detection'
                assert rec['true_label']==('ai' if row['is_ai'] else 'real') and rec['classes']==['real','ai']
                records.append(dict(rec,role=role,condition='clean',**{k:row[k] for k in ['group_id','is_ai','source_label','semantic_class']}))
    else:
        roles=read(folder/'roles.json');validate_saved_roles(manifest,recipe,roles)
        records=read(folder/'predictions.jsonl',True)
        for path,digest in provenance['code'].items():assert sha256(path)==digest
        head=folder/('knn.pth' if recipe.get('estimator')=='knn' else 'linear/head.pth')
        assert sha256(head)==metric['head_asset_sha256']
        verification=metric['actual_image_verification'];assert verification['full_offline_backbone_reload'] is True and verification['images_opened']==2
        assert set(verification['selection_sample_ids'])<={r['sample_id'] for r in parts['selection']}
        assert all(v<=1e-5 for k,v in verification.items() if k.endswith('_max_error'))
        if recipe.get('estimator')=='knn':
            history=read(folder/'knn_selection.json');assert [h['k'] for h in history]==list(range(1,21))
            assert metric['best_k']==max(history,key=lambda h:h['selection_macro_f1'])['k'] and metric['epoch'] is None
        else:
            history=read(folder/'linear/history.jsonl',True);assert [h['epoch'] for h in history]==list(range(1,31))
            selection_metric='auroc' if recipe['task']=='detection' else 'macro_f1'
            best=max(history,key=lambda h:h['selection'][selection_metric])['epoch']
            saved=metric.get('closed_known_head',metric)
            assert saved['best_epoch']==best and metric['training_epochs']==30
        if recipe['mode']=='cross_semantic':
            assert all(r['semantic_class']!='FFHQ' for role in ['train','selection','calibration'] for r in parts[role])
        if recipe['mode']=='open_set':
            assert not any(r['source_label'] in ['kling','grok3'] for role in ['train','selection'] for r in parts[role])
    evaluation=selected(records,manifest,recipe)
    classes=next(iter(evaluation.values()))['classes']
    score_differences={}
    threshold_decisions={}
    for role in ['calibration','evaluation']:
        predictions=indexed([r for r in records if r['role']==role]);assert set(predictions)=={r['sample_id'] for r in parts[role]}
        probs=validate_probabilities([predictions[r['sample_id']]['scores'] for r in parts[role]],len(parts[role]),len(classes))
        canonical=probs
        if name!='OmniAID' and recipe.get('estimator')!='knn' and recipe['mode']!='open_set' and (recipe['task']=='detection' or role=='evaluation'):
            lp=folder/'linear'/('calibration/predictions.jsonl' if role=='calibration' else 'predictions.jsonl')
            linear=indexed(read(lp,True));assert set(linear)==set(predictions)
            for row in parts[role]:
                rec=linear[row['sample_id']];assert rec['status']=='ok' and rec['split']=='dev' and rec['classes']==classes
                assert rec['true_label']==(('ai' if row['is_ai'] else 'real') if recipe['task']=='detection' else row['source_label'])
            canonical=validate_probabilities([linear[r['sample_id']]['scores'] for r in parts[role]],len(parts[role]),len(classes))
            error=float(np.abs(canonical-probs).max());assert error<=1e-5
            score_differences[role]=error
        if recipe['task']=='detection':
            negative=np.array([not r['is_ai'] for r in parts[role]])
            threshold_decisions[role]={}
            for target in ['0.01','0.05']:
                threshold=metric['calibrated'][target]['evaluation']['threshold']
                old,new=canonical[:,1]>=threshold,probs[:,1]>=threshold
                threshold_decisions[role][target]=dict(canonical_fp=int((old & negative).sum()),root_fp=int((new & negative).sum()),negative_count=int(negative.sum()),all_decision_changes=int((old!=new).sum()))
        if role=='evaluation':
            if recipe['task']=='detection':
                actual=binary_report([int(r['is_ai']) for r in parts[role]],probs[:,1],.5)
                for key in ['accuracy','macro_f1','auroc','fpr','tpr']:close(actual[key],metric['fixed_0_5'][key])
            elif recipe['mode']!='open_set':
                actual=class_report(parts[role],classes,probs)
                for key in ['top1','top3','macro_f1']:close(actual[key],metric[key])
        elif recipe['task']=='detection':
            negatives=canonical[[not r['is_ai'] for r in parts[role]],1]
            for target in ['0.01','0.05']:
                threshold=choose_threshold(negatives,float(target));close(threshold,metric['calibrated'][target]['evaluation']['threshold'])
                close(float((negatives>=threshold).mean()),metric['calibrated'][target]['calibration_fpr'])
    checks[name]=dict(role_counts={k:len(v) for k,v in parts.items()},epoch=metric.get('epoch'),all_completed_and_verified=True,root_vs_canonical_linear_score_max_error=score_differences,threshold_decisions=threshold_decisions,calibrated_metric_source='linear/calibration and linear evaluation scores' if name!='OmniAID' and recipe['task']=='detection' else 'root saved scores')
for name in ['closed-comparison','rejection-comparison']:
    base=Path('outputs/image-next-routes-20261029')/name
    completion=read(base/'completion.json');assert completion['status']=='completed' and completion['exit_code']==0
    assert sha256(base/'summary.json')==completion['summary_sha256'] and sha256(base/'provenance.json')==completion['provenance_sha256']
    for path,digest in read(base/'provenance.json')['inputs'].items():assert sha256(path)==digest
    summary=read(base/'summary.json')
    boots=[v['group_bootstrap'] for v in summary.values()] if name=='closed-comparison' else [summary['paired_uncertainty']]
    assert all(b['repeats']==1000 for b in boots)
    if name=='rejection-comparison':assert summary['paired_uncertainty']['valid_repeats']==1000
inputs[str(Path(__file__).relative_to(ROOT))]=sha256(__file__)
for path,digest in inputs.items():assert sha256(path)==digest
args.output.write_text(json.dumps(dict(status='completed',exit_code=0,checks=checks,inputs=inputs,images_read=False,model_inference_performed=False,holdout_scored=False,queue_stopped=True),indent=2)+'\n')
print(json.dumps({'status':'completed','routes':list(checks),'input_hashes':len(inputs),'output':str(args.output)}))
