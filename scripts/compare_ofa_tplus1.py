#!/usr/bin/env python3
"""Read-only same-role OFA20+1 rejection comparison; no inference or fitting on eval."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_image_rejection_routes import (
    MANIFEST,MANIFEST_SHA,RECIPE,FIELDS,report,bootstrap,
)
from scripts.compare_image_scale_epochs import indexed,sha256
from research.image_attribution.followups import partition_rows,validate_probabilities
from research.image_attribution.openclip_route import digest


def extract(records,parts,known,kind):
    expected_classes=known+['simulated_unknown'] if kind=='tplus1' else known
    if any(r.get('role') not in ('calibration','evaluation') or r.get('condition','clean')!='clean' for r in records):
        raise ValueError('only declared clean calibration/evaluation scores allowed')
    route={}
    for role in ('calibration','evaluation'):
        selected=indexed([r for r in records if r['role']==role])
        if set(selected)!={r['sample_id'] for r in parts[role]}:
            raise ValueError('missing or extra role IDs; intersection fallback forbidden')
        values=[];proto=[]
        for row in parts[role]:
            r=selected[row['sample_id']]
            if row['split']!='dev' or r['classes']!=expected_classes:
                raise ValueError('nondev ID or changed vocabulary')
            if any(r[k]!=row[k] for k in ('group_id','source_label','is_ai','semantic_class')):
                raise ValueError('metadata differs from frozen role')
            values.append(r['scores'])
            if kind=='openclip':proto.append(r['prototype_cosine'])
        scores=validate_probabilities(values,len(parts[role]),len(expected_classes))
        confidence=1-scores[:,-1] if kind=='tplus1' else scores.max(1)
        route[role]=dict(confidence=confidence,pred=scores[:,:len(known)].argmax(1),
            full_pred=scores.argmax(1),prototype=np.asarray(proto,dtype=float))
    return route


def oscr(rows,classes,confidence,prediction):
    """Area of correct-known acceptance versus unknown acceptance; tied scores move together."""
    confidence=np.asarray(confidence,float);prediction=np.asarray(prediction,int)
    known=np.array([r['source_label'] in classes for r in rows])
    if (confidence.shape!=(len(rows),) or prediction.shape!=(len(rows),)
            or not np.isfinite(confidence).all() or not known.any() or known.all()
            or (prediction<0).any() or (prediction>=len(classes)).any()):
        raise ValueError('invalid OSCR inputs')
    correct=np.array([classes[p]==r['source_label'] for p,r in zip(prediction,rows)])
    x,y=[0.],[0.]
    for threshold in np.unique(confidence)[::-1]:
        accepted=confidence>=threshold
        x.append(float(accepted[~known].mean()))
        y.append(float((accepted & correct)[known].mean()))
    return float(np.trapezoid(y,x))


def validate_saved_tplus(recomputed,metrics):
    saved=metrics['rejection']
    if (metrics.get('holdout_scored') is not False
            or metrics.get('training_epochs')!=30
            or saved.get('threshold_fitted_role')!='calibration'
            or not np.isclose(recomputed['threshold'],saved['acceptance_threshold'],rtol=0,atol=1e-10)
            or any(not np.isclose(recomputed[f],saved['evaluation'][f],rtol=0,atol=1e-10) for f in FIELDS)):
        raise ValueError('OFA T+1 aggregate or protocol mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve existing comparison')
    inputs={}
    def read(relative,lines=False):
        p=ROOT/relative;inputs[str(p.relative_to(ROOT))]=sha256(p)
        return [json.loads(s) for s in p.read_text().splitlines()] if lines else json.loads(p.read_text())
    rows=read(MANIFEST,True)
    if inputs[MANIFEST]!=MANIFEST_SHA:raise ValueError('frozen manifest changed')
    parts=partition_rows(rows,RECIPE);known=sorted({r['source_label'] for r in parts['train']})
    if len(known)!=20 or len(parts['evaluation'])!=1275 or len(parts['calibration'])!=1482:
        raise ValueError('registered open20 scope changed')
    specs={
        'resnet':'outputs/image_experiments/followups-v7-200-q179-e30/A1/evaluate',
        'openclip':'outputs/image_experiments/openclip-q179-20261029/A-OpenCLIP-open20',
        'tplus1':'outputs/image_experiments/ofa-tplus1-q179-20261029',
    }
    routes={};metrics={}
    for name,directory in specs.items():
        folder=Path(directory);status=read(folder/'status.json')
        if status.get('status')!='completed' or status.get('exit_code')!=0:raise ValueError(f'{name} incomplete')
        roles=read(folder/'roles.json')
        if roles!={k:[r['sample_id'] for r in v] for k,v in parts.items()}:
            raise ValueError('original known roles differ from frozen recipe; synthetic rows must be separate')
        metrics[name]=read(folder/'metrics.json')
        routes[name]=extract(read(folder/'predictions.jsonl',True),parts,known,name)
    methods={
        'resnet_msp':tuple(routes['resnet'][r][k] for r,k in [('calibration','confidence'),('evaluation','confidence'),('evaluation','pred')]),
        'openclip_msp':tuple(routes['openclip'][r][k] for r,k in [('calibration','confidence'),('evaluation','confidence'),('evaluation','pred')]),
        'openclip_prototype':tuple(routes['openclip'][r][k] for r,k in [('calibration','prototype'),('evaluation','prototype'),('evaluation','pred')]),
        'ofa_tplus1_known_mass':tuple(routes['tplus1'][r][k] for r,k in [('calibration','confidence'),('evaluation','confidence'),('evaluation','pred')]),
    }
    reports={k:report(parts['calibration'],parts['evaluation'],known,*v) for k,v in methods.items()}
    # Require the already published references to agree with recomputed saved scores.
    legacy=metrics['resnet']['conditions']['clean']
    for field,key in [('threshold','acceptance_threshold'),('known_retention','known_acceptance'),
        ('known_correct_and_retained','known_correct_and_accepted_rate'),
        ('unknown_false_acceptance','unknown_false_acceptance'),('unknown_auroc','unknown_auroc')]:
        if not np.isclose(reports['resnet_msp'][field],legacy[key],rtol=0,atol=1e-10):raise ValueError('ResNet aggregate mismatch')
    for name,method in [('openclip_msp','msp'),('openclip_prototype','prototype_cosine')]:
        saved=metrics['openclip']['rejection'][method]['evaluation']
        if any(not np.isclose(reports[name][f],saved[f],rtol=0,atol=1e-10) for f in FIELDS):
            raise ValueError('OpenCLIP aggregate mismatch')
    validate_saved_tplus(reports['ofa_tplus1_known_mass'],metrics['tplus1'])
    tplus=Path(specs['tplus1']);provenance=read(tplus/'provenance.json')
    registration=read(tplus/'registration.json')
    if (provenance['config']!=registration or provenance['config_sha256']!=digest(registration)
            or registration['manifest_sha256']!=MANIFEST_SHA
            or provenance['holdout_read_for_inference'] is not False
            or provenance['role_ids_sha256']!={k:digest([r['sample_id'] for r in v]) for k,v in parts.items()}):
        raise ValueError('OFA input provenance mismatch')
    for filename,key in [('head.pth','head_sha256'),('architecture_traces.json','architecture_traces_sha256'),
                         ('derived_training.jsonl','synthetic_manifest_sha256')]:
        path=tplus/filename;actual_hash=sha256(ROOT/path);inputs[str(path)]=actual_hash
        if actual_hash!=provenance[key]:raise ValueError('OFA model or derived-training identity changed')
    history=read(tplus/'history.jsonl',True)
    if [r['epoch'] for r in history]!=list(range(1,31)):raise ValueError('OFA training history incomplete')
    verification=read(tplus/'actual_selection_verification.json')
    if (verification['images_opened']!=2 or verification['full_offline_backbone_reload'] is not True
            or verification['selection_sample_ids']!=[r['sample_id'] for r in parts['selection'][:2]]
            or any(not np.isfinite(v) or v>1e-5 for k,v in verification.items() if k.endswith('max_error'))):
        raise ValueError('OFA actual-image reload invalid')
    for name,(_,ep,pred) in methods.items():reports[name]['oscr']=oscr(parts['evaluation'],known,ep,pred)
    intervals=bootstrap(parts['calibration'],parts['evaluation'],known,methods,repeats=1000)
    actual=routes['tplus1']['evaluation'];known_mask=np.array([r['source_label'] in known for r in parts['evaluation']])
    correct=np.array([int(pred)<20 and known[int(pred)]==row['source_label'] for pred,row in zip(actual['full_pred'],parts['evaluation'])])
    reports['ofa_tplus1_known_mass']['known_argmax_including_unknown_accuracy']=float(correct[known_mask].mean())
    inputs[str(Path(__file__).relative_to(ROOT))]=sha256(__file__)
    for helper in ['scripts/compare_image_rejection_routes.py','research/image_attribution/followups.py','scripts/compare_image_scale_epochs.py','research/image_attribution/openclip_route.py']:
        inputs[helper]=sha256(ROOT/helper)
    for path,expected_hash in inputs.items():
        if sha256(ROOT/path)!=expected_hash:raise ValueError('comparison input changed')
    args.output.mkdir(parents=True,exist_ok=False)
    summary=dict(methods=reports,paired_uncertainty=intervals,holdout_scored=False,
        note='OFA20+1 is frozen official simulator/fingerprint plus a project head, not full OFA pretraining. Known class prediction for calibrated rejection is argmax of the first20 logits; the21-way argmax accuracy is separately reported. Thresholds refit only on64 kling calibration samples;63 grok3 evaluation samples are never used to select thresholds. Single-seed previously inspected developer roles; no universal unknown guarantee.')
    for filename,value in [('summary.json',summary),('provenance.json',dict(inputs=inputs,recipe=RECIPE,manifest_sha256=MANIFEST_SHA,images_read=False))]:
        (args.output/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (args.output/'completion.json').write_text(json.dumps(dict(status='completed',exit_code=0,
        summary_sha256=sha256(args.output/'summary.json'),provenance_sha256=sha256(args.output/'provenance.json'),holdout_scored=False),indent=2)+'\n')

if __name__=='__main__':main()
