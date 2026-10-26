"""Manual-only D1-D3/A1-A3 preparation and execution; defaults to a read-only plan."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.image_attribution.protocol import sha256, validate_manifest, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path, name):
    config = read_json(path)
    if config.get('automatic_execution') is not False:
        raise ValueError('follow-up config must disable automatic execution')
    experiment = dict(config['experiments'][name], seed=config['seed'])
    if config['optimizer'] != 'Adam(lr=0.001)' or config['scheduler'] != 'StepLR(step_size=1,gamma=0.9)':
        raise ValueError('unsupported training recipe; existing runner uses fixed Adam/StepLR')
    return config, experiment


def validate_parts(parts, experiment):
    task = experiment['task']
    expected = ({'real','ai'} if task == 'detection' else
                {r['source_label'] for r in parts['train']})
    if len(expected) < 2:
        raise ValueError('training vocabulary has fewer than two classes')
    for role, rows in parts.items():
        if not rows:
            raise ValueError(f'empty {role} partition; do not silently relax protocol')
        observed = {('ai' if r['is_ai'] else 'real') if task == 'detection' else r['source_label'] for r in rows}
        if not expected <= observed:
            raise ValueError(f'{role} lacks required classes: {sorted(expected-observed)}')
    if experiment['mode'] == 'open_set':
        for role, key in [('calibration','unknown_calibration'), ('evaluation','unknown_evaluation')]:
            if not set(experiment[key]) <= {r['source_label'] for r in parts[role]}:
                raise ValueError(f'{role} lacks designated unknown sources')


def prepare(config, experiment):
    from research.image_attribution.followups import partition_rows
    dataset = ROOT / config['dataset']
    if sha256(dataset/'selection.json') != config['selection_sha256']:
        raise ValueError('selection identity differs from frozen config')
    if read_json(dataset/'download_status.json')['status'] != 'completed':
        raise ValueError('dataset not complete')
    manifest = dataset/'manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    validate_manifest(rows, dataset, verify_files=False)
    parts = partition_rows(rows, experiment)
    validate_parts(parts, experiment)
    return manifest, parts


def class_report(rows, classes, p):
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from scripts.run_image_experiment import metrics
    y = [classes.index(r['source_label']) for r in rows]
    result = metrics(y, p, 'attribution', calibrate_detection=False)
    precision, recall, f1, counts = precision_recall_fscore_support(
        y, p.argmax(1), labels=list(range(len(classes))), zero_division=0)
    result['per_source'] = {s:dict(n=int(counts[i]), precision=float(precision[i]),
        recall=float(recall[i]), f1=float(f1[i])) for i,s in enumerate(classes)}
    result['confusion_matrix'] = confusion_matrix(y,p.argmax(1),labels=list(range(len(classes)))).tolist()
    result['classes'] = classes
    result['per_semantic'] = {}
    for semantic in sorted({r['semantic_class'] for r in rows}):
        ids = [i for i,r in enumerate(rows) if r['semantic_class']==semantic]
        result['per_semantic'][semantic] = metrics(np.array(y)[ids],p[ids],'attribution',False)
    return result


def evaluate_scores(experiment, parts, classes, scores, mapping=None):
    """Pure evaluator: calibration scores fit thresholds; evaluation scores report results."""
    import numpy as np
    from research.image_attribution.followups import (binary_report, choose_threshold,
        family_scores, open_set_report, threshold_stability, validate_probabilities)
    cal, query = parts['calibration'], parts['evaluation']
    cp = validate_probabilities(scores['calibration'], len(cal), len(classes))
    qp = validate_probabilities(scores['evaluation'], len(query), len(classes))
    if experiment['task'] == 'detection':
        if classes != ['real','ai']:
            raise ValueError('incorrect detection class order')
        truth = [r['is_ai'] for r in query]
        result = {'fixed_0_5':binary_report(truth, qp[:,1], .5), 'per_semantic':{}, 'per_source':{}}
        for semantic in sorted({r['semantic_class'] for r in query}):
            ids = [i for i,r in enumerate(query) if r['semantic_class']==semantic]
            yy = np.array(truth)[ids]
            result['per_semantic'][semantic] = (binary_report(yy,qp[ids,1],.5) if len(set(yy))==2
                                               else dict(n=len(ids),note='one truth class only'))
        for source in sorted({r['source_label'] or 'real' for r in query}):
            ids = [i for i,r in enumerate(query) if (r['source_label'] or 'real')==source]
            result['per_source'][source] = dict(n=len(ids), predicted_ai_rate=float((qp[ids,1]>=.5).mean()))
        if experiment['mode'] == 'calibration':
            negatives = cp[[not r['is_ai'] for r in cal],1]
            result['calibrated'] = {}
            for target in experiment['target_fprs']:
                threshold = choose_threshold(negatives,target)
                result['calibrated'][str(target)] = dict(
                    calibration_negative_n=len(negatives), calibration_fpr=float((negatives>=threshold).mean()),
                    evaluation=binary_report(truth,qp[:,1],threshold),
                    threshold_stability=threshold_stability(cal,cp[:,1],target,experiment['seed'],experiment['bootstrap_repeats']))
        return result
    if experiment['mode'] == 'open_set':
        unknown = [r['source_label'] not in classes for r in cal]
        threshold = choose_threshold(cp[unknown].max(1),experiment['unknown_false_acceptance_target'])
        result = open_set_report([r['source_label'] for r in query],classes,qp,threshold)
        result['calibration_unknown_n'] = int(sum(unknown))
        result['calibration_unknown_false_acceptance'] = float((cp[unknown].max(1)>=threshold).mean())
        result['per_source'] = {}
        for source in sorted({r['source_label'] for r in query}):
            ids = [i for i,r in enumerate(query) if r['source_label']==source]
            result['per_source'][source] = dict(n=len(ids), known=source in classes,
                acceptance_rate=float((qp[ids].max(1)>=threshold).mean()))
        result['per_semantic'] = {}
        for semantic in sorted({r['semantic_class'] for r in query}):
            ids = [i for i,r in enumerate(query) if r['semantic_class']==semantic]
            sources = [query[i]['source_label'] for i in ids]
            has_known = [source in classes for source in sources]
            result['per_semantic'][semantic] = (open_set_report(sources,classes,qp[ids],threshold)
                if any(has_known) and not all(has_known) else dict(n=len(ids),note='known or unknown truth absent'))
        from sklearn.metrics import confusion_matrix
        vocabulary = classes+['__unknown__']
        truth = [r['source_label'] if r['source_label'] in classes else '__unknown__' for r in query]
        pred = [classes[int(p.argmax())] if p.max()>=threshold else '__unknown__' for p in qp]
        result['confusion_classes'] = vocabulary
        result['confusion_matrix'] = confusion_matrix(truth,pred,labels=vocabulary).tolist()
        return result
    result = class_report(query,classes,qp)
    if experiment['mode'] == 'hierarchy':
        names, fp = family_scores(classes,qp,mapping)
        family_rows = [dict(r,source_label=mapping[r['source_label']]['family']) for r in query]
        result['family'] = class_report(family_rows,names,fp)
        result['hierarchy_note'] = 'Family probability sums from one source classifier; source labels are publisher labels, not verified exact backend versions'
    return result


def run(args, config, experiment, manifest, parts, out):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from research.image_attribution.followups import Postprocess, TrainingAugmentation
    from research.image_attribution.models import load_checkpoint, resnet_transform
    from scripts.run_image_experiment import Images, probabilities, train
    torch.set_num_threads(4)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    roles = {role:[r['sample_id'] for r in rows] for role,rows in parts.items()}
    write_json(out/'roles.json',roles)
    identity = dict(experiment=args.experiment, config_sha256=sha256(args.config),
        manifest_sha256=sha256(manifest), roles_sha256=sha256(out/'roles.json'))
    write_json(out/'resolved_config.json',dict(identity, config=config, experiment_config=experiment,
        action=args.action, device=args.device, command=sys.argv,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
            ROOT/'research/image_attribution/followups.py',ROOT/'scripts/run_image_experiment.py',
            ROOT/'research/image_attribution/models.py',ROOT/'research/image_attribution/protocol.py']},
        pretrained_sha256=sha256(ROOT/config['weights']),
        holdout_evaluated=False,
        limitation='Exploratory dev-only pilot. Baseline-backed evaluations inherit D0/A0 dev checkpoint-selection bias.'))
    if experiment['mode']=='hierarchy':
        write_json(out/'family_mapping.json',read_json(ROOT/experiment['family_mapping']))
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    roles_used = ('train','selection') if args.action=='train' else ('calibration','evaluation')
    for role in roles_used:
        for row in parts[role]:
            if sha256(manifest.parent/row['relative_path']) != row['sha256']:
                raise ValueError(f'file hash mismatch: {row["sample_id"]}')
    if args.action=='train':
        training_rows = [dict(r,split='train') for r in parts['train']]+[dict(r,split='dev') for r in parts['selection']]
        with (out/'effective_training_rows.jsonl').open('w') as stream:
            for row in training_rows: stream.write(json.dumps(row)+'\n')
        training_args = SimpleNamespace(task=experiment['task'], manifest=manifest, weights=ROOT/config['weights'],
            device=args.device,batch_size=config['batch_size'],workers=config['workers'],epochs=config['epochs'])
        if experiment['mode']=='augmentation':
            training_args.train_transform = transforms.Compose([TrainingAugmentation(),resnet_transform()])
        result = train(training_args,training_rows,out)
        checkpoint = torch.load(out/'model.pth',map_location='cpu',weights_only=True)
        checkpoint['followup_identity'] = identity
        torch.save(checkpoint,out/'model.pth')
        result['evaluation_role'] = 'selection_only; run explicit evaluate later'
        return result
    if experiment['requires_training']:
        run_dir=ROOT/config['output_root']/args.experiment/'train'
    else:
        run_dir=ROOT/'outputs/image_experiments'/experiment['baseline']
    if read_json(run_dir/'status.json')['status']!='completed':
        raise ValueError('checkpoint run is not complete')
    model, checkpoint = load_checkpoint(run_dir/'model.pth',args.device)
    if checkpoint['task']!=experiment['task'] or checkpoint['manifest_sha256']!=sha256(manifest):
        raise ValueError('checkpoint task/manifest mismatch')
    if experiment['requires_training'] and checkpoint.get('followup_identity')!=identity:
        raise ValueError('checkpoint experiment/config/role identity mismatch')
    classes=checkpoint['classes']
    expected = ['real','ai'] if experiment['task']=='detection' else sorted({r['source_label'] for r in parts['train']})
    if classes!=expected:
        raise ValueError('checkpoint vocabulary mismatch (unknown sources must not have been trained)')
    mapping=None
    if experiment['mode']=='hierarchy':
        mapping=read_json(ROOT/experiment['family_mapping'])
        if mapping.get('reviewed') is not True: raise ValueError('A3 family mapping awaits evidence review')
        mapping=mapping['sources']
        from research.image_attribution.followups import family_scores
        family_scores(classes,np.full((1,len(classes)),1/len(classes)),mapping)
    conditions=experiment.get('conditions',[dict(name='clean')])
    results={}
    with (out/'predictions.jsonl').open('w') as stream:
        for condition in conditions:
            transform=transforms.Compose([Postprocess(condition.get('jpeg_quality'),condition.get('scale',1.)),resnet_transform()])
            scores={}
            for role in ('calibration','evaluation'):
                rows=parts[role]
                loader=DataLoader(Images(rows,[0]*len(rows),manifest.parent,transform),
                    batch_size=config['batch_size'],num_workers=config['workers'],shuffle=False)
                scores[role]=probabilities(model,loader,args.device)
                for row,p in zip(rows,scores[role]):
                    stream.write(json.dumps(dict(sample_id=row['sample_id'],group_id=row['group_id'],
                        role=role,condition=condition['name'],source_label=row['source_label'],is_ai=row['is_ai'],
                        semantic_class=row['semantic_class'],classes=classes,scores=p.tolist()))+'\n')
            results[condition['name']]=evaluate_scores(experiment,parts,classes,scores,mapping)
    if experiment['mode']=='postprocess':
        for name,result in results.items():
            result['accuracy_delta_from_clean']=result['accuracy']-results['clean']['accuracy']
            result['macro_f1_delta_from_clean']=result['macro_f1']-results['clean']['macro_f1']
    return dict(conditions=results, checkpoint_sha256=sha256(run_dir/'model.pth'),
                checkpoint=str(run_dir/'model.pth'),holdout_evaluated=False,
                evaluation_note='Parent dev only; baseline-backed evaluation inherits earlier checkpoint selection bias')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/image_experiments/iab_followups_v4_20.json')
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--action',choices=['plan','train','evaluate'],default='plan')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    config,experiment=load_config(args.config,args.experiment)
    if args.action=='plan':
        print(json.dumps(dict(experiment=args.experiment,protocol=experiment,execution_enabled=False,
            epochs=config['epochs'],holdout_used=False,
            note='No files written, no data downloaded, no model loaded. Await baseline results and experiment selection.'),ensure_ascii=False))
        return
    if not args.execute: parser.error('manual --execute is required; later experiments are not automatically scheduled')
    if args.action=='train' and (not experiment['requires_training'] or args.device!='cuda'):
        parser.error('this experiment does not require training, or CPU training is outside resource policy')
    # Keep current baseline pipeline first; refuse even explicitly requested later runs until complete.
    for name in ('d0-resnet50-iab-v4-20','a0-resnet50-iab-v4-20','ofa-iab-v4-20'):
        if read_json(ROOT/'outputs/image_experiments'/name/'status.json')['status']!='completed':
            raise ValueError('D0/A0/OFA baselines must finish first')
    manifest,parts=prepare(config,experiment)
    if experiment['mode']=='hierarchy':
        mapping=read_json(ROOT/experiment['family_mapping'])
        if mapping.get('reviewed') is not True:
            raise ValueError('A3 family mapping awaits evidence review; no model loaded')
        from research.image_attribution.followups import family_scores
        import numpy as np
        classes=sorted({r['source_label'] for r in parts['train']})
        family_scores(classes,np.full((1,len(classes)),1/len(classes)),mapping['sources'])
    out=ROOT/config['output_root']/args.experiment/args.action
    lock_path=ROOT/'outputs/image_experiments/gpu0-research.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.device=='cuda':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
                parser.error('set CUDA_VISIBLE_DEVICES=0; GPU1 service must remain untouched')
            used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used>1024: raise RuntimeError('GPU0 busy; preserve other processes')
        out.mkdir(parents=True,exist_ok=False)
        start=time.monotonic()
        try:
            with (out/'run.log').open('w',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result=run(args,config,experiment,manifest,parts,out)
            write_json(out/'metrics.json',result)
            write_json(out/'asset_manifest.json',{str(p.relative_to(out)):dict(sha256=sha256(p),bytes=p.stat().st_size)
                for p in sorted(out.rglob('*')) if p.is_file()})
            write_json(out/'status.json',dict(status='completed',exit_code=0,seconds=time.monotonic()-start))
        except Exception as exc:
            write_json(out/'status.json',dict(status='failed',exit_code=1,error=repr(exc),seconds=time.monotonic()-start))
            raise
    print(json.dumps(dict(output=str(out),status='completed')))


if __name__=='__main__': main()
