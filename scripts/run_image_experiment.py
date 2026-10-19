"""Run separate paper-based ResNet tasks or complete frozen OFA verification offline."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader, Dataset

from research.image_attribution.models import (ResNetClassifier, load_checkpoint, load_feature_cache, load_ofa,
    ofa_features, ofa_transform, reference_score, resnet_transform)
from research.image_attribution.protocol import load_manifest, sha256, task_rows, write_json


class Images(Dataset):
    def __init__(self, rows, labels, root, transform):
        self.rows, self.labels, self.root, self.transform = rows, labels, Path(root), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]['relative_path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, self.labels[index]


def probabilities(model, loader, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for inputs, _ in loader:
            values.append(model(inputs.to(device)).softmax(1).cpu().numpy())
    return np.concatenate(values)


def metrics(labels, probs, task):
    pred = probs.argmax(1)
    result = dict(n=len(labels), accuracy=float(accuracy_score(labels, pred)),
                  macro_f1=float(f1_score(labels, pred, average='macro', zero_division=0)))
    if task == 'detection':
        truth = np.asarray(labels)
        fpr, tpr, thresholds = roc_curve(truth, probs[:, 1])
        finite = np.isfinite(thresholds)
        eligible = np.flatnonzero((fpr <= .02) & finite)
        # No finite point qualifies: threshold above maximum score, explicitly all-negative.
        threshold = float(thresholds[eligible[-1]]) if len(eligible) else float(
            np.nextafter(probs[:,1].max(), np.inf, dtype=probs.dtype))
        result.update(auroc=float(roc_auc_score(truth, probs[:, 1])), threshold_0_5=.5,
            fpr_0_5=float((probs[truth == 0, 1] >= .5).mean()),
            tpr_0_5=float((probs[truth == 1, 1] >= .5).mean()),
            dev_target_fpr=.02, dev_threshold=threshold,
            dev_fpr=float((probs[truth == 0, 1] >= threshold).mean()),
            dev_tpr=float((probs[truth == 1, 1] >= threshold).mean()),
            real_count=int((truth == 0).sum()), ai_count=int((truth == 1).sum()),
            threshold_note='Exploratory development calibration, not an independent low-FPR claim')
    else:
        result['top1'] = result['accuracy']
        result['top3'] = float(np.mean([y in np.argsort(p)[-min(3, len(p)):] for y, p in zip(labels, probs)]))
    return result


def write_predictions(out, rows, labels, classes, probs, task):
    with (out / 'predictions.jsonl').open('w') as stream:
        for row, label, p in zip(rows, labels, probs):
            record = dict(sample_id=row['sample_id'], split=row['split'], status='ok', task=task,
                          true_label=classes[label], predicted_label=classes[int(p.argmax())],
                          classes=classes, scores=p.tolist())
            stream.write(json.dumps(record) + '\n')
    matrix = confusion_matrix(labels, probs.argmax(1), labels=list(range(len(classes))))
    with (out / 'confusion_matrix.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true/predicted'] + classes)
        writer.writerows([[name] + row.tolist() for name, row in zip(classes, matrix)])


def train(args, rows, out):
    rows, labels, classes = task_rows(rows, args.task)
    if args.task == 'detection' and set(labels) != {0, 1}:
        raise ValueError('both truth classes are required')
    if len(classes) < 2:
        raise ValueError('at least two classes are required')
    partitions = {}
    for split in ('train', 'dev'):
        selected = [(r, y) for r, y in zip(rows, labels) if r['split'] == split]
        rr, yy = [x[0] for x in selected], [x[1] for x in selected]
        if set(yy) != set(range(len(classes))):
            raise ValueError(f'{split} lacks a class')
        partitions[split] = (rr, yy)
    datasets = {s: Images(r, y, args.manifest.parent, resnet_transform()) for s, (r, y) in partitions.items()}
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == 'train', num_workers=0)
               for s, d in datasets.items()}
    model = ResNetClassifier(len(classes), args.weights).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    criterion = torch.nn.CrossEntropyLoss()
    best, history = -1., []
    key = 'auroc' if args.task == 'detection' else 'macro_f1'
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        total_loss, count = 0., 0
        for x, y in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(args.device))
            loss = criterion(logits, y.to(args.device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            count += len(y)
        p = probabilities(model, loaders['dev'], args.device)
        result = metrics(partitions['dev'][1], p, args.task)
        record = dict(epoch=epoch, loss=total_loss/count, lr=optimizer.param_groups[0]['lr'],
                      seconds=time.monotonic()-start, dev=result)
        print(json.dumps(record), flush=True)
        history.append(record)
        write_json(out / 'history.json', history)
        # Earliest epoch wins ties; no holdout access.
        if result[key] > best:
            best = result[key]
            checkpoint = dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                              task=args.task, classes=classes, epoch=epoch,
                              manifest_sha256=sha256(args.manifest),
                              preprocessing='iab-resnet50-v1', pretrained_sha256=sha256(args.weights))
            torch.save(checkpoint, out / 'model.pth')
            np.save(out / 'best_dev_probabilities.npy', p)
        scheduler.step()
    del model, optimizer
    model, checkpoint = load_checkpoint(out / 'model.pth', args.device)
    p = probabilities(model, loaders['dev'], args.device)
    saved = np.load(out / 'best_dev_probabilities.npy')
    reload_error = float(np.max(np.abs(p - saved)))
    single_loader = DataLoader(torch.utils.data.Subset(datasets['dev'], [0]), batch_size=1)
    single = probabilities(model, single_loader, args.device)
    single_error = float(np.max(np.abs(single[0] - p[0])))
    if reload_error > 1e-5 or single_error > 1e-4:
        raise RuntimeError(f'inference consistency failed: reload={reload_error} single={single_error}')
    result = metrics(partitions['dev'][1], p, args.task)
    result.update(task=args.task, evaluation_split='dev', selected_epoch=checkpoint['epoch'],
        selection_metric=key, tie_break='earliest epoch', train_n=len(datasets['train']),
        train_counts=dict(Counter(classes[y] for y in partitions['train'][1])),
        classes=classes, failed_samples=0, reload_max_abs_error=reload_error,
        single_vs_batch_max_abs_error=single_error, holdout_evaluated=False)
    write_predictions(out, partitions['dev'][0], partitions['dev'][1], classes, p, args.task)
    write_json(out / 'single_prediction.json', dict(sample_id=partitions['dev'][0][0]['sample_id'],
               task=args.task, classes=classes, scores=single[0].tolist()))
    return result


def verify_ofa(args, rows, out):
    rows, labels, classes = task_rows(rows, 'attribution')
    rows_labels = [(r,y) for r,y in zip(rows,labels) if r['split'] in {'train','dev'}]
    rows, labels = [r for r,y in rows_labels], [y for r,y in rows_labels]
    encoder, pair, denoiser = load_ofa(args.weights, args.denoiser, args.device)
    dataset = Images(rows, labels, args.manifest.parent, ofa_transform)
    cache_identity = dict(manifest_sha256=sha256(args.manifest), weight_sha256=sha256(args.weights),
        denoiser_sha256=sha256(args.denoiser), preprocessing='ofa-author-3780cbf-v1',
        sample_ids=[r['sample_id'] for r in rows], model_code_sha256=sha256(Path(__file__).resolve().parents[1]/'research/image_attribution/models.py'))
    values = []
    with torch.inference_mode():
        for i, (inputs, _) in enumerate(DataLoader(dataset, batch_size=args.batch_size)):
            values.append(ofa_features(inputs.to(args.device), encoder, denoiser).cpu())
            print(json.dumps(dict(feature_batch=i, batch_size=len(inputs))), flush=True)
    features = torch.cat(values)
    torch.save(dict(identity=cache_identity, features=features), out/'features.pt')
    reloaded = load_feature_cache(out/'features.pt', cache_identity)
    if not torch.equal(features, reloaded):
        raise ValueError('feature cache identity/contents mismatch')
    reference_indices = {label:[i for i,(r,y) in enumerate(zip(rows,labels)) if y==label and r['split']=='train'][:100]
                         for label in range(len(classes))}
    queries = [i for i,r in enumerate(rows) if r['split']=='dev']
    if any(not ids for ids in reference_indices.values()) or not queries:
        raise ValueError('empty support or dev set')
    if set(queries) & {i for ids in reference_indices.values() for i in ids}:
        raise ValueError('support/query overlap')
    scores = []
    with torch.inference_mode():
        for query in queries:
            values = []
            for label in range(len(classes)):
                support = features[reference_indices[label]].to(args.device)
                values.append(float(reference_score(pair, support, features[query].to(args.device)).cpu()))
            scores.append(values)
        one = ofa_features(dataset[queries[0]][0].unsqueeze(0).to(args.device), encoder, denoiser).cpu()
    feature_error = float((one[0] - features[queries[0]]).abs().max())
    if not torch.allclose(one[0], features[queries[0]], rtol=1e-3, atol=1e-4):
        raise RuntimeError(f'OFA single/batch mismatch {feature_error}')
    scores = np.asarray(scores)
    truth = np.asarray([labels[i] for i in queries])
    aucs, pairwise = {}, {}
    for label,name in enumerate(classes):
        aucs[name] = float(roc_auc_score(truth==label, scores[:,label]))
        for other,other_name in enumerate(classes):
            if label == other:
                continue
            mask = (truth==label)|(truth==other)
            pairwise[f'{name} vs {other_name}'] = float(roc_auc_score(truth[mask]==label, scores[mask,label]))
    write_json(out/'reference_manifest.json', {name:[rows[i]['sample_id'] for i in reference_indices[label]]
                                             for label,name in enumerate(classes)})
    with (out/'predictions.jsonl').open('w') as stream:
        for index,score in zip(queries,scores):
            stream.write(json.dumps(dict(sample_id=rows[index]['sample_id'],split='dev',task='source_verification',
                status='ok',true_label=classes[labels[index]],classes=classes,match_scores=score.tolist()))+'\n')
    return dict(task='source_verification', evaluation_split='dev', query_n=len(queries),
        reference_counts={classes[k]:len(v) for k,v in reference_indices.items()},
        per_source_auroc=aucs, macro_auroc=float(np.mean(list(aucs.values()))),
        ordered_pair_aurocs=pairwise, macro_ordered_pair_auroc=float(np.mean(list(pairwise.values()))),
        single_vs_batch_feature_max_abs_error=feature_error, failed_samples=0,
        holdout_evaluated=False, note='Complete pretrained OFA inference on custom development subset; no claim of full paper reproduction or truth detection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['detection','attribution','ofa'], required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--denoiser', type=Path)
    parser.add_argument('--device', choices=['cpu','mps','cuda'], required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.device == 'cpu' and args.task != 'ofa':
        parser.error('CPU full-network training is outside this first-round resource policy')
    if args.task == 'ofa' and not args.denoiser:
        parser.error('--denoiser is required for complete OFA')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_status=subprocess.check_output(['git','status','--porcelain'],text=True),
        source_hashes={str(p):sha256(p) for p in [Path(__file__),
            Path(__file__).resolve().parents[1]/'research/image_attribution/models.py',
            Path(__file__).resolve().parents[1]/'research/image_attribution/protocol.py']},
        manifest_sha256=sha256(args.manifest), pretrained_sha256=sha256(args.weights),
        python=sys.version, torch=torch.__version__, platform=platform.platform(),
        command=sys.argv, cpu_threads=4, workers=0, training_protocol='Adam lr=.001, CE, StepLR(1,.9), full backbone; OFA frozen',
        limitation='Fixed seed does not guarantee bitwise reproducibility across devices')
    write_json(args.output/'resolved_config.json',config)
    (args.output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    try:
        rows = load_manifest(args.manifest)
        result = verify_ofa(args,rows,args.output) if args.task=='ofa' else train(args,rows,args.output)
        write_json(args.output/'metrics.json',result)
        write_json(args.output/'asset_manifest.json',{p.name:dict(sha256=sha256(p),bytes=p.stat().st_size)
            for p in sorted(args.output.iterdir()) if p.is_file()})
        write_json(args.output/'status.json',dict(status='completed',elapsed_seconds=time.monotonic()-start,exit_code=0))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(args.output/'status.json',dict(status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-start,exit_code=1))
        raise


if __name__=='__main__':
    main()
