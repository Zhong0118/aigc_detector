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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, Dataset

from research.image_attribution.models import (ResNetClassifier, load_checkpoint, load_feature_cache, load_ofa,
    ofa_features, ofa_transform, reference_score, resnet_transform)
from research.image_attribution.protocol import load_manifest, sha256, task_rows, write_json
from research.image_attribution.iab_protocol import reference_indices as select_references


class Images(Dataset):
    def __init__(self, rows, labels, root, transform):
        self.rows, self.labels, self.root, self.transform = rows, labels, Path(root), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]['relative_path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, self.labels[index]



def image_datasets(partitions, root, train_transform=None):
    """Keep development and train-metric inputs clean when training augmentation is enabled."""
    datasets = {s: Images(r, y, root, resnet_transform()) for s, (r, y) in partitions.items()}
    clean_train = datasets['train']
    if train_transform is not None:
        rows, labels = partitions['train']
        datasets['train'] = Images(rows, labels, root, train_transform)
    return datasets, clean_train


def probabilities(model, loader, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for inputs, _ in loader:
            values.append(model(inputs.to(device)).softmax(1).cpu().numpy())
    return np.concatenate(values)



def evaluate_epoch(model, loader, device, task, classes, criterion, calibrate=False):
    model.eval()
    labels, probabilities = [], []
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for inputs, target in loader:
            logits = model(inputs.to(device))
            loss = criterion(logits, target.to(device))
            loss_sum += float(loss.cpu()) * len(target)
            count += len(target)
            labels.extend(target.numpy().tolist())
            probabilities.append(logits.softmax(1).cpu().numpy())
    scores = np.concatenate(probabilities)
    result = metrics(labels, scores, task, calibrate_detection=calibrate)
    result.update(loss=loss_sum / count,
                  class_counts=dict(Counter(classes[label] for label in labels)))
    return result


def score_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(n=int(values.size), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), p05=float(np.quantile(values, .05)),
                median=float(np.quantile(values, .50)), p95=float(np.quantile(values, .95)),
                max=float(values.max()))


def metrics(labels, probs, task, calibrate_detection=True):
    pred = probs.argmax(1)
    result = dict(n=len(labels), accuracy=float(accuracy_score(labels, pred)),
                  macro_f1=float(f1_score(labels, pred, average='macro', zero_division=0)))
    if task == 'detection':
        truth = np.asarray(labels)
        result.update(auroc=float(roc_auc_score(truth, probs[:, 1])), threshold_0_5=.5,
            fpr_0_5=float((probs[truth == 0, 1] >= .5).mean()),
            tpr_0_5=float((probs[truth == 1, 1] >= .5).mean()),
            real_count=int((truth == 0).sum()), ai_count=int((truth == 1).sum()))
        if calibrate_detection:
            fpr, tpr, thresholds = roc_curve(truth, probs[:, 1])
            finite = np.isfinite(thresholds)
            eligible = np.flatnonzero((fpr <= .02) & finite)
            threshold = float(thresholds[eligible[-1]]) if len(eligible) else float(
                np.nextafter(probs[:,1].max(), np.inf, dtype=probs.dtype))
            result.update(dev_target_fpr=.02, dev_threshold=threshold,
                dev_fpr=float((probs[truth == 0, 1] >= threshold).mean()),
                dev_tpr=float((probs[truth == 1, 1] >= threshold).mean()),
                threshold_note='Exploratory development calibration, not an independent low-FPR claim')
    else:
        result['top1'] = result['accuracy']
        result['top3'] = float(np.mean([y in np.argsort(p)[-min(3, len(p)):] for y, p in zip(labels, probs)]))
    return result


def stratified_metrics(rows, labels, classes, scores, task):
    truth=np.asarray(labels);pred=scores.argmax(1)
    result=dict(per_source={},per_semantic={})
    for semantic in sorted({str(r.get('semantic_class')) for r in rows}):
        ids=[i for i,r in enumerate(rows) if str(r.get('semantic_class'))==semantic]
        if task=='detection' and len(set(truth[ids]))<2:
            result['per_semantic'][semantic]=dict(n=len(ids),auroc=None,note='only one truth class present')
        else:result['per_semantic'][semantic]=metrics(truth[ids],scores[ids],task)
    if task=='attribution':
        precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=list(range(len(classes))),zero_division=0)
        for i,name in enumerate(classes):
            ids=np.flatnonzero(truth==i)
            result['per_source'][name]=dict(n=int(support[i]),precision=float(precision[i]),
                recall=float(recall[i]),f1=float(f1[i]),top3=float(np.mean([i in np.argsort(scores[j])[-min(3,len(classes)):] for j in ids])) if len(ids) else None)
    else:
        for name in sorted({r.get('source_label') or 'real' for r in rows}):
            ids=[i for i,r in enumerate(rows) if (r.get('source_label') or 'real')==name]
            positive=float((scores[ids,1]>=.5).mean())
            result['per_source'][name]=dict(n=len(ids),**({'fpr_0_5':positive} if name=='real' else {'tpr_0_5':positive}))
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
    datasets, clean_train = image_datasets(partitions, args.manifest.parent, getattr(args, 'train_transform', None))
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == 'train', num_workers=args.workers, pin_memory=args.device == 'cuda')
               for s, d in datasets.items()}
    train_eval_loader = DataLoader(clean_train, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=args.device == 'cuda')
    model = ResNetClassifier(len(classes), args.weights).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    train_counts = np.bincount(partitions['train'][1], minlength=len(classes))
    class_weights = (len(partitions['train'][1]) / (len(classes) * train_counts)
                     if args.task == 'detection' else None)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=args.device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
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
        train_result = evaluate_epoch(model, train_eval_loader, args.device, args.task, classes,
                                      criterion, calibrate=False)
        dev_result = evaluate_epoch(model, loaders['dev'], args.device, args.task, classes,
                                    criterion, calibrate=args.task == 'detection')
        p = probabilities(model, loaders['dev'], args.device)
        result = {key: value for key, value in dev_result.items()
                  if key not in {'loss', 'class_counts'}}
        record = dict(epoch=epoch, train_loss=train_result['loss'], train=train_result,
                      dev_loss=dev_result['loss'], dev=dev_result,
                      train_class_counts=train_result['class_counts'],
                      dev_class_counts=dev_result['class_counts'],
                      lr=optimizer.param_groups[0]['lr'], scheduler='StepLR(step_size=1,gamma=0.9)',
                      optimizer='Adam(lr=0.001)', seconds=time.monotonic()-start)
        print(json.dumps(record), flush=True)
        history.append(record)
        write_json(out / 'history.json', history)
        # Earliest epoch wins ties; no holdout access.
        if result[key] > best:
            best = result[key]
            checkpoint = dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                              task=args.task, classes=classes, epoch=epoch,
                              manifest_sha256=sha256(args.manifest),
                              preprocessing='iab-resnet50-v1', pretrained_sha256=sha256(args.weights),
                              class_weights=class_weights.tolist() if class_weights is not None else None)
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
        dev_counts=dict(Counter(classes[y] for y in partitions['dev'][1])),
        class_weights=class_weights.tolist() if class_weights is not None else None,
        classes=classes, failed_samples=0, reload_max_abs_error=reload_error,
        single_vs_batch_max_abs_error=single_error, holdout_evaluated=False)
    write_json(out/'stratified_metrics.json', stratified_metrics(partitions['dev'][0], partitions['dev'][1], classes, p, args.task))
    write_predictions(out, partitions['dev'][0], partitions['dev'][1], classes, p, args.task)
    write_json(out / 'single_prediction.json', dict(sample_id=partitions['dev'][0][0]['sample_id'],
               task=args.task, classes=classes, scores=single[0].tolist()))
    return result


def verify_ofa(args, rows, out):
    ofa_start=time.monotonic()
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
        for i, (inputs, _) in enumerate(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)):
            values.append(ofa_features(inputs.to(args.device), encoder, denoiser).cpu())
            print(json.dumps(dict(feature_batch=i, batch_size=len(inputs))), flush=True)
    features = torch.cat(values)
    torch.save(dict(identity=cache_identity, features=features), out/'features.pt')
    reloaded = load_feature_cache(out/'features.pt', cache_identity)
    if not torch.equal(features, reloaded):
        raise ValueError('feature cache identity/contents mismatch')
    reference_indices = select_references(rows, labels, len(classes), limit=100, seed=args.seed)
    bank = dict(classes=classes, identity=cache_identity,
                references={classes[k]: features[v] for k,v in reference_indices.items()})
    torch.save(bank, out/'reference_bank.pt')
    saved_bank = torch.load(out/'reference_bank.pt', map_location='cpu', weights_only=True)
    if saved_bank['classes'] != classes or any(not torch.equal(bank['references'][c], saved_bank['references'][c]) for c in classes):
        raise ValueError('OFA reference bank reload mismatch')
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
    closed_set = metrics(truth, scores, 'attribution')
    closed_set['score_note'] = 'Argmax of independently trained pair match scores; scores are not class probabilities'
    closed_out = out/'closed_set'; closed_out.mkdir()
    write_json(out/'stratified_metrics.json', stratified_metrics([rows[i] for i in queries], truth, classes, scores, 'attribution'))
    write_predictions(closed_out, [rows[i] for i in queries], truth, classes, scores, 'source_attribution_from_match_scores')
    # Reopen the persisted reference bank and author weights for a complete single-image check.
    del encoder, pair, denoiser
    encoder, pair, denoiser = load_ofa(args.weights, args.denoiser, args.device)
    with torch.inference_mode():
        one = ofa_features(dataset[queries[0]][0].unsqueeze(0).to(args.device), encoder, denoiser)[0]
        single_scores = np.asarray([float(reference_score(pair, saved_bank['references'][c].to(args.device), one).cpu()) for c in classes])
    score_error = float(np.abs(single_scores-scores[0]).max())
    if score_error > 1e-4: raise RuntimeError(f'OFA saved-bank single/batch score mismatch {score_error}')
    write_json(out/'single_prediction.json', dict(sample_id=rows[queries[0]]['sample_id'], classes=classes,
        match_scores=single_scores.tolist(), predicted_label=classes[int(single_scores.argmax())]))
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
    score_distributions={}
    for label,name in enumerate(classes):
        positive=scores[truth==label,label]
        negative=scores[truth!=label,label]
        score_distributions[name]=dict(matching=score_distribution(positive),
                                       nonmatching=score_distribution(negative))
    with (out/'predictions.jsonl').open('w') as stream:
        for index,score in zip(queries,scores):
            stream.write(json.dumps(dict(sample_id=rows[index]['sample_id'],split='dev',task='source_verification',
                status='ok',true_label=classes[labels[index]],classes=classes,match_scores=score.tolist()))+'\n')
    return dict(task='source_verification', epoch=None, evaluation_split='dev', query_n=len(queries),
        reference_counts={classes[k]:len(v) for k,v in reference_indices.items()},
        reference_selection='seeded round-robin by semantic category, training partition only',
        closed_set=closed_set, saved_bank_single_vs_batch_score_max_abs_error=score_error,
        per_source_auroc=aucs, macro_auroc=float(np.mean(list(aucs.values()))),
        ordered_pair_aurocs=pairwise, macro_ordered_pair_auroc=float(np.mean(list(pairwise.values()))),
        score_distributions=score_distributions, inference_seconds=time.monotonic()-ofa_start,
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
    parser.add_argument('--workers', type=int, default=4)
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
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_status=subprocess.check_output(['git','status','--porcelain'],text=True),
        source_hashes={str(p):sha256(p) for p in [Path(__file__),
            Path(__file__).resolve().parents[1]/'research/image_attribution/models.py',
            Path(__file__).resolve().parents[1]/'research/image_attribution/protocol.py',
            Path(__file__).resolve().parents[1]/'research/image_attribution/iab_protocol.py']},
        manifest_sha256=sha256(args.manifest), pretrained_sha256=sha256(args.weights),
        python=sys.version, torch=torch.__version__, platform=platform.platform(),
        command=sys.argv, cpu_threads=4, workers=args.workers, cudnn_tf32=False, matmul_tf32=False,
        ofa_denoiser_microbatch=1, class_weight_rule='binary detection: N/(K*class_count); attribution: unweighted',
        per_epoch_metrics='train/dev loss, accuracy, macro_f1, task metrics and class counts',
        training_protocol='Adam lr=.001, CE, StepLR(1,.9), full backbone; OFA frozen',
        limitation='Fixed seed does not guarantee bitwise reproducibility across devices')
    write_json(args.output/'resolved_config.json',config)
    (args.output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    try:
        rows = load_manifest(args.manifest)
        result = verify_ofa(args,rows,args.output) if args.task=='ofa' else train(args,rows,args.output)
        write_json(args.output/'metrics.json',result)
        write_json(args.output/'asset_manifest.json',{str(p.relative_to(args.output)):dict(sha256=sha256(p),bytes=p.stat().st_size)
            for p in sorted(args.output.rglob('*')) if p.is_file()})
        write_json(args.output/'status.json',dict(status='completed',elapsed_seconds=time.monotonic()-start,exit_code=0))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(args.output/'status.json',dict(status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-start,exit_code=1))
        raise


if __name__=='__main__':
    main()
