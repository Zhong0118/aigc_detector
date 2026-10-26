"""Frozen RGB/OFA source classification. This is not OFA pair verification."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from research.image_attribution.followups import partition_rows
from research.image_attribution.models import load_feature_cache
from research.image_attribution.protocol import sha256, validate_manifest, write_json
from scripts.run_image_experiment import evaluate_epoch, metrics, stratified_metrics, write_predictions


DIMENSIONS = {'rgb': 2048, 'fingerprint': 128, 'fusion': 2176}
EXTRACT_ROLES = ('train', 'selection', 'evaluation')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_features(features, dimensions):
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != dimensions:
        raise ValueError('feature shape mismatch')
    if not torch.is_floating_point(features) or not torch.isfinite(features).all():
        raise ValueError('non-finite or non-floating features')
    if (features.norm(dim=1) == 0).any():
        raise ValueError('zero-norm feature')
    return features


def combine_features(variant, rgb=None, fingerprint=None):
    branches = {'rgb': rgb, 'fingerprint': fingerprint}
    names = ('rgb', 'fingerprint') if variant == 'fusion' else (variant,)
    normalized = [F.normalize(checked_features(branches[name], DIMENSIONS[name]), dim=1) for name in names]
    if len({len(x) for x in normalized}) != 1:
        raise ValueError('branch sample count mismatch')
    return F.normalize(torch.cat(normalized, dim=1), dim=1)


def freeze_encoder(model):
    return model.eval().requires_grad_(False)


def load_bound_features(path, identity, dimensions):
    return checked_features(load_feature_cache(path, identity), dimensions)


def source_partitions(rows, seed, expected_classes):
    parts = partition_rows(rows, dict(mode='clean', task='attribution', seed=seed))
    classes = sorted({row['source_label'] for row in parts['train']})
    if classes != list(expected_classes):
        raise ValueError('training vocabulary differs from registered classes')
    for role, selected in parts.items():
        if not selected or any(row['source_label'] not in classes for row in selected):
            raise ValueError(f'empty role or unseen label: {role}')
    return parts, classes


def require_execution(config, execute):
    if config.get('execution_enabled') is not True:
        raise PermissionError('execution disabled: prepared only; await user selection after baselines')
    if not execute:
        raise PermissionError('explicit --execute is required')


def prepare(root, config, dataset):
    """Read hashes and manifest metadata only; never open any image or load a model."""
    root = Path(root)
    spec = config['datasets'][dataset]
    data = root / spec['path']
    if sha256(data / 'selection.json') != spec['selection_sha256']:
        raise ValueError('selection hash mismatch')
    plan = dict(dataset=dataset, dataset_path=str(data), manifest_bound=False,
                images_opened=False, training_started=False, holdout_scored=False,
                execution_enabled=config.get('execution_enabled', False),
                selection_sha256=spec['selection_sha256'])
    manifest = data / 'manifest.jsonl'
    if not manifest.exists():
        plan['status'] = 'awaiting_manifest'
        return plan, None, None
    actual_hash = sha256(manifest)
    plan['manifest_sha256'] = actual_hash
    if not spec.get('manifest_sha256'):
        plan['status'] = 'manifest_binding_required'
        return plan, None, None
    if actual_hash != spec['manifest_sha256']:
        raise ValueError('manifest hash mismatch')
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    validate_manifest(rows, data, verify_files=False)
    parts, classes = source_partitions(rows, config['seed'], config['classes'])
    plan.update(status='prepared_disabled' if not config.get('execution_enabled') else 'bound',
                manifest_bound=True, classes=classes,
                role_counts={role: len(selected) for role, selected in parts.items()},
                role_ids_sha256={role: digest([r['sample_id'] for r in selected]) for role, selected in parts.items()},
                per_source={role: dict(Counter(r['source_label'] for r in selected)) for role, selected in parts.items()},
                calibration_used=False, role_protocol='sha256(seed:group_id)%3; original train preserved')
    return plan, parts, classes


def verify_runtime(root, config, plan):
    if not plan['manifest_bound']:
        raise ValueError('manifest must be explicitly bound before execution')
    for relative in config['prerequisites']:
        completion = json.loads((root / relative).read_text())
        if completion.get('status') != 'completed' or not (root / completion['summary']).is_file():
            raise ValueError('baseline prerequisite incomplete')
    for relative, expected in config['implementation_sha256'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'implementation changed: {relative}')
    for name, asset in config['assets'].items():
        if sha256(root / asset['path']) != asset['sha256']:
            raise ValueError(f'pretrained weight hash mismatch: {name}')


def code_identity(root):
    paths = ['research/image_attribution/feature_ablation.py', 'scripts/run_image_feature_ablation.py',
             'research/image_attribution/models.py', 'research/image_attribution/protocol.py',
             'research/image_attribution/followups.py', 'scripts/run_image_experiment.py']
    return {path: sha256(root / path) for path in paths}


def fit_head(features, parts, classes, config, out, device, variant, provenance):
    """Train only a linear head; selection picks epoch, evaluation is scored once."""
    out = Path(out)
    expected_n = sum(len(parts[role]) for role in EXTRACT_ROLES)
    checked_features(features, DIMENSIONS[variant])
    if len(features) != expected_n:
        raise ValueError('features do not match registered role IDs')
    started = time.monotonic()
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    labels = {name: i for i, name in enumerate(classes)}
    loaders, tensors = {}, {}
    offset = 0
    for role in EXTRACT_ROLES:
        rows = parts[role]
        x = features[offset:offset + len(rows)]
        y = torch.tensor([labels[row['source_label']] for row in rows], dtype=torch.long)
        offset += len(rows)
        tensors[role] = (x, y)
        loaders[role] = DataLoader(TensorDataset(x, y), batch_size=config['training']['batch_size'],
                                  shuffle=False, num_workers=0)
    # A separate seeded loader controls training order and does not affect evaluation order.
    train_loader = DataLoader(TensorDataset(*tensors['train']), batch_size=config['training']['batch_size'],
                              shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    head = nn.Linear(DIMENSIONS[variant], len(classes)).to(device)
    training = config['training']
    optimizer = torch.optim.Adam(head.parameters(), lr=training['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training['scheduler_step'],
                                               gamma=training['scheduler_gamma'])
    criterion = nn.CrossEntropyLoss()
    best_score, best_state, best_epoch = -1.0, None, None
    write_json(out / 'status.json', dict(status='running', epoch=0))
    with (out / 'history.jsonl').open('x') as history:
        for epoch in range(1, training['epochs'] + 1):
            tick = time.monotonic()
            head.train()
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(x.to(device)), y.to(device))
                loss.backward()
                optimizer.step()
            train = evaluate_epoch(head, loaders['train'], device, 'attribution', classes, criterion)
            selection = evaluate_epoch(head, loaders['selection'], device, 'attribution', classes, criterion)
            if selection['macro_f1'] > best_score:
                best_score, best_epoch = selection['macro_f1'], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            record = dict(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'], train=train,
                          selection=selection, elapsed_seconds=time.monotonic() - tick)
            history.write(json.dumps(record, allow_nan=False) + '\n')
            history.flush()
            write_json(out / 'status.json', dict(status='running', epoch=epoch, best_epoch=best_epoch))
            scheduler.step()
    head.load_state_dict(best_state)
    head.eval()
    checkpoint = dict(schema='frozen-source-linear-v1', variant=variant, classes=classes,
                      input_dimensions=DIMENSIONS[variant], state_dict=best_state, best_epoch=best_epoch,
                      preprocessing='per-branch-l2-concat-l2-v1', provenance=provenance)
    torch.save(checkpoint, out / 'head.pth')
    saved = torch.load(out / 'head.pth', map_location='cpu', weights_only=True)
    reloaded = nn.Linear(saved['input_dimensions'], len(saved['classes'])).to(device)
    reloaded.load_state_dict(saved['state_dict'], strict=True)
    reloaded.eval()
    x, y = tensors['evaluation']
    # Same batch with original/reloaded head, plus single-vs-batch numerical check.
    probe = x[:min(8, len(x))].to(device)
    with torch.inference_mode():
        original = head(probe).softmax(1)
        reload_error = float((original - reloaded(probe).softmax(1)).abs().max().cpu())
        single_error = float((reloaded(probe).softmax(1) - torch.cat([
            reloaded(item.unsqueeze(0)).softmax(1) for item in probe])).abs().max().cpu())
    if reload_error > 1e-6 or single_error > 1e-5:
        raise ValueError('head reload or single/batch consistency failed')
    scores = []
    with torch.inference_mode():
        for batch, _ in loaders['evaluation']:
            scores.append(reloaded(batch.to(device)).softmax(1).cpu().numpy())
    scores = np.concatenate(scores)
    result = metrics(y.numpy(), scores, 'attribution')
    result.update(best_epoch=best_epoch, selection_macro_f1=best_score,
                  stratified=stratified_metrics(parts['evaluation'], y.numpy(), classes, scores, 'attribution'),
                  head_parameters=sum(p.numel() for p in head.parameters()),
                  reload_max_error=reload_error, single_batch_max_error=single_error,
                  holdout_scored=False, calibration_used=False,
                  note='Exploratory parent-dev evaluation; parent dev informed earlier experiments')
    write_predictions(out, parts['evaluation'], y.tolist(), classes, scores, 'attribution')
    write_json(out / 'metrics.json', result)
    write_json(out / 'provenance.json', dict(**provenance, head_sha256=sha256(out / 'head.pth'),
               encoder_training=False, head_training=True))
    write_json(out / 'status.json', dict(status='completed', exit_code=0, best_epoch=best_epoch,
                                        elapsed_seconds=time.monotonic() - started))
    return result
