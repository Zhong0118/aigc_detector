"""Independent frozen-feature detection/source heads; not OFA pair verification."""
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
DETECTION_EXTRACT_ROLES = ('train', 'selection', 'calibration', 'evaluation')


def extract_roles(task='attribution'):
    if task == 'detection':
        return DETECTION_EXTRACT_ROLES
    if task == 'attribution':
        return EXTRACT_ROLES
    raise ValueError(f'unsupported feature-head task: {task}')


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


def combine_features(variant, rgb=None, fingerprint=None, dimensions=None):
    dimensions = DIMENSIONS if dimensions is None else dimensions
    branches = {'rgb': rgb, 'fingerprint': fingerprint}
    names = ('rgb', 'fingerprint') if variant == 'fusion' else (variant,)
    normalized = [F.normalize(checked_features(branches[name], dimensions[name]), dim=1) for name in names]
    if len({len(x) for x in normalized}) != 1:
        raise ValueError('branch sample count mismatch')
    return checked_features(F.normalize(torch.cat(normalized, dim=1), dim=1), dimensions[variant])


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
    task = config.get('task', 'attribution')
    roles = extract_roles(task)
    dimensions = config.get('dimensions', DIMENSIONS)
    if task == 'detection' and classes != ['real', 'ai']:
        raise ValueError('detection classes must be [real, ai] in that order')
    for role in roles:
        if role not in parts or not parts[role]:
            raise ValueError(f'missing or empty feature role: {role}')
        if any(row.get('split') == 'holdout' for row in parts[role]):
            raise ValueError('holdout rows must not enter feature-head roles')
        if task == 'detection':
            truth = {row.get('is_ai') for row in parts[role]}
            if not truth <= {False, True} or (role != 'calibration' and truth != {False, True}):
                raise ValueError(f'both binary labels required for detection role: {role}')
            if role == 'calibration' and False not in truth:
                raise ValueError('real calibration samples are required')
    expected_n = sum(len(parts[role]) for role in roles)
    checked_features(features, dimensions[variant])
    if len(features) != expected_n:
        raise ValueError('features do not match registered role IDs')
    started = time.monotonic()
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    labels = {name: i for i, name in enumerate(classes)}
    loaders, tensors = {}, {}
    offset = 0
    for role in roles:
        rows = parts[role]
        x = features[offset:offset + len(rows)]
        y = torch.tensor([int(row['is_ai']) if task == 'detection' else labels[row['source_label']]
                          for row in rows], dtype=torch.long)
        offset += len(rows)
        tensors[role] = (x, y)
        loaders[role] = DataLoader(TensorDataset(x, y), batch_size=config['training']['batch_size'],
                                  shuffle=False, num_workers=0)
    # A separate seeded loader controls training order and does not affect evaluation order.
    train_loader = DataLoader(TensorDataset(*tensors['train']), batch_size=config['training']['batch_size'],
                              shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    head = nn.Linear(dimensions[variant], len(classes)).to(device)
    training = config['training']
    optimizer = torch.optim.Adam(head.parameters(), lr=training['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training['scheduler_step'],
                                               gamma=training['scheduler_gamma'])
    criterion = nn.CrossEntropyLoss()
    selection_metric = 'auroc' if task == 'detection' else 'macro_f1'
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
            train = evaluate_epoch(head, loaders['train'], device, task, classes, criterion, calibrate=False)
            selection = evaluate_epoch(head, loaders['selection'], device, task, classes, criterion, calibrate=False)
            if selection[selection_metric] > best_score:
                best_score, best_epoch = selection[selection_metric], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            record = dict(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'], train=train,
                          selection=selection, elapsed_seconds=time.monotonic() - tick)
            history.write(json.dumps(record, allow_nan=False) + '\n')
            history.flush()
            write_json(out / 'status.json', dict(status='running', epoch=epoch, best_epoch=best_epoch))
            scheduler.step()
    head.load_state_dict(best_state)
    head.eval()
    score_semantics = ('softmax probabilities ordered [real, ai]; column 1 is p_ai'
                       if task == 'detection' else 'softmax source probabilities conditional on known AI input')
    checkpoint = dict(schema='frozen-detection-linear-v1' if task == 'detection' else 'frozen-source-linear-v1',
                      task=task, score_semantics=score_semantics, variant=variant, classes=classes,
                      input_dimensions=dimensions[variant], state_dict=best_state, best_epoch=best_epoch,
                      preprocessing='per-branch-l2-concat-l2-v1', provenance=provenance)
    torch.save(checkpoint, out / 'head.pth')
    saved = torch.load(out / 'head.pth', map_location='cpu', weights_only=True)
    reloaded = nn.Linear(saved['input_dimensions'], len(saved['classes'])).to(device)
    reloaded.load_state_dict(saved['state_dict'], strict=True)
    reloaded.eval()
    x, y = tensors['evaluation']
    # Consistency probes use selection; evaluation is visited only for the final report.
    selection_x = tensors['selection'][0]
    probe = selection_x[:min(8, len(selection_x))].to(device)
    with torch.inference_mode():
        original = head(probe).softmax(1)
        reload_error = float((original - reloaded(probe).softmax(1)).abs().max().cpu())
        single_error = float((reloaded(probe).softmax(1) - torch.cat([
            reloaded(item.unsqueeze(0)).softmax(1) for item in probe])).abs().max().cpu())
    if reload_error > 1e-6 or single_error > 1e-5:
        raise ValueError('head reload or single/batch consistency failed')
    final_scores = {}
    for role in (('calibration', 'evaluation') if task == 'detection' else ('evaluation',)):
        batches = []
        with torch.inference_mode():
            for batch, _ in loaders[role]:
                batches.append(reloaded(batch.to(device)).softmax(1).cpu().numpy())
        final_scores[role] = np.concatenate(batches)
    scores = final_scores['evaluation']
    result = metrics(y.numpy(), scores, task, calibrate_detection=False)
    if task == 'detection':
        from scripts.run_image_followup import evaluate_scores
        protocol = dict(task='detection', mode='calibration', target_fprs=config.get('target_fprs', [.01, .05]),
                        seed=seed, bootstrap_repeats=config.get('bootstrap_repeats', 200))
        result.update(evaluate_scores(protocol, parts, classes, final_scores))
        result['calibration_protocol'] = protocol
        calibration_out = out / 'calibration'
        calibration_out.mkdir()
        write_predictions(calibration_out, parts['calibration'], tensors['calibration'][1].tolist(),
                          classes, final_scores['calibration'], task)
    else:
        result['stratified'] = stratified_metrics(parts['evaluation'], y.numpy(), classes, scores, task)
    result.update(task=task, score_semantics=score_semantics, best_epoch=best_epoch,
                  head_parameters=sum(p.numel() for p in head.parameters()),
                  reload_max_error=reload_error, single_batch_max_error=single_error,
                  holdout_scored=False, calibration_used=(task == 'detection'),
                  note='Exploratory parent-dev evaluation; parent dev informed earlier experiments')
    result[f'selection_{selection_metric}'] = best_score
    write_predictions(out, parts['evaluation'], y.tolist(), classes, scores, task)
    write_json(out / 'metrics.json', result)
    write_json(out / 'provenance.json', dict(**provenance, head_sha256=sha256(out / 'head.pth'),
               encoder_training=False, head_training=True))
    write_json(out / 'status.json', dict(status='completed', exit_code=0, best_epoch=best_epoch,
                                        elapsed_seconds=time.monotonic() - started))
    return result
