"""Frozen OFA 20+1 adaptation: train-only reconstructions and isolated head fitting."""
from collections import defaultdict, deque
from contextlib import contextmanager
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from research.image_attribution.followups import choose_threshold, validate_probabilities
from research.image_attribution.openclip_route import digest, normalized, rejection_report

UNKNOWN = 'simulated_unknown'
DIMENSION = 128
RECIPE = dict(epochs=30, batch_size=32, seed=42, optimizer='Adam', learning_rate=.001,
              scheduler_step=1, scheduler_gamma=.9, loss='unweighted_cross_entropy')


def safe_load(path):
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if unsafe:
        raise ValueError(f'unsafe checkpoint globals: {unsafe}')
    return torch.load(path, map_location='cpu', weights_only=True)


@contextmanager
def architecture_rng(seed):
    original = random.getstate()
    random.setstate(random.Random(seed).getstate())
    try:
        yield
    finally:
        random.setstate(original)


def reconstruct(encoder, decoder, images, seed, expected_trace=None):
    with torch.inference_mode(), architecture_rng(seed):
        result, trace = decoder(encoder(images))
    if result.shape != images.shape or not torch.isfinite(result).all():
        raise ValueError('invalid simulator reconstruction shape or values')
    if expected_trace is not None and trace != expected_trace:
        raise ValueError('simulator architecture trace changed')
    return result, trace


def select_reconstructions(rows, seed):
    """One parent per group/hash; no parent may share identities with any nontrain row."""
    fields = ('group_id', 'sha256', 'pixel_sha256')
    protected = {(key, row[key]) for row in rows if row['split'] != 'train'
                 for key in fields if row.get(key)}
    candidates = [row for row in rows if row['split'] == 'train' and not row['is_ai']]
    if any((key, row.get(key)) in protected for row in candidates for key in fields):
        raise ValueError('real training parent identity overlaps a protected role')
    ordered = sorted(candidates, key=lambda r: (digest([seed, r['group_id'], r['sample_id']]), r['sample_id']))
    used, pools = set(), defaultdict(deque)
    for row in ordered:
        if any(not row.get(key) for key in fields):
            raise ValueError('parent group and file/pixel hashes are required')
        keys = {(key, row[key]) for key in fields}
        if keys & used:
            continue
        used.update(keys)
        pools[row['semantic_class']].append(row)
    semantics = sorted(pools)
    if not semantics:
        raise ValueError('no real train parents')
    selected = []
    architectures = [i % 8 for i in range(1024)]
    random.Random(seed).shuffle(architectures)
    for i in range(1024):
        available = [s for s in semantics if pools[s]]
        if not available:
            raise ValueError('1024 distinct real train parent groups required')
        semantic = available[i % len(available)]
        parent = pools[semantic].popleft()
        architecture = architectures[i]
        identity = digest([seed, parent['sample_id'], parent['sha256'], architecture])
        selected.append(dict(sample_id='ofa-sim-' + identity, group_id=parent['group_id'],
            split='train', role='train', source_label=UNKNOWN, is_ai=True,
            semantic_class=parent['semantic_class'], parent=dict(parent), architecture_index=architecture))
    if {r['sample_id'] for r in selected} & {r['sample_id'] for r in rows}:
        raise ValueError('synthetic IDs collide with original data')
    return selected


def checked_role(rows, classes, role, unknown):
    if not rows or any(r.get('role') != role or r['split'] != 'dev' or not r['is_ai']
                       or r['source_label'] not in set(classes) | set(unknown) for r in rows):
        raise ValueError(f'invalid {role} rows or source labels')


def selection_f1(rows, classes, probabilities):
    checked_role(rows, classes, 'selection', [])
    p = validate_probabilities(probabilities, len(rows), len(classes) + 1)
    truth = [classes.index(r['source_label']) for r in rows]
    return float(f1_score(truth, p.argmax(1), labels=list(range(len(classes))),
                          average='macro', zero_division=0))


def calibrate(rows, classes, probabilities):
    checked_role(rows, classes, 'calibration', ['kling'])
    p = validate_probabilities(probabilities, len(rows), len(classes) + 1)
    unknown = np.array([r['source_label'] == 'kling' for r in rows])
    if not unknown.any():
        raise ValueError('kling calibration rows required')
    return choose_threshold(1 - p[unknown, -1], .05)


def report(rows, classes, probabilities, threshold):
    roles = {r.get('role') for r in rows}
    if roles == {'calibration'}:
        checked_role(rows, classes, 'calibration', ['kling'])
    elif roles == {'evaluation'}:
        checked_role(rows, classes, 'evaluation', ['grok3'])
    else:
        raise ValueError('one complete calibration or evaluation role required')
    p = validate_probabilities(probabilities, len(rows), len(classes) + 1)
    confidence = 1 - p[:, -1]
    prediction = p[:, :-1].argmax(1)
    result = rejection_report(rows, classes, confidence, prediction, threshold)
    known = np.array([r['source_label'] in classes for r in rows])
    correct = np.array([classes[i] == r['source_label'] for i, r in zip(prediction, rows)])
    accepted = confidence >= threshold
    result['counts'] = {name: dict(successes=int(success[mask].sum()), total=int(mask.sum()))
        for name, mask, success in [('known_retention', known, accepted),
            ('known_correct_and_retained', known, accepted & correct),
            ('unknown_false_acceptance', ~known, accepted), ('unknown_rejection', ~known, ~accepted)]}
    result['threshold_fitted_role'] = 'calibration'
    result['threshold_unknown_source'] = 'kling'
    result['score_note'] = '1 - simulated_unknown softmax score is acceptance confidence, not calibrated origin probability'
    return result


def head_probabilities(head, features, device):
    with torch.inference_mode():
        return np.concatenate([head(x.to(device)).softmax(1).cpu().numpy() for x in features.split(128)])


def fit_head(train_x, train_y, selection_x, selection_rows, classes, out, device):
    """Accepts no calibration/evaluation data; reject predictions count as known selection errors."""
    checked_role(selection_rows, classes, 'selection', [])
    train_x, selection_x = normalized(train_x, DIMENSION), normalized(selection_x, DIMENSION)
    if (len(train_x) != len(train_y) or len(selection_x) != len(selection_rows)
            or train_y.dtype != torch.long or set(train_y.tolist()) != set(range(len(classes) + 1))):
        raise ValueError('training must contain all known classes and simulator unknown')
    torch.manual_seed(42)
    head = torch.nn.Linear(DIMENSION, len(classes) + 1).to(device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(42), num_workers=0)
    optimizer = torch.optim.Adam(head.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    criterion = torch.nn.CrossEntropyLoss()
    best_score, best_epoch, state = -1., None, None
    with (Path(out) / 'history.jsonl').open('x') as history:
        for epoch in range(1, 31):
            head.train(); total_loss = 0.
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(x.to(device)), y.to(device))
                loss.backward(); optimizer.step()
                total_loss += float(loss.detach()) * len(x)
            head.eval()
            score = selection_f1(selection_rows, classes, head_probabilities(head, selection_x, device))
            if score > best_score:
                best_score, best_epoch = score, epoch
                state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            history.write(json.dumps(dict(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'],
                train_loss=total_loss/len(train_x), selection_macro_f1=score)) + '\n'); history.flush()
            scheduler.step()
    head.load_state_dict(state, strict=True); head.eval()
    metadata = dict(schema='ofa-simulator-tplus1-linear-v1', classes=classes + [UNKNOWN],
        known_classes=classes, input_dimensions=DIMENSION, training_epochs=30,
        best_epoch=best_epoch, selection_macro_f1=best_score,
        selection_note='Fixed known labels; 21-way argmax unknown is an error; earliest best epoch')
    torch.save(dict(metadata, state_dict=state), Path(out) / 'head.pth')
    return head, metadata
