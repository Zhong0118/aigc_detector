"""Predeclared follow-up splits, corruptions and score-only evaluation.

Never opens the sealed holdout. Group assignment is independent of model scores.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import math
import random

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def partition_rows(rows, config):
    groups, ids = {}, set()
    for row in rows:
        if row['sample_id'] in ids or not row.get('group_id'):
            raise ValueError('duplicate sample or missing leakage group')
        ids.add(row['sample_id'])
        for field in ('group_id', 'sha256', 'pixel_sha256'):
            if not row.get(field):
                continue
            key = (field, row[field])
            if key in groups and groups[key] != row['split']:
                raise ValueError('identity crosses original splits')
            groups[key] = row['split']
    cal_unknown = set(config.get('unknown_calibration', []))
    eval_unknown = set(config.get('unknown_evaluation', []))
    if cal_unknown & eval_unknown:
        raise ValueError('unknown calibration/evaluation sources must differ')
    mode, task = config['mode'], config['task']
    all_sources = {r['source_label'] for r in rows if r['is_ai']}
    all_semantics = {r['semantic_class'] for r in rows}
    held_sources = set(config.get('held_sources', []))
    held_semantics = set(config.get('held_semantics', []))
    if not (cal_unknown | eval_unknown | held_sources) <= all_sources:
        raise ValueError('requested source is absent from manifest')
    if not held_semantics <= all_semantics:
        raise ValueError('requested semantic is absent from manifest')
    if mode == 'open_set' and (not cal_unknown or not eval_unknown):
        raise ValueError('open set requires separate unknown source lists')
    if mode == 'cross_source' and not held_sources:
        raise ValueError('missing held sources')
    if mode == 'cross_semantic' and not held_semantics:
        raise ValueError('missing held semantics')
    result = {name: [] for name in ('train', 'selection', 'calibration', 'evaluation')}
    for row in sorted(rows, key=lambda r: r['sample_id']):
        if row['split'] == 'holdout' or (task == 'attribution' and not row['is_ai']):
            continue
        if row['split'] == 'train':
            role = 'train'
        elif row['split'] == 'dev':
            # Whole groups go to one of three development roles, before scoring.
            key = f"{config['seed']}:{row['group_id']}".encode()
            role = ('selection', 'calibration', 'evaluation')[int(hashlib.sha256(key).hexdigest(), 16) % 3]
        else:
            raise ValueError('invalid parent split')
        source = row['source_label']
        if mode == 'open_set':
            if source in cal_unknown | eval_unknown:
                if not ((role == 'calibration' and source in cal_unknown) or
                        (role == 'evaluation' and source in eval_unknown)):
                    continue
        elif mode == 'cross_source':
            if row['is_ai'] and ((source in held_sources) != (role == 'evaluation')):
                continue
        elif mode == 'cross_semantic':
            if (row['semantic_class'] in held_semantics) != (role == 'evaluation'):
                continue
        result[role].append(dict(row))
    identities = {}
    for role, selected in result.items():
        for row in selected:
            for field in ('group_id', 'sha256', 'pixel_sha256'):
                if not row.get(field):
                    continue
                key = (field, row[field])
                if key in identities and identities[key] != role:
                    raise ValueError('identity crosses development roles')
                identities[key] = role
    return result


def validate_probabilities(scores, n, k):
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.shape != (n, k) or not np.isfinite(scores).all() or
            np.any(scores < 0) or np.any(scores > 1) or
            not np.allclose(scores.sum(1), 1, atol=1e-5)):
        raise ValueError('invalid probability matrix or class order/width')
    return scores


def choose_threshold(negative_scores, target_fpr):
    values = np.asarray(negative_scores, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or not 0 <= target_fpr < 1:
        raise ValueError('invalid calibration negatives or target FPR')
    # Prediction is score >= threshold. Move past ties; never exceed the budget.
    candidates = np.nextafter(np.unique(values), np.inf)
    return float(next(t for t in candidates if np.mean(values >= t) <= target_fpr))


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    center = (p + z*z/(2*total)) / (1+z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [max(0., center-half), min(1., center+half)]


def binary_report(labels, scores, threshold):
    truth, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if set(truth) != {0, 1} or scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError('both binary classes and finite scores required')
    pred = scores >= threshold
    neg, pos = truth == 0, truth == 1
    fp, tp = int(pred[neg].sum()), int(pred[pos].sum())
    return dict(n=len(truth), accuracy=float(accuracy_score(truth, pred)),
        macro_f1=float(f1_score(truth, pred, average='macro', zero_division=0)),
        auroc=float(roc_auc_score(truth, scores)), threshold=float(threshold),
        negative_count=int(neg.sum()), positive_count=int(pos.sum()),
        fpr=fp/int(neg.sum()), tpr=tp/int(pos.sum()),
        fpr_resolution=1/int(neg.sum()), fpr_wilson95=wilson(fp, int(neg.sum())),
        uncertainty_note='Wilson interval treats images as independent; correlated groups reduce effective sample size')


def threshold_stability(rows, scores, target_fpr, seed, repeats=200):
    """Bootstrap calibration groups only, never evaluation samples."""
    groups = defaultdict(list)
    for row, score in zip(rows, scores):
        if not row['is_ai']:
            groups[row['group_id']].append(float(score))
    keys = sorted(groups)
    if not keys:
        raise ValueError('no real calibration groups')
    rng = random.Random(seed)
    values = [choose_threshold([v for key in rng.choices(keys, k=len(keys)) for v in groups[key]], target_fpr)
              for _ in range(repeats)]
    return dict(repeats=repeats, calibration_negative_groups=len(keys),
                median=float(np.median(values)), p025=float(np.quantile(values, .025)),
                p975=float(np.quantile(values, .975)), method='group bootstrap of calibration negatives only')


def open_set_report(sources, classes, probabilities, acceptance_threshold):
    p = validate_probabilities(probabilities, len(sources), len(classes))
    known = np.array([s in classes for s in sources])
    if known.all() or not known.any():
        raise ValueError('known and unknown evaluation examples required')
    confidence = p.max(1)
    accepted = confidence >= acceptance_threshold
    correct = np.array([classes[i] == s for i, s in zip(p.argmax(1), sources)])
    return dict(n=len(sources), known_n=int(known.sum()), unknown_n=int((~known).sum()),
        unknown_auroc=float(roc_auc_score(~known, 1-confidence)),
        acceptance_threshold=float(acceptance_threshold),
        known_acceptance=float(accepted[known].mean()),
        known_closed_set_accuracy=float(correct[known].mean()),
        known_correct_and_accepted_rate=float((correct & accepted)[known].mean()),
        unknown_rejection_rate=float((~accepted)[~known].mean()),
        unknown_false_acceptance=float(accepted[~known].mean()),
        accepted_known_accuracy=float(correct[known & accepted].mean()) if (known & accepted).any() else None,
        score_note='1 - maximum softmax probability is a rejection score, not calibrated unknown probability')


def family_scores(classes, probabilities, mapping):
    for source in classes:
        item = mapping.get(source, {})
        if not item.get('family') or not item.get('evidence'):
            raise ValueError(f'unreviewed family mapping: {source}')
    names = sorted({mapping[s]['family'] for s in classes})
    p = validate_probabilities(probabilities, len(probabilities), len(classes))
    out = np.zeros((len(p), len(names)))
    for i, source in enumerate(classes):
        out[:, names.index(mapping[source]['family'])] += p[:, i]
    return names, out


class Postprocess:
    """Deterministic image-domain JPEG and downsample/restore before preprocessing."""
    def __init__(self, jpeg_quality=None, scale=1.):
        if jpeg_quality is not None and not 1 <= jpeg_quality <= 100:
            raise ValueError('invalid JPEG quality')
        if not 0 < scale <= 1:
            raise ValueError('invalid resize scale')
        self.jpeg_quality, self.scale = jpeg_quality, scale

    def __call__(self, image):
        result = image.convert('RGB').copy()
        if self.scale != 1:
            size = result.size
            result = result.resize(tuple(max(1, round(n*self.scale)) for n in size), Image.Resampling.BICUBIC)
            result = result.resize(size, Image.Resampling.BICUBIC)
        if self.jpeg_quality is not None:
            buffer = io.BytesIO()
            result.save(buffer, 'JPEG', quality=self.jpeg_quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                result = decoded.convert('RGB').copy()
        return result


class TrainingAugmentation:
    """One fixed D1 factor: independently apply JPEG and resize, probability .5 each."""
    def __call__(self, image):
        quality = random.choice([50, 75, 95]) if random.random() < .5 else None
        scale = random.choice([.5, .75]) if random.random() < .5 else 1.
        return Postprocess(quality, scale)(image)
