"""Fixed, auditable development splits and separate image task definitions."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def assign_splits(rows, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    strata = defaultdict(list)
    for group, members in sorted(groups.items()):
        strata[tuple(sorted({str(r['source_label']) for r in members}))].append(group)
    mapping = {}
    rng = random.Random(seed)
    for _, ids in sorted(strata.items()):
        rng.shuffle(ids)
        n_train, n_dev = int(.7 * len(ids)), int(.15 * len(ids))
        for i, group in enumerate(ids):
            mapping[group] = 'train' if i < n_train else 'dev' if i < n_train + n_dev else 'holdout'
    return [dict(row, split=mapping[row['group_id']]) for row in rows]


def validate_manifest(rows, root, verify_files=True):
    root = Path(root).resolve()
    ids, seen = set(), {}
    for row in rows:
        if row['sample_id'] in ids:
            raise ValueError('duplicate sample_id')
        ids.add(row['sample_id'])
        if row['split'] not in {'train', 'dev', 'holdout'}:
            raise ValueError('invalid split')
        if row['is_ai'] not in (0, 1) or (row['is_ai'] == 1 and not row['source_label']):
            raise ValueError('unverified task label')
        path = Path(row['relative_path'])
        if path.is_absolute() or not (root / path).resolve().is_relative_to(root):
            raise ValueError('unsafe relative path')
        for key in ('group_id', 'sha256', 'pixel_sha256'):
            if not row.get(key):
                continue
            identity = (key, row[key])
            if identity in seen and seen[identity] != row['split']:
                raise ValueError(f'{key} crosses splits')
            seen[identity] = row['split']
        if verify_files and sha256(root / path) != row['sha256']:
            raise ValueError(f'file hash mismatch: {row["sample_id"]}')
    return rows


def load_manifest(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return validate_manifest(rows, path.parent)


def task_rows(rows, task):
    if task == 'detection':
        return rows, [r['is_ai'] for r in rows], ['real', 'ai']
    if task != 'attribution':
        raise ValueError('unsupported task')
    selected = [r for r in rows if r['is_ai'] == 1]
    classes = sorted({r['source_label'] for r in selected})
    return selected, [classes.index(r['source_label']) for r in selected], classes
