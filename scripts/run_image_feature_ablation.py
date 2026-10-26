#!/usr/bin/env python3
"""Prepare-only by default. This entry is deliberately absent from all supervisors."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.image_attribution.feature_ablation import (
    DIMENSIONS, EXTRACT_ROLES, code_identity, combine_features, fit_head, freeze_encoder,
    load_bound_features, prepare, require_execution, verify_runtime,
)
from research.image_attribution.protocol import sha256, write_json


def branch_features(root, data, rows, classes, config, plan, branch, cache_dir, device):
    import torch
    from torch.utils.data import DataLoader
    from research.image_attribution.models import ResNetClassifier, resnet_transform, load_ofa, ofa_transform, ofa_features
    from scripts.run_image_experiment import Images

    weights = {key: value['sha256'] for key, value in config['assets'].items()
               if key in (('rgb',) if branch == 'rgb' else ('fingerprint', 'denoiser'))}
    identity = dict(schema='frozen-source-cache-v1', branch=branch, dtype='float32',
                    manifest_sha256=plan['manifest_sha256'], selection_sha256=plan['selection_sha256'],
                    sample_ids=[row['sample_id'] for row in rows], weights=weights,
                    preprocessing=config['preprocessing'][branch], code_sha256=code_identity(root))
    cache = cache_dir / (branch + '.pt')
    if cache.exists():
        return load_bound_features(cache, identity, DIMENSIONS[branch]), identity
    for row in rows:
        if row['split'] == 'holdout' or not row['is_ai']:
            raise ValueError('forbidden extraction member')
        if sha256(data / row['relative_path']) != row['sha256']:
            raise ValueError('image hash mismatch: ' + row['sample_id'])
    assets = config['assets']
    if branch == 'rgb':
        encoder = freeze_encoder(ResNetClassifier(len(classes), root / assets['rgb']['path']).backbone.to(device))
        transform = resnet_transform()
        denoiser = pair = None
    else:
        encoder, pair, denoiser = load_ofa(root / assets['fingerprint']['path'], root / assets['denoiser']['path'], device)
        freeze_encoder(encoder)
        freeze_encoder(denoiser)
        transform = ofa_transform
    labels = [classes.index(row['source_label']) for row in rows]
    loader = DataLoader(Images(rows, labels, data, transform), batch_size=config['extraction_batch_size'][branch],
                        shuffle=False, num_workers=0)
    batches = []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            result = encoder(images) if branch == 'rgb' else ofa_features(images, encoder, denoiser)
            batches.append(result.float().cpu())
    features = torch.cat(batches)
    from research.image_attribution.feature_ablation import checked_features
    checked_features(features, DIMENSIONS[branch])
    temporary = cache.with_suffix('.pt.partial')
    if temporary.exists():
        raise FileExistsError('unfinished cache retained: ' + str(temporary))
    with temporary.open('xb') as stream:
        torch.save(dict(identity=identity, features=features), stream)
    temporary.replace(cache)
    return features, identity


def run(root, config, dataset, variant, plan, parts, classes, config_hash):
    # Configuration permission and all metadata checks precede locks/GPU/output creation.
    verify_runtime(root, config, plan)
    lock_path = root / 'outputs/image_experiments/gpu0-research.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_index'])
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('registered CUDA GPU unavailable; no silent CPU fallback')
        device = torch.device('cuda:0')
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        base = root / 'outputs/image_experiments' / ('feature-ablation-' + dataset)
        out = base / variant
        out.mkdir(parents=True, exist_ok=False)
        try:
            write_json(out / 'resolved_config.json', dict(config=config, config_sha256=config_hash,
                       dataset=dataset, variant=variant, plan=plan, physical_gpu=config['gpu_index']))
            write_json(out / 'role_ids.json', {role: [row['sample_id'] for row in selected]
                                               for role, selected in parts.items()})
            write_json(out / 'status.json', dict(status='extracting', time=time.time()))
            rows = [row for role in EXTRACT_ROLES for row in parts[role]]
            cache_dir = base / 'feature-cache'
            cache_dir.mkdir(exist_ok=True)
            branches = ('rgb', 'fingerprint') if variant == 'fusion' else (variant,)
            features, identities = {}, {}
            for branch in branches:
                features[branch], identities[branch] = branch_features(
                    root, root / config['datasets'][dataset]['path'], rows, classes,
                    config, plan, branch, cache_dir, device)
            joined = combine_features(variant, **features)
            provenance = dict(manifest_sha256=plan['manifest_sha256'], selection_sha256=plan['selection_sha256'],
                              config_sha256=config_hash, code_sha256=code_identity(root),
                              cache_identities=identities, role_ids_sha256=plan['role_ids_sha256'],
                              seed=config['seed'], gpu_index=config['gpu_index'])
            fit_head(joined, parts, classes, config, out, device, variant, provenance)
        except Exception as error:
            write_json(out / 'status.json', dict(status='failed', exit_code=1,
                       error_type=type(error).__name__, error=str(error), time=time.time()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/image_experiments/iab_feature_ablation.json'))
    parser.add_argument('--dataset', choices=['v5-50', 'v6-100'], required=True)
    parser.add_argument('--variant', choices=list(DIMENSIONS), required=True)
    parser.add_argument('--action', choices=['plan', 'run'], default='plan')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if args.action == 'run':
        require_execution(config, args.execute)
    plan, parts, classes = prepare(ROOT, config, args.dataset)
    plan.update(variant=args.variant, input_dimensions=DIMENSIONS[args.variant],
                epochs=config['training']['epochs'], automatic_execution=False)
    if args.action == 'plan':
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    run(ROOT, config, args.dataset, args.variant, plan, parts, classes, sha256(config_path))


if __name__ == '__main__':
    main()
