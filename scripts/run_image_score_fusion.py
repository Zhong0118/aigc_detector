#!/usr/bin/env python3
"""Calibrate a convex combination of existing attribution probabilities only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probabilities(value):
    value = np.asarray(value, dtype=float)
    if (value.ndim != 2 or not len(value) or value.shape[1] < 2 or
            not np.isfinite(value).all() or (value < 0).any() or (value > 1).any() or
            not np.allclose(value.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError('invalid probability matrix')
    return value


def mix_scores(first, second, alpha):
    first, second = probabilities(first), probabilities(second)
    if first.shape != second.shape or not 0 <= alpha <= 1:
        raise ValueError('incompatible scores or weight')
    return alpha * first + (1-alpha) * second


def fit_weight(truth, first, second):
    """The first model is ResNet. No evaluation scores/labels are accepted here."""
    first, second = probabilities(first), probabilities(second)
    truth = np.asarray(truth)
    if (truth.shape != (len(first),) or not np.issubdtype(truth.dtype, np.integer) or
            (truth < 0).any() or (truth >= first.shape[1]).any()):
        raise ValueError('invalid calibration labels')
    losses = []
    for alpha in np.linspace(0, 1, 21):
        p = mix_scores(first, second, float(alpha))
        loss = -np.log(np.clip(p[np.arange(len(truth)), truth], 1e-12, 1)).mean()
        losses.append(dict(alpha=float(alpha), calibration_nll=float(loss)))
    best = min(x['calibration_nll'] for x in losses)
    # Numerical ties favor the unchanged ResNet endpoint, before seeing evaluation.
    chosen = max(x['alpha'] for x in losses if np.isclose(x['calibration_nll'], best, atol=1e-12, rtol=0))
    return chosen, losses


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def cached_fusion(root, config, parts):
    import torch
    from research.image_attribution.feature_ablation import combine_features
    torch.set_num_threads(4)
    base = root / config['feature_run']
    checkpoint = torch.load(base/'A-Fusion/head.pth', map_location='cpu', weights_only=True)
    rgb = torch.load(base/'feature-cache/rgb.pt', map_location='cpu', weights_only=True)
    fp = torch.load(root/config['ofa_cache'], map_location='cpu', weights_only=True)
    manifest = config['manifest_sha256']
    if any(x['identity']['manifest_sha256'] != manifest for x in (rgb, fp)):
        raise ValueError('cache manifest mismatch')
    registered = read(root/config['research_config'])
    if (fp['identity']['weight_sha256'] != registered['assets']['fingerprint']['sha256'] or
            fp['identity']['denoiser_sha256'] != registered['assets']['denoiser']['sha256'] or
            fp['identity']['model_code_sha256'] != sha(root/'research/image_attribution/models.py') or
            fp['identity']['preprocessing'] != 'ofa-author-3780cbf-v1'):
        raise ValueError('OFA cache implementation or assets mismatch')
    lookups = [{sid: i for i, sid in enumerate(x['identity']['sample_ids'])} for x in (rgb, fp)]
    if any(len(lookup) != len(cache['features']) for lookup, cache in zip(lookups, (rgb, fp))):
        raise ValueError('cache contains missing or duplicate IDs')
    head = torch.nn.Linear(checkpoint['input_dimensions'], len(checkpoint['classes']))
    head.load_state_dict(checkpoint['state_dict'], strict=True)
    head.eval()
    scores = {}
    for role in ('calibration', 'evaluation'):
        ids = [row['sample_id'] for row in parts[role]]
        x = combine_features('fusion', rgb=rgb['features'][[lookups[0][sid] for sid in ids]],
                             fingerprint=fp['features'][[lookups[1][sid] for sid in ids]],
                             dimensions=registered['dimensions'])
        with torch.inference_mode():
            scores[role] = head(x).softmax(1).numpy()
    return checkpoint['classes'], scores


def run(root, config, config_path):
    from scripts.run_image_research_queue import load_protocol, verify_registered_files
    from scripts.compare_image_scale_epochs import align, indexed, stats, group_bootstrap
    registered = read(root/config['research_config'])
    verify_registered_files(root, registered)
    plan, all_parts = load_protocol(root, registered)
    if plan['manifest_sha256'] != config['manifest_sha256'] or not all_parts:
        raise ValueError('registered protocol is not ready')
    for name, digest in config['inputs_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('registered fusion input changed: ' + name)
    base = root/config['feature_run']
    if (read(base/'state.json')['status'] != 'completed' or
            read(base/'A-Fusion/status.json')['exit_code'] != 0 or
            sha(base/'A-Fusion/head.pth') != read(base/'A-Fusion/provenance.json')['head_sha256']):
        raise ValueError('fusion model is not certified complete')
    parts = all_parts['attribution']
    cal_groups = {x['group_id'] for x in parts['calibration']}
    if cal_groups & {x['group_id'] for x in parts['evaluation']}:
        raise ValueError('calibration/evaluation group overlap')
    classes, fusion = cached_fusion(root, config, parts)
    baseline = records(root/config['resnet_predictions'])
    vectors, truths = {}, {}
    for role in ('calibration', 'evaluation'):
        selected = indexed(parts[role])
        actual = [x for x in baseline if x['sample_id'] in selected]
        ids, y, p, _, vocabulary = align(selected, actual, actual, 'A0')
        if vocabulary != classes:
            raise ValueError('source vocabulary mismatch')
        order = {row['sample_id']: i for i, row in enumerate(parts[role])}
        fusion[role] = fusion[role][[order[sid] for sid in ids]]
        vectors[role], truths[role] = p, y
    saved = records(base/'A-Fusion/predictions.jsonl')
    ev = indexed(parts['evaluation'])
    ids, y, old_fusion, _, _ = align(ev, saved, saved, 'A0')
    error = float(np.abs(old_fusion-fusion['evaluation']).max())
    if error > 1e-5:
        raise ValueError('reused OFA cache does not reproduce saved fusion predictions')
    alpha, grid = fit_weight(truths['calibration'], vectors['calibration'], fusion['calibration'])
    scores = dict(ResNet50=vectors['evaluation'], FeatureFusion=fusion['evaluation'],
                  Uniform=mix_scores(vectors['evaluation'], fusion['evaluation'], .5),
                  Calibrated=mix_scores(vectors['evaluation'], fusion['evaluation'], alpha))
    out = root/config['output']
    out.mkdir(parents=True, exist_ok=False)
    try:
        write(out/'resolved_config.json', dict(config=config, config_sha256=sha(config_path)))
        write(out/'calibration.json', dict(alpha_resnet=alpha, alpha_feature_fusion=1-alpha,
              objective='calibration negative log likelihood', grid=grid, calibration_n=len(truths['calibration']),
              tie_break='larger ResNet weight within absolute NLL tolerance 1e-12'))
        write(out/'role_ids.json', {k: [x['sample_id'] for x in parts[k]] for k in ('calibration', 'evaluation')})
        report = {name: stats(y, p, 'A0', classes) for name, p in scores.items()}
        comparisons = {name: group_bootstrap(y, p, scores['Calibrated'], [ev[sid]['group_id'] for sid in ids],
                        repeats=1000, seed=42) for name, p in scores.items() if name != 'Calibrated'}
        for name, p in scores.items():
            with (out/(name+'-predictions.jsonl')).open('x') as f:
                for sid, label, values in zip(ids, y, p):
                    f.write(json.dumps(dict(sample_id=sid, split='dev', role='evaluation', status='ok',
                        true_label=classes[int(label)], classes=classes, scores=values.tolist()))+'\n')
        result = dict(status='completed', metrics=report, alpha_resnet=alpha, paired_group_bootstrap=comparisons,
            evaluation_n=len(ids), calibration_n=len(truths['calibration']), cache_score_max_error=error,
            holdout_scored=False, backbone_trained=False,
            limitation='Exploratory parent-dev comparison; ResNet used parent dev for checkpoint selection; single seed; no unknown rejection claim')
        write(out/'summary.json', result)
        write(out/'provenance.json', dict(inputs_sha256=config['inputs_sha256'], config_sha256=sha(config_path),
              script_sha256=sha(Path(__file__)), model_manifest_sha256=plan['manifest_sha256']))
        write(out/'status.json', dict(status='completed', exit_code=0))
        return result
    except Exception as error:
        write(out/'failed.json', dict(status='failed', error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    if not args.execute:
        print(json.dumps(dict(status='registered', config=config), ensure_ascii=False))
        return
    print(json.dumps(run(ROOT, config, args.config), ensure_ascii=False))


if __name__ == '__main__':
    main()
