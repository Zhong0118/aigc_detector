#!/usr/bin/env python3
"""Separate research detection and attribution outputs; does not change the live API."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_completed_run(run):
    from research.image_attribution.protocol import sha256
    status = json.loads((run / 'status.json').read_text())
    if status.get('status') != 'completed' or status.get('exit_code') != 0:
        raise ValueError('run has not completed model and evaluation verification')
    provenance = json.loads((run / 'provenance.json').read_text())
    if sha256(run / 'head.pth') != provenance['head_sha256']:
        raise ValueError('head hash differs from completed provenance')
    return provenance


def task_output(task, classes, scores, threshold=.5, top_k=5):
    import numpy as np
    p = np.asarray(scores, dtype=float)
    if p.shape != (len(classes),) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError('invalid classifier scores')
    if task == 'detection':
        if classes != ['real', 'ai'] or not (np.isfinite(threshold) and threshold >= 0):
            raise ValueError('invalid detection vocabulary or threshold')
        return dict(task='detection', score_ai=float(p[1]), threshold=threshold,
                    label='ai' if p[1] >= threshold else 'real',
                    score_type='binary_softmax_uncalibrated', source_scores_used=False)
    if task != 'attribution' or 'real' in classes:
        raise ValueError('invalid attribution task or vocabulary')
    order = np.argsort(-p, kind='stable')[:top_k]
    return dict(task='attribution', conditional_on_ai=True,
                top_k=[dict(source=classes[i], score=float(p[i])) for i in order],
                score_type='closed_set_softmax_uncalibrated', label_level='publisher_directory',
                unknown=dict(status='not_evaluated', decision=None),
                note='Candidate source ranking is not evidence that this image is AI generated. Exact backend versions are not inferred.')


def predict(run_directory, images, device='cpu', batch_size=8, target_fpr=None, top_k=5):
    import torch
    from PIL import Image
    from research.image_attribution.feature_ablation import combine_features
    from research.image_attribution.models import load_ofa, ofa_features, ofa_transform
    from research.image_attribution.vision_backbones import load_dinov2
    from scripts.run_image_research_queue import verify_registered_files
    run = Path(run_directory)
    provenance = verify_completed_run(run)
    resolved = json.loads((run / 'resolved_config.json').read_text())
    if provenance['config_sha256'] != resolved['config_sha256']:
        raise ValueError('resolved configuration differs from completed provenance')
    cfg, job = resolved['config'], resolved['job']
    verify_registered_files(ROOT, cfg)
    saved = torch.load(run / 'head.pth', map_location='cpu', weights_only=True)
    if saved.get('task', 'attribution') != job['task'] or saved['variant'] != job['variant']:
        raise ValueError('head task does not match registered job')
    classes = saved['classes']
    expected_classes = ['real', 'ai'] if job['task'] == 'detection' else cfg['classes']
    if classes != expected_classes or saved['input_dimensions'] != cfg['dimensions'][job['variant']]:
        raise ValueError('head classes or dimensions differ from registered configuration')
    threshold = .5
    operating_point = None
    if target_fpr is not None and job['task'] == 'detection':
        report = json.loads((run / 'metrics.json').read_text())
        operating_point = report['calibrated'][str(target_fpr)]
        threshold = operating_point['evaluation']['threshold']
    head = torch.nn.Linear(saved['input_dimensions'], len(classes)).to(device)
    head.load_state_dict(saved['state_dict'], strict=True)
    head.eval()
    branches = ('rgb', 'fingerprint') if job['variant'] == 'fusion' else (job['variant'],)
    if 'rgb' in branches:
        rgb, transform, _ = load_dinov2(ROOT / cfg['dinov2_directory'], device)
    if 'fingerprint' in branches:
        fp, _, denoiser = load_ofa(ROOT / cfg['assets']['fingerprint']['path'], ROOT / cfg['assets']['denoiser']['path'], device)
    result = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            selected = images[start:start + batch_size]
            pixels = []
            for path in selected:
                with Image.open(path) as image:
                    pixels.append(image.convert('RGB'))
            values = {}
            if 'rgb' in branches:
                values['rgb'] = rgb(torch.stack([transform(x) for x in pixels]).to(device))
            if 'fingerprint' in branches:
                values['fingerprint'] = ofa_features(torch.stack([ofa_transform(x) for x in pixels]).to(device), fp, denoiser)
            features = combine_features(job['variant'], **values, dimensions=cfg['dimensions'])
            scores = head(features).softmax(1).cpu().numpy()
            result.extend(dict(image=str(path), **task_output(job['task'], classes, p, threshold=threshold, top_k=top_k)) for path, p in zip(selected, scores))
    if operating_point is not None:
        for row in result:
            row['operating_point'] = dict(calibration_target_fpr=target_fpr,
                calibration_negative_n=operating_point['calibration_negative_n'],
                calibration_actual_fpr=operating_point['calibration_fpr'],
                note='Empirical calibration target; no guarantee on arbitrary new images.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detection-run')
    parser.add_argument('--attribution-run')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--detection-target-fpr', type=float, choices=[.01, .05])
    args = parser.parse_args()
    if not (args.detection_run or args.attribution_run) or args.batch_size < 1:
        parser.error('at least one completed run and positive batch size are required')
    outputs = {task: predict(directory, args.images, batch_size=args.batch_size, target_fpr=args.detection_target_fpr)
               for task, directory in [('detection', args.detection_run), ('attribution', args.attribution_run)] if directory}
    for task, records in outputs.items():
        if any(row['task'] != task for row in records):
            raise ValueError('run passed to incorrect task argument')
    print(json.dumps(dict(schema='separate-image-tasks-v1', **outputs), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
