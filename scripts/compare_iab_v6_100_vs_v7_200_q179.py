#!/usr/bin/env python3
"""Read-only same-sample comparison: saved IAB 100/30 versus 200-q179/30 results.

No training, image reads, GPU use or holdout scoring. The original v6 sample IDs,
roles, labels and original group IDs fix every paired denominator; the new full
dev and new evaluation results are reported separately without deltas. The output
directory is exclusive and existing reports or model outputs are never overwritten.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_image_scale_epochs import (Evidence, align, auc, indexed, stats, group_bootstrap,
    load_manifest, load_run, dev_rows, sha256, require, verify_extension, PROTOCOL_FIELDS,
    confusion, f1_from_cm, top3_hits)
from scripts.run_image_research_queue import load_protocol, verify_registered_files
from scripts.run_image_score_fusion import fit_weight, mix_scores, cached_fusion
from research.image_attribution.followups import choose_threshold, binary_report, threshold_stability, wilson

ROLES = ('train', 'selection', 'calibration', 'evaluation')
RUNS = (('D0', 'd0-resnet50'), ('A0', 'a0-resnet50'), ('OFA', 'ofa'))
FEATURES = ('D-DINO', 'A-RGB', 'A-FP', 'A-Fusion')
FUSION_METHODS = ('ResNet50', 'FeatureFusion', 'Uniform', 'Calibrated')
TARGETS = (.01, .05)
REPEATS, SEED = 1000, 42
OLD = dict(name='100/30', suffix='v6-100-e30', data='outputs/image_experiments/iab-balanced-v6-100',
    manifest_sha='e8152a198db5b39481482cab18be9a56d173ba6a25dc330bc40cf1fe7cd4b06e',
    selection_sha='286a3c34f5291a355259cfed7170e9b5a6d9d17ddf483f6d32c535cecc4f64b6',
    research='configs/image_experiments/iab_research_v6_100_e30.json',
    research_sha='71f68509698de5a39210841135c8e8b01c7f5df5557b646791dcac6d506580fc',
    fusion='configs/image_experiments/iab_score_fusion_v6_100_e30.json',
    monitor='outputs/iab-v6-100-e30-monitor', samples=42000, train=dict(D0=25446, A0=11456),
    dev=dict(D0=4960, A0=1968), fusion_n=dict(calibration=757, evaluation=606),
    roles=dict(detection=dict(train=25446, selection=1604, calibration=1771, evaluation=1585),
               attribution=dict(train=11456, selection=605, calibration=757, evaluation=606)))
NEW = dict(name='200-q179/30', suffix='v7-200-q179-e30', data='outputs/image_experiments/iab-balanced-v7-200-q179',
    manifest_sha='9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3',
    selection_sha='1c8dee2e17cc128867562186cee7bc4f8cb82042b26ddb6d5569d703ab77876b',
    research='configs/image_experiments/iab_research_v7_200_q179_e30.json',
    research_sha='e293747ef188eb48ace4270fa3158d95d0b04c04d14616164f028497d0b148c9',
    fusion='configs/image_experiments/iab_score_fusion_v7_200_q179_e30.json',
    monitor='outputs/iab-v7-200-q179-e30-monitor', samples=63821, train=dict(D0=37338, A0=23348),
    dev=dict(D0=7189, A0=4197), fusion_n=dict(calibration=1554, evaluation=1331),
    roles=dict(detection=dict(train=37338, selection=2311, calibration=2568, evaluation=2310),
               attribution=dict(train=23348, selection=1312, calibration=1554, evaluation=1331)))
PINNED = {'scripts/compare_image_scale_epochs.py': '752764997b36a0b664170251df9ac0494de45fb692b7d1b0b705b7e5a6e53e4e',
          'scripts/run_image_score_fusion.py': '7377b90a5e27a0cfac1a3063bd7e291aa0f633457cdebd427110d52e0c829add'}
DEFAULT_OUTPUT = 'outputs/image-followups-20261028/v6-100-e30-vs-v7-200-q179-e30'


def verify_preserved_roles(old_parts, new_parts):
    """Every original role ID keeps its role, split, label and group in the extended plan."""
    counts = {}
    for task, roles in old_parts.items():
        lookup = {r['sample_id']: (role, r) for role in ROLES for r in new_parts[task][role]}
        for role in ROLES:
            for before in roles[role]:
                sid = before['sample_id']
                require(sid in lookup, f'original {task} ID missing: {sid}')
                after_role, after = lookup[sid]
                require(after_role == role, f'original {task} role changed: {sid}')
                require(all(before[k] == after[k] for k in ('split', 'group_id', 'is_ai', 'source_label')),
                        f'original {task} identity changed: {sid}')
        counts[task] = {role: len(roles[role]) for role in ROLES}
    return counts


def reference_diff(old, new):
    require(set(old) == set(new), 'OFA reference vocabulary changed')
    rows = []
    for source in sorted(old):
        a, b = set(old[source]), set(new[source])
        rows.append(dict(source=source, old_n=len(a), new_n=len(b), shared=len(a & b),
                         removed=len(a - b), added=len(b - a)))
    return rows


def _group_index(groups):
    names = sorted(set(groups))
    lookup = {g: i for i, g in enumerate(names)}
    return names, np.array([lookup[g] for g in groups])


def threshold_bootstrap(binary, p, q, tp, tq, groups, repeats=REPEATS, seed=SEED):
    """Paired original-group draws of FPR/TPR at each model's fixed calibration threshold."""
    binary = np.asarray(binary, dtype=bool); p = np.asarray(p, float); q = np.asarray(q, float)
    require(len(binary) == len(p) == len(q) == len(groups) and binary.any() and (~binary).any(),
            'threshold bootstrap requires aligned binary samples')
    names, index = _group_index(groups)
    predictions = (p >= tp, q >= tq)
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[index]
        pos, neg = weights[binary].sum(), weights[~binary].sum()
        require(pos > 0 and neg > 0, 'bootstrap missing binary class')
        values = [[weights[~binary & pred].sum() / neg, weights[binary & pred].sum() / pos] for pred in predictions]
        deltas.append(np.subtract(values[1], values[0]))
    bounds = np.quantile(deltas, [.025, .975], axis=0)
    return dict(group_n=len(names), repeats=repeats, seed=seed,
                delta_95_ci=dict(fpr=bounds[:, 0].tolist(), tpr=bounds[:, 1].tolist()),
                note='Conditional on each model threshold fitted on calibration real images; '
                     'threshold refit uncertainty is reported separately')


def detection_bootstrap(y, p, q, groups, classes, repeats=REPEATS, seed=SEED):
    """Paired group bootstrap for accuracy, macro-F1, weighted AUROC and FPR/TPR at 0.5."""
    base = group_bootstrap(y, p, q, groups, repeats=repeats, seed=seed)
    ci = {'accuracy': base['delta_95_ci']['top1'], 'macro_f1': base['delta_95_ci']['macro_f1']}
    ai = classes.index('ai'); truth = y == ai
    names, index = _group_index(groups)
    preparations = []
    for scores in (p[:, ai], q[:, ai]):
        order = np.argsort(scores, kind='stable')
        starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
        preparations.append((order, starts, scores >= .5))
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[index]
        pos, neg = weights[truth].sum(), weights[~truth].sum()
        require(pos > 0 and neg > 0, 'bootstrap missing binary class')
        pair = []
        for order, starts, pred in preparations:
            wp = np.add.reduceat(weights[order] * truth[order], starts)
            wn = np.add.reduceat(weights[order] * (~truth[order]), starts)
            weighted_auc = float((wp * (np.cumsum(wn) - wn + .5 * wn)).sum() / (pos * neg))
            pair.append([weighted_auc, float(weights[~truth & pred].sum() / neg),
                         float(weights[truth & pred].sum() / pos)])
        deltas.append(np.array(pair[1]) - pair[0])
    bounds = np.quantile(deltas, [.025, .975], axis=0)
    ci.update({name: bounds[:, i].tolist() for i, name in enumerate(('auroc', 'fpr_0_5', 'tpr_0_5'))})
    return dict(group_n=len(names), repeats=repeats, seed=seed, method=base['method'], delta_95_ci=ci)


def source_rows(name, y, p, q, classes, groups):
    f1a, f1b = f1_from_cm(confusion(y, p)), f1_from_cm(confusion(y, q)); rows = []
    for i, source in enumerate(classes):
        ix = y == i
        row = dict(comparison=name, source=source, n=int(ix.sum()),
                   group_n=len({groups[j] for j in np.flatnonzero(ix)}))
        if ix.any():
            for field, a, b in [('top1', float((p[ix].argmax(1) == i).mean()), float((q[ix].argmax(1) == i).mean())),
                                ('top3', float(top3_hits(y[ix], p[ix]).mean()), float(top3_hits(y[ix], q[ix]).mean())),
                                ('f1', float(f1a[i]), float(f1b[i]))]:
                row.update({'before_' + field: a, 'after_' + field: b, 'delta_' + field: b - a})
        rows.append(row)
    return rows


def worker_alive(state):
    """Same PID and kernel start time; a recycled PID is not the recorded worker."""
    try:
        start = Path(f'/proc/{state["pid"]}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, KeyError, IndexError):
        return False
    return state.get('identity') is None or str(state['identity'][1]) == start


def restrict(predictions, rows):
    return [x for x in predictions if x['sample_id'] in rows]


def csv_text(rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    f = io.StringIO(); w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
    return f.getvalue()


def dump(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def check_saved_metrics(saved, computed):
    for k, v in computed.items():
        require(k in saved and np.isclose(v, saved[k], atol=1e-8, rtol=0), 'saved metric mismatch: ' + k)


def analyze():
    E = Evidence()

    def bind(path, expected=None):
        p = ROOT / path
        value = sha256(p)
        require(expected is None or value == expected, 'input SHA mismatch: ' + str(path))
        E.inputs[str(p.resolve())] = value
        return value

    def read(path, lines=False):
        return E.read(ROOT / path, lines=lines)

    require(ROOT.name == 'attribution-pilot-baseline', 'unexpected research tree')
    for name, digest in PINNED.items():
        bind(name, digest)
    versions = {}
    for spec in (OLD, NEW):
        manifest_path = ROOT / spec['data'] / 'manifest.jsonl'
        bind(manifest_path, spec['manifest_sha']); bind(Path(spec['data']) / 'selection.json', spec['selection_sha'])
        manifest = load_manifest(manifest_path, E)
        require(len(manifest) == spec['samples'], 'manifest size changed')
        require(len(dev_rows(manifest, 'D0')) == spec['dev']['D0'] and
                len(dev_rows(manifest, 'A0')) == spec['dev']['A0'], 'dev counts changed')
        cfg = read(spec['research']); bind(spec['research'], spec['research_sha'])
        verify_registered_files(ROOT, cfg)
        for name, digest in cfg['implementation_sha256'].items():
            bind(name, digest)
        for asset in cfg['assets'].values():
            bind(asset['path'], asset['sha256'])
        bind(cfg['parent_manifest']['path'], cfg['parent_manifest']['sha256'])
        plan, parts = load_protocol(ROOT, cfg)
        require(plan['status'] == 'ready' and plan['manifest_sha256'] == spec['manifest_sha'] and
                plan['roles'] == spec['roles'], 'role plan differs from registered counts')
        for task in ('detection', 'attribution'):
            sets = [{r['group_id'] for r in parts[task][role]} for role in ROLES]
            require(not any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)), 'role group overlap')
        monitor = Path(spec['monitor'])
        completion = read(monitor / 'completion.json'); validated = read(monitor / 'validated-summary.json')
        summary = read(completion['summary']); pipeline = read(monitor / 'pipeline.json')
        require(all(x['status'] == 'completed' for x in (completion, validated, summary, pipeline)) and
                pipeline['exit_code'] == 0, 'baseline completion chain invalid')
        require(completion['manifest_sha256'] == validated['manifest_sha256'] == summary['manifest_sha256'] ==
                spec['manifest_sha'], 'baseline certification SHA mismatch')
        if 'summary_sha256' in completion:
            require(completion['summary_sha256'] == sha256(ROOT / completion['summary']) ==
                    sha256(ROOT / monitor / 'validated-summary.json'), 'baseline summary seal mismatch')
        versions[spec['name']] = dict(spec=spec, manifest=manifest, cfg=cfg, parts=parts, plan=plan, runs={},
            features={}, certification=dict(status='verified', monitor=str(monitor),
            scope=completion.get('completion_scope', 'baseline'), pipeline_exit_code=pipeline['exit_code']))
    old, new = versions[OLD['name']], versions[NEW['name']]
    preserved = verify_extension(old['manifest'], new['manifest'])
    role_counts = verify_preserved_roles(old['parts'], new['parts'])
    real_cal = [{r['sample_id'] for r in v['parts']['detection']['calibration'] if not r['is_ai']} for v in (old, new)]
    require(real_cal[0] == real_cal[1] and len(real_cal[0]) == 1014, 'calibration real images changed')

    epochs = []

    def collect_epochs(model, version, history, best, scope):
        for entry in history:
            row = dict(model=model, version=version, epoch=entry['epoch'], selected=entry['epoch'] == best,
                       selection_scope=scope)
            for key in ('lr', 'learning_rate', 'seconds'):
                if key in entry:
                    row[key] = entry[key]
            for role in ('train', 'dev', 'selection'):
                for field in ('n', 'loss', 'accuracy', 'macro_f1', 'auroc', 'top1', 'top3', 'fpr_0_5', 'tpr_0_5'):
                    if field in entry.get(role, {}):
                        row[role + '_' + field] = entry[role][field]
            epochs.append(row)

    training = {}
    for name, v in versions.items():
        spec = v['spec']; training[name] = {}
        for task, prefix in RUNS:
            folder = ROOT / f'outputs/image_experiments/{prefix}-iab-{spec["suffix"]}'
            item = load_run(folder, v['manifest'], spec['manifest_sha'], task, 30, E)
            require(item['status'] == 'verified', 'baseline not verified: ' + str(folder))
            for asset, record in item['declared_model_assets'].items():
                bind(folder / asset, record['sha256'])
            v['runs'][task] = item
            if task != 'OFA':
                require(all(h['train']['n'] == spec['train'][task] for h in item['history']), 'baseline training size')
                metrics = read(folder / 'metrics.json')
                training[name][task] = dict(train_n=spec['train'][task], selected_epoch=item['selected_epoch'],
                    train_counts=metrics.get('train_counts') or item['history'][0].get('train_class_counts'),
                    class_weights=metrics.get('class_weights'), elapsed_seconds=item['elapsed_seconds'])
                collect_epochs(task, name, item['history'], item['selected_epoch'], 'own_full_dev')
    for task, _ in RUNS:
        a, b = old['runs'][task]['config'], new['runs'][task]['config']
        require(all(a[k] == b[k] for k in PROTOCOL_FIELDS), 'baseline training protocol changed: ' + task)

    for name, v in versions.items():
        spec, cfg = v['spec'], v['cfg']
        base = Path(cfg['output'])
        state = read(base / 'state.json'); root_summary = read(base / 'summary.json')
        bound = read(base / 'bound-protocol.json')
        require(state['status'] == 'completed' and state['exit_code'] == 0 and
                state['config_sha256'] == spec['research_sha'], 'feature root not complete: ' + name)
        require(not worker_alive(state), 'feature worker still active')
        require(bound['manifest_sha256'] == spec['manifest_sha'], 'bound protocol manifest mismatch')
        v['feature_cache_sha256'] = {b: bind(base / 'feature-cache' / f'{b}.pt') for b in ('rgb', 'fingerprint')}
        for job in cfg['jobs']:
            job_name = job['id']; folder = base / job_name; task = 'D0' if job['task'] == 'detection' else 'A0'
            status = read(folder / 'status.json'); resolved = read(folder / 'resolved_config.json')
            metrics = read(folder / 'metrics.json'); provenance = read(folder / 'provenance.json')
            history = read(folder / 'history.jsonl', True); preds = read(folder / 'predictions.jsonl', True)
            roleids = read(folder / 'role_ids.json'); check = read(folder / 'image-inference-check.json')
            require(status['status'] == 'completed' and status['exit_code'] == 0, 'feature status invalid ' + job_name)
            require(resolved['config_sha256'] == spec['research_sha'] and resolved['config']['task'] == job['task'],
                    'feature resolved config mismatch')
            require(provenance['config_sha256'] == spec['research_sha'] and
                    provenance['manifest_sha256'] == spec['manifest_sha'] and
                    provenance['selection_sha256'] == spec['selection_sha'], 'feature provenance identity mismatch')
            bind(folder / 'head.pth', provenance['head_sha256'])
            require(provenance['encoder_training'] is False and provenance['head_training'] is True,
                    'feature training range changed')
            require([h['epoch'] for h in history] == list(range(1, 31)), 'incomplete feature history')
            rolemap = v['parts'][job['task']]
            require(roleids == {r: [x['sample_id'] for x in rows] for r, rows in rolemap.items()}, 'role IDs mismatch')
            require(check['role'] == 'selection' and check['sample_ids'] == roleids['selection'][:2] and
                    check['holdout_scored'] is False, 'image check role mismatch')
            require(check['cached_vs_image_max_error'] <= 1e-5 and check['image_single_batch_max_error'] <= 1e-5,
                    'failed image consistency check')
            require(metrics['holdout_scored'] is False and metrics['image_inference_verified'] is True,
                    'feature inference certification missing')
            criterion = 'auroc' if task == 'D0' else 'macro_f1'
            best = max(history, key=lambda h: h['selection'][criterion])['epoch']
            require(best == metrics['best_epoch'] == status['best_epoch'], 'feature best epoch mismatch')
            require(all(h['train']['n'] == len(rolemap['train']) and h['selection']['n'] == len(rolemap['selection'])
                        for h in history), 'history role size mismatch')
            ids, y, p, _, classes = align(indexed(rolemap['evaluation']), preds, preds, task)
            computed = stats(y, p, task, classes); check_saved_metrics(metrics, computed)
            require(metrics == root_summary['results'][job_name], 'root/job metric mismatch')
            item = dict(status='verified', path=str(folder), selected_epoch=best, metrics=computed,
                        head_sha256=provenance['head_sha256'], predictions=preds, history=history)
            if task == 'D0':
                calpred = read(folder / 'calibration/predictions.jsonl', True)
                ci, cy, cp, _, cc = align(indexed(rolemap['calibration']), calpred, calpred, 'D0')
                require(cc == classes, 'detection calibration vocabulary')
                item['calibration'] = dict(ids=ci, y=cy, p=cp, classes=cc)
            v['features'][job_name] = item
            collect_epochs(job_name, name, history, best, 'selection_role_only')

    fusion = {}
    for name, v in versions.items():
        spec = v['spec']; cfg = read(spec['fusion']); folder = Path(cfg['output'])
        result = read(folder / 'summary.json'); status = read(folder / 'status.json')
        provenance = read(folder / 'provenance.json'); resolved = read(folder / 'resolved_config.json')
        calibration = read(folder / 'calibration.json'); roleids = read(folder / 'role_ids.json')
        require(result['status'] == 'completed' and status['exit_code'] == 0 and result['holdout_scored'] is False and
                result['backbone_trained'] is False, 'fusion not complete: ' + name)
        require(resolved['config'] == cfg and resolved['config_sha256'] == provenance['config_sha256'] ==
                bind(spec['fusion']), 'fusion config SHA mismatch')
        require(provenance['script_sha256'] == PINNED['scripts/run_image_score_fusion.py'] ==
                cfg['inputs_sha256']['scripts/run_image_score_fusion.py'], 'fusion script changed')
        for path, digest in cfg['inputs_sha256'].items():
            bind(path, digest)
        require(provenance['inputs_sha256'] == cfg['inputs_sha256'], 'fusion registered input mismatch')
        parts = v['parts']['attribution']
        require(roleids == {r: [x['sample_id'] for x in parts[r]] for r in ('calibration', 'evaluation')},
                'fusion role IDs mismatch')
        require(result['calibration_n'] == calibration['calibration_n'] == spec['fusion_n']['calibration'] and
                result['evaluation_n'] == spec['fusion_n']['evaluation'], 'fusion role size mismatch')
        classes, cached = cached_fusion(ROOT, cfg, parts)
        resnet = v['runs']['A0']['predictions']; cal = indexed(parts['calibration'])
        ids, y, p, _, cv = align(cal, restrict(resnet, cal), restrict(resnet, cal), 'A0')
        require(cv == classes, 'fusion cache class vocabulary')
        order = {r['sample_id']: i for i, r in enumerate(parts['calibration'])}
        alpha, grid = fit_weight(y, p, cached['calibration'][[order[sid] for sid in ids]])
        require(np.isclose(alpha, result['alpha_resnet'], atol=1e-12, rtol=0) and grid == calibration['grid'],
                'calibration NLL fit differs')
        require(result['cache_score_max_error'] <= 1e-5, 'fusion cache score reproduction failed')
        ev = indexed(parts['evaluation']); scores = {}; preds = {}
        for method in FUSION_METHODS:
            pred = read(folder / (method + '-predictions.jsonl'), True)
            ids, y, s, _, cc = align(ev, pred, pred, 'A0'); require(cc == classes, 'fusion evaluation vocabulary')
            check_saved_metrics(result['metrics'][method], stats(y, s, 'A0', cc)); scores[method] = s; preds[method] = pred
        require(np.allclose(scores['Uniform'], mix_scores(scores['ResNet50'], scores['FeatureFusion'], .5),
                            atol=1e-12, rtol=0), 'uniform fusion formula changed')
        require(np.allclose(scores['Calibrated'], mix_scores(scores['ResNet50'], scores['FeatureFusion'], alpha),
                            atol=1e-12, rtol=0), 'calibrated fusion formula changed')
        for method in FUSION_METHODS[:3]:
            redo = group_bootstrap(y, scores[method], scores['Calibrated'], [ev[sid]['group_id'] for sid in ids],
                                   repeats=REPEATS, seed=SEED)
            require(redo == result['paired_group_bootstrap'][method], 'saved fusion bootstrap differs')
        fusion[name] = dict(alpha_resnet=alpha, calibration_n=result['calibration_n'],
            evaluation_n=result['evaluation_n'], calibration_nll=min(x['calibration_nll'] for x in grid),
            metrics=result['metrics'], calibrated_minus_other=result['paired_group_bootstrap'],
            cache_score_max_error=result['cache_score_max_error'], config=spec['fusion'], output=str(folder),
            predictions=preds)

    summary = dict(status='completed', before=OLD['name'], after=NEW['name'], seed=SEED, bootstrap_repeats=REPEATS,
        holdout_scored=False, training_executed=False, image_inference_executed=False, gpu_used=False,
        manifests={n: dict(path=v['spec']['data'], manifest_sha256=v['spec']['manifest_sha'],
                           selection_sha256=v['spec']['selection_sha'], samples=v['spec']['samples'],
                           roles=v['plan']['roles'], full_dev=v['spec']['dev']) for n, v in versions.items()},
        original_rows_preserved=preserved, original_roles_preserved=role_counts,
        calibration_real_images=dict(n=len(real_cal[0]), identical=True),
        certifications={n: v['certification'] for n, v in versions.items()}, training=training,
        gpu_index={n: v['cfg'].get('gpu_index') for n, v in versions.items()},
        paired={}, new_only={}, detection_thresholds={}, ofa_references=None,
        score_fusion={n: {k: x for k, x in f.items() if k != 'predictions'} for n, f in fusion.items()},
        feature_cache_sha256={n: v['feature_cache_sha256'] for n, v in versions.items()})
    sources = []

    def paired(label, rows, a, b, task):
        ids, y, p, q, classes = align(rows, restrict(a, rows), restrict(b, rows), task)
        before, after = stats(y, p, task, classes), stats(y, q, task, classes)
        groups = [old['manifest'][sid]['group_id'] for sid in ids]
        result = dict(n=len(ids), group_n=len(set(groups)), before=before, after=after,
                      delta={k: after[k] - x for k, x in before.items() if k != 'n'},
                      same_sample_ids=True, same_labels=True, same_class_vocabulary=True)
        if task == 'A0':
            result['paired_group_bootstrap'] = group_bootstrap(y, p, q, groups, repeats=REPEATS, seed=SEED)
            sources.extend(source_rows(label, y, p, q, classes, groups))
        elif task == 'D0':
            result['paired_group_bootstrap'] = detection_bootstrap(y, p, q, groups, classes)
        else:
            for i, source in enumerate(classes):
                sources.append(dict(comparison=label, source=source, n=int((y == i).sum()),
                    before_auroc=auc(y == i, p[:, i]), after_auroc=auc(y == i, q[:, i]),
                    delta_auroc=auc(y == i, q[:, i]) - auc(y == i, p[:, i])))
        summary['paired'][label] = result
        return ids, y, p, q, classes

    parent_eval = {t: indexed(old['parts'][r]['evaluation']) for t, r in (('D0', 'detection'), ('A0', 'attribution'))}
    parent_eval['OFA'] = parent_eval['A0']
    for task, _ in RUNS:
        paired(task + '/parent_full_dev', dev_rows(old['manifest'], task), old['runs'][task]['predictions'],
               new['runs'][task]['predictions'], task)
        paired(task + '/parent_evaluation', parent_eval[task], old['runs'][task]['predictions'],
               new['runs'][task]['predictions'], task)
        summary['new_only'][task + '/new_full_dev'] = new['runs'][task]['metrics']
    for job in FEATURES:
        task = 'D0' if job == 'D-DINO' else 'A0'
        paired(job + '/parent_evaluation', parent_eval[task], old['features'][job]['predictions'],
               new['features'][job]['predictions'], task)
        summary['new_only'][job + '/new_evaluation'] = new['features'][job]['metrics']
    for method in FUSION_METHODS:
        paired('score-fusion/' + method + '/parent_evaluation', parent_eval['A0'], fusion[OLD['name']]['predictions'][method],
               fusion[NEW['name']]['predictions'][method], 'A0')
        summary['new_only']['score-fusion/' + method + '/new_evaluation'] = fusion[NEW['name']]['metrics'][method]

    references = [read(Path(v['runs']['OFA']['path']) / 'reference_manifest.json') for v in (old, new)]
    diff = reference_diff(*references)
    summary['ofa_references'] = dict(per_source=diff, total_old=sum(r['old_n'] for r in diff),
        total_new=sum(r['new_n'] for r in diff), shared=sum(r['shared'] for r in diff),
        changed=sum(r['added'] for r in diff),
        note='Same per-source reference count; reference IDs re-drawn from each version training partition')

    for model in ('ResNet50', 'D-DINO'):
        fitted = {}
        for name, v in versions.items():
            rows = v['parts']['detection']
            if model == 'ResNet50':
                preds = v['runs']['D0']['predictions']; cal = indexed(rows['calibration'])
                ci, cy, cp, _, classes = align(cal, restrict(preds, cal), restrict(preds, cal), 'D0')
            else:
                c = v['features']['D-DINO']['calibration']; ci, cy, cp, classes = c['ids'], c['y'], c['p'], c['classes']
                preds = v['features']['D-DINO']['predictions']
            ai = classes.index('ai'); negative = cp[cy != ai, ai]
            require(len(negative) == 1014, 'calibration negative size changed')
            cal_rows = [indexed(rows['calibration'])[sid] for sid in ci]
            scopes = {'parent_evaluation': parent_eval['D0'], 'own_evaluation': indexed(rows['evaluation'])}
            entry = dict(calibration_n=len(ci), calibration_real_n=len(negative), targets={})
            for scope, ev in scopes.items():
                ids, y, p, _, _ = align(ev, restrict(preds, ev), restrict(preds, ev), 'D0')
                binary = (y == ai).astype(int); report = binary_report(binary, p[:, ai], .5)
                report['tpr_wilson95'] = wilson(int((p[binary == 1, ai] >= .5).sum()), int(binary.sum()))
                entry[scope + '_fixed_0.5'] = report
            for target in TARGETS:
                threshold = choose_threshold(negative, target)
                t = dict(threshold=threshold, calibration_fpr=float((negative >= threshold).mean()),
                         calibration_fpr_wilson95=wilson(int((negative >= threshold).sum()), len(negative)),
                         threshold_stability=threshold_stability(cal_rows, cp[:, ai], target, SEED, repeats=REPEATS))
                for scope, ev in scopes.items():
                    ids, y, p, _, _ = align(ev, restrict(preds, ev), restrict(preds, ev), 'D0')
                    binary = (y == ai).astype(int); report = binary_report(binary, p[:, ai], threshold)
                    report['tpr_wilson95'] = wilson(int((p[binary == 1, ai] >= threshold).sum()), int(binary.sum()))
                    t[scope] = report
                    if scope == 'parent_evaluation':
                        t['_parent'] = (ids, binary, p[:, ai])
                if model == 'D-DINO':
                    saved = read(Path(v['cfg']['output']) / 'D-DINO/metrics.json')['calibrated'][str(target)]
                    require(np.isclose(saved['evaluation']['threshold'], threshold, atol=1e-12, rtol=0),
                            'saved threshold mismatch')
                    for key in ('fpr', 'tpr', 'auroc', 'accuracy'):
                        require(np.isclose(saved['evaluation'][key], t['own_evaluation'][key], atol=1e-12, rtol=0),
                                'saved calibrated evaluation mismatch')
                entry['targets'][str(target)] = t
            fitted[name] = entry
        for target in TARGETS:
            a, b = fitted[OLD['name']]['targets'][str(target)], fitted[NEW['name']]['targets'][str(target)]
            ids, binary, p = a.pop('_parent'); ids2, binary2, q = b.pop('_parent')
            require(ids == ids2 and np.array_equal(binary, binary2), 'threshold comparison identity mismatch')
            a_thr, b_thr = a['threshold'], b['threshold']
            fitted.setdefault('paired_parent_evaluation', {})[str(target)] = dict(
                delta_fpr=b['parent_evaluation']['fpr'] - a['parent_evaluation']['fpr'],
                delta_tpr=b['parent_evaluation']['tpr'] - a['parent_evaluation']['tpr'],
                paired_group_bootstrap=threshold_bootstrap(binary, p, q, a_thr, b_thr,
                    [old['manifest'][sid]['group_id'] for sid in ids]))
        summary['detection_thresholds'][model] = fitted

    distribution = []
    for name, v in versions.items():
        for role in ROLES:
            rows = v['parts']['detection'][role]
            counts = Counter(r['source_label'] if r['is_ai'] else 'real' for r in rows)
            for source in sorted(counts):
                distribution.append(dict(version=name, role=role, source=source, n=counts[source],
                    original_group_n=len({r['group_id'] for r in rows if (r['source_label'] if r['is_ai'] else 'real') == source})))
    role_ids = {t: {r: [x['sample_id'] for x in rows] for r, rows in old['parts'][t].items()} for t in ('detection', 'attribution')}
    summary['limitations'] = [
        'Single seed 42; exploratory development comparison, not an independent test, not SOTA',
        'ResNet D0/A0 selected checkpoints on their own full dev, which contains the evaluation role; parent dev informed model choice',
        'The 100-version ran on physical GPU1 and the 200-q179 version on physical GPU0 (same RTX 4090 D model)',
        'Old 100-version small-stratum floor allocation is retained; only new unanchored groups use the corrected extension split',
        '179 conflicting new images are quarantined; actual AI cells hold 187-200 images instead of a nominal 200',
        'Training data, optimisation steps and D0 class weights change together; no pure data-scale causal claim',
        'Frozen feature heads use unweighted cross-entropy while the AI share of training rose from 45.0% to 62.5%; '
        'compare D-DINO at calibrated thresholds rather than at 0.5',
        'OFA keeps 100 references per source, but reference IDs are re-drawn; OFA AUROC is source verification, not a classifier probability',
        'Bootstrap intervals are paired percentiles over original groups, conditional on selected checkpoints, thresholds and alpha',
        'Full-dev results of the two versions have different denominators and are only listed side by side',
    ]
    E.unchanged()
    return summary, epochs, sources, distribution, role_ids, diff, dict(E.inputs)


def pct(x):
    return '—' if x is None else f'{100 * x:.4f}%'


LIMITATIONS_ZH = [
    '单一seed 42；探索性开发比较，不是独立测试，也不声称SOTA。',
    'ResNet D0/A0各自在完整dev选checkpoint，完整dev包含evaluation角色；父dev曾参与选模。',
    '100版在物理GPU1、200-q179版在物理GPU0运行（同型号RTX 4090 D）。',
    '旧100版小分层floor分配偏差保留；仅新无锚点组采用修正后的扩展划分。',
    '179张冲突新增图被隔离，AI小格实际为187–200张，并非名义200张。',
    '训练数据量、优化步数与D0类别权重同时变化（D0权重由[0.909,1.111]变为[1.334,0.800]），不作纯数据量因果宣称。',
    '冻结特征头使用未加权交叉熵；训练集中AI占比由45.0%升至62.5%，D-DINO固定0.5阈值的FPR上升与此一致，校准阈值比较更可比。',
    'OFA每来源仍100张参考，但参考ID重新抽取（2200个中仅237个保留）；OFA宏AUROC是来源验证分数，不是分类概率。',
    'bootstrap为原group配对百分位区间，以已选checkpoint、阈值和alpha为条件，不含重新训练或重选模型的不确定性。',
    '两版完整dev分母不同，仅并列，不用于宣称提升。',
]


def render(summary, sources):
    o, n = OLD['name'], NEW['name']
    lines = [f'# IAB {o} 与 {n} 同样本比较', '',
        '状态：全部保存结果核验通过。未训练、未读取图片、未用GPU、未评分holdout。', '',
        f'原100版manifest `{OLD["manifest_sha"]}`；200-q179版manifest `{NEW["manifest_sha"]}`。原42,000个样本ID、split、group、标签与图像hash全部保留；原train/selection/calibration/evaluation角色在新协议中完全不变。', '',
        '配对分母固定为原100版：D0 4960张dev（2992真实、1968 AI）、A0/OFA 1968张AI dev；新头与分数融合固定原1585张真假evaluation、606张AI evaluation。group_id均为原分组，bootstrap 1000次、seed42、95%百分位区间，差值为200-q179减100。', '',
        '## 同样本配对结果', '', '|模型/范围|n|原group|指标|100/30|200-q179/30|变化 pp|95%区间 pp|', '|---|---:|---:|---|---:|---:|---:|---|']
    for label, item in summary['paired'].items():
        boot = item.get('paired_group_bootstrap', {}).get('delta_95_ci', {})
        delta = item['delta']
        fields = (('auroc', 'fpr_0_5', 'tpr_0_5', 'accuracy', 'macro_f1') if 'auroc' in delta else
                  ('macro_auroc',) if 'macro_auroc' in delta else ('top1', 'top3', 'macro_f1'))
        for field in fields:
            ci = boot.get(field)
            span = f'[{100 * ci[0]:+.4f}, {100 * ci[1]:+.4f}]' if ci else '—'
            lines.append(f'|{label}|{item["n"]}|{item["group_n"]}|{field}|{pct(item["before"][field])}|'
                         f'{pct(item["after"][field])}|{100 * item["delta"][field]:+.4f}|{span}|')
    lines += ['', 'OFA为冻结参考匹配，epoch=null；宏AUROC是来源验证分数，不与分类概率混用，未做bootstrap。逐来源AUROC见per-source.csv。', '',
        '## 检测阈值（仅calibration真实图拟合）', '',
        '两版calibration真实图为同一组1014张；目标FPR阈值只用它们选择，再在固定1585 evaluation与各自完整evaluation报告。Wilson区间按图片独立计算；阈值稳定性为calibration真实图group bootstrap 1000次；配对区间以各自阈值为条件。', '',
        '|模型|版本|阈值方式|阈值|cal FPR|原1585 FPR|原1585 TPR|FPR Wilson95|TPR Wilson95|阈值bootstrap 2.5–97.5%|',
        '|---|---|---|---:|---:|---:|---:|---|---|---|']

    def interval(ci):
        return '—' if not ci else f'[{100 * ci[0]:.3f}, {100 * ci[1]:.3f}]%'
    notes = []
    for model, fitted in summary['detection_thresholds'].items():
        for version in (o, n):
            entry = fitted[version]; fixed = entry['parent_evaluation_fixed_0.5']
            lines.append(f'|{model}|{version}|固定0.5|0.5|—|{pct(fixed["fpr"])}|{pct(fixed["tpr"])}|'
                         f'{interval(fixed["fpr_wilson95"])}|{interval(fixed["tpr_wilson95"])}|—|')
            for target, t in entry['targets'].items():
                ev = t['parent_evaluation']; st = t['threshold_stability']
                lines.append(f'|{model}|{version}|目标FPR {100 * float(target):g}%|{t["threshold"]:.6f}|'
                             f'{pct(t["calibration_fpr"])}|{pct(ev["fpr"])}|{pct(ev["tpr"])}|'
                             f'{interval(ev["fpr_wilson95"])}|{interval(ev["tpr_wilson95"])}|'
                             f'[{st["p025"]:.6f}, {st["p975"]:.6f}]|')
        for target, item in fitted['paired_parent_evaluation'].items():
            ci = item['paired_group_bootstrap']['delta_95_ci']
            notes.append(f'- {model} 目标FPR {100 * float(target):g}%：原1585 evaluation FPR变化{100 * item["delta_fpr"]:+.4f} pp '
                         f'[{100 * ci["fpr"][0]:+.4f}, {100 * ci["fpr"][1]:+.4f}]，TPR变化{100 * item["delta_tpr"]:+.4f} pp '
                         f'[{100 * ci["tpr"][0]:+.4f}, {100 * ci["tpr"][1]:+.4f}]。')
    lines += [''] + notes
    lines += ['', '## 来源分数融合', '',
        '只融合ResNet A0与A-Fusion的22类来源概率；alpha网格0..1步长0.05，各版只用自身AI calibration最小NLL选择，数值平分倾向ResNet；50/50为事前固定对照，不按evaluation反选。', '',
        '|版本|calibration n|alpha_resnet|calibration NLL|evaluation n|方法|Top1|Top3|Macro-F1|', '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for version, f in summary['score_fusion'].items():
        for method, m in f['metrics'].items():
            lines.append(f'|{version}|{f["calibration_n"]}|{f["alpha_resnet"]:.2f}|{f["calibration_nll"]:.4f}|'
                         f'{f["evaluation_n"]}|{method}|{pct(m["top1"])}|{pct(m["top3"])}|{pct(m["macro_f1"])}|')
    for version, f in summary['score_fusion'].items():
        ci = f['calibrated_minus_other']['ResNet50']
        delta = f['metrics']['Calibrated']['top1'] - f['metrics']['ResNet50']['top1']
        lines.append(f'- {version}：校准融合减ResNet Top1 {100 * delta:+.4f} pp，自身evaluation {ci["group_n"]}个group区间'
                     f'[{100 * ci["delta_95_ci"]["top1"][0]:+.4f}, {100 * ci["delta_95_ci"]["top1"][1]:+.4f}] pp。')
    lines += ['', '## 逐来源改善与退化（Top1，原同样本）', '', '|比较|来源数|改善|持平|退化|最大退化来源|最大改善来源|', '|---|---:|---:|---:|---:|---|---|']
    for label in dict.fromkeys(r['comparison'] for r in sources if 'delta_top1' in r):
        rows = [r for r in sources if r['comparison'] == label and 'delta_top1' in r]
        worst, best = min(rows, key=lambda r: r['delta_top1']), max(rows, key=lambda r: r['delta_top1'])
        lines.append(f'|{label}|{len(rows)}|{sum(r["delta_top1"] > 0 for r in rows)}|{sum(r["delta_top1"] == 0 for r in rows)}|'
                     f'{sum(r["delta_top1"] < 0 for r in rows)}|{worst["source"]} {100 * worst["delta_top1"]:+.2f}pp (n={worst["n"]})|'
                     f'{best["source"]} {100 * best["delta_top1"]:+.2f}pp (n={best["n"]})|')
    lines += ['', '逐来源样本量小（原evaluation每来源约20–35张），单来源差异不稳定，仅作误差分析。']
    lines += ['', '## 200-q179完整dev/evaluation（另列，不与100版分母混比）', '', '|模型/范围|n|AUROC/宏AUC|Top1/Accuracy|Top3|Macro-F1|FPR@0.5|TPR@0.5|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, m in summary['new_only'].items():
        lines.append(f'|{label}|{m["n"]}|{pct(m.get("auroc", m.get("macro_auroc")))}|{pct(m.get("top1", m.get("accuracy")))}|'
                     f'{pct(m.get("top3"))}|{pct(m.get("macro_f1"))}|{pct(m.get("fpr_0_5"))}|{pct(m.get("tpr_0_5"))}|')
    lines += ['', '## 训练量、类别权重与OFA参考', '']
    for version, tasks in summary['training'].items():
        for task, t in tasks.items():
            lines.append(f'- {version} {task}：train {t["train_n"]}，类别计数 {json.dumps(t["train_counts"], ensure_ascii=False)}，'
                         f'类别权重 {t["class_weights"]}，最优epoch {t["selected_epoch"]}。')
    refs = summary['ofa_references']
    lines += [f'- OFA参考：每来源数量不变，共{refs["total_old"]}→{refs["total_new"]}；保留{refs["shared"]}个、更换{refs["changed"]}个ID，逐来源见ofa-reference-diff.csv。',
        f'- GPU登记：{json.dumps(summary["gpu_index"], ensure_ascii=False)}（物理GPU编号）。', '',
        '## 限制', ''] + ['- ' + x for x in LIMITATIONS_ZH] + [
        '', '逐epoch曲线见epochs.csv；逐来源改善与退化见per-source.csv；角色分布见source-distribution.csv。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    out = (ROOT / args.output).resolve()
    if out.exists():
        print(f'Refusing existing output directory: {out}', file=sys.stderr)
        return 1
    summary, epochs, sources, distribution, role_ids, diff, inputs = analyze()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'summary.json', summary)
    for name, rows in [('epochs.csv', epochs), ('per-source.csv', sources), ('source-distribution.csv', distribution),
                       ('ofa-reference-diff.csv', diff)]:
        with (out / name).open('x') as f:
            f.write(csv_text(rows))
    dump(out / 'role_ids.json', role_ids)
    with (out / 'report.md').open('x') as f:
        f.write(render(summary, sources))
    source = Path(__file__).read_text()
    provenance = dict(inputs_sha256=inputs, outputs_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        analysis_script=str(Path(__file__).resolve().relative_to(ROOT)),
        analysis_script_sha256=hashlib.sha256(source.encode()).hexdigest(), numpy_version=np.__version__,
        model_binary_hashes_verified=True, original_files_overwritten=False, holdout_scored=False,
        image_inference_executed=False, training_executed=False)
    dump(out / 'provenance.json', provenance)
    dump(out / 'completion.json', dict(status='completed', exit_code=0,
        manifest_sha256={OLD['name']: OLD['manifest_sha'], NEW['name']: NEW['manifest_sha']},
        same_sample_ids_verified=True, summary_sha256=sha256(out / 'summary.json'),
        provenance_sha256=sha256(out / 'provenance.json'), holdout_scored=False, training_executed=False,
        image_inference_executed=False))
    print(json.dumps(dict(status='completed', output=str(out), summary_sha256=sha256(out / 'summary.json')),
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
