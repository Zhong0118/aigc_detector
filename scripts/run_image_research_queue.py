#!/usr/bin/env python3
"""Registered frozen-feature experiments, gated by completed 100/cell baselines.

Default is a metadata-only plan. --advance launches at most one detached worker;
failed or abandoned runs require diagnosis, never an automatic reset.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.monitor_iab_v2 import alive, identity, lock, read, write
from research.image_attribution.protocol import sha256, validate_manifest

REQUIRED_CODE = ('scripts/run_image_research_queue.py', 'research/image_attribution/feature_ablation.py',
    'research/image_attribution/vision_backbones.py', 'research/image_attribution/models.py',
    'research/image_attribution/protocol.py', 'research/image_attribution/followups.py',
    'scripts/run_image_experiment.py', 'scripts/run_image_followup.py', 'scripts/monitor_iab_v2.py',
    'scripts/compare_image_scale_epochs.py', 'scripts/predict_image_research.py')


def validate_parent(rows, parent):
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate current sample IDs')
    fields = ('split', 'group_id', 'sha256', 'pixel_sha256', 'is_ai', 'source_label', 'semantic_class')
    for old in parent:
        new = by_id.get(old['sample_id'])
        if new is None or any(new[key] != old[key] for key in fields):
            raise ValueError('parent identity, label or partition changed: ' + old['sample_id'])


def safe_rows(rows, task):
    if task not in ('detection', 'attribution'):
        raise ValueError('unknown task')
    for row in rows:
        if row['split'] not in ('train', 'dev') or (task == 'attribution' and not row['is_ai']):
            raise ValueError('forbidden extraction member: ' + row['sample_id'])


def prerequisite_manifest(root, cfg):
    """Only bind the new manifest when completion, validated summary and summary agree."""
    latest = None
    for index, name in enumerate(cfg['prerequisites']):
        path = root / name
        state = read(path.parent / 'pipeline.json', {})
        if state.get('status') == 'failed':
            raise ValueError('baseline failed; preserve its outputs')
        if not path.exists():
            return None
        completion = json.loads(path.read_text())
        validated = json.loads((path.parent / 'validated-summary.json').read_text())
        summary = json.loads((root / completion['summary']).read_text())
        if any(item.get('status') != 'completed' for item in (completion, validated, summary)):
            raise ValueError('incomplete baseline certification')
        latest = completion['manifest_sha256']
        if any(item.get('manifest_sha256') != latest for item in (validated, summary)):
            raise ValueError('baseline certification hash mismatch')
        if index == 0 and latest != cfg['parent_manifest']['sha256']:
            raise ValueError('50/cell prerequisite is not the registered parent manifest')
    return latest


def load_protocol(root, cfg):
    bound = prerequisite_manifest(root, cfg)
    if not bound:
        return dict(status='waiting_baselines', holdout_scored=False), None
    data = root / cfg['dataset']['path']
    if sha256(data / 'selection.json') != cfg['dataset']['selection_sha256']:
        raise ValueError('selection hash mismatch')
    if sha256(data / 'manifest.jsonl') != bound:
        raise ValueError('manifest differs from certified baseline')
    if cfg['dataset'].get('manifest_sha256') and bound != cfg['dataset']['manifest_sha256']:
        raise ValueError('certified manifest differs from registered dataset')
    parent = root / cfg['parent_manifest']['path']
    if sha256(parent) != cfg['parent_manifest']['sha256']:
        raise ValueError('parent manifest hash mismatch')
    rows = [json.loads(line) for line in (data / 'manifest.jsonl').read_text().splitlines() if line]
    validate_manifest(rows, data, verify_files=False)
    validate_parent(rows, [json.loads(line) for line in parent.read_text().splitlines() if line])
    from research.image_attribution.followups import partition_rows
    parts = {task: partition_rows(rows, dict(mode='clean', task=task, seed=cfg['seed']))
             for task in ('detection', 'attribution')}
    for task, role_map in parts.items():
        for role, selected in role_map.items():
            safe_rows(selected, task)
            if not selected:
                raise ValueError('empty role: ' + task + '/' + role)
        classes = sorted({r['source_label'] for r in role_map['train'] if r['is_ai']})
        if classes != cfg['classes']:
            raise ValueError('source vocabulary mismatch')
    return dict(status='ready', manifest_sha256=bound, selection_sha256=cfg['dataset']['selection_sha256'],
                roles={t: {k: len(v) for k, v in p.items()} for t, p in parts.items()},
                holdout_scored=False), parts


def verify_registered_files(root, cfg):
    if not set(REQUIRED_CODE).issubset(cfg['implementation_sha256']):
        raise ValueError('missing implementation hash registration')
    for name, digest in cfg['implementation_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError('registered implementation changed: ' + name)
    for asset in cfg['assets'].values():
        if sha256(root / asset['path']) != asset['sha256']:
            raise ValueError('registered asset changed: ' + asset['path'])


def features(root, cfg, plan, branch, rows, cache_dir, device):
    import torch
    from torch.utils.data import DataLoader
    from research.image_attribution.feature_ablation import checked_features, load_bound_features
    from research.image_attribution.models import load_ofa, ofa_features, ofa_transform
    from research.image_attribution.vision_backbones import load_dinov2
    from scripts.run_image_experiment import Images

    safe_rows(rows, 'detection' if branch == 'rgb' else 'attribution')
    asset_keys = ('dinov2_model', 'dinov2_config', 'dinov2_processor') if branch == 'rgb' else ('fingerprint', 'denoiser')
    binding = dict(schema='task-separated-features-v1', branch=branch,
                   manifest_sha256=plan['manifest_sha256'], sample_ids=[r['sample_id'] for r in rows],
                   weights={k: cfg['assets'][k]['sha256'] for k in asset_keys},
                   code_sha256=cfg['implementation_sha256'], preprocessing=cfg['preprocessing'][branch])
    path = cache_dir / (branch + '.pt')
    if path.exists():
        return load_bound_features(path, binding, cfg['dimensions'][branch]), binding
    data = root / cfg['dataset']['path']
    for row in rows:
        if sha256(data / row['relative_path']) != row['sha256']:
            raise ValueError('image hash changed: ' + row['sample_id'])
    if branch == 'rgb':
        encoder, transform, _ = load_dinov2(root / cfg['dinov2_directory'], device)
        denoiser = None
    else:
        encoder, _, denoiser = load_ofa(root / cfg['assets']['fingerprint']['path'],
                                        root / cfg['assets']['denoiser']['path'], device)
        transform = ofa_transform
    loader = DataLoader(Images(rows, [int(r['is_ai']) for r in rows], data, transform),
                        batch_size=cfg['extraction_batch_size'][branch], shuffle=False, num_workers=0)
    batches, count = [], 0
    started = time.monotonic()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            value = encoder(images) if branch == 'rgb' else ofa_features(images, encoder, denoiser)
            batches.append(value.float().cpu())
            count += len(images)
            write(cache_dir / (branch + '-progress.json'), dict(status='extracting', n=count,
                  total=len(rows), elapsed_seconds=time.monotonic() - started))
    tensor = checked_features(torch.cat(batches), cfg['dimensions'][branch])
    partial = path.with_suffix('.partial')
    with partial.open('xb') as stream:
        torch.save(dict(identity=binding, features=tensor), stream)
    partial.replace(path)
    write(cache_dir / (branch + '-progress.json'), dict(status='completed', n=count,
          elapsed_seconds=time.monotonic() - started, bytes=path.stat().st_size, sha256=sha256(path)))
    return tensor, binding


def run_jobs(root, cfg, config_hash, plan, parts, base, device=None):
    import random
    import numpy as np
    import torch
    from research.image_attribution.feature_ablation import combine_features, digest, fit_head
    device = torch.device('cuda:0' if device is None else device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('registered CUDA GPU unavailable')
    torch.set_num_threads(4)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    roles = ('train', 'selection', 'calibration', 'evaluation')
    rgb_rows = [r for role in roles for r in parts['detection'][role]]
    fp_rows = [r for role in ('train', 'selection', 'evaluation') for r in parts['attribution'][role]]
    cache = base / 'feature-cache'; cache.mkdir(exist_ok=True)
    cache_values = {}
    rgb_lookup = {r['sample_id']: i for i, r in enumerate(rgb_rows)}
    results = {}
    for job in cfg['jobs']:
        out = base / job['id']
        if out.exists():
            raise FileExistsError('existing job output retained: ' + str(out))
        out.mkdir()
        job_cfg = dict(cfg, task=job['task'])
        p = parts[job['task']]
        classes = ['real', 'ai'] if job['task'] == 'detection' else cfg['classes']
        ordered = rgb_rows if job['task'] == 'detection' else fp_rows
        write(out / 'resolved_config.json', dict(config=job_cfg, job=job, plan=plan, config_sha256=config_hash))
        write(out / 'role_ids.json', {role: [r['sample_id'] for r in selected] for role, selected in p.items()})
        write(base / 'progress.json', dict(status='running', job=job['id'], phase='extracting'))
        branches = ('rgb', 'fingerprint') if job['variant'] == 'fusion' else (job['variant'],)
        values, bindings = {}, {}
        for branch in branches:
            rows = rgb_rows if branch == 'rgb' else fp_rows
            if branch not in cache_values:
                cache_values[branch] = features(root, cfg, plan, branch, rows, cache, device)
            value, binding = cache_values[branch]
            if branch == 'rgb' and job['task'] == 'attribution':
                value = value[[rgb_lookup[r['sample_id']] for r in ordered]]
            values[branch], bindings[branch] = value, binding
        joined = combine_features(job['variant'], **values, dimensions=cfg['dimensions'])
        provenance = dict(config_sha256=config_hash, manifest_sha256=plan['manifest_sha256'],
            selection_sha256=plan['selection_sha256'], code_sha256=cfg['implementation_sha256'],
            cache_identities=bindings, seed=cfg['seed'], gpu_index=cfg['gpu_index'],
            role_ids_sha256={k: digest([r['sample_id'] for r in v]) for k, v in p.items()})
        write(base / 'progress.json', dict(status='running', job=job['id'], phase='training_head'))
        results[job['id']] = fit_head(joined, p, classes, job_cfg, out, device, job['variant'], provenance)
        try:
            from scripts.predict_image_research import predict
            probe_rows = p['selection'][:2]
            paths = [root / cfg['dataset']['path'] / r['relative_path'] for r in probe_rows]
            batch = predict(out, paths, device=device, batch_size=2, top_k=len(classes))
            single = predict(out, paths, device=device, batch_size=1, top_k=len(classes))
            def vectors(records):
                if job['task'] == 'detection':
                    return np.array([[1-r['score_ai'], r['score_ai']] for r in records])
                return np.array([[dict((x['source'],x['score']) for x in r['top_k'])[c] for c in classes] for r in records])
            checkpoint = torch.load(out / 'head.pth', map_location='cpu', weights_only=True)
            head = torch.nn.Linear(cfg['dimensions'][job['variant']], len(classes)).to(device)
            head.load_state_dict(checkpoint['state_dict'], strict=True); head.eval()
            with torch.no_grad():
                cached = head(joined[len(p['train']):len(p['train'])+len(probe_rows)].to(device)).softmax(1).cpu().numpy()
            check = dict(sample_ids=[r['sample_id'] for r in probe_rows], role='selection',
                cached_vs_image_max_error=float(np.abs(cached-vectors(batch)).max()),
                image_single_batch_max_error=float(np.abs(vectors(single)-vectors(batch)).max()),
                batch=batch, single=single, holdout_scored=False)
            write(out / 'image-inference-check.json', check)
            if max(check['cached_vs_image_max_error'], check['image_single_batch_max_error']) > 1e-5:
                raise ValueError('full image reload/single/batch consistency failed')
            results[job['id']]['image_inference_verified'] = True
            write(out / 'metrics.json', results[job['id']])
        except Exception as error:
            write(out / 'status.json', dict(status='failed', exit_code=1, phase='image_reload_check', error=str(error)))
            raise
    comparisons = compare_heads(root, cfg, parts, base)
    write(base / 'summary.json', dict(status='completed', results=results, comparisons=comparisons, plan=plan,
        config_sha256=config_hash, holdout_scored=False,
        limitations=['Exploratory development roles; not an independent test.',
                     'Single seed. Fusion has a larger head. Frozen probes are project adaptations.']))


def compare_heads(root, cfg, parts, base):
    from scripts.compare_image_scale_epochs import align, indexed, stats, group_bootstrap
    def predictions(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    comparisons = {}
    for task, first, second, path in [
        ('A0', 'A-RGB', 'A-FP', base / 'A-RGB/predictions.jsonl'),
        ('A0', 'A-RGB', 'A-Fusion', base / 'A-RGB/predictions.jsonl'),
        ('A0', 'A0-ResNet50', 'A-RGB', root / cfg['baseline_predictions']['attribution']),
        ('D0', 'D0-ResNet50', 'D-DINO', root / cfg['baseline_predictions']['detection'])]:
        rows = indexed(parts['detection' if task == 'D0' else 'attribution']['evaluation'])
        baseline_predictions = predictions(path)
        old = [r for r in baseline_predictions if r['sample_id'] in rows]
        new = predictions(base / second / 'predictions.jsonl')
        ids, y, p, q, classes = align(rows, old, new, task)
        record = dict(first=first, second=second, first_metrics=stats(y, p, task, classes),
                      second_metrics=stats(y, q, task, classes), sample_ids=ids)
        if task == 'A0':
            record['paired_group_bootstrap'] = group_bootstrap(y, p, q,
                [rows[sid]['group_id'] for sid in ids], repeats=1000, seed=cfg['seed'])
        else:
            from scripts.run_image_followup import evaluate_scores
            cal = indexed(parts['detection']['calibration'])
            selected = [r for r in baseline_predictions if r['sample_id'] in cal]
            cal_ids, _, cp, _, cal_classes = align(cal, selected, selected, 'D0')
            if cal_classes != classes:
                raise ValueError('baseline calibration vocabulary mismatch')
            protocol = dict(task='detection', mode='calibration', seed=cfg['seed'],
                            target_fprs=cfg['target_fprs'], bootstrap_repeats=cfg['bootstrap_repeats'])
            record['first_calibrated'] = evaluate_scores(protocol,
                dict(calibration=[cal[sid] for sid in cal_ids], evaluation=[rows[sid] for sid in ids]),
                classes, dict(calibration=cp, evaluation=p))
            record['second_calibrated'] = json.loads((base / second / 'metrics.json').read_text())['calibrated']
            record['limitation'] = 'D0 previously used parent dev for model selection; calibration comparison remains exploratory.'
        comparisons[first + '_vs_' + second] = record
    write(base / 'paired-comparison.json', comparisons)
    return comparisons


def worker(root, cfg, config_hash, base):
    with lock(base / 'worker.lock') as owned:
        if owned is None:
            return
        if read(base / 'state.json') is not None:
            raise FileExistsError('existing worker state retained; no worker re-entry')
        # Parent records launch immediately after Popen; do not allow direct unregistered --worker.
        launch = None
        for _ in range(100):
            launch = read(base / 'launch.json')
            if launch is not None:
                break
            time.sleep(.05)
        if not launch or launch.get('pid') != os.getpid() or launch.get('identity') != identity(os.getpid()) or launch.get('config_sha256') != config_hash:
            raise PermissionError('worker lacks a matching supervisor launch registration')
        state = dict(status='running', pid=os.getpid(), identity=identity(os.getpid()),
                     started=time.time(), config_sha256=config_hash)
        write(base / 'state.json', state)
        try:
            verify_registered_files(root, cfg)
            plan, parts = load_protocol(root, cfg)
            if not parts:
                raise ValueError('baseline gate no longer ready')
            if shutil.disk_usage(root).free < cfg['minimum_free_bytes']:
                raise RuntimeError('insufficient free disk for registered run')
            write(base / 'bound-protocol.json', plan)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu_index'])
            while True:
                with lock(root / 'outputs/image_experiments/gpu0-research.lock') as gpu_lock:
                    if gpu_lock is not None:
                        run_jobs(root, cfg, config_hash, plan, parts, base)
                        break
                write(base / 'progress.json', dict(status='waiting_gpu_lock'))
                time.sleep(30)
            write(base / 'state.json', dict(state, status='completed', exit_code=0, finished=time.time()))
        except Exception as error:
            write(base / 'state.json', dict(state, status='failed', exit_code=1, finished=time.time(),
                  error_type=type(error).__name__, error=str(error)))
            raise


def advance(root, cfg, config_path, base):
    with lock(base / 'supervisor.lock') as owned:
        if owned is None:
            return {'status': 'supervisor_busy'}
        state = read(base / 'state.json', {})
        launched = read(base / 'launch.json', {})
        if state.get('status') in ('failed', 'completed'):
            return state
        if alive(state) or alive(launched):
            return dict(state or launched, active=True)
        if launched or state.get('status') == 'running':
            raise RuntimeError('previous worker exited without successful completion; preserve outputs')
        plan, _ = load_protocol(root, cfg)
        write(base / 'plan.json', plan)
        if plan['status'] != 'ready':
            return plan
        if not cfg.get('execution_enabled') or not cfg.get('automatic_execution'):
            return dict(plan, status='execution_disabled')
        verify_registered_files(root, cfg)
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu_index'])
        with (base / 'worker.log').open('xb') as log:
            proc = subprocess.Popen([str(root / '.venv-image/bin/python'), str(Path(__file__).resolve()),
                '--config', str(config_path), '--worker'], cwd=root, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        launched = dict(status='launched', pid=proc.pid, identity=identity(proc.pid),
                        config_sha256=sha256(config_path), time=time.time())
        write(base / 'launch.json', launched)
        return launched


def watch(root, cfg, config_path, base):
    """One server-side half-hour watcher; application closure does not cancel the queue."""
    with lock(base / 'watch.lock') as owned:
        if owned is None:
            return
        path = base / 'watch.json'
        if path.exists():
            raise FileExistsError('prior watcher state retained; inspect before restarting')
        state = dict(status='running', pid=os.getpid(), identity=identity(os.getpid()),
                     config_sha256=sha256(config_path), started=time.time())
        write(path, state)
        try:
            while True:
                if sha256(config_path) != state['config_sha256']:
                    raise ValueError('watcher configuration changed; preserve state for review')
                result = advance(root, cfg, config_path, base)
                write(path, dict(state, last_check=time.time(), queue_status=result['status']))
                print(json.dumps(result), flush=True)
                if result['status'] in ('completed', 'failed'):
                    write(path, dict(state, status=result['status'], finished=time.time()))
                    return
                time.sleep(1800)
        except Exception as error:
            write(path, dict(state, status='failed', finished=time.time(), error=str(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/image_experiments/iab_research_v6_100.json'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--advance', action='store_true')
    group.add_argument('--worker', action='store_true')
    group.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    base = ROOT / cfg['output']
    if args.watch:
        watch(ROOT, cfg, config_path, base)
    elif args.worker:
        if not cfg['execution_enabled']:
            raise PermissionError('execution disabled')
        worker(ROOT, cfg, sha256(config_path), base)
    else:
        result = advance(ROOT, cfg, config_path, base) if args.advance else load_protocol(ROOT, cfg)[0]
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
