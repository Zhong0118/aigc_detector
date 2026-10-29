"""Manual q179 frozen OmniAID route with real-image verification and GPU 0 gates."""
import argparse
from contextlib import contextmanager
import hashlib
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import evaluate_omniaid as core
from scripts.prepare_omniaid_q179 import ASSET_ROOT, CONFIG, LEGACY, asset_path, asset_reference

DATASET = 'outputs/image_experiments/iab-balanced-v7-200-q179'
OUTPUT = 'outputs/image_experiments/omniaid-v2-q179-20261029'
MANIFEST_SHA = '9dd85636e2a19dbe833d9ebe2bee0c2a6b6eaf43015699e4fd9f27db3694b4c3'
SELECTION_SHA = '1c8dee2e17cc128867562186cee7bc4f8cb82042b26ddb6d5569d703ab77876b'


def process_identity():
    return dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                process_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                commandline=Path('/proc/self/cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=str(Path.cwd()), script_sha256=core.sha256(Path(__file__)),
                launched_unix=time.time())


@contextmanager
def stage_status(path, stage, config):
    from scripts.prepare_omniaid_q179 import write_json
    identity = process_identity()
    record = dict(status='running', exit_code=None, epoch=None, stage=stage, pid=identity['pid'],
                  launch=identity, config_sha256=hashlib.sha256(
                      json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    with path.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    started = time.monotonic()
    try:
        yield
    except BaseException as error:
        record.update(status='failed', exit_code=1, error_type=type(error).__name__,
                      error=str(error)[:2000], elapsed_seconds=time.monotonic() - started)
        write_json(path, record)
        raise
    else:
        record.update(status='completed', exit_code=0, elapsed_seconds=time.monotonic() - started)
        write_json(path, record)


def validate_route(root, config):
    if config.get('output') != OUTPUT:
        raise ValueError('output must be the unique registered q179 output')
    if config.get('dataset') != dict(path=DATASET, per_cell=200,
                                    manifest_sha256=MANIFEST_SHA, selection_sha256=SELECTION_SHA):
        raise ValueError('dataset differs from the frozen q179 registration')
    if (config.get('seed') != 42 or config.get('gpu_index') != 0
            or config.get('automatic_execution') is not False or config.get('epoch') is not None
            or config.get('task') != 'detection' or config.get('batch_size') != 4
            or config.get('bootstrap_repeats') != 1000 or config.get('target_fprs') != [0.01, 0.05]
            or config.get('classes') != ['real', 'ai']):
        raise ValueError('frozen detection recipe mismatch')
    pinned = json.loads((Path(root) / LEGACY).read_text())['assets']
    for name in ('source', 'checkpoint', 'model_config', 'clip_config'):
        if config['assets'][name] != pinned[name]:
            raise ValueError('official asset identity differs from reviewed registration')
    clip = config['assets']['clip_weights']
    if clip['derived_from'] != pinned['clip_weights']['derived_from']:
        raise ValueError('base CLIP identity mismatch')
    for spec in config['assets'].values():
        asset_path(root, spec['path'])
    asset_path(root, config['verification_receipt']['path'])
    review_spec = config.get('source_review_receipt', {})
    if not review_spec.get('sha256'):
        raise ValueError('manual source review receipt is unbound')
    review_path = asset_path(root, review_spec['path'])
    if core.sha256(review_path) != review_spec['sha256']:
        raise ValueError('manual source review receipt hash mismatch')
    review = json.loads(review_path.read_text())
    if review.get('source_sha256') != core.SOURCE_SHA256 or not review.get('manual_source_review'):
        raise ValueError('manual source review does not match the registered source')
    license_spec = review['license']
    license_path = asset_path(root, license_spec['path'])
    if core.sha256(license_path) != license_spec['sha256']:
        raise ValueError('official license evidence hash mismatch')
    receipt_spec = config.get('conversion_receipt', {})
    if not receipt_spec.get('sha256'):
        raise ValueError('safe conversion receipt is unbound')
    path = asset_path(root, receipt_spec['path'])
    if core.sha256(path) != receipt_spec['sha256']:
        raise ValueError('safe conversion receipt hash mismatch')
    receipt = json.loads(path.read_text())
    if (receipt.get('tensor_equality_verified') is not True or receipt.get('torch_load_weights_only') is not True
            or receipt.get('sha256') != clip['sha256'] or receipt.get('bytes') != clip['bytes']
            or receipt.get('source_sha256') != clip['derived_from']['sha256']):
        raise ValueError('safe conversion receipt does not verify registered tensors')


def gpu_guard(config, device):
    if config.get('gpu_index') != 0:
        raise ValueError('only physical GPU 0 is authorized')
    if device == 'cpu':
        return None
    if device != 'cuda':
        raise ValueError('use cpu or cuda; GPU 0 isolation is configured explicitly')
    used = int(subprocess.check_output(
        ['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        text=True, timeout=15).strip())
    if used > 1024:
        raise RuntimeError(f'physical GPU 0 is busy ({used} MiB > 1024 MiB)')
    return used


def probe_images(root, config):
    data = Path(root) / config['dataset']['path']
    for name, expected in [('manifest.jsonl', MANIFEST_SHA), ('selection.json', SELECTION_SHA)]:
        if core.sha256(data / name) != expected:
            raise ValueError('frozen dataset identity mismatch')
    rows = [json.loads(line) for line in (data / 'manifest.jsonl').read_text().splitlines() if line.strip()]
    core.validate_manifest(rows, data, verify_files=False)
    selection = core.partition_rows(rows, dict(task='detection', mode='calibration', seed=42))['selection']
    selected = []
    for is_ai in (False, True):
        row = next((row for row in selection if row['is_ai'] == is_ai), None)
        if row is None or row['split'] != 'dev':
            raise ValueError('selection probe requires one real and one AI development image')
        path = (data / row['relative_path']).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError('probe image escapes dataset')
        if core.sha256(path) != row['sha256']:
            raise ValueError('selection probe image hash mismatch')
        selected.append(row)
    transform = core.transforms.Compose([
        core.transforms.Resize((336, 336)), core.transforms.ToTensor(),
        core.transforms.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])])
    dataset = core.Images(selected, [0, 1], data, transform)
    images = torch.stack([dataset[index][0] for index in range(2)])
    return selected, images


def validate_actual_receipt(root, config, receipt):
    if (receipt.get('actual_images_verified') is not True or receipt.get('probe_role') != 'selection'
            or len(receipt.get('sample_ids', [])) != 2):
        raise ValueError('receipt must verify two actual selection images')
    if (receipt.get('manifest_sha256') != config['dataset']['manifest_sha256']
            or receipt.get('selection_sha256') != config['dataset']['selection_sha256']):
        raise ValueError('actual-image receipt dataset mismatch')
    for key, tolerance in [('reload_max_error', 1e-6), ('single_batch_max_error', 1e-5),
                           ('cache_max_error', 1e-5)]:
        value = receipt.get(key, float('inf'))
        if not np.isfinite(value) or value < 0 or value > tolerance:
            raise ValueError('actual-image consistency tolerance exceeded')
    cache = asset_path(root, receipt['cache']['path'])
    if core.sha256(cache) != receipt['cache']['sha256']:
        raise ValueError('actual-image score cache hash mismatch')
    cached = np.load(cache, allow_pickle=False)
    expected = np.asarray(receipt['scores'])
    if cached.shape != (2, 2) or not np.isfinite(cached).all() or not np.array_equal(cached, expected):
        raise ValueError('actual-image score cache contents mismatch')


def verify_load(root, config, requested):
    validate_route(root, config)
    path = asset_path(root, config['verification_receipt']['path'])
    cache_path = path.with_name('actual_selection_scores_q179_20261029.npy')
    if path.exists() or cache_path.exists():
        raise FileExistsError('preserve prior actual-image verification evidence')
    with core.research_lock(root), stage_status(
            path.with_name('verification_status_q179_20261029.json'), 'verify-load', config):
        idle = gpu_guard(config, requested)
        device = core.resolve_device(config, requested)
        selected, images = probe_images(root, config)
        model, audit = core.load_detector(root, config, device)
        first = core.score_loader(model, [(images, None)], device)
        with cache_path.open('xb') as stream:
            np.save(stream, first, allow_pickle=False)
        cached = np.load(cache_path, allow_pickle=False)
        same_model = core.score_loader(model, [(images, None)], device)
        del model
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        model, second_audit = core.load_detector(root, config, device)
        if second_audit['asset_digest'] != audit['asset_digest']:
            raise ValueError('assets changed between model loads')
        reloaded = core.score_loader(model, [(images, None)], device)
        singles = core.score_loader(model, [(image[None], None) for image in images], device)
        receipt = dict(status='verified', reload_verified=True, actual_images_verified=True,
                       probe_role='selection', sample_ids=[row['sample_id'] for row in selected],
                       image_sha256=[row['sha256'] for row in selected],
                       manifest_sha256=MANIFEST_SHA, selection_sha256=SELECTION_SHA, **audit,
                       reload_max_error=float(np.abs(cached - reloaded).max()),
                       single_batch_max_error=float(np.abs(reloaded - singles).max()),
                       cache_max_error=float(np.abs(cached - same_model).max()),
                       cache=dict(path=asset_reference(root, cache_path), sha256=core.sha256(cache_path)),
                       scores=cached.tolist(), device=device, gpu_index=config['gpu_index'],
                       gpu_memory_before_mib=idle, epoch=None, encoder_training=False)
        validate_actual_receipt(root, config, receipt)
        with path.open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return dict(receipt=str(path), sha256=core.sha256(path), verified=True)


def run(root, config, requested):
    validate_route(root, config)
    if config.get('execution_enabled') is not True or config.get('status') != 'ready':
        raise PermissionError('actual-image verification must explicitly enable this configuration')
    spec = config['verification_receipt']
    path = asset_path(root, spec['path'])
    if not spec.get('sha256') or core.sha256(path) != spec['sha256']:
        raise ValueError('actual-image verification receipt is unbound or changed')
    validate_actual_receipt(root, config, json.loads(path.read_text()))
    with core.research_lock(root):
        idle = gpu_guard(config, requested)
        if shutil.disk_usage(root).free < 20 * 1024**3:
            raise RuntimeError('less than 20 GiB disk free')
        device = core.resolve_device(config, requested)
        out = Path(root) / config['output']
        out.mkdir(parents=True, exist_ok=False)
        with stage_status(out / 'status.json', 'execute', config):
            core.write_json(out / 'resolved_config.json', dict(config, runtime_device=device,
                            gpu_memory_before_mib=idle, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
            result = core._evaluate(root, config, device, out)
            result['pretraining_overlap_caveat'] = config['protocol']['overlap_caveat']
            core.write_json(out / 'metrics.json', result)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / CONFIG)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--verify-load', action='store_true')
    actions.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.config.resolve() != (ROOT / CONFIG).resolve():
        raise ValueError('this wrapper only mutates its new registered q179 configuration')
    config = json.loads(args.config.read_text())
    if args.verify_load:
        result = verify_load(ROOT, config, args.device)
        config['verification_receipt']['sha256'] = result['sha256']
        config.update(status='ready', execution_enabled=True, blockers=[])
        from scripts.prepare_omniaid_q179 import write_json
        write_json(args.config, config)
        result['execution_enabled'] = True
    elif args.execute:
        result = run(ROOT, config, args.device)
    else:
        result = dict(status=config['status'], execution_enabled=config['execution_enabled'],
                      automatic_execution=config['automatic_execution'], images_read=False,
                      blockers=config['blockers'])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
