"""Manual, offline evaluation of the pinned author OmniAID CLIP v2 checkpoint."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from research.image_attribution.followups import partition_rows
from research.image_attribution.protocol import sha256, validate_manifest, write_json
from scripts.run_image_experiment import Images, write_predictions
from scripts.run_image_followup import evaluate_scores

SOURCE_SHA256 = '0280750b22d1d60815b89d6d35cff8ee6a474a5615a3f8d95a3db9998d028596'
PATCHED_SHA256 = 'e70258fdb053424add1a2d0eeff6573a209a56c003a3958f66d04ef868b1f212'
ASSET_NAMES = ('source', 'checkpoint', 'model_config', 'clip_config', 'clip_weights')
MODEL_PARAMETERS = dict(num_experts=6, rank_per_expert=1, moe_top_k=2, moe_router_hidden_dim=256)
CLASSES = ['real', 'ai']


@contextmanager
def research_lock(root):
    path = Path(root) / 'outputs/image_experiments/gpu0-research.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('research lock busy; baseline or another experiment is running') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_device(config, requested):
    if requested == 'cpu':
        return 'cpu'
    if torch.device(requested).type != 'cuda':
        raise ValueError('only cpu or an explicit cuda device is supported')
    index = config.get('gpu_index')
    if type(index) is not int or index < 0:
        raise ValueError('a physical gpu_index must be configured')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before configured GPU isolation')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
    return 'cuda:0'


def verify_assets(root, config):
    records = {}
    for name in ASSET_NAMES:
        spec = config.get('assets', {}).get(name, {})
        if not spec.get('sha256') or not spec.get('bytes') or not spec.get('path'):
            raise ValueError(f'unbound asset: {name}')
        path = (Path(root) / spec['path']).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'missing asset: {path}')
        if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise ValueError(f'asset hash/size mismatch: {name}')
        records[name] = dict(path=str(path), sha256=spec['sha256'], bytes=spec['bytes'])
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return records, hashlib.sha256(encoded).hexdigest()


def reviewed_source(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('source does not match the reviewed author Git blob')
    old = b'CLIPModel.from_pretrained(pretrained_path).vision_model'
    new = b'CLIPModel.from_pretrained(pretrained_path, local_files_only=True, trust_remote_code=False, use_safetensors=True).vision_model'
    if source.count(old) != 1:
        raise ValueError('offline patch anchor mismatch')
    patched = source.replace(old, new)
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
        raise ValueError('reviewed offline patch hash mismatch')
    module = ModuleType('_reviewed_omniaid_v2')
    module.__file__ = str(path)
    exec(compile(patched, str(path), 'exec'), module.__dict__)
    return module


def load_detector(root, config, device):
    records, asset_digest = verify_assets(root, config)
    paths = {key: Path(value['path']) for key, value in records.items()}
    if (paths['clip_config'].name != 'config.json' or paths['clip_weights'].name != 'model.safetensors'
            or paths['clip_config'].parent != paths['clip_weights'].parent):
        raise ValueError('CLIP config and safetensors must share their registered local model directory')
    published = json.loads(paths['model_config'].read_text())
    if any(published.get(key) != value for key, value in MODEL_PARAMETERS.items()):
        raise ValueError('checkpoint configuration is not the registered CLIP v2 variant')
    module = reviewed_source(paths['source'])
    model = module.OmniAID(SimpleNamespace(**MODEL_PARAMETERS, is_hybrid=True,
                                         CLIP_path=str(paths['clip_weights'].parent)))
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    state = checkpoint.get('model', checkpoint)
    if not isinstance(state, dict) or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError('checkpoint must contain a tensor state_dict; unsafe pickle fallback is disabled')
    cleaned = {key.removeprefix('module.'): value for key, value in state.items()}
    if len(cleaned) != len(state):
        raise ValueError('checkpoint keys collide after module-prefix removal')
    model.load_state_dict(cleaned, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device), dict(assets=records, asset_digest=asset_digest,
                                 original_source_sha256=SOURCE_SHA256, patched_source_sha256=PATCHED_SHA256)


def score_loader(model, loader, device):
    batches = []
    with torch.inference_mode():
        for images, _ in loader:
            probability = model(images.to(device))
            if (not torch.is_tensor(probability) or probability.shape != (len(images),)
                    or not torch.isfinite(probability).all()
                    or ((probability < 0) | (probability > 1)).any()):
                raise ValueError('author model must return one finite AI probability per image')
            # The author already returns softmax[:, 1]. Do not sigmoid/softmax again.
            p = probability.cpu().numpy()
            batches.append(np.stack((1 - p, p), axis=1))
    return np.concatenate(batches)


def verify_load(root, config, device):
    """Explicit asset smoke check only; creates a receipt, never enables evaluation."""
    with research_lock(root):
        return _verify_load(root, config, resolve_device(config, device))


def _verify_load(root, config, device):
    images = torch.zeros(2, 3, 336, 336)
    images[1].fill_(.25)
    model, audit = load_detector(root, config, device)
    first = score_loader(model, [(images, None)], device)
    del model
    model, _ = load_detector(root, config, device)
    second = score_loader(model, [(images, None)], device)
    singles = score_loader(model, [(image[None], None) for image in images], device)
    errors = dict(reload_max_error=float(np.abs(first - second).max()),
                  single_batch_max_error=float(np.abs(second - singles).max()))
    if errors['reload_max_error'] > 1e-6 or errors['single_batch_max_error'] > 1e-5:
        raise ValueError('full reload or single/batch consistency check failed')
    receipt = dict(status='verified', reload_verified=True, **audit, **errors)
    path = Path(root) / config['verification_receipt']['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('verification receipt already exists; preserve prior evidence')
    write_json(path, receipt)
    return dict(receipt=str(path), sha256=sha256(path), execution_enabled=False)


def load_parts(root, config, seed):
    dataset = config['dataset']
    if not dataset.get('manifest_sha256'):
        raise ValueError('the fixed 100-per-cell manifest must be bound before evaluation')
    data = Path(root) / dataset['path']
    if sha256(data / 'manifest.jsonl') != dataset['manifest_sha256']:
        raise ValueError('manifest hash mismatch')
    if sha256(data / 'selection.json') != dataset.get('selection_sha256'):
        raise ValueError('fixed dataset selection hash mismatch')
    rows = [json.loads(line) for line in (data / 'manifest.jsonl').read_text().splitlines() if line.strip()]
    validate_manifest(rows, data, verify_files=False)
    all_parts = partition_rows(rows, dict(task='detection', mode='calibration', seed=seed))
    parts = {role: all_parts[role] for role in ('calibration', 'evaluation')}
    for role, selected in parts.items():
        if not selected or {row['is_ai'] for row in selected} != {False, True}:
            raise ValueError(f'{role} requires both binary classes')
        if any(row['split'] != 'dev' for row in selected):
            raise ValueError('only registered development calibration/evaluation images may be read')
        for row in selected:
            path = (data / row['relative_path']).resolve()
            if not path.is_relative_to(data.resolve()):
                raise ValueError('selected image path escapes the registered dataset')
            if not row.get('sha256') or sha256(path) != row['sha256']:
                raise ValueError(f'{role} image hash mismatch: {row["sample_id"]}')
    return data, parts


def run(root, config, device):
    if config.get('execution_enabled') is not True or config.get('status') != 'ready':
        raise PermissionError('OmniAID evaluation is disabled until assets, manifest and reload receipt are registered')
    if config.get('automatic_execution') is not False or config.get('task') != 'detection' or config.get('epoch') is not None:
        raise ValueError('this is a manual pretrained detection evaluation, with epoch=null')
    with research_lock(root):
        device = resolve_device(config, device)
        out = Path(root) / config['output']
        # Outside the failure handler: repeated calls must never rewrite prior status.
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        write_json(out / 'status.json', dict(status='running', exit_code=None, task='detection', epoch=None))
        try:
            write_json(out / 'resolved_config.json', dict(config, runtime_device=device,
                       cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
            result = _evaluate(root, config, device, out)
            write_json(out / 'status.json', dict(status='completed', exit_code=0,
                       elapsed_seconds=time.monotonic() - started))
            return result
        except BaseException as error:
            write_json(out / 'status.json', dict(status='failed', exit_code=1,
                       error_type=type(error).__name__, error=str(error)[:2000],
                       elapsed_seconds=time.monotonic() - started))
            raise


def _evaluate(root, config, device, out):
    _, asset_digest = verify_assets(root, config)
    receipt_spec = config['verification_receipt']
    receipt_path = Path(root) / receipt_spec['path']
    if not receipt_spec.get('sha256') or sha256(receipt_path) != receipt_spec['sha256']:
        raise ValueError('verified reload receipt must be explicitly hash-bound')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'verified' or receipt.get('reload_verified') is not True or receipt.get('asset_digest') != asset_digest:
        raise ValueError('reload receipt does not verify the current assets')
    if receipt.get('reload_max_error', float('inf')) > 1e-6 or receipt.get('single_batch_max_error', float('inf')) > 1e-5:
        raise ValueError('reload receipt failed consistency tolerance')
    data, parts = load_parts(root, config, config['seed'])
    model, audit = load_detector(root, config, device)
    transform = transforms.Compose([transforms.Resize((336, 336)), transforms.ToTensor(),
                                    transforms.Normalize([.48145466, .4578275, .40821073],
                                                         [.26862954, .26130258, .27577711])])
    scores = {}
    for role, selected in parts.items():
        labels = [int(row['is_ai']) for row in selected]
        loader = DataLoader(Images(selected, labels, data, transform), batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)
        scores[role] = score_loader(model, loader, device)
        directory = out / role
        directory.mkdir()
        write_predictions(directory, selected, labels, CLASSES, scores[role], 'detection')
    result = evaluate_scores(dict(task='detection', mode='calibration', seed=config['seed'],
                                  target_fprs=config['target_fprs'], bootstrap_repeats=config['bootstrap_repeats']),
                             parts, CLASSES, scores)
    result.update(task='detection', epoch=None, pretrained=True, encoder_training=False,
                  holdout_scored=False, calibration_used=True, score_semantics='p_ai; author softmax[:,1]')
    write_json(out / 'metrics.json', result)
    write_json(out / 'provenance.json', dict(**audit, manifest_sha256=config['dataset']['manifest_sha256'],
                                           verification_receipt_sha256=receipt_spec['sha256'],
                                           role_ids={role: [row['sample_id'] for row in selected] for role, selected in parts.items()}))
    write_json(out / 'config.json', config)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/image_experiments/omniaid_v2_evaluation.json')
    parser.add_argument('--device', default='cpu')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--execute', action='store_true')
    actions.add_argument('--verify-load', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.verify_load:
        result = verify_load(ROOT, config, args.device)
    elif args.execute:
        result = run(ROOT, config, args.device)
    else:
        result = dict(status=config['status'], execution_enabled=config['execution_enabled'],
                      assets=config['assets'], blockers=config['blockers'], images_read=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
