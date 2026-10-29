"""Download pinned official assets and safely convert CLIP; never initializes CUDA."""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path('outputs/image_experiments/assets/omniaid_v2')
CONFIG = Path('configs/image_experiments/omniaid_v2_q179_20261029.json')
LEGACY = Path('configs/image_experiments/omniaid_v2_evaluation.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def asset_path(root, path):
    owner = (Path(root) / ASSET_ROOT).resolve()
    target = (Path(root) / path).resolve()
    if not target.is_relative_to(owner) or target == owner:
        raise ValueError('path must remain inside the owned asset directory')
    return target


def asset_reference(root, path):
    owner = (Path(root) / ASSET_ROOT).resolve()
    return str(ASSET_ROOT / Path(path).resolve().relative_to(owner))


def verify_file(path, spec):
    if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
        raise ValueError('asset hash/size mismatch')


def convert_clip(source, destination, spec):
    import torch
    from safetensors.torch import load_file, save_file
    source, destination = Path(source), Path(destination)
    verify_file(source, spec)
    state = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError('CLIP binary must be a tensor state dictionary')
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in state.items()}
    partial = destination.with_name(destination.name + '.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError('preserve existing conversion artifact')
    save_file(tensors, str(partial), metadata={'format': 'pt'})
    converted = load_file(str(partial))
    if set(tensors) != set(converted) or any(
            tensors[key].dtype != converted[key].dtype or tensors[key].shape != converted[key].shape
            or not torch.equal(tensors[key], converted[key]) for key in tensors):
        raise ValueError('safetensors roundtrip tensor equality failed')
    result = dict(bytes=partial.stat().st_size, sha256=sha256(partial),
                  tensor_equality_verified=True, tensor_count=len(tensors),
                  source_sha256=spec['sha256'], torch_load_weights_only=True)
    os.link(partial, destination)
    partial.unlink()
    return result


def download(root, spec, blob=False):
    path = asset_path(root, spec['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify_file(path, spec)
        return
    partial = path.with_name(path.name + '.partial')
    # No automatic retry or resume: failed/partial evidence is retained.
    with partial.open('xb') as output:
        request = urllib.request.Request(spec['source_url'], headers={'User-Agent': 'aigc-research-pinned-assets/1'})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 200:
                raise ValueError('unexpected download HTTP status')
            if blob:
                document = json.load(response)
                if document.get('sha') != spec['git_blob'] or document.get('encoding') != 'base64':
                    raise ValueError('GitHub source blob identity mismatch')
                output.write(base64.b64decode(document['content']))
            else:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
    verify_file(partial, spec)
    os.link(partial, path)
    partial.unlink()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def import_stream(root, spec, stream, name, resume=None):
    path = asset_path(root, spec['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    receipt_path = Path(root) / ASSET_ROOT / f'import_{name}_q179_20261029.json'
    if path.exists() or partial.exists():
        raise FileExistsError('preserve existing imported asset or partial')
    record = dict(status='running', transport='explicit official HTTPS stream over SSH',
                  source_url=spec['source_url'], sha256=spec['sha256'], bytes=spec['bytes'],
                  path=spec['path'], received_bytes=0, tls_verification='normal; no bypass',
                  launch=dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    process_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    commandline=Path('/proc/self/cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                    script_sha256=sha256(Path(__file__)), started_unix=time.time()))
    if resume:
        record['resume'] = resume
        record['received_bytes'] = resume['prefix_bytes']
    with receipt_path.open('x') as output:
        json.dump(record, output, indent=2)
        output.write('\n')
    try:
        with partial.open('xb') as output:
            if resume:
                prefix = asset_path(root, resume['prefix_path'])
                if prefix.stat().st_size != resume['prefix_bytes'] or sha256(prefix) != resume['prefix_sha256']:
                    raise ValueError('archived prefix changed before controlled resume')
                with prefix.open('rb') as original:
                    shutil.copyfileobj(original, output, 8 * 1024 * 1024)
            last_reported = record['received_bytes']
            while chunk := stream.read(8 * 1024 * 1024):
                output.write(chunk)
                record['received_bytes'] += len(chunk)
                if record['received_bytes'] > spec['bytes']:
                    raise ValueError('import hash/size mismatch: stream exceeds pinned size')
                if record['received_bytes'] - last_reported >= 64 * 1024 * 1024:
                    write_json(receipt_path, record)
                    last_reported = record['received_bytes']
        verify_file(partial, spec)
        os.link(partial, path)
        partial.unlink()
        record.update(status='completed', completed_unix=time.time())
        write_json(receipt_path, record)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, completed_unix=time.time())
        write_json(receipt_path, record)
        raise
    return record


def resume_import(root, spec, stream, name, archive):
    archive = asset_path(root, archive)
    old_path = asset_path(root, archive / f'import_{name}_q179_20261029.json')
    old = json.loads(old_path.read_text())
    if old.get('status') != 'failed' or any(old.get(key) != spec[key]
            for key in ('source_url', 'sha256', 'bytes', 'path')):
        raise ValueError('prior import does not match pinned source identity')
    if Path('/proc/' + str(old['launch']['pid'])).exists():
        raise RuntimeError('prior importer has not exited')
    prefix = asset_path(root, archive / (Path(spec['path']).name + '.partial'))
    length = prefix.stat().st_size
    if not 0 < length < spec['bytes'] or length != old.get('received_bytes'):
        raise ValueError('prior import is not a bounded incomplete stream')
    resume = dict(prefix_path=asset_reference(root, prefix), prefix_bytes=length,
                  prefix_sha256=sha256(prefix), prior_receipt_path=asset_reference(root, old_path),
                  prior_receipt_sha256=sha256(old_path), expected_content_range_start=length,
                  expected_total_bytes=spec['bytes'], policy='one explicit controlled resume; no automatic retry')
    return import_stream(root, spec, stream, name, resume=resume)


def verify_import(root, spec, name):
    verify_file(asset_path(root, spec['path']), spec)
    path = Path(root) / ASSET_ROOT / f'import_{name}_q179_20261029.json'
    receipt = json.loads(path.read_text())
    if receipt.get('status') != 'completed' or any(receipt.get(key) != spec[key]
            for key in ('source_url', 'sha256', 'bytes', 'path')):
        raise ValueError('import receipt identity mismatch')
    if receipt.get('transport') != 'explicit official HTTPS stream over SSH':
        raise ValueError('unregistered asset import transport')
    return dict(path=str(path.relative_to(root)), sha256=sha256(path))


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--download-and-convert', action='store_true')
    actions.add_argument('--convert-imported', action='store_true')
    actions.add_argument('--import-stdin', choices=('model_config', 'clip_config', 'checkpoint', 'clip_binary'))
    actions.add_argument('--resume-stdin', choices=('checkpoint',))
    parser.add_argument('--conversion-owner', choices=('serial-queue',))
    args = parser.parse_args()
    if args.convert_imported and args.conversion_owner != 'serial-queue':
        raise PermissionError('offline conversion belongs to the serial queue; explicit owner required')
    if args.download_and_convert or args.convert_imported or args.import_stdin or args.resume_stdin:
        with preparation_lock(ROOT):
            return _prepare(args)
    return _prepare(args)


def _prepare(args):
    config_path = ROOT / CONFIG
    config = json.loads(config_path.read_text())
    pinned = json.loads((ROOT / LEGACY).read_text())['assets']
    # Every network URL, upstream hash and source path must remain the reviewed original.
    for name in ('source', 'checkpoint', 'model_config', 'clip_config'):
        if config['assets'][name] != pinned[name]:
            raise ValueError('asset specification differs from pinned legacy evidence')
    if config['assets']['clip_weights']['derived_from'] != pinned['clip_weights']['derived_from']:
        raise ValueError('CLIP source differs from pinned legacy evidence')
    if not (args.download_and_convert or args.convert_imported or args.import_stdin or args.resume_stdin):
        print(json.dumps({'status': config['status'], 'network_used': False}))
        return
    if config.get('execution_enabled') is not False:
        raise ValueError('asset preparation requires a disabled configuration')
    base = dict(pinned['clip_weights']['derived_from'],
                path=str(ASSET_ROOT / 'clip-vit-large-patch14-336/pytorch_model.bin'))
    if args.resume_stdin:
        archive = ROOT / ASSET_ROOT / 'failed-import-checkpoint-20261029'
        print(json.dumps(resume_import(ROOT, pinned[args.resume_stdin], sys.stdin.buffer, args.resume_stdin, archive)))
        return
    if args.import_stdin:
        spec = base if args.import_stdin == 'clip_binary' else pinned[args.import_stdin]
        print(json.dumps(import_stream(ROOT, spec, sys.stdin.buffer, args.import_stdin)))
        return
    directory = ROOT / ASSET_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / ('preparation_imported_q179_20261029.json' if args.convert_imported
                               else 'preparation_q179_20261029.json')
    if status_path.exists():
        raise FileExistsError('preserve previous preparation status')
    status = dict(status='running', pid=os.getpid(), started_unix=time.time(),
                  script_sha256=sha256(Path(__file__)), config_sha256=sha256(config_path),
                  cuda_used=False, retry_policy='none; preserve partials',
                  mode='convert_imported_offline' if args.convert_imported else 'direct_download',
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  process_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                  commandline=Path('/proc/self/cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                  cwd=str(Path.cwd()))
    write_json(status_path, status)
    try:
        for name in ('source', 'model_config', 'clip_config', 'checkpoint'):
            status['current_asset'] = name
            write_json(status_path, status)
            if args.convert_imported:
                if name == 'source':
                    verify_file(asset_path(ROOT, pinned[name]['path']), pinned[name])
                else:
                    config.setdefault('transport_receipts', {})[name] = verify_import(ROOT, pinned[name], name)
            else:
                download(ROOT, pinned[name], blob=name == 'source')
        status['current_asset'] = 'clip_binary'
        write_json(status_path, status)
        if args.convert_imported:
            config.setdefault('transport_receipts', {})['clip_binary'] = verify_import(ROOT, base, 'clip_binary')
        else:
            download(ROOT, base)
        status['current_asset'] = 'safe_conversion'
        write_json(status_path, status)
        converted = convert_clip(asset_path(ROOT, base['path']),
                                 asset_path(ROOT, config['assets']['clip_weights']['path']), base)
        receipt_path = directory / 'clip_conversion_q179_20261029.json'
        with receipt_path.open('x') as stream:
            json.dump(converted, stream, indent=2)
            stream.write('\n')
        config['assets']['clip_weights'].update(bytes=converted['bytes'], sha256=converted['sha256'],
                                               status='safe_conversion_verified')
        config['conversion_receipt'] = dict(path=str(receipt_path.relative_to(ROOT)), sha256=sha256(receipt_path))
        config['status'] = 'actual_image_verification_pending'
        config['blockers'] = ['Actual selection-image full reload receipt has not been produced or bound.']
        write_json(config_path, config)
        status.update(status='completed', completed_unix=time.time(), conversion=converted,
                      config_sha256=sha256(config_path), execution_enabled=False)
        write_json(status_path, status)
    except BaseException as error:
        # Do not print exception URLs: redirect failures may contain signed credentials.
        status.update(status='failed', error_type=type(error).__name__,
                      http_code=getattr(error, 'code', None), completed_unix=time.time())
        write_json(status_path, status)
        print(json.dumps(status), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(status))


@contextmanager
def preparation_lock(root):
    path = Path(root) / ASSET_ROOT / 'preparation.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('asset preparation already running') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    _main()


if __name__ == '__main__':
    main()
