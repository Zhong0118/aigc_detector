#!/usr/bin/env python3
"""Read-only IAB v2 snapshot by default; --advance enables lock-protected work."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlsplit

PROXY_URL = "http://127.0.0.1:17898"
PROBE_BYTES = 1  # Connectivity only; throughput comparisons use retained image files.
FREE_SPACE_RESERVE_FRACTION = 0.20


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default



def monitor_directory(root):
    return root/Path(os.environ.get('IAB_MONITOR_STATE_DIR','outputs/iab-v2-monitor'))


def settings(root):
    config = dict(dataset='outputs/image_experiments/iab-balanced-v2-200',
                  suffix='v2-200', per_cell=200, workers=4,
                  reuse_from=str(root.parent/'image-m0-m2-paper/outputs/image_experiments/iab-balanced-v1'),
                  prefix_selection=False, route='auto', epochs=20, gpu_index=0)
    config.update(read(monitor_directory(root)/'config.json', {}))
    if (config['route'] not in {'auto', 'direct', 'proxy'} or
        min(config['per_cell'], config['workers'], config['epochs']) < 1 or
        not isinstance(config['epochs'], int) or not isinstance(config['gpu_index'], int) or config['gpu_index'] < 0):
        raise ValueError('invalid monitor configuration')
    config['summary'] = f"outputs/image_experiments/iab-m2-{config['suffix']}-summary.json"
    return config


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, separators=(',', ':')) + '\n')


def identity(pid):
    """Linux boot ID and process start ticks prevent PID reuse confusion."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return [Path('/proc/sys/kernel/random/boot_id').read_text().strip(), stat[19]]
    except OSError:
        return None


def alive(state):
    return bool(state and state.get('identity') and identity(state['pid']) == state['identity'])


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        yield handle


def expiry(path):
    """Summarize URL expiry without returning or logging any URL value."""
    values = []
    for row in read(path, {}).get('results', []):
        query = parse_qs(urlsplit(row.get('url', '')).query)
        for key in ('Expires', 'expires', 'e'):
            try:
                values.append(int(query[key][0]))
            except (KeyError, ValueError):
                pass
        try:
            start = datetime.strptime(query['X-Amz-Date'][0], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
            values.append(int(start.timestamp()) + int(query['X-Amz-Expires'][0]))
        except (KeyError, ValueError):
            pass
    return dict(known=len(values), earliest_unix=min(values) if values else None,
                seconds_remaining=int(min(values)-time.time()) if values else None)


def request_probe(file_id, total_size, proxy_url=None):
    import requests

    length = min(PROBE_BYTES, int(total_size))
    session = requests.Session()
    session.trust_env = False
    if proxy_url:
        session.proxies.update(http=proxy_url, https=proxy_url)
    started = time.monotonic()
    result = dict(success=False, requests_attempted=1, requests_succeeded=0,
                  request_failures=0, timeouts=0, bytes=0)
    try:
        url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        with session.get(url, headers={'Range': f'bytes=0-{length-1}', 'Accept-Encoding': 'identity'},
                         stream=True, timeout=(8, 20)) as response:
            result['status_code'] = response.status_code
            expected = f'bytes 0-{length-1}/{total_size}'
            body = response.raw.read(length + 1)
            result['bytes'] = len(body)
            result['success'] = (response.status_code == 206 and
                                 response.headers.get('Content-Range') == expected and
                                 response.headers.get('Content-Encoding', 'identity') == 'identity' and
                                 len(body) == length)
    except requests.Timeout:
        result['timeouts'] = 1
        result['request_failures'] = 1
        result['error_type'] = 'Timeout'
    except requests.RequestException as exc:
        result['request_failures'] = 1
        result['error_type'] = type(exc).__name__
    elapsed = time.monotonic() - started
    result['requests_succeeded'] = int(result['success'])
    result['request_failures'] = int(not result['success'])
    result['elapsed_seconds'] = elapsed
    result['throughput_mib_s'] = result['bytes'] / (1024 * 1024 * elapsed) if elapsed else 0.0
    return result


def probe_transports(metadata_path, proxy_url=PROXY_URL):
    metadata = read(metadata_path)
    if not metadata or not metadata.get('files'):
        failure = dict(success=False, request_failures=1, error_type='metadata_missing',
                       requests_attempted=0, requests_succeeded=0, timeouts=0, bytes=0)
        return dict(sample_bytes=PROBE_BYTES, direct=failure, proxy=failure)
    sample = next((item for item in metadata['files'] if not item.get('restricted')), None)
    if sample is None:
        failure = dict(success=False, request_failures=1, error_type='no_public_file',
                       requests_attempted=0, requests_succeeded=0, timeouts=0, bytes=0)
        return dict(sample_bytes=PROBE_BYTES, direct=failure, proxy=failure)
    return dict(sample_bytes=min(PROBE_BYTES, int(sample['size'])),
                direct=request_probe(sample['id'], sample['size']),
                proxy=request_probe(sample['id'], sample['size'], proxy_url))


def worker_environment(verified_proxy_url=None):
    env = os.environ.copy()
    for key in ('HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy',
                'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'):
        env.pop(key, None)
    if verified_proxy_url:
        env['HTTP_PROXY'] = verified_proxy_url
        env['HTTPS_PROXY'] = verified_proxy_url
    return env


def gpu(index=0):
    try:
        result = subprocess.run(['nvidia-smi', f'--id={index}', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
        used, total, utilization = map(int, result.stdout.strip().split(','))
        return dict(index=index, used_mib=used, total_mib=total, utilization=utilization)
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(index=index, unavailable=True)



def remaining_selected_bytes(data):
    selection = read(data/'selection.json', {})
    completed = set()
    for receipt_path in (data/'receipts').glob('*.json'):
        receipt = read(receipt_path, {})
        completed.update(row.get('sample_id') for row in receipt.get('rows', [])
                         if row.get('sample_id'))
    total = 0
    for key, entries in selection.get('entries', {}).items():
        source, semantic = key.split('/', 1)
        for entry in entries:
            identifier = hashlib.sha256(
                f"{source}/{semantic}/{entry['filename']}".encode()).hexdigest()[:24]
            if f'iab-{identifier}' not in completed:
                total += int(entry.get('size', 0))
    return total

def snapshot(root):
    state_dir = monitor_directory(root)
    config = settings(root)
    data = root/config['dataset']
    stages = {}
    for name in ('d0-resnet50', 'a0-resnet50', 'ofa'):
        folder = root/f"outputs/image_experiments/{name}-iab-{config['suffix']}"
        history = read(folder/'history.json', [])
        stages[name] = dict(status=read(folder/'status.json'), epochs=len(history),
                            latest_epoch=history[-1] if history else None)
    workers = {kind: read(state_dir/f'{kind}.json') for kind in ('download', 'pipeline')}
    for value in workers.values():
        if value:
            value['alive'] = alive(value)
    metadata_path = state_dir/'iab-public-v3.2-metadata.json'
    free_bytes = shutil.disk_usage(root).free
    report = dict(time=time.time(), dataset=config['dataset'], run_suffix=config['suffix'], disk_free_bytes=free_bytes,
                remaining_selected_bytes=remaining_selected_bytes(data), gpu=gpu(config['gpu_index']),
                download=read(data/'download_status.json'), receipts=len(list((data/'receipts').glob('*.json'))),
                verified_sidecars=len(list((data/'images').rglob('*.json'))),
                sidecar_verification='count of downloader verification records; monitor does not rehash images',
                failure_count=len(read(data/'download_failures.json', [])),
                transport=probe_transports(metadata_path), stages=stages, workers=workers,
                completion=read(state_dir/'completion.json'))
    if config['gpu_index'] == 0:
        report['gpu0'] = report['gpu']  # Preserve older GPU0 monitor consumers.
    return report


def command(root, kind):
    config = settings(root)
    py = str(root/'.venv-image/bin/python')
    data = root/config['dataset']
    if kind == 'download':
        args = [py, '-u', 'scripts/download_iab_subset.py',
                '--metadata', str(monitor_directory(root)/'iab-public-v3.2-metadata.json'),
                '--output', str(data), '--reuse-from', str(root/config['reuse_from']),
                '--per-cell', str(config['per_cell']), '--workers', str(config['workers'])]
        if config['prefix_selection']:
            args.append('--prefix-selection')
        return args
    args=['bash', 'scripts/run_iab_m2.sh', str(data), config['suffix'], config['summary']]
    args.extend([str(root/config['parent_manifest']) if config.get('parent_manifest') else '',
                 str(config['epochs']), str(config['gpu_index'])])
    return args


def launch(root, kind, proxy_url=None):
    folder = monitor_directory(root)
    state = dict(status='launch_intent', time=time.time(),
                 verified_proxy=bool(proxy_url), proxy_url=proxy_url)
    write(folder/f'{kind}.json', state)
    args = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
            '--worker', kind]
    if proxy_url:
        args.extend(['--proxy-url', proxy_url])
    with (folder/f'{kind}.log').open('ab') as log:
        proc = subprocess.Popen(args, cwd=root, stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    state.update(status='running', pid=proc.pid, identity=identity(proc.pid))
    write(folder/f'{kind}.json', state)


def required_assets(root):
    assets = root/'outputs/image_experiments/assets'
    paths = ('resnet50-0676ba61.pth', 'ofa_fingerprint.pth', 'model_best.th')
    return [assets/name for name in paths if not (assets/name).is_file()]



def matches_worker(args, data, kind, cwd=None):
    """Match a worker's output/manifest, never a reused parent dataset."""
    scripts = {'download_iab_subset.py'} if kind == 'download' else {'run_iab_m2.sh', 'run_image_experiment.py'}
    for index, arg in enumerate(args):
        name = Path(os.fsdecode(arg)).name
        if name not in scripts:
            continue
        try:
            flag = b'--output' if kind == 'download' else b'--manifest'
            position = index + 1 if name == 'run_iab_m2.sh' else args.index(flag) + 1
            target = Path(os.fsdecode(args[position]))
        except (ValueError, IndexError):
            return False
        if not target.is_absolute():
            target = (cwd or Path.cwd()) / target
        expected = data/'manifest.jsonl' if name == 'run_image_experiment.py' else data
        try:
            return target.resolve() == expected.resolve()
        except OSError:
            return False
    return False


def advance(root, report):
    config = settings(root)
    folder = monitor_directory(root)
    if report['completion']:
        return
    download = report['download'] or {}
    kind = 'pipeline' if download.get('status') == 'completed' else 'download'
    state = read(folder/f'{kind}.json')
    if state:
        if state.get('status') == 'failed':
            report['needs_attention'] = f'previous {kind} worker failed; inspect state/log before explicit reset'
        elif not alive(state) and state.get('status') != 'completed':
            report['needs_attention'] = f'{kind} stopped or launch ambiguous; inspect state/log before manual reset'
        return
    proxy_url = None
    if kind == 'pipeline':
        selected_gpu = report.get('gpu', report.get('gpu0', {}))
        if selected_gpu.get('unavailable') or selected_gpu.get('used_mib', 999999) > 1024:
            report['waiting'] = f"GPU {config['gpu_index']} unavailable or busy"
            return
        missing = required_assets(root)
        if missing:
            report['needs_attention'] = 'required image model assets missing'
            return
    else:
        metadata_path = folder/'iab-public-v3.2-metadata.json'
        if not metadata_path.is_file():
            report['needs_attention'] = 'public dataset metadata missing'
            return
        transport = report.get('transport') or {}
        direct = (transport.get('direct') or {}).get('success', False)
        proxy = (transport.get('proxy') or {}).get('success', False)
        if config['route'] == 'direct':
            if not direct:
                report['waiting'] = 'configured direct route unavailable'
                return
        elif config['route'] == 'proxy':
            if not proxy:
                report['waiting'] = 'configured proxy route unavailable'
                return
            proxy_url = PROXY_URL
        elif proxy:
            proxy_url = PROXY_URL
        elif not direct:
            report['waiting'] = 'public Range request failed over direct and configured proxy paths'
            return
        free_bytes = int(report.get('disk_free_bytes', shutil.disk_usage(root).free))
        remaining = int(report.get('remaining_selected_bytes',
                                   remaining_selected_bytes(root/config['dataset'])))
        reserve = int(free_bytes * FREE_SPACE_RESERVE_FRACTION)
        if free_bytes < remaining + reserve:
            report['waiting'] = 'insufficient free disk for remaining selected data and reserve'
            report['disk_reserve_bytes'] = reserve
            return
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = proc.read_bytes().split(b'\0')
        except OSError:
            continue
        if matches_worker(args, root/config['dataset'], kind, proc.parent/'cwd'):
            report['waiting'] = f'existing {kind} process detected'
            return
    launch(root, kind, proxy_url=proxy_url)
    report['launched'] = kind
    report['verified_proxy'] = bool(proxy_url)


def worker(root, kind, proxy_url=None):
    config = settings(root)
    folder = monitor_directory(root)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'supervisor.lock').open('a') as supervisor:
        fcntl.flock(supervisor, fcntl.LOCK_EX)
    with lock(folder/f'{kind}.lock') as held:
        if held is None:
            return 3
        result = subprocess.run(command(root, kind), cwd=root,
                                env=worker_environment(proxy_url))
        state = read(folder/f'{kind}.json', {})
        state.update(status='completed' if result.returncode == 0 else 'failed',
                     exit_code=result.returncode, finished=time.time())
        if kind == 'pipeline' and result.returncode == 0:
            verified = folder/'validated-summary.json'
            if verified.exists():
                verified.unlink()
            verify = subprocess.run([str(root/'.venv-image/bin/python'), 'scripts/summarize_iab_m2.py',
                                     '--dataset', config['dataset'],
                                     '--run-suffix', config['suffix'], '--output', str(verified)],
                                    cwd=root, env=worker_environment())
            if verify.returncode:
                state.update(status='failed', exit_code=verify.returncode)
            else:
                write(folder/'completion.json', dict(status='completed', time=time.time(),
                      manifest_sha256=read(verified)['manifest_sha256'],
                      summary=config['summary']))
        write(folder/f'{kind}.json', state)
        return state['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--state-dir',type=Path,help='independent monitor state; existing completed runs remain untouched')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--advance', action='store_true')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--interval', type=int, default=600)
    parser.add_argument('--worker', choices=('download', 'pipeline'), help=argparse.SUPPRESS)
    parser.add_argument('--proxy-url', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.state_dir:
        os.environ['IAB_MONITOR_STATE_DIR']=str(args.state_dir)
    if args.worker:
        return worker(root, args.worker, args.proxy_url)
    if args.interval < 1:
        parser.error('interval must be positive')
    folder = monitor_directory(root)
    loop_lock = None
    if args.loop:
        folder.mkdir(parents=True, exist_ok=True)
        loop_lock = (folder/'loop.lock').open('a')
        try:
            fcntl.flock(loop_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 3
        write(folder/'loop.json', dict(pid=os.getpid(), identity=identity(os.getpid()),
                                       interval=args.interval))
    while True:
        if args.advance:
            with lock(folder/'supervisor.lock') as held:
                report = snapshot(root)
                if held is not None:
                    append_jsonl(folder/'transport.jsonl',
                                 dict(time=report['time'], **report['transport']))
                    advance(root, report)
                    write(folder/'snapshot.json', report)
        else:
            report = snapshot(root)
        print(json.dumps(report), flush=True)
        if not args.loop or report.get('completion'):
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
