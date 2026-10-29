#!/usr/bin/env python3
"""One-shot serial queue for the approved 2026-10-29 image experiments.

No retry/resume or network download: preserve any failed child and all outputs.
The separate official asset importer must complete before this queue can advance.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = Path('outputs/image-next-routes-20261029/queue')
ASSETS = Path('outputs/image_experiments/assets/omniaid_v2')
OMNI_CONFIG = Path('configs/image_experiments/omniaid_v2_q179_20261029.json')
SOURCES = [
    'scripts/run_image_next_routes.py', 'scripts/prepare_omniaid_q179.py',
    'scripts/evaluate_omniaid_q179.py', 'scripts/evaluate_omniaid.py',
    'scripts/run_image_openclip.py', 'research/image_attribution/openclip_route.py',
    'research/image_attribution/feature_ablation.py', 'research/image_attribution/protocol.py',
    'research/image_attribution/followups.py', 'research/image_attribution/models.py',
    'scripts/run_image_experiment.py', 'scripts/compare_image_scale_epochs.py',
    'scripts/compare_iab_v6_100_vs_v7_200_q179.py',
    'scripts/run_image_followup.py', 'scripts/compare_image_next_routes.py',
    'scripts/compare_image_rejection_routes.py',
    'configs/image_experiments/iab_openclip_q179_20261029.json',
    'configs/image_experiments/iab_next_routes_comparison_20261029.json',
    'configs/image_experiments/omniaid_v2_evaluation.json',
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                process_start_ticks=fields[19],
                commandline=(proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'))


def alive(record):
    try:
        current = identity(record['pid'])
        fields = (Path('/proc') / str(record['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and all(str(current[k]) == str(record[k])
                                      for k in ('pid', 'boot_id', 'process_start_ticks'))
    except (OSError, KeyError, ValueError):
        return False


def import_ready(path, spec=None):
    path = Path(path)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get('status') == 'failed':
        raise RuntimeError(f'asset import failed; preserve and diagnose: {path}')
    if record.get('status') == 'running':
        if not alive(record.get('launch', {})):
            raise RuntimeError(f'asset importer exited without completion: {path}')
        return False
    required = ('path', 'bytes', 'sha256', 'completed_unix', 'received_bytes', 'source_url')
    if record.get('status') != 'completed' or any(k not in record for k in required):
        raise ValueError('invalid completed asset receipt')
    if record['received_bytes'] != record['bytes']:
        raise ValueError('receipt byte count differs')
    if spec is not None and any(record[k] != spec[k] for k in ('path', 'bytes', 'sha256', 'source_url')):
        raise ValueError('receipt differs from pinned official asset')
    target = (ROOT / record['path']).resolve()
    if not target.is_relative_to((ROOT / ASSETS).resolve()):
        raise ValueError('receipt path escapes asset directory')
    if not target.is_file() or target.stat().st_size != record['bytes'] or sha256(target) != record['sha256']:
        raise ValueError('receipt file size or SHA mismatch')
    return True


def gpu_used():
    return int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip())


def completed(path, require_exit=True):
    data = json.loads(path.read_text())
    if data.get('status') != 'completed' or (require_exit and data.get('exit_code') != 0):
        raise ValueError(f'child did not complete successfully: {path}')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def stages():
    python = '.venv-image/bin/python'
    return [
        ('omniaid-conversion', [python, 'scripts/prepare_omniaid_q179.py', '--convert-imported',
                               '--conversion-owner', 'serial-queue'], False,
         ASSETS / 'preparation_imported_q179_20261029.json', True),
        ('omniaid-verify', [python, 'scripts/evaluate_omniaid_q179.py', '--verify-load', '--device', 'cuda'], True,
         ASSETS / 'verification_status_q179_20261029.json', True),
        ('omniaid-evaluate', [python, 'scripts/evaluate_omniaid_q179.py', '--execute', '--device', 'cuda'], True,
         Path('outputs/image_experiments/omniaid-v2-q179-20261029/status.json'), False),
        ('openclip', ['.venv-image-next/bin/python', 'scripts/run_image_openclip.py', '--execute'], True,
         Path('outputs/image_experiments/openclip-q179-20261029/status.json'), False),
        ('closed-comparison', [python, 'scripts/compare_image_next_routes.py', '--config',
                              'configs/image_experiments/iab_next_routes_comparison_20261029.json',
                              '--output', 'outputs/image-next-routes-20261029/closed-comparison'], False,
         Path('outputs/image-next-routes-20261029/closed-comparison/completion.json'), False),
        ('rejection-comparison', [python, 'scripts/compare_image_rejection_routes.py', '--output',
                                 'outputs/image-next-routes-20261029/rejection-comparison'], False,
         Path('outputs/image-next-routes-20261029/rejection-comparison/completion.json'), False),
    ]


def execute():
    directory = ROOT / STATE
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory.mkdir(exist_ok=False)
        launch = dict(**identity(), started_unix=time.time(), source_sha256=sha256(__file__))
        write(directory / 'launch.json', launch)
        hashes = {p: sha256(ROOT / p) for p in SOURCES}
        omni_hash = sha256(ROOT / OMNI_CONFIG)
        write(directory / 'registration.json', dict(sources=hashes, initial_omniaid_config_sha256=omni_hash,
              git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              stages=[dict(name=s[0], command=s[1]) for s in stages()], holdout_scored=False))
        shutil.copyfile(ROOT / OMNI_CONFIG, directory / 'initial_omniaid_config.json')
        state = dict(status='waiting_assets', launch=launch, completed_stages=[], started_unix=time.time())
        state_path = directory / 'status.json'

        def save(**values):
            state.update(values, updated_unix=time.time())
            write(state_path, state)

        def invariant():
            for p, expected in hashes.items():
                if sha256(ROOT / p) != expected:
                    raise ValueError(f'registered source/config changed: {p}')
            if sha256(ROOT / OMNI_CONFIG) != omni_hash:
                raise ValueError('OmniAID config changed outside a registered preparation stage')
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise RuntimeError('less than 20 GiB disk free')

        save()
        try:
            config = json.loads((ROOT / OMNI_CONFIG).read_text())
            specs = {n: config['assets'][n] for n in ('model_config', 'clip_config', 'checkpoint')}
            specs['clip_binary'] = dict(config['assets']['clip_weights']['derived_from'],
                path=str(ASSETS / 'clip-vit-large-patch14-336/pytorch_model.bin'))
            ready = set()
            while len(ready) != len(specs):
                invariant()
                for name, spec in specs.items():
                    if name not in ready and import_ready(ROOT / ASSETS / f'import_{name}_q179_20261029.json', spec):
                        ready.add(name)
                save(status='waiting_assets', ready_assets=sorted(ready))
                if len(ready) != len(specs):
                    time.sleep(30)
            for name, command, needs_gpu, receipt, changes_config in stages():
                invariant()
                if (ROOT / receipt).exists():
                    raise FileExistsError(f'preserve pre-existing stage receipt: {receipt}')
                if needs_gpu:
                    while gpu_used() > 1024:
                        save(status='waiting_gpu0', stage=name)
                        time.sleep(30)
                        invariant()
                save(status='running', stage=name, command=command)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4',
                                   MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
                with (directory / f'{name}.log').open('x') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        child_identity = identity(child.pid)
                    except OSError:
                        child_identity = dict(pid=child.pid)
                    write(directory / f'{name}-launch.json', dict(identity=child_identity, command=command, started_unix=time.time()))
                    save(child=child_identity)
                    code = child.wait()
                save(child_exit_code=code)
                if code != 0:
                    raise RuntimeError(f'{name} exited {code}; do not retry or overwrite')
                evidence = completed(ROOT / receipt, require_exit=name != 'omniaid-conversion')
                if changes_config:
                    omni_hash = sha256(ROOT / OMNI_CONFIG)
                    shutil.copyfile(ROOT / OMNI_CONFIG, directory / f'{name}-config.json')
                invariant()
                state['completed_stages'].append(dict(name=name, receipt=evidence, child=child_identity,
                    config_sha256=omni_hash, exit_code=code, completed_unix=time.time()))
                save()
            save(status='awaiting_git_archive', finished_unix=time.time(), child=None, exit_code=0)
        except BaseException as error:
            save(status='failed', error_type=type(error).__name__, error=str(error),
                 failed_unix=time.time(), exit_code=1)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print(json.dumps(dict(stages=[s[0] for s in stages()], execution=False, network=False)))


if __name__ == '__main__':
    main()
