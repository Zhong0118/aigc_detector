#!/usr/bin/env python3
"""Run a registered follow-up whose completed baselines are named in its config.

Same plan/train/evaluate protocol, GPU0 gate, research lock and exclusive outputs as
scripts/run_image_followup.py, whose functions are reused unchanged (its registered
hash stays intact). The only difference: required baselines come from
config['baseline_runs'] and must match the config manifest, instead of the
hard-coded 20-image runs. A3 remains blocked while the family mapping is unreviewed.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.image_attribution.protocol import sha256, write_json
from scripts import run_image_followup as base


def required_baselines(config):
    names = config.get('baseline_runs')
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError('registered follow-up config must list its completed baseline runs')
    if any(not isinstance(n, str) or Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('baseline runs must be plain run directory names')
    return names


def check_baselines(config, root=ROOT):
    manifest = root / config['dataset'] / 'manifest.jsonl'
    if sha256(manifest) != config['manifest_sha256']:
        raise ValueError('dataset manifest differs from registered config')
    for name in required_baselines(config):
        folder = root / 'outputs/image_experiments' / name
        status = base.read_json(folder / 'status.json')
        if status.get('status') != 'completed' or status.get('exit_code') != 0:
            raise ValueError('baseline must finish first: ' + name)
        if base.read_json(folder / 'resolved_config.json').get('manifest_sha256') != config['manifest_sha256']:
            raise ValueError('baseline manifest differs from registered config: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--action', choices=['plan', 'train', 'evaluate'], default='plan')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    config, experiment = base.load_config(args.config, args.experiment)
    if args.action == 'plan':
        print(json.dumps(dict(experiment=args.experiment, protocol=experiment, execution_enabled=False,
            epochs=config['epochs'], holdout_used=False, baselines=required_baselines(config),
            note='No files written, no model loaded.'), ensure_ascii=False))
        return
    if not args.execute:
        parser.error('manual --execute is required')
    if args.action == 'train' and (not experiment['requires_training'] or args.device != 'cuda'):
        parser.error('this experiment does not require training, or CPU training is outside resource policy')
    check_baselines(config)
    manifest, parts = base.prepare(config, experiment)
    if experiment['mode'] == 'hierarchy':
        mapping = base.read_json(ROOT / experiment['family_mapping'])
        if mapping.get('reviewed') is not True:
            raise ValueError('A3 family mapping awaits evidence review; no model loaded')
    out = ROOT / config['output_root'] / args.experiment / args.action
    with (ROOT / 'outputs/image_experiments/gpu0-research.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.device == 'cuda':
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
                parser.error('set CUDA_VISIBLE_DEVICES=0; GPU1 service must remain untouched')
            used = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                                                '--format=csv,noheader,nounits'], text=True).strip())
            if used > 1024:
                raise RuntimeError('GPU0 busy; preserve other processes')
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'launcher.json', dict(launcher=str(Path(__file__).relative_to(ROOT)),
            launcher_sha256=sha256(Path(__file__)), runner_sha256=sha256(Path(base.__file__)),
            baselines=required_baselines(config), config_sha256=sha256(args.config)))
        start = time.monotonic()
        try:
            with (out / 'run.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), \
                    contextlib.redirect_stderr(log):
                result = base.run(args, config, experiment, manifest, parts, out)
            write_json(out / 'metrics.json', result)
            write_json(out / 'asset_manifest.json', {str(p.relative_to(out)): dict(sha256=sha256(p), bytes=p.stat().st_size)
                for p in sorted(out.rglob('*')) if p.is_file()})
            write_json(out / 'status.json', dict(status='completed', exit_code=0, seconds=time.monotonic() - start))
        except Exception as exc:
            write_json(out / 'status.json', dict(status='failed', exit_code=1, error=repr(exc),
                                                 seconds=time.monotonic() - start))
            raise
    print(json.dumps(dict(output=str(out), status='completed')))


if __name__ == '__main__':
    main()
