"""Registered follow-up launcher: baseline list and manifest identity fail closed."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_image_followup_registered as launcher

ROOT = Path(__file__).resolve().parents[1]


def fake_root(tmp_path, status='completed', manifest_sha=None):
    data = tmp_path / 'data'; data.mkdir()
    (data / 'manifest.jsonl').write_text('{"sample_id": "x"}\n')
    digest = hashlib.sha256((data / 'manifest.jsonl').read_bytes()).hexdigest()
    run = tmp_path / 'outputs/image_experiments/d0-run'; run.mkdir(parents=True)
    (run / 'status.json').write_text(json.dumps(dict(status=status, exit_code=0 if status == 'completed' else 1)))
    (run / 'resolved_config.json').write_text(json.dumps(dict(manifest_sha256=manifest_sha or digest)))
    return dict(dataset='data', manifest_sha256=digest, baseline_runs=['d0-run'])


@pytest.mark.parametrize('names', [None, [], ['a', 'a'], ['../escape'], ['sub/run']])
def test_baseline_list_must_be_explicit_unique_plain_names(names):
    with pytest.raises(ValueError):
        launcher.required_baselines({} if names is None else dict(baseline_runs=names))


def test_completed_matching_baseline_passes(tmp_path):
    launcher.check_baselines(fake_root(tmp_path), root=tmp_path)


@pytest.mark.parametrize('case', ['incomplete', 'other_manifest', 'config_manifest'])
def test_incomplete_or_mismatched_baseline_is_rejected(tmp_path, case):
    config = fake_root(tmp_path, status='failed' if case == 'incomplete' else 'completed',
                       manifest_sha='0' * 64 if case == 'other_manifest' else None)
    if case == 'config_manifest':
        config['manifest_sha256'] = '1' * 64
    with pytest.raises(ValueError):
        launcher.check_baselines(config, root=tmp_path)


def test_q179_config_keeps_template_protocol_and_only_rebinds_data():
    old = json.loads((ROOT / 'configs/image_experiments/iab_followups_v6_100_e30.json').read_text())
    new = json.loads((ROOT / 'configs/image_experiments/iab_followups_v7_200_q179_e30.json').read_text())
    assert new['automatic_execution'] is False
    for key in ('seed', 'epochs', 'batch_size', 'workers', 'optimizer', 'scheduler', 'weights', 'development_roles'):
        assert new[key] == old[key]
    assert set(new['experiments']) == set(old['experiments'])
    for name, item in old['experiments'].items():
        changed = {k for k in set(item) | set(new['experiments'][name]) if item.get(k) != new['experiments'][name].get(k)}
        assert changed <= {'baseline'}
    assert all(r.endswith('iab-v7-200-q179-e30') for r in new['baseline_runs'])
