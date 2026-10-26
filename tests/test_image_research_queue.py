import copy
import json
from pathlib import Path

import pytest

from scripts.run_image_research_queue import validate_parent, safe_rows, prerequisite_manifest, verify_registered_files, worker


def row(i='a', split='dev'):
    return dict(sample_id=i, split=split, group_id=i, sha256='h', pixel_sha256='p',
                is_ai=1, source_label='source', semantic_class='cat')


def test_parent_labels_and_partitions_cannot_move():
    parent = [row()]
    validate_parent(parent + [row('b')], parent)
    for key, value in [('split', 'train'), ('group_id', 'new'), ('source_label', 'renamed'),
                       ('sha256', 'new'), ('is_ai', 0)]:
        changed = copy.deepcopy(parent)
        changed[0][key] = value
        with pytest.raises(ValueError):
            validate_parent(changed, parent)
    with pytest.raises(ValueError):
        validate_parent([], parent)


def test_holdout_and_real_cannot_enter_attribution_extractor():
    safe_rows([row()], 'attribution')
    with pytest.raises(ValueError):
        safe_rows([row(split='holdout')], 'detection')
    real = dict(row(), is_ai=0, source_label=None)
    safe_rows([real], 'detection')
    with pytest.raises(ValueError):
        safe_rows([real], 'attribution')


def test_missing_baseline_is_pending_without_creating_files(tmp_path):
    cfg = {'prerequisites': ['monitor/completion.json']}
    assert prerequisite_manifest(tmp_path, cfg) is None
    assert list(tmp_path.iterdir()) == []


def test_failed_or_inconsistent_prerequisite_is_not_accepted(tmp_path):
    monitor = tmp_path / 'monitor'
    monitor.mkdir()
    (monitor / 'completion.json').write_text(json.dumps(dict(status='completed',
        summary='summary.json', manifest_sha256='aaa')))
    (monitor / 'validated-summary.json').write_text(json.dumps(dict(status='completed', manifest_sha256='bbb')))
    (tmp_path / 'summary.json').write_text(json.dumps(dict(status='completed', manifest_sha256='aaa')))
    with pytest.raises(ValueError):
        prerequisite_manifest(tmp_path, {'prerequisites': ['monitor/completion.json']})


def test_wrong_parent_certificate_and_empty_code_registry_are_rejected(tmp_path):
    monitor = tmp_path / 'monitor'; monitor.mkdir()
    cert = dict(status='completed', summary='summary.json', manifest_sha256='wrong-parent')
    for path in (monitor / 'completion.json', monitor / 'validated-summary.json', tmp_path / 'summary.json'):
        path.write_text(json.dumps(cert))
    with pytest.raises(ValueError, match='parent manifest'):
        prerequisite_manifest(tmp_path, dict(prerequisites=['monitor/completion.json'], parent_manifest={'sha256':'expected'}))
    with pytest.raises(ValueError, match='hash registration'):
        verify_registered_files(tmp_path, dict(implementation_sha256={}, assets={}))


def test_worker_never_overwrites_existing_terminal_state(tmp_path):
    for status in ('failed', 'completed', 'running'):
        original = json.dumps(dict(status=status, error='keep original evidence'))
        (tmp_path / 'state.json').write_text(original)
        with pytest.raises(FileExistsError):
            worker(tmp_path, {}, 'hash', tmp_path)
        assert (tmp_path / 'state.json').read_text() == original


def test_server_watcher_checks_each_half_hour_then_stops(tmp_path, monkeypatch):
    from scripts import run_image_research_queue as queue
    config_path = tmp_path / 'config.json'; config_path.write_text('{}')
    states = iter([{'status':'waiting_baselines'}, {'status':'completed'}])
    sleeps = []
    monkeypatch.setattr(queue, 'advance', lambda *args: next(states))
    monkeypatch.setattr(queue.time, 'sleep', sleeps.append)
    queue.watch(tmp_path, {}, config_path, tmp_path)
    assert sleeps == [1800]
    assert json.loads((tmp_path / 'watch.json').read_text())['status'] == 'completed'
    with pytest.raises(FileExistsError):
        queue.watch(tmp_path, {}, config_path, tmp_path)
