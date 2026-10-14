import json
import pytest
from scripts import fuse_window_a0_classical as fusion


def row(key='one', true='openai', reverse=False, split='validation'):
    probs = {'openai': 0.8, 'qwen': 0.2}
    if reverse:
        probs = dict(reversed(list(probs.items())))
    return {'request_key': key, 'true_family': true, 'family_probabilities': probs, 'split': split}


@pytest.mark.parametrize('mismatch', ['keys', 'labels', 'order'])
def test_fusion_refuses_misaligned_branches(mismatch):
    left = {'one': row()}
    right = {'one': row()}
    if mismatch == 'keys':
        right = {'other': row('other')}
    elif mismatch == 'labels':
        right = {'one': row(true='qwen')}
    else:
        right = {'one': row(reverse=True)}
    with pytest.raises(ValueError, match='align|label|key'):
        fusion.mix(['one'], {'A0': left, 'B0': right}, {'A0': 0.5, 'B0': 0.5})


def test_load_jsonl_defaults_to_validation_and_rejects_duplicate_keys(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in [row(), row('test', split='test')]))
    assert list(fusion.load_jsonl(path)) == ['one']
    path.write_text('\n'.join(json.dumps(r) for r in [row(), row()]))
    with pytest.raises(ValueError, match='duplicate'):
        fusion.load_jsonl(path)


def test_aligned_fusion_keeps_all_documents():
    rows = {'one': row(), 'two': row('two')}
    assert fusion.mix(['one', 'two'], {'A': rows, 'B': rows}, {'A': 0.5, 'B': 0.5}) == ['openai', 'openai']


def test_offline_comparator_requires_explicit_unverified_acknowledgement(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['compare', '--seed', '42', '--output', 'unused.json'])
    with pytest.raises(SystemExit) as error:
        fusion.main()
    assert error.value.code == 2
    assert 'unverified' in capsys.readouterr().err


def test_offline_comparator_labels_historical_results(tmp_path, monkeypatch):
    monkeypatch.setattr(fusion, '__file__', str(tmp_path / 'scripts/comparator.py'))
    paths = [
        'family-window-split/A0-closed-seed42/predictions.jsonl',
        'family-phase3/A0/A0-closed-seed42/predictions.jsonl',
        'family-phase4/fast-track-v1/classical/closed-seed42/B0/predictions.jsonl',
        'family-phase4/fast-track-v1/classical/closed-seed42/B1/predictions.jsonl',
        'family-phase1/final/predictions.jsonl',
    ]
    for name in paths:
        path = tmp_path / 'outputs/attribution' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        experiment = 'B3' if 'family-phase1' in name else None
        payload = {**row(), 'predicted_family': 'openai'}
        if experiment:
            payload['experiment_id'] = experiment
        path.write_text(json.dumps(payload) + '\n')
    metrics_path = tmp_path / 'outputs/attribution/family-phase4/fast-track-v1/E1-fusion/closed-seed42/metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({'overall': {'accuracy': 1.0, 'macro_f1': 1.0, 'support': 1}}))
    out = tmp_path / 'comparison.json'
    monkeypatch.setattr('sys.argv', ['compare', '--seed', '42', '--output', str(out),
                                    '--acknowledge-unverified-inputs'])
    assert fusion.main() == 0
    result = json.loads(out.read_text())
    assert result['provenance_verified'] is False
    assert result['purpose'] == 'historical_comparison_only'
    assert len(result['input_sha256']) == 5
    assert 'E1_A4+B0+B1_this_seed' not in result['systems']
