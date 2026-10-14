from types import SimpleNamespace
import json
from unittest.mock import Mock

import pytest

from provenance import window_a0_fusion as production
from scripts import tune_window_a0_fusion_weights as tuner


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf')])
def test_explicit_weights_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        production.load_branch_weights({'A0': bad, 'B0': 1, 'B1': 1, 'B3': 1})


def test_production_does_not_implicitly_load_old_lock(tmp_path, monkeypatch):
    path = tmp_path / 'old-lock.json'
    path.write_text(json.dumps({'weights': {'A0': 1, 'B0': 0, 'B1': 0, 'B3': 0}}))
    monkeypatch.setattr(production, 'WEIGHTS_ARTIFACT', path)
    assert production.load_branch_weights() == production.DEFAULT_BRANCH_WEIGHTS


def test_legacy_same_count_cache_is_rejected(tmp_path):
    cache = tmp_path / 'cache.json'
    cache.write_text(json.dumps({'n': 1, 'split': 'validation'}))
    fusion = Mock()
    with pytest.raises(ValueError, match='stale|provenance'):
        tuner._load_or_score_branches(fusion=fusion, samples=[SimpleNamespace(request_key='new', text='new', family_id='openai')], cache_path=cache, split='validation')


def _fake_run(monkeypatch):
    labels = ['openai', 'qwen']
    samples = [SimpleNamespace(request_key='v1', text='validation text', family_id='openai')]
    test_samples = [SimpleNamespace(request_key='t1', text='test text', family_id='qwen')]
    monkeypatch.setattr(tuner, 'apply_split_manifest', lambda rows, manifest: {'validation': samples, 'test': test_samples})
    loader = Mock(return_value=SimpleNamespace(samples=samples + test_samples))
    monkeypatch.setattr(tuner, 'load_training_dataset', loader)
    monkeypatch.setattr(tuner, 'load_split_manifest', lambda path: SimpleNamespace(dataset_fingerprint='dataset', group_assignments={'v': 'validation', 't': 'test'}))
    monkeypatch.setattr(tuner, 'WindowA0EqualFusion', lambda cfg: SimpleNamespace(family_labels=labels))
    monkeypatch.setattr(tuner, '_cuda', lambda: False)
    monkeypatch.setattr(tuner, '_hash_file', lambda path: 'file-hash')
    monkeypatch.setattr(tuner, '_inference_identity', lambda fusion: {'model': 'hash'})
    score = Mock(return_value={'n': 1, 'family_labels': labels, 'request_keys': ['v1'], 'true_family': ['openai'], 'branches': {b: [[0.8, 0.2]] for b in tuner.BRANCHES}, 'provenance': {'rows': 'hash'}})
    monkeypatch.setattr(tuner, '_load_or_score_branches', score)
    return loader, score


def test_default_tuning_only_validation_and_lock_is_immutable(tmp_path, monkeypatch):
    loader, score = _fake_run(monkeypatch)
    assert tuner.main(['--output-dir', str(tmp_path), '--step', '0.25']) == 0
    assert [sample.request_key for sample in score.call_args.kwargs['samples']] == ['v1']
    assert [call.kwargs['split'] for call in score.call_args_list] == ['validation']
    lock = tmp_path / 'seed42-weights.json'
    original = lock.read_bytes()
    with pytest.raises(FileExistsError):
        tuner.main(['--output-dir', str(tmp_path)])
    assert lock.read_bytes() == original
    assert not (tmp_path / 'seed42-test-evaluation.json').exists()


def test_final_evaluation_consumes_lock_without_retuning(tmp_path, monkeypatch):
    loader, score = _fake_run(monkeypatch)
    tuner.main(['--output-dir', str(tmp_path), '--step', '0.25'])
    original = (tmp_path / 'seed42-weights.json').read_bytes()
    monkeypatch.setattr(tuner, '_simplex', Mock(side_effect=AssertionError('must not retune')))
    assert tuner.main(['--output-dir', str(tmp_path), '--mode', 'evaluate-test']) == 0
    assert [sample.request_key for sample in score.call_args.kwargs['samples']] == ['t1']
    assert score.call_args.kwargs['split'] == 'test'
    assert (tmp_path / 'seed42-weights.json').read_bytes() == original
    with pytest.raises(FileExistsError, match='duplicate'):
        tuner.main(['--output-dir', str(tmp_path), '--mode', 'evaluate-test'])
    assert score.call_count == 2


@pytest.mark.parametrize('change', ['text', 'key', 'label', 'model', 'dataset', 'order', 'max_length'])
def test_cache_rejects_same_count_changed_inputs(tmp_path, monkeypatch, change):
    samples = [SimpleNamespace(request_key='one', text='a', family_id='openai'), SimpleNamespace(request_key='two', text='b', family_id='qwen')]
    fusion = SimpleNamespace(family_labels=['openai', 'qwen'], _ensure_loaded=Mock(), _classical={b: None for b in ('B0', 'B1', 'B3')}, _predict_a0=lambda text: __import__('numpy').array([0.8, 0.2]))
    from research.text_attribution import classical_models
    monkeypatch.setattr(classical_models, 'predict_family_probabilities', lambda *args: __import__('numpy').array([[0.8, 0.2], [0.8, 0.2]]))
    provenance = {'dataset': 'd1', 'model': 'm1', 'max_length': 512}
    cache = tmp_path / 'cache.json'
    tuner._load_or_score_branches(fusion=fusion, samples=samples, cache_path=cache, split='validation', provenance=provenance)
    assert tuner._load_or_score_branches(fusion=fusion, samples=samples, cache_path=cache, split='validation', provenance=provenance)['n'] == 2
    if change in ('text', 'key', 'label'):
        setattr(samples[0], {'text': 'text', 'key': 'request_key', 'label': 'family_id'}[change], 'changed')
    elif change == 'order':
        samples.reverse()
    else:
        provenance[change] = 'changed'
    with pytest.raises(ValueError, match='stale'):
        tuner._load_or_score_branches(fusion=fusion, samples=samples, cache_path=cache, split='validation', provenance=provenance)


def test_failed_test_attempt_stays_reserved_and_preserves_lock(tmp_path, monkeypatch):
    _, score = _fake_run(monkeypatch)
    tuner.main(['--output-dir', str(tmp_path), '--step', '0.25'])
    original = (tmp_path / 'seed42-weights.json').read_bytes()
    score.side_effect = RuntimeError('inference interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        tuner.main(['--output-dir', str(tmp_path), '--mode', 'evaluate-test'])
    with pytest.raises(FileExistsError):
        tuner.main(['--output-dir', str(tmp_path), '--mode', 'evaluate-test'])
    assert (tmp_path / 'seed42-weights.json').read_bytes() == original
    assert score.call_count == 2


def test_changed_artifact_bytes_change_inference_identity(tmp_path):
    paths = {}
    for name in production.DEFAULT_PATHS:
        path = tmp_path / name
        path.write_bytes(b'first')
        paths[name] = path
    fusion = production.WindowA0EqualFusion(paths)
    initial = tuner._inference_identity(fusion)
    paths['b0_model_path'].write_bytes(b'other')
    assert tuner._inference_identity(fusion) != initial


@pytest.mark.parametrize('step', [0, -0.25, 0.3, 2, float('nan'), float('inf')])
def test_simplex_rejects_invalid_step(step):
    with pytest.raises(ValueError):
        tuner._simplex(step)


@pytest.mark.parametrize('field,value', [
    ('recipe_id', 'A0'), ('recipe_id', 'A4'),
    ('window_mode', 'head_tail'), ('validation_window_mode', 'single'),
    ('max_length', 256), ('train_window_stride', None),
])
def test_production_rejects_incompatible_window_recipe(tmp_path, monkeypatch, field, value):
    import transformers
    fusion = production.WindowA0EqualFusion({'a0_model_dir': tmp_path, 'device': 'cpu'})
    recipe = SimpleNamespace(recipe_id='A0-window-split', window_mode='single',
                             validation_window_mode='nonoverlap-probability-mean',
                             max_length=512, train_window_stride=510)
    setattr(recipe, field, value)
    model = SimpleNamespace(recipe=recipe, family_labels=fusion.family_labels, eval=lambda: None)
    monkeypatch.setattr(production, 'load_phase3_model', lambda *a, **k: model)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(production, '_load_sklearn', lambda path: object())
    with pytest.raises(ValueError, match='window-A0 recipe'):
        fusion._ensure_loaded()
    assert not fusion._loaded
