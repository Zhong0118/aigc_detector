"""Synthetic CPU checks for detection/attribution role and score separation."""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from research.image_attribution import feature_ablation as fa


class UnusedRows:
    def __len__(self):
        raise AssertionError('reserved rows must not be consumed')

    def __iter__(self):
        raise AssertionError('reserved rows must not be consumed')


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup_task(task='detection', epochs=20):
    roles = ('train', 'selection', 'calibration', 'evaluation') if task == 'detection' else fa.EXTRACT_ROLES
    classes = ['real', 'ai'] if task == 'detection' else ['generator_a', 'generator_b', 'generator_c']
    parts, chunks = {}, []
    for marker, role in enumerate(roles, 1):
        rows = []
        for i in range(6):
            rows.append(dict(sample_id=f'{role}-{i}', group_id=f'{role}-group-{i}',
                             split='train' if role == 'train' else 'dev',
                             is_ai=bool(i % 2) if task == 'detection' else True,
                             source_label=('generator_a' if i % 2 else None) if task == 'detection' else classes[i % 3],
                             semantic_class='synthetic'))
        parts[role] = rows
        x = torch.ones(6, 384)
        x[:, 0] = marker
        x[:, 1] = torch.tensor([(-1.0 if i % 2 else 1.0) for i in range(6)])
        chunks.append(x)
    parts['holdout'] = UnusedRows()
    if task == 'attribution':
        parts['calibration'] = UnusedRows()
    config = dict(task=task, seed=42, dimensions={'rgb': 384, 'fingerprint': 128, 'fusion': 512},
                  training=dict(epochs=epochs, batch_size=3, learning_rate=.001,
                                scheduler_step=1, scheduler_gamma=.9))
    return torch.cat(chunks), parts, classes, config


def test_task_roles_and_dimensions_preserve_old_defaults():
    assert fa.EXTRACT_ROLES == ('train', 'selection', 'evaluation')
    assert fa.extract_roles('attribution') == fa.EXTRACT_ROLES
    assert fa.extract_roles('detection') == ('train', 'selection', 'calibration', 'evaluation')
    assert fa.combine_features('rgb', torch.ones(2, 2048)).shape == (2, 2048)
    custom = dict(rgb=384, fingerprint=128, fusion=512)
    fused = fa.combine_features('fusion', torch.ones(2, 384), torch.ones(2, 128), dimensions=custom)
    assert fused.shape == (2, 512)
    torch.testing.assert_close(fused.norm(dim=1), torch.ones(2))


def test_detection_selects_auroc_and_scores_evaluation_once_without_holdout(tmp_path, monkeypatch):
    features, parts, classes, config = setup_task()
    real_evaluate = fa.evaluate_epoch
    selection_calls, evaluation_rows_seen = [], []
    original_linear = torch.nn.Linear

    class CountingLinear(original_linear):
        def forward(self, inputs):
            evaluation_rows_seen.append(int((inputs[:, 0] == 4).sum()))
            return super().forward(inputs)

    def evaluate(model, loader, device, task, names, criterion, calibrate=False):
        assert task == 'detection' and calibrate is False
        assert set(loader.dataset.tensors[1].tolist()) == {0, 1}
        result = real_evaluate(model, loader, device, task, names, criterion, calibrate=calibrate)
        if loader.dataset.tensors[0][0, 0] == 2:
            selection_calls.append(result)
            # AUROC favors epoch 1 while macro-F1 favors every subsequent epoch.
            result.update(auroc=.9 if len(selection_calls) == 1 else .6,
                          macro_f1=.1 if len(selection_calls) == 1 else .8)
        return result

    monkeypatch.setattr(fa.nn, 'Linear', CountingLinear)
    monkeypatch.setattr(fa, 'evaluate_epoch', evaluate)
    result = fa.fit_head(features, parts, classes, config, tmp_path, 'cpu', 'rgb', {})
    assert result['best_epoch'] == 1 and result['selection_auroc'] == .9
    assert len(selection_calls) == 20
    assert sum(evaluation_rows_seen) == len(parts['evaluation'])
    assert result['holdout_scored'] is False and result['calibration_used'] is True
    assert 'selection_macro_f1' not in result
    assert 'dev_threshold' not in result
    assert 'fpr_wilson95' in result['fixed_0_5']
    assert set(result['calibrated']) == {'0.01', '0.05'}
    assert result['calibrated']['0.01']['threshold_stability']['repeats'] == 200
    saved = torch.load(tmp_path / 'head.pth', weights_only=True)
    assert saved['task'] == 'detection' and saved['classes'] == ['real', 'ai']
    assert saved['input_dimensions'] == 384
    assert 'ai' in saved['score_semantics']
    cal = [json.loads(line) for line in (tmp_path / 'calibration/predictions.jsonl').read_text().splitlines()]
    evaluation = [json.loads(line) for line in (tmp_path / 'predictions.jsonl').read_text().splitlines()]
    assert [r['sample_id'] for r in cal] == [r['sample_id'] for r in parts['calibration']]
    assert [r['sample_id'] for r in evaluation] == [r['sample_id'] for r in parts['evaluation']]
    assert [r['true_label'] for r in cal] == ['real', 'ai'] * 3


def test_thresholds_are_fit_only_on_calibration_not_evaluation(tmp_path):
    features, parts, classes, config = setup_task(epochs=2)
    (tmp_path / 'one').mkdir()
    (tmp_path / 'two').mkdir()
    one = fa.fit_head(features, parts, classes, config, tmp_path / 'one', 'cpu', 'rgb', {})
    changed = features.clone()
    changed[-len(parts['evaluation']):] *= 100
    two = fa.fit_head(changed, parts, classes, config, tmp_path / 'two', 'cpu', 'rgb', {})
    assert (tmp_path / 'one/predictions.jsonl').read_text() != (tmp_path / 'two/predictions.jsonl').read_text()
    records = [json.loads(line) for line in (tmp_path / 'one/calibration/predictions.jsonl').read_text().splitlines()]
    expected = np.nextafter(max(row['scores'][1] for row in records if row['true_label'] == 'real'), np.inf)
    for target in ('0.01', '0.05'):
        assert one['calibrated'][target]['evaluation']['threshold'] == expected
        assert two['calibrated'][target]['evaluation']['threshold'] == expected
        assert one['calibrated'][target]['threshold_stability'] == two['calibrated'][target]['threshold_stability']


def test_attribution_keeps_source_labels_macro_f1_top3_and_reserved_calibration(tmp_path, monkeypatch):
    features, parts, classes, config = setup_task('attribution', epochs=2)
    del config['task']  # Old configurations default to attribution.
    real_evaluate = fa.evaluate_epoch
    selections = []

    def evaluate(model, loader, device, task, names, criterion, calibrate=False):
        assert task == 'attribution'
        assert set(loader.dataset.tensors[1].tolist()) == {0, 1, 2}
        result = real_evaluate(model, loader, device, task, names, criterion, calibrate=calibrate)
        if loader.dataset.tensors[0][0, 0] == 2:
            selections.append(result)
            result['macro_f1'] = .2 if len(selections) == 1 else .8
        return result

    monkeypatch.setattr(fa, 'evaluate_epoch', evaluate)
    result = fa.fit_head(features, parts, classes, config, tmp_path, 'cpu', 'rgb', {})
    assert result['best_epoch'] == 2 and result['selection_macro_f1'] == .8
    assert result['top3'] == 1.0 and 'top1' in result
    assert result['calibration_used'] is False and 'calibrated' not in result
    assert not (tmp_path / 'calibration').exists()
    saved = torch.load(tmp_path / 'head.pth', weights_only=True)
    assert saved['task'] == 'attribution' and saved['classes'] == classes


@pytest.mark.parametrize('invalid', ['missing_calibration', 'empty_calibration', 'class_order', 'holdout_in_role'])
def test_detection_rejects_incomplete_roles_or_task_mismatch_before_training(tmp_path, invalid):
    features, parts, classes, config = setup_task(epochs=1)
    if invalid == 'missing_calibration':
        del parts['calibration']
    elif invalid == 'empty_calibration':
        parts['calibration'] = []
    elif invalid == 'class_order':
        classes.reverse()
    else:
        parts['evaluation'][0]['split'] = 'holdout'
    with pytest.raises(ValueError):
        fa.fit_head(features, parts, classes, config, tmp_path, 'cpu', 'rgb', {})
    assert not (tmp_path / 'history.jsonl').exists()
