"""Offline protocol regressions; synthetic fixtures are never experiment evidence."""
import importlib.util
import json
import pytest


def test_image_protocol_module_exists():
    assert importlib.util.find_spec("research.image_attribution") is not None


def test_split_is_order_independent_and_groups_never_cross():
    from research.image_attribution.protocol import assign_splits
    rows = [dict(sample_id=str(i), group_id=str(i // 2), source_label="a") for i in range(40)]
    a = assign_splits(rows, 42)
    b = assign_splits(list(reversed(rows)), 42)
    assert {r['sample_id']: r['split'] for r in a} == {r['sample_id']: r['split'] for r in b}
    assert all(len({r['split'] for r in a if r['group_id'] == g}) == 1 for g in {r['group_id'] for r in a})
    assert {r['split'] for r in a} == {'train', 'dev', 'holdout'}


def test_manifest_rejects_leakage_and_unsafe_path(tmp_path):
    from research.image_attribution.protocol import validate_manifest
    rows = [dict(sample_id='a', group_id='g', sha256='abc', split='train', relative_path='a.png', is_ai=1, source_label='A'),
            dict(sample_id='b', group_id='g', sha256='def', split='dev', relative_path='b.png', is_ai=1, source_label='A')]
    with pytest.raises(ValueError, match='crosses'): validate_manifest(rows, tmp_path, verify_files=False)
    rows[1]['group_id'] = 'h'
    rows[1]['relative_path'] = '../outside.png'
    with pytest.raises(ValueError, match='path'): validate_manifest(rows, tmp_path, verify_files=False)


def test_detection_labels_do_not_follow_source_class_order():
    from research.image_attribution.protocol import task_rows
    rows = [dict(is_ai=0, source_label=None), dict(is_ai=1, source_label='Z'), dict(is_ai=1, source_label='A')]
    _, labels, classes = task_rows(rows, 'detection')
    assert labels == [0, 1, 1] and classes == ['real', 'ai']
    _, labels, classes = task_rows(rows, 'attribution')
    assert labels == [1, 0] and classes == ['A', 'Z']


def test_missing_ofa_pair_head_fails_closed(tmp_path):
    import torch
    from research.image_attribution.models import load_ofa
    p = tmp_path / 'incomplete.pth'
    torch.save({'model': {}}, p)
    with pytest.raises(ValueError, match='pair_classifier'): load_ofa(p, p, 'cpu')


def test_checkpoint_class_order_is_verified(tmp_path):
    import torch
    from research.image_attribution.models import load_checkpoint
    p = tmp_path / 'invalid.pth'
    torch.save({'task': 'detection', 'classes': ['ai', 'real'], 'state_dict': {}}, p)
    with pytest.raises(ValueError, match='class order'): load_checkpoint(p, 'cpu')


def test_schema_labels_override_historical_fallback():
    from scripts.train_image_attribution import decode_dataset_label
    features = {'generator': {'names': ['new-source', 'another-source']}}
    assert decode_dataset_label(features, 'generator', 0) == 'new-source'
    with pytest.raises(ValueError): decode_dataset_label(features, 'generator', 3)
