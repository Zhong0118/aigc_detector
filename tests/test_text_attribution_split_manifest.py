from __future__ import annotations

from collections import Counter

import pytest

from research.text_attribution.split_manifest import (
    SPLIT_NAMES,
    apply_split_manifest,
    create_split_manifest,
    load_split_manifest,
    save_split_manifest,
)
from research.text_attribution.training_data import load_training_dataset
from tests.text_attribution_test_utils import write_dataset, write_model_catalog


def _load_fixture_dataset(tmp_path):
    data_path = tmp_path / "samples.jsonl"
    catalog_path = tmp_path / "catalog.json"
    write_dataset(data_path, groups=20)
    write_model_catalog(catalog_path)
    return load_training_dataset(data_path, catalog_path)


def test_creates_deterministic_leakage_safe_balanced_manifest(tmp_path) -> None:
    dataset = _load_fixture_dataset(tmp_path)

    manifest = create_split_manifest(dataset.samples, seed=42)
    same_manifest = create_split_manifest(dataset.samples, seed=42)
    split_samples = apply_split_manifest(dataset.samples, manifest)

    assert manifest == same_manifest
    assert tuple(split_samples) == SPLIT_NAMES
    assert Counter(manifest.group_assignments.values()) == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }
    assert Counter(sample.request_key for samples in split_samples.values() for sample in samples) == Counter(
        sample.request_key for sample in dataset.samples
    )
    group_splits = {
        sample.semantic_prompt_group: split_name
        for split_name, samples in split_samples.items()
        for sample in samples
    }
    assert len(group_splits) == 20
    assert all(
        group_splits[sample.semantic_prompt_group] == split_name
        for split_name, samples in split_samples.items()
        for sample in samples
    )
    for samples in split_samples.values():
        assert {sample.family_id for sample in samples} == set(dataset.family_labels)
        assert {sample.model_id for sample in samples} == set(dataset.model_labels)


def test_persisted_manifest_rejects_dataset_with_missing_sample(tmp_path) -> None:
    dataset = _load_fixture_dataset(tmp_path)
    manifest_path = tmp_path / "split-manifest.json"
    manifest = create_split_manifest(dataset.samples, seed=42)
    save_split_manifest(manifest_path, manifest)

    loaded_manifest = load_split_manifest(manifest_path)

    assert loaded_manifest == manifest
    with pytest.raises(ValueError, match="fingerprint"):
        apply_split_manifest(dataset.samples[:-1], loaded_manifest)
