from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from math import ceil
import re

import pytest

from research.text_attribution.split_manifest import (
    SPLIT_NAMES,
    apply_split_manifest,
    create_split_manifest,
    dataset_fingerprint,
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


def test_three_semantic_groups_can_form_complete_nonempty_splits(tmp_path) -> None:
    data_path = tmp_path / "samples.jsonl"
    catalog_path = tmp_path / "catalog.json"
    write_dataset(data_path, groups=3)
    write_model_catalog(catalog_path)
    dataset = load_training_dataset(data_path, catalog_path)

    splits = apply_split_manifest(dataset.samples, create_split_manifest(dataset.samples, seed=42))

    assert all(splits.values())
    for samples in splits.values():
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


def test_fingerprint_covers_full_sample_content_and_ignores_input_order(tmp_path) -> None:
    dataset = _load_fixture_dataset(tmp_path)
    manifest = create_split_manifest(dataset.samples, seed=42)
    changed_text_samples = (
        replace(dataset.samples[0], text="Different text with identical provenance IDs."),
        *dataset.samples[1:],
    )

    assert dataset_fingerprint(dataset.samples) == dataset_fingerprint(tuple(reversed(dataset.samples)))
    assert dataset_fingerprint(dataset.samples) != dataset_fingerprint(changed_text_samples)
    with pytest.raises(ValueError, match="fingerprint"):
        apply_split_manifest(changed_text_samples, manifest)


def test_rejects_duplicate_request_keys_before_fingerprinting(tmp_path) -> None:
    dataset = _load_fixture_dataset(tmp_path)
    duplicate_key_samples = (
        dataset.samples[0],
        replace(dataset.samples[1], request_key=dataset.samples[0].request_key),
        *dataset.samples[2:],
    )

    with pytest.raises(ValueError, match="duplicate request_key"):
        dataset_fingerprint(duplicate_key_samples)
    with pytest.raises(ValueError, match="duplicate request_key"):
        create_split_manifest(duplicate_key_samples)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2},
        {"schema_version": 1.0},
        {"schema_version": 1, "seed": True},
    ],
)
def test_load_rejects_future_or_malformed_manifest_without_coercion(tmp_path, payload) -> None:
    manifest_path = tmp_path / "invalid-manifest.json"
    valid_payload = {
        "schema_version": 1,
        "seed": 42,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "dataset_fingerprint": "0" * 64,
        "group_assignments": {"group-1": "train"},
    }
    manifest_path.write_text(json.dumps({**valid_payload, **payload}), encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(str(manifest_path))):
        load_split_manifest(manifest_path)


def test_concentrated_model_groups_meet_coverage_tolerance_or_report_details(tmp_path) -> None:
    dataset = _load_fixture_dataset(tmp_path)
    concentrated_samples = tuple(
        sample
        for sample in dataset.samples
        if sample.model_id != "gpt-balanced-test" or int(sample.semantic_prompt_group.rsplit("-", 1)[1]) < 5
    )

    try:
        manifest = create_split_manifest(concentrated_samples, seed=42)
    except ValueError as exc:
        assert "underrepresented" in str(exc)
        assert "split=" in str(exc)
        assert "actual=" in str(exc)
        assert "expected=" in str(exc)
        assert "minimum=" in str(exc)
    else:
        splits = apply_split_manifest(concentrated_samples, manifest)
        total = sum(sample.model_id == "gpt-balanced-test" for sample in concentrated_samples)
        for split_name, samples in splits.items():
            actual = sum(sample.model_id == "gpt-balanced-test" for sample in samples)
            assert actual >= ceil(total * manifest.ratios[split_name] * 0.8)
