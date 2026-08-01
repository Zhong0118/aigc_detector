from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from research.text_attribution.training_data import AttributionSample


CATALOG_PATH = (
    Path(__file__).parents[1]
    / "configs/text_attribution/zhizengzeng_model_catalog.v5.json"
)


def _sample(model_id: str, family_id: str, split: str, index: int) -> AttributionSample:
    return AttributionSample(
        request_key=f"{split}-{model_id}-{index}",
        text=f"text {model_id} {index}",
        family_id=family_id,
        model_id=model_id,
        prompt_id=f"prompt-{index}",
        semantic_prompt_group=f"group-{index % 2}",
        language="zh" if index % 2 else "en",
        task_type="qa",
        target_length="medium",
        temperature=0.7,
        repeat_index=index,
    )


def _catalog_rows():
    from research.text_attribution.phase3_data import CatalogVariant, VariantCatalog

    models = (
        CatalogVariant("family-a", "a-fast", "fast"),
        CatalogVariant("family-a", "a-balanced", "balanced"),
        CatalogVariant("family-b", "b-fast", "fast"),
        CatalogVariant("family-b", "b-balanced", "balanced"),
    )
    return VariantCatalog(models=models)


def _splits() -> dict[str, tuple[AttributionSample, ...]]:
    pairs = (
        ("family-a", "a-fast"),
        ("family-a", "a-balanced"),
        ("family-b", "b-fast"),
        ("family-b", "b-balanced"),
    )
    return {
        split: tuple(
            _sample(model_id, family_id, split, index)
            for index, (family_id, model_id) in enumerate(pairs)
        )
        for split in ("train", "validation")
    }


def test_fast_to_balanced_filters_by_explicit_catalog_role() -> None:
    from research.text_attribution.phase3_data import build_phase3_data_view

    view = build_phase3_data_view(
        splits=_splits(),
        catalog=_catalog_rows(),
        view_id="fast-to-balanced",
        test_request_keys=frozenset({"test-secret"}),
    )

    assert {row.model_id for row in view.train_samples} == {"a-fast", "b-fast"}
    assert {row.model_id for row in view.validation_samples} == {
        "a-balanced",
        "b-balanced",
    }
    assert tuple(view.train_visible_model_to_id) == ("a-fast", "b-fast")
    assert all(view.model_label_for(row) == -100 for row in view.validation_samples)


def test_balanced_to_fast_reverses_roles() -> None:
    from research.text_attribution.phase3_data import build_phase3_data_view

    view = build_phase3_data_view(
        _splits(), _catalog_rows(), "balanced-to-fast", frozenset()
    )
    assert {row.model_id for row in view.train_samples} == {
        "a-balanced",
        "b-balanced",
    }
    assert {row.model_id for row in view.validation_samples} == {"a-fast", "b-fast"}


def test_closed_view_uses_all_train_visible_models_in_catalog_order() -> None:
    from research.text_attribution.phase3_data import build_phase3_data_view

    view = build_phase3_data_view(_splits(), _catalog_rows(), "closed", frozenset())
    assert tuple(view.family_to_id) == ("family-a", "family-b")
    assert tuple(view.train_visible_model_to_id) == (
        "a-fast",
        "a-balanced",
        "b-fast",
        "b-balanced",
    )
    assert [view.model_to_family_id[model] for model in view.train_visible_model_to_id] == [
        0,
        0,
        1,
        1,
    ]


def test_v5_catalog_has_ten_families_and_twenty_visible_models() -> None:
    from research.text_attribution.phase3_data import (
        build_train_visible_model_maps,
        load_variant_catalog,
    )

    catalog = load_variant_catalog(CATALOG_PATH)
    closed, _ = build_train_visible_model_maps(catalog, "closed")
    fast, _ = build_train_visible_model_maps(catalog, "fast-to-balanced")
    balanced, _ = build_train_visible_model_maps(catalog, "balanced-to-fast")

    assert len(catalog.family_ids) == 10
    assert len(closed) == 20
    assert len(fast) == len(balanced) == 10
    for family_id in catalog.family_ids:
        roles = {row.variant_role for row in catalog.models if row.family_id == family_id}
        assert roles == {"fast", "balanced"}


@pytest.mark.parametrize(
    ("models", "message"),
    [
        (
            (
                ("family-a", "model-1", "fast"),
                ("family-a", "model-2", "fast"),
            ),
            "duplicate.*role|exactly one",
        ),
        ((("family-a", "model-1", "other"),), "variant role"),
    ],
)
def test_catalog_rejects_invalid_role_contract(models, message: str) -> None:
    from research.text_attribution.phase3_data import CatalogVariant, VariantCatalog

    with pytest.raises(ValueError, match=message):
        VariantCatalog(models=tuple(CatalogVariant(*row) for row in models))


def test_view_rejects_test_request_key_leakage() -> None:
    from research.text_attribution.phase3_data import build_phase3_data_view

    splits = _splits()
    leaked_key = splits["train"][0].request_key
    with pytest.raises(ValueError, match="test.*leak|leak.*test"):
        build_phase3_data_view(
            splits,
            _catalog_rows(),
            "closed",
            test_request_keys=frozenset({leaked_key}),
        )


def test_view_rejects_wrong_family_and_unknown_model() -> None:
    from research.text_attribution.phase3_data import build_phase3_data_view

    splits = _splits()
    wrong = replace(splits["train"][0], family_id="family-b")
    with pytest.raises(ValueError, match="family mismatch"):
        build_phase3_data_view(
            {**splits, "train": (wrong, *splits["train"][1:])},
            _catalog_rows(),
            "closed",
            frozenset(),
        )

    unknown = replace(splits["train"][0], model_id="unknown")
    with pytest.raises(ValueError, match="missing from catalog"):
        build_phase3_data_view(
            {**splits, "train": (unknown, *splits["train"][1:])},
            _catalog_rows(),
            "closed",
            frozenset(),
        )


def test_view_requires_both_families_on_each_side() -> None:
    from research.text_attribution.phase3_data import build_phase3_data_view

    splits = _splits()
    only_a_validation = tuple(
        row for row in splits["validation"] if row.family_id == "family-a"
    )
    with pytest.raises(ValueError, match="validation.*famil"):
        build_phase3_data_view(
            {**splits, "validation": only_a_validation},
            _catalog_rows(),
            "closed",
            frozenset(),
        )
