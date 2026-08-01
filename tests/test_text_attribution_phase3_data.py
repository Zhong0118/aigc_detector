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


class RecordingWindowTokenizer:
    pad_token_id = 99

    def __init__(self, token_ids: list[int] | range) -> None:
        self.token_ids = list(token_ids)

    def __call__(self, text: str, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return {"input_ids": list(self.token_ids)}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [101, *token_ids, 102]


def test_three_window_long_text_uses_head_center_tail_and_mask() -> None:
    from research.text_attribution.phase3_data import tokenize_phase3_windows

    encoded = tokenize_phase3_windows(
        RecordingWindowTokenizer(range(30)),
        "long",
        max_length=8,
        window_mode="head-center-tail",
    )

    assert encoded["window_mask"] == [1, 1, 1]
    assert encoded["input_ids"][0] == [101, 0, 1, 2, 3, 4, 5, 102]
    assert encoded["input_ids"][1] == [101, 12, 13, 14, 15, 16, 17, 102]
    assert encoded["input_ids"][2] == [101, 24, 25, 26, 27, 28, 29, 102]
    assert all(mask == [1] * 8 for mask in encoded["attention_mask"])


def test_short_text_is_not_duplicated_across_three_windows() -> None:
    from research.text_attribution.phase3_data import tokenize_phase3_windows

    encoded = tokenize_phase3_windows(
        RecordingWindowTokenizer([7, 8, 9]),
        "short",
        max_length=8,
        window_mode="head-center-tail",
    )

    assert encoded["window_mask"] == [1, 0, 0]
    assert encoded["input_ids"][0] == [101, 7, 8, 9, 102, 99, 99, 99]
    assert encoded["attention_mask"][0] == [1, 1, 1, 1, 1, 0, 0, 0]
    assert encoded["input_ids"][1:] == [[99] * 8, [99] * 8]
    assert encoded["attention_mask"][1:] == [[0] * 8, [0] * 8]


def test_overlapping_windows_are_deduplicated() -> None:
    from research.text_attribution.phase3_data import tokenize_phase3_windows

    encoded = tokenize_phase3_windows(
        RecordingWindowTokenizer(range(7)),
        "medium",
        max_length=8,
        window_mode="head-center-tail",
    )
    assert encoded["window_mask"] == [1, 1, 0]
    assert encoded["input_ids"][0] != encoded["input_ids"][1]


def test_single_window_has_one_window_and_preserves_special_tokens() -> None:
    from research.text_attribution.phase3_data import tokenize_phase3_windows

    encoded = tokenize_phase3_windows(
        RecordingWindowTokenizer(range(20)), "long", 8, "single"
    )
    assert encoded == {
        "input_ids": [[101, 0, 1, 2, 3, 4, 5, 102]],
        "attention_mask": [[1] * 8],
        "window_mask": [1],
    }


def test_window_tokenization_rejects_bad_tokenizer_and_mode() -> None:
    from research.text_attribution.phase3_data import tokenize_phase3_windows

    with pytest.raises(ValueError, match="window mode"):
        tokenize_phase3_windows(RecordingWindowTokenizer([1]), "x", 8, "sliding")
    with pytest.raises(ValueError, match="max_length"):
        tokenize_phase3_windows(RecordingWindowTokenizer([1]), "x", 2, "single")


def test_dataset_returns_only_model_inputs_and_labels() -> None:
    from research.text_attribution.phase3_data import (
        Phase3AttributionDataset,
        build_phase3_data_view,
    )

    view = build_phase3_data_view(_splits(), _catalog_rows(), "closed", frozenset())
    dataset = Phase3AttributionDataset(
        samples=view.train_samples,
        data_view=view,
        tokenizer=RecordingWindowTokenizer([7, 8]),
        max_length=8,
        window_mode="single",
    )
    item = dataset[0]

    assert set(item) == {
        "input_ids",
        "attention_mask",
        "window_mask",
        "family_labels",
        "model_labels",
        "contrastive_family_labels",
        "contrastive_group_ids",
        "contrastive_model_ids",
    }
    assert item["family_labels"] == view.family_to_id[view.train_samples[0].family_id]
    assert item["model_labels"] == view.train_visible_model_to_id[
        view.train_samples[0].model_id
    ]
    assert isinstance(item["contrastive_group_ids"], int)
    assert "text" not in item and "request_key" not in item


def test_holdout_validation_dataset_ignores_unseen_model_labels() -> None:
    from research.text_attribution.phase3_data import (
        Phase3AttributionDataset,
        build_phase3_data_view,
    )

    view = build_phase3_data_view(
        _splits(), _catalog_rows(), "fast-to-balanced", frozenset()
    )
    dataset = Phase3AttributionDataset(
        samples=view.validation_samples,
        data_view=view,
        tokenizer=RecordingWindowTokenizer([7]),
        max_length=8,
        window_mode="single",
    )
    assert {dataset[index]["model_labels"] for index in range(len(dataset))} == {-100}
    assert {dataset[index]["contrastive_model_ids"] for index in range(len(dataset))} == {
        -100
    }
