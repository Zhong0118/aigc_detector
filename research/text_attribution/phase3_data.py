"""Leakage-safe data contracts for Phase3 family attribution.

This module deliberately contains no PyTorch or Transformers imports.  Catalog and
split validation can therefore run before a GPU process (or a model download) is
started.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from research.text_attribution.training_data import (
    AttributionSample,
    load_model_catalog,
)


PHASE3_VIEW_ROLES: Mapping[str, tuple[str | None, str | None]] = MappingProxyType(
    {
        "closed": (None, None),
        "fast-to-balanced": ("fast", "balanced"),
        "balanced-to-fast": ("balanced", "fast"),
    }
)
_ALLOWED_ROLES = frozenset({"fast", "balanced"})


@dataclass(frozen=True)
class CatalogVariant:
    """One enabled concrete model and its explicit version-holdout role."""

    family_id: str
    model_id: str
    variant_role: str


@dataclass(frozen=True)
class VariantCatalog:
    """Validated catalog whose input order defines every Phase3 label order."""

    models: tuple[CatalogVariant, ...]

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("variant catalog must contain enabled models")

        seen_models: set[str] = set()
        seen_family_roles: set[tuple[str, str]] = set()
        family_roles: dict[str, set[str]] = {}
        for row in self.models:
            if not row.family_id or not row.model_id:
                raise ValueError("catalog family_id and model_id must be nonempty")
            if row.variant_role not in _ALLOWED_ROLES:
                raise ValueError(
                    "Phase3 variant role must be exactly 'fast' or 'balanced': "
                    f"{row.variant_role!r}"
                )
            if row.model_id in seen_models:
                raise ValueError(f"duplicate catalog model_id: {row.model_id}")
            seen_models.add(row.model_id)

            pair = (row.family_id, row.variant_role)
            if pair in seen_family_roles:
                raise ValueError(
                    "duplicate family variant role; expected exactly one model for "
                    f"{row.family_id!r}/{row.variant_role!r}"
                )
            seen_family_roles.add(pair)
            family_roles.setdefault(row.family_id, set()).add(row.variant_role)

        for family_id, roles in family_roles.items():
            if roles != _ALLOWED_ROLES:
                raise ValueError(
                    "each family must have exactly one fast and one balanced model; "
                    f"{family_id!r} has {sorted(roles)!r}"
                )

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.family_id for row in self.models))

    @property
    def by_model(self) -> Mapping[str, CatalogVariant]:
        return MappingProxyType({row.model_id: row for row in self.models})


@dataclass(frozen=True)
class Phase3DataView:
    """The exact train/validation rows and labels owned by one Phase3 run."""

    view_id: str
    train_samples: tuple[AttributionSample, ...]
    validation_samples: tuple[AttributionSample, ...]
    family_to_id: Mapping[str, int]
    id_to_family: Mapping[int, str]
    train_visible_model_to_id: Mapping[str, int]
    id_to_train_visible_model: Mapping[int, str]
    model_to_family_id: Mapping[str, int]

    def model_label_for(self, sample: AttributionSample) -> int:
        """Return the auxiliary label, ignoring concrete models unseen in training."""

        return self.train_visible_model_to_id.get(sample.model_id, -100)


class Phase3AttributionDataset:
    """Tokenizer-backed dataset exposing only tensors/labels to the trainer."""

    def __init__(
        self,
        *,
        samples: Sequence[AttributionSample],
        data_view: Phase3DataView,
        tokenizer: Any,
        max_length: int,
        window_mode: str,
    ) -> None:
        self._samples = tuple(samples)
        self._data_view = data_view
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._window_mode = window_mode
        group_names = tuple(
            dict.fromkeys(row.semantic_prompt_group for row in self._samples)
        )
        self._group_to_id = {
            group_name: index for index, group_name in enumerate(group_names)
        }

        for row in self._samples:
            if row.family_id not in data_view.family_to_id:
                raise ValueError(
                    f"dataset family missing from data view: {row.family_id!r}"
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        encoded = tokenize_phase3_windows(
            self._tokenizer,
            sample.text,
            max_length=self._max_length,
            window_mode=self._window_mode,
        )
        family_label = self._data_view.family_to_id[sample.family_id]
        model_label = self._data_view.model_label_for(sample)
        return {
            **encoded,
            "family_labels": family_label,
            "model_labels": model_label,
            "contrastive_family_labels": family_label,
            "contrastive_group_ids": self._group_to_id[
                sample.semantic_prompt_group
            ],
            "contrastive_model_ids": model_label,
        }


class StructuredContrastiveBatchSampler:
    """Deterministic 2-family x 2-group rectangle batches for Phase3 A3.

    Each four-item block is a complete rectangle.  Consequently every emitted
    anchor has a same-family/different-group positive and a
    same-group/different-family hard negative.  Four blocks form the fixed formal
    batch size of 16.
    """

    def __init__(
        self,
        samples: Sequence[AttributionSample],
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size != 16:
            raise ValueError("structured Phase3 batch_size must be exactly 16")
        if not isinstance(seed, int):
            raise ValueError("structured sampler seed must be an integer")
        self._samples = tuple(samples)
        self._batch_size = batch_size
        self._seed = seed
        self._epoch = 0

        cells: dict[tuple[str, str], list[int]] = defaultdict(list)
        for original_index, sample in sorted(
            enumerate(self._samples),
            key=lambda item: (
                item[1].family_id,
                item[1].semantic_prompt_group,
                item[1].model_id,
                item[1].request_key,
                item[0],
            ),
        ):
            cells[(sample.family_id, sample.semantic_prompt_group)].append(
                original_index
            )
        self._cells = {key: tuple(value) for key, value in cells.items()}

        families = sorted({sample.family_id for sample in self._samples})
        groups_by_family = {
            family_id: {
                group_id
                for candidate_family, group_id in self._cells
                if candidate_family == family_id
            }
            for family_id in families
        }
        valid_pairs: list[tuple[str, str, tuple[str, ...]]] = []
        for left_index, left_family in enumerate(families):
            for right_family in families[left_index + 1 :]:
                common_groups = tuple(
                    sorted(
                        groups_by_family[left_family]
                        & groups_by_family[right_family]
                    )
                )
                if len(common_groups) >= 2:
                    valid_pairs.append(
                        (left_family, right_family, common_groups)
                    )
        if not valid_pairs:
            raise ValueError(
                "structured contrastive batching is impossible: need two families "
                "sharing at least two semantic groups"
            )
        self._valid_pairs = tuple(valid_pairs)

    def __len__(self) -> int:
        return len(self._samples) // self._batch_size

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")
        self._epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"epoch"}:
            raise ValueError("sampler state must contain exactly epoch")
        self.set_epoch(state["epoch"])

    def __iter__(self):
        rng = random.Random(_sampler_epoch_seed(self._seed, self._epoch))
        pair_order = list(self._valid_pairs)
        rng.shuffle(pair_order)
        blocks_per_batch = self._batch_size // 4
        block_cursor = 0

        for _ in range(len(self)):
            batch: list[int] = []
            for _ in range(blocks_per_batch):
                if block_cursor and block_cursor % len(pair_order) == 0:
                    rng.shuffle(pair_order)
                family_left, family_right, common_groups = pair_order[
                    block_cursor % len(pair_order)
                ]
                block_cursor += 1
                group_left, group_right = rng.sample(common_groups, 2)
                if rng.randrange(2):
                    family_left, family_right = family_right, family_left
                if rng.randrange(2):
                    group_left, group_right = group_right, group_left
                batch.extend(
                    self._sample_rectangle(
                        rng,
                        family_left,
                        family_right,
                        group_left,
                        group_right,
                    )
                )
            yield batch

    def _sample_rectangle(
        self,
        rng: random.Random,
        family_left: str,
        family_right: str,
        group_left: str,
        group_right: str,
    ) -> tuple[int, int, int, int]:
        left_anchor = rng.choice(self._cells[(family_left, group_left)])
        left_positive = _choose_preferred_model(
            rng,
            self._cells[(family_left, group_right)],
            self._samples,
            avoid_model=self._samples[left_anchor].model_id,
        )
        right_anchor = rng.choice(self._cells[(family_right, group_left)])
        right_positive = _choose_preferred_model(
            rng,
            self._cells[(family_right, group_right)],
            self._samples,
            avoid_model=self._samples[right_anchor].model_id,
        )
        return left_anchor, left_positive, right_anchor, right_positive


def summarize_contrastive_batches(
    samples: Sequence[AttributionSample],
    batches: Sequence[Sequence[int]],
) -> dict[str, int | float | bool]:
    """Audit structured batches without importing the training framework."""

    anchor_count = 0
    positive_count = 0
    hard_negative_count = 0
    different_model_positive_count = 0
    all_batches_valid = True
    for batch in batches:
        if len(batch) != 16:
            all_batches_valid = False
        rows = [samples[index] for index in batch]
        for anchor in rows:
            anchor_count += 1
            positives = [
                other
                for other in rows
                if other.family_id == anchor.family_id
                and other.semantic_prompt_group
                != anchor.semantic_prompt_group
            ]
            negatives = [
                other
                for other in rows
                if other.family_id != anchor.family_id
                and other.semantic_prompt_group
                == anchor.semantic_prompt_group
            ]
            if positives:
                positive_count += 1
                if any(other.model_id != anchor.model_id for other in positives):
                    different_model_positive_count += 1
            else:
                all_batches_valid = False
            if negatives:
                hard_negative_count += 1
            else:
                all_batches_valid = False

    denominator = max(anchor_count, 1)
    return {
        "batch_count": len(batches),
        "anchor_count": anchor_count,
        "all_batches_valid": all_batches_valid,
        "cross_group_positive_anchor_rate": positive_count / denominator,
        "same_group_hard_negative_anchor_rate": hard_negative_count / denominator,
        "different_model_positive_anchor_rate": (
            different_model_positive_count / denominator
        ),
    }


def tokenize_phase3_windows(
    tokenizer: Any,
    text: str,
    max_length: int,
    window_mode: str,
) -> dict[str, list[list[int]] | list[int]]:
    """Tokenize one text as a single window or deduplicated head/center/tail.

    Content is tokenized without special tokens, sliced, and then wrapped using the
    tokenizer's native special-token method.  Inactive A4 windows are all padding
    with an all-zero attention mask, so no synthetic duplicate reaches the encoder.
    """

    if window_mode not in {"single", "head-center-tail"}:
        raise ValueError(
            "window mode must be exactly 'single' or 'head-center-tail'"
        )
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_length = max_length - special_count
    if content_length < 1:
        raise ValueError(
            "max_length must leave room for at least one non-special token"
        )
    pad_token_id = tokenizer.pad_token_id
    if not isinstance(pad_token_id, int):
        raise ValueError("tokenizer must define an integer pad_token_id")

    raw = tokenizer(text, add_special_tokens=False, truncation=False)
    if not isinstance(raw, Mapping) or "input_ids" not in raw:
        raise ValueError("tokenizer must return an input_ids field")
    token_ids = [int(token_id) for token_id in raw["input_ids"]]

    if window_mode == "single":
        slices = [token_ids[:content_length]]
        target_windows = 1
    else:
        target_windows = 3
        maximum_start = max(0, len(token_ids) - content_length)
        starts = (0, maximum_start // 2, maximum_start)
        slices = []
        seen: set[tuple[int, ...]] = set()
        for start in starts:
            window = tuple(token_ids[start : start + content_length])
            if window in seen:
                continue
            seen.add(window)
            slices.append(list(window))

    input_windows: list[list[int]] = []
    attention_windows: list[list[int]] = []
    window_mask: list[int] = []
    for token_slice in slices:
        wrapped = [
            int(token_id)
            for token_id in tokenizer.build_inputs_with_special_tokens(token_slice)
        ]
        if len(wrapped) > max_length:
            raise ValueError(
                "tokenizer special-token wrapping exceeded configured max_length"
            )
        active_length = len(wrapped)
        input_windows.append(
            wrapped + [pad_token_id] * (max_length - active_length)
        )
        attention_windows.append(
            [1] * active_length + [0] * (max_length - active_length)
        )
        window_mask.append(1)

    while len(input_windows) < target_windows:
        input_windows.append([pad_token_id] * max_length)
        attention_windows.append([0] * max_length)
        window_mask.append(0)

    return {
        "input_ids": input_windows,
        "attention_mask": attention_windows,
        "window_mask": window_mask,
    }


def load_variant_catalog(path: str | Path) -> VariantCatalog:
    """Load the existing V5 catalog and enforce the Phase3 role invariant."""

    rows = load_model_catalog(path)
    return VariantCatalog(
        models=tuple(
            CatalogVariant(
                family_id=row.family_id,
                model_id=row.model_id,
                variant_role=row.variant_role,
            )
            for row in rows
        )
    )


def build_train_visible_model_maps(
    catalog: VariantCatalog,
    view_id: str,
) -> tuple[Mapping[str, int], Mapping[int, str]]:
    """Build the auxiliary-head label maps in catalog order."""

    train_role, _ = _view_roles(view_id)
    visible = tuple(
        row.model_id
        for row in catalog.models
        if train_role is None or row.variant_role == train_role
    )
    model_to_id = MappingProxyType({model_id: index for index, model_id in enumerate(visible)})
    id_to_model = MappingProxyType({index: model_id for model_id, index in model_to_id.items()})
    return model_to_id, id_to_model


def build_phase3_data_view(
    splits: Mapping[str, Sequence[AttributionSample]],
    catalog: VariantCatalog,
    view_id: str,
    test_request_keys: frozenset[str] = frozenset(),
) -> Phase3DataView:
    """Create one closed or bidirectional version-holdout validation view.

    The function receives only request keys for the frozen test split.  It never
    consumes test text or labels, and rejects any overlap before filtering rows.
    """

    train_role, validation_role = _view_roles(view_id)
    if "train" not in splits or "validation" not in splits:
        raise ValueError("splits must contain train and validation samples")

    unfiltered_train = tuple(splits["train"])
    unfiltered_validation = tuple(splits["validation"])
    overlap = {
        row.request_key for row in (*unfiltered_train, *unfiltered_validation)
    } & set(test_request_keys)
    if overlap:
        raise ValueError(
            "test request-key leakage into Phase3 train/validation: "
            f"{sorted(overlap)[:3]!r}"
        )

    by_model = catalog.by_model
    _validate_rows_against_catalog(unfiltered_train, by_model, "train")
    _validate_rows_against_catalog(unfiltered_validation, by_model, "validation")

    train_samples = tuple(
        row
        for row in unfiltered_train
        if train_role is None or by_model[row.model_id].variant_role == train_role
    )
    validation_samples = tuple(
        row
        for row in unfiltered_validation
        if validation_role is None
        or by_model[row.model_id].variant_role == validation_role
    )

    expected_families = set(catalog.family_ids)
    _validate_view_family_coverage(train_samples, expected_families, "train")
    _validate_view_family_coverage(
        validation_samples, expected_families, "validation"
    )

    family_to_id = MappingProxyType(
        {family_id: index for index, family_id in enumerate(catalog.family_ids)}
    )
    id_to_family = MappingProxyType(
        {index: family_id for family_id, index in family_to_id.items()}
    )
    model_to_id, id_to_model = build_train_visible_model_maps(catalog, view_id)
    model_to_family_id = MappingProxyType(
        {
            model_id: family_to_id[by_model[model_id].family_id]
            for model_id in model_to_id
        }
    )

    return Phase3DataView(
        view_id=view_id,
        train_samples=train_samples,
        validation_samples=validation_samples,
        family_to_id=family_to_id,
        id_to_family=id_to_family,
        train_visible_model_to_id=model_to_id,
        id_to_train_visible_model=id_to_model,
        model_to_family_id=model_to_family_id,
    )


def _view_roles(view_id: str) -> tuple[str | None, str | None]:
    try:
        return PHASE3_VIEW_ROLES[view_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown Phase3 data view {view_id!r}; expected {tuple(PHASE3_VIEW_ROLES)!r}"
        ) from exc


def _sampler_epoch_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"phase3\0{seed}\0{epoch}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _choose_preferred_model(
    rng: random.Random,
    candidates: Sequence[int],
    samples: Sequence[AttributionSample],
    *,
    avoid_model: str,
) -> int:
    preferred = [
        index for index in candidates if samples[index].model_id != avoid_model
    ]
    return rng.choice(preferred or list(candidates))


def _validate_rows_against_catalog(
    rows: Sequence[AttributionSample],
    by_model: Mapping[str, CatalogVariant],
    split_name: str,
) -> None:
    seen_request_keys: set[str] = set()
    for row in rows:
        if row.request_key in seen_request_keys:
            raise ValueError(
                f"duplicate request_key in {split_name}: {row.request_key!r}"
            )
        seen_request_keys.add(row.request_key)
        catalog_row = by_model.get(row.model_id)
        if catalog_row is None:
            raise ValueError(
                f"{split_name} model missing from catalog: {row.model_id!r}"
            )
        if row.family_id != catalog_row.family_id:
            raise ValueError(
                "catalog family mismatch in "
                f"{split_name}: model {row.model_id!r} is "
                f"{catalog_row.family_id!r}, row is {row.family_id!r}"
            )


def _validate_view_family_coverage(
    rows: Sequence[AttributionSample],
    expected: set[str],
    split_name: str,
) -> None:
    present = {row.family_id for row in rows}
    if present != expected:
        raise ValueError(
            f"{split_name} view family coverage mismatch; "
            f"missing={sorted(expected - present)!r}, extra={sorted(present - expected)!r}"
        )
