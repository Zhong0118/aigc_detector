"""Leakage-safe data contracts for Phase3 family attribution.

This module deliberately contains no PyTorch or Transformers imports.  Catalog and
split validation can therefore run before a GPU process (or a model download) is
started.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

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
