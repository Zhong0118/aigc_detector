"""Canonical family-attribution training data and model catalog contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from research.text_attribution.baseline_data import load_clean_samples


@dataclass(frozen=True)
class CatalogModel:
    family_id: str
    model_id: str
    variant_role: str


@dataclass(frozen=True)
class AttributionSample:
    request_key: str
    text: str
    family_id: str
    model_id: str
    prompt_id: str
    semantic_prompt_group: str
    language: str
    task_type: str
    target_length: str
    temperature: float
    repeat_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingDataset:
    samples: tuple[AttributionSample, ...]
    family_labels: tuple[str, ...]
    model_labels: tuple[str, ...]
    model_to_family: Mapping[str, str]
    audit: Mapping[str, Any]


def load_model_catalog(path: str | Path) -> tuple[CatalogModel, ...]:
    """Load enabled model records from a JSON model catalog."""

    catalog_path = Path(path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid model catalog JSON: {catalog_path}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Model catalog must contain a models list")

    models: list[CatalogModel] = []
    seen_model_ids: set[str] = set()
    for row in payload["models"]:
        if not isinstance(row, dict) or not row.get("enabled"):
            continue
        family_id = str(row.get("family_id", "")).strip()
        model_id = str(row.get("model_id", "")).strip()
        if not family_id or not model_id:
            raise ValueError("Enabled catalog models require nonempty family_id and model_id")
        if model_id in seen_model_ids:
            raise ValueError(f"Duplicate catalog model_id: {model_id}")
        seen_model_ids.add(model_id)
        models.append(
            CatalogModel(
                family_id=family_id,
                model_id=model_id,
                variant_role=str(row.get("variant_role", "")).strip(),
            )
        )

    if not models:
        raise ValueError("Model catalog has no enabled models")
    return tuple(models)


def load_training_dataset(
    data_path: str | Path,
    catalog_path: str | Path,
) -> TrainingDataset:
    """Load cleaned samples, enforcing family labels from the formal catalog."""

    catalog = load_model_catalog(catalog_path)
    catalog_by_model = {model.model_id: model for model in catalog}
    cleaned_samples, clean_audit = load_clean_samples(data_path)

    samples: list[AttributionSample] = []
    for sample in cleaned_samples:
        catalog_model = catalog_by_model.get(sample.model)
        if catalog_model is None:
            raise ValueError(f"clean sample model missing from catalog: {sample.model}")
        if sample.label != catalog_model.family_id:
            raise ValueError(
                "catalog family mismatch: "
                f"model {sample.model!r} is {catalog_model.family_id!r}, "
                f"sample is {sample.label!r}"
            )
        samples.append(
            AttributionSample(
                request_key=sample.request_key,
                text=sample.text,
                family_id=sample.label,
                model_id=sample.model,
                prompt_id=sample.prompt_id,
                semantic_prompt_group=sample.semantic_prompt_group,
                language=sample.language,
                task_type=sample.task_type,
                target_length=sample.target_length,
                temperature=sample.temperature,
                repeat_index=sample.repeat_index,
            )
        )

    present_models = sorted({sample.model_id for sample in samples})
    model_to_family = {
        model_id: catalog_by_model[model_id].family_id for model_id in present_models
    }
    audit = dict(clean_audit)
    audit["counts_by_model"] = dict(
        sorted(Counter(sample.model_id for sample in samples).items())
    )
    audit["catalog_models_absent_from_clean_data"] = sorted(
        set(catalog_by_model) - set(present_models)
    )

    return TrainingDataset(
        samples=tuple(samples),
        family_labels=tuple(sorted({sample.family_id for sample in samples})),
        model_labels=tuple(present_models),
        model_to_family=model_to_family,
        audit=audit,
    )
