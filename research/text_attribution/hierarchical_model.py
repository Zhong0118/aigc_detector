"""Hierarchical XLM-R model used by the Phase3 A0--A4 recipes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from research.text_attribution.phase3_config import (
    ResolvedPhase3Recipe,
    load_resolved_recipe,
    write_resolved_recipe,
)


try:  # Keep control-plane imports available before the CUDA environment is built.
    import torch
    from torch import nn
    import torch.nn.functional as functional
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
    from transformers import AutoConfig, AutoModel
except ImportError as exc:  # pragma: no cover - local controller has no torch
    torch = None
    nn = None
    functional = None
    load_safetensors = None
    save_safetensors = None
    AutoConfig = None
    AutoModel = None
    _RUNTIME_IMPORT_ERROR: ImportError | None = exc
else:
    _RUNTIME_IMPORT_ERROR = None


@dataclass
class Phase3ModelOutput:
    family_logits: Any
    model_logits: Any | None
    projections: Any | None
    pooled_representation: Any


if nn is not None:

    class _XLMRClassificationHead(nn.Module):
        def __init__(self, hidden_size: int, output_size: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, output_size)

        def forward(self, features):
            features = self.dropout(features)
            features = torch.tanh(self.dense(features))
            features = self.dropout(features)
            return self.out_proj(features)


    class HierarchicalAttributionModel(nn.Module):
        """Shared encoder with recipe-controlled family/model/projection heads."""

        def __init__(
            self,
            *,
            backbone: nn.Module,
            recipe: ResolvedPhase3Recipe,
            family_labels: tuple[str, ...],
            model_labels: tuple[str, ...],
        ) -> None:
            super().__init__()
            if len(family_labels) < 2 or len(set(family_labels)) != len(family_labels):
                raise ValueError("family labels must contain at least two unique values")
            if len(set(model_labels)) != len(model_labels):
                raise ValueError("model labels must be unique")
            hidden_size = int(backbone.config.hidden_size)
            dropout_value = getattr(backbone.config, "classifier_dropout", None)
            if dropout_value is None:
                dropout_value = float(
                    getattr(backbone.config, "hidden_dropout_prob", 0.1)
                )

            needs_model_head = (
                recipe.auxiliary_weight > 0 or recipe.consistency_weight > 0
            )
            if needs_model_head and not model_labels:
                raise ValueError(
                    "recipe with auxiliary/consistency loss requires model labels"
                )
            if not needs_model_head and model_labels:
                raise ValueError("A0-like recipe must not receive auxiliary model labels")

            self.backbone = backbone
            self.recipe = recipe
            self.family_labels = tuple(family_labels)
            self.model_labels = tuple(model_labels)
            self.family_head = _XLMRClassificationHead(
                hidden_size, len(family_labels), dropout_value
            )
            self.model_head = (
                _XLMRClassificationHead(
                    hidden_size, len(model_labels), dropout_value
                )
                if needs_model_head
                else None
            )
            self.projection_head = (
                nn.Linear(hidden_size, recipe.projection_dimension)
                if recipe.contrastive_weight > 0
                else None
            )
            self._initialize_new_heads()

        def _initialize_new_heads(self) -> None:
            initializer = getattr(self.backbone, "_init_weights", None)
            if initializer is None:
                return
            self.family_head.apply(initializer)
            if self.model_head is not None:
                self.model_head.apply(initializer)
            if self.projection_head is not None:
                self.projection_head.apply(initializer)

        def forward(self, input_ids, attention_mask, window_mask):
            if input_ids.ndim != 3:
                raise ValueError("Phase3 input_ids must have shape (batch, windows, seq)")
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            if window_mask.shape != input_ids.shape[:2]:
                raise ValueError("window_mask must have shape (batch, windows)")
            if bool((window_mask.sum(dim=1) <= 0).any().item()):
                raise ValueError("every Phase3 sample must contain an active window")

            batch_size, window_count, sequence_length = input_ids.shape
            encoded = self.backbone(
                input_ids=input_ids.reshape(-1, sequence_length),
                attention_mask=attention_mask.reshape(-1, sequence_length),
                return_dict=True,
            )
            window_representations = encoded.last_hidden_state[:, 0, :].reshape(
                batch_size, window_count, -1
            )
            float_mask = window_mask.to(
                dtype=window_representations.dtype,
                device=window_representations.device,
            ).unsqueeze(-1)
            pooled = (window_representations * float_mask).sum(dim=1)
            pooled = pooled / float_mask.sum(dim=1).clamp_min(1.0)

            family_logits = self.family_head(pooled)
            model_logits = (
                None if self.model_head is None else self.model_head(pooled)
            )
            projections = (
                None
                if self.projection_head is None
                else functional.normalize(self.projection_head(pooled), p=2, dim=-1)
            )
            return Phase3ModelOutput(
                family_logits=family_logits,
                model_logits=model_logits,
                projections=projections,
                pooled_representation=pooled,
            )

else:

    class HierarchicalAttributionModel:  # pragma: no cover - dependency message only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_runtime()


def save_phase3_model(
    model: HierarchicalAttributionModel,
    destination: str | Path,
) -> Path:
    """Save a self-contained, hash-bound model without pickle weights."""

    _require_runtime()
    output_path = Path(destination)
    output_path.mkdir(parents=True, exist_ok=False)
    backbone_config_path = output_path / "backbone_config"
    model.backbone.config.save_pretrained(backbone_config_path)
    write_resolved_recipe(output_path / "resolved_recipe.json", model.recipe)
    metadata = {
        "schema_version": 1,
        "recipe_hash": model.recipe.recipe_hash,
        "family_labels": list(model.family_labels),
        "model_labels": list(model.model_labels),
    }
    (output_path / "phase3_model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_safetensors(state, str(output_path / "model.safetensors"))
    return output_path


def load_phase3_model(
    source: str | Path,
    *,
    expected_recipe_hash: str | None = None,
    expected_family_labels: tuple[str, ...] | None = None,
    expected_model_labels: tuple[str, ...] | None = None,
    device: str = "cpu",
) -> HierarchicalAttributionModel:
    """Reload and validate a complete Phase3 model artifact."""

    _require_runtime()
    source_path = Path(source)
    metadata_path = source_path / "phase3_model_config.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Phase3 model metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version",
        "recipe_hash",
        "family_labels",
        "model_labels",
    }:
        raise ValueError("Phase3 model metadata has unknown or missing fields")
    if metadata["schema_version"] != 1:
        raise ValueError("unsupported Phase3 model schema_version")

    recipe = load_resolved_recipe(source_path / "resolved_recipe.json")
    if metadata["recipe_hash"] != recipe.recipe_hash:
        raise ValueError("model metadata recipe hash does not match resolved recipe")
    if expected_recipe_hash is not None and recipe.recipe_hash != expected_recipe_hash:
        raise ValueError("saved model recipe hash does not match requested recipe hash")
    family_labels = _metadata_labels(metadata["family_labels"], "family")
    model_labels = _metadata_labels(metadata["model_labels"], "model")
    if expected_family_labels is not None and family_labels != tuple(
        expected_family_labels
    ):
        raise ValueError("saved model family labels do not match expected family labels")
    if expected_model_labels is not None and model_labels != tuple(
        expected_model_labels
    ):
        raise ValueError("saved model model labels do not match expected model labels")

    config = AutoConfig.from_pretrained(
        source_path / "backbone_config", local_files_only=True
    )
    backbone = AutoModel.from_config(config)
    model = HierarchicalAttributionModel(
        backbone=backbone,
        recipe=recipe,
        family_labels=family_labels,
        model_labels=model_labels,
    )
    state = load_safetensors(str(source_path / "model.safetensors"), device=device)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def _metadata_labels(value: Any, label_kind: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(label, str) or not label for label in value
    ):
        raise ValueError(f"saved {label_kind} labels must be nonempty strings")
    labels = tuple(value)
    if len(labels) != len(set(labels)):
        raise ValueError(f"saved {label_kind} labels contain duplicates")
    return labels


def _require_runtime() -> None:
    if _RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Phase3 model execution requires PyTorch, Transformers, and safetensors"
        ) from _RUNTIME_IMPORT_ERROR
