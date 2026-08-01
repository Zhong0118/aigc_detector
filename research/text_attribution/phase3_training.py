"""Unified training lifecycle for one Phase3 recipe, seed, and data view."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from research.text_attribution.attribution_losses import compute_phase3_loss
from research.text_attribution.hierarchical_model import (
    HierarchicalAttributionModel,
    save_phase3_model,
)
from research.text_attribution.phase3_config import (
    ResolvedPhase3Recipe,
    write_resolved_recipe,
)
from research.text_attribution.phase3_data import (
    Phase3AttributionDataset,
    Phase3DataView,
    StructuredContrastiveBatchSampler,
)


try:  # Imported only in the server training environment.
    import torch
    from torch.utils.data import Sampler, WeightedRandomSampler
    from transformers import (
        AutoModel,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )
except ImportError as exc:  # pragma: no cover - controller environment
    torch = None
    Sampler = None
    WeightedRandomSampler = None
    AutoModel = None
    AutoTokenizer = None
    EarlyStoppingCallback = None
    Trainer = None
    TrainerCallback = None
    TrainingArguments = None
    default_data_collator = None
    set_seed = None
    _RUNTIME_IMPORT_ERROR: ImportError | None = exc
else:
    _RUNTIME_IMPORT_ERROR = None


@dataclass(frozen=True)
class Phase3TrainingRequest:
    recipe: ResolvedPhase3Recipe
    data_view: Phase3DataView | None
    output_dir: Path
    seed: int
    hf_cache: Path | None = None
    bf16: bool = True
    smoke_test: bool = False
    require_cuda: bool = True
    resume_from_checkpoint: Path | None = None
    model_source: str | Path | None = None
    tokenizer_source: str | Path | None = None
    max_steps: int | None = None
    gradient_checkpointing: bool = False
    preflight_only: bool = False


@dataclass(frozen=True)
class Phase3TrainingResult:
    model_dir: Path
    tokenizer_dir: Path
    validation_logits: np.ndarray
    validation_probabilities: np.ndarray
    validation_labels: np.ndarray
    trainer_history: tuple[Mapping[str, Any], ...]
    loss_components: tuple[Mapping[str, Any], ...]
    best_checkpoint: str | None


if Trainer is not None:

    class _FlattenedStructuredSampler(Sampler):
        def __init__(self, batch_sampler: StructuredContrastiveBatchSampler) -> None:
            self.batch_sampler = batch_sampler

        def __len__(self) -> int:
            return len(self.batch_sampler) * self.batch_sampler._batch_size

        def __iter__(self):
            for batch in self.batch_sampler:
                yield from batch

        def set_epoch(self, epoch: int) -> None:
            self.batch_sampler.set_epoch(epoch)


    class _RecipeCheckpointCallback(TrainerCallback):
        def __init__(self, recipe_hash: str) -> None:
            self.recipe_hash = recipe_hash

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            _atomic_write_json(
                checkpoint / "phase3_checkpoint.json",
                {
                    "schema_version": 1,
                    "recipe_hash": self.recipe_hash,
                    "global_step": int(state.global_step),
                },
            )
            return control


    class Phase3Trainer(Trainer):
        """Trainer that owns Phase3 loss composition and deterministic sampling."""

        def __init__(
            self,
            *args: Any,
            recipe: ResolvedPhase3Recipe,
            train_samples: tuple[Any, ...],
            model_to_family_ids: tuple[int, ...],
            seed: int,
            **kwargs: Any,
        ) -> None:
            self.phase3_recipe = recipe
            self.phase3_train_samples = train_samples
            self.phase3_model_to_family_ids = model_to_family_ids
            self.phase3_seed = seed
            self.phase3_loss_history: list[dict[str, Any]] = []
            super().__init__(*args, **kwargs)

        def _get_train_sampler(self, train_dataset=None):
            selected = self.train_dataset if train_dataset is None else train_dataset
            if selected is None:
                raise ValueError("Phase3Trainer requires a training dataset")
            if self.phase3_recipe.contrastive_weight > 0:
                structured = StructuredContrastiveBatchSampler(
                    self.phase3_train_samples,
                    batch_size=self.phase3_recipe.per_device_train_batch_size,
                    seed=self.phase3_seed,
                )
                return _FlattenedStructuredSampler(structured)

            counts = Counter(row.model_id for row in self.phase3_train_samples)
            weights = torch.as_tensor(
                [1.0 / counts[row.model_id] for row in self.phase3_train_samples],
                dtype=torch.double,
            )
            generator = torch.Generator()
            generator.manual_seed(self.phase3_seed)
            return WeightedRandomSampler(
                weights,
                num_samples=len(self.phase3_train_samples),
                replacement=True,
                generator=generator,
            )

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            values = dict(inputs)
            family_labels = values.pop("family_labels")
            model_labels = values.pop("model_labels")
            contrastive_family_labels = values.pop(
                "contrastive_family_labels"
            )
            contrastive_group_ids = values.pop("contrastive_group_ids")
            values.pop("contrastive_model_ids")
            output = model(**values)
            contrastive_weight = (
                self.phase3_recipe.contrastive_weight if model.training else 0.0
            )
            losses = compute_phase3_loss(
                family_logits=output.family_logits,
                family_labels=family_labels,
                label_smoothing_factor=(
                    self.phase3_recipe.label_smoothing_factor
                ),
                auxiliary_weight=self.phase3_recipe.auxiliary_weight,
                model_logits=output.model_logits,
                model_labels=model_labels,
                consistency_weight=self.phase3_recipe.consistency_weight,
                model_to_family_ids=self.phase3_model_to_family_ids,
                contrastive_weight=contrastive_weight,
                projections=output.projections,
                contrastive_family_labels=contrastive_family_labels,
                contrastive_group_ids=contrastive_group_ids,
                contrastive_temperature=(
                    self.phase3_recipe.contrastive_temperature
                ),
            )
            if model.training:
                self.phase3_loss_history.append(
                    {
                        "global_step": int(self.state.global_step),
                        "total": float(losses.total.detach().cpu()),
                        "family": float(losses.family.detach().cpu()),
                        "model": float(losses.model.detach().cpu()),
                        "consistency": float(losses.consistency.detach().cpu()),
                        "contrastive": float(losses.contrastive.detach().cpu()),
                    }
                )
            trainer_output = {"logits": output.family_logits}
            return (losses.total, trainer_output) if return_outputs else losses.total

else:

    class Phase3Trainer:  # pragma: no cover - dependency message only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_runtime()


def validate_phase3_checkpoint(
    checkpoint: str | Path,
    expected_recipe_hash: str,
) -> Mapping[str, Any]:
    metadata_path = Path(checkpoint) / "phase3_checkpoint.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Phase3 checkpoint metadata: {metadata_path}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "recipe_hash",
        "global_step",
    }:
        raise ValueError("Phase3 checkpoint metadata has unknown or missing fields")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported Phase3 checkpoint schema_version")
    if payload["recipe_hash"] != expected_recipe_hash:
        raise ValueError("checkpoint recipe hash does not match requested recipe hash")
    if not isinstance(payload["global_step"], int) or payload["global_step"] < 0:
        raise ValueError("checkpoint global_step must be a nonnegative integer")
    return payload


def validate_phase3_runtime_policy(
    *,
    require_cuda: bool,
    smoke_test: bool,
    cuda_available: bool,
    bf16_requested: bool,
    bf16_supported: bool,
) -> bool:
    """Validate formal/smoke device rules and return effective BF16."""

    if require_cuda and not cuda_available:
        raise RuntimeError("formal Phase3 training requires a CUDA device")
    if not smoke_test and not require_cuda:
        raise RuntimeError("CPU execution is restricted to Phase3 smoke/dry-run")
    if bf16_requested and cuda_available and not bf16_supported:
        raise RuntimeError("requested Phase3 BF16 is unsupported by this CUDA device")
    return bool(bf16_requested and cuda_available and bf16_supported)


def train_phase3_experiment(request: Phase3TrainingRequest) -> Phase3TrainingResult:
    """Train and publish exactly one recipe × seed × data-view run."""

    output_dir = Path(request.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Phase3 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        _require_runtime()
        if request.data_view is None:
            raise ValueError("Phase3 training request requires a data view")
        if request.seed < 0:
            raise ValueError("Phase3 seed must be nonnegative")
        if request.max_steps is not None and request.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if request.resume_from_checkpoint is not None:
            validate_phase3_checkpoint(
                request.resume_from_checkpoint, request.recipe.recipe_hash
            )
        if (
            request.recipe.contrastive_weight > 0
            and request.recipe.per_device_train_batch_size not in {4, 16}
        ):
            raise ValueError(
                "contrastive Phase3 recipes require batch 16 (A3) or batch 4 (A4)"
            )

        cuda_available = bool(torch.cuda.is_available())
        bf16_supported = bool(
            cuda_available
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        effective_bf16 = validate_phase3_runtime_policy(
            require_cuda=request.require_cuda,
            smoke_test=request.smoke_test,
            cuda_available=cuda_available,
            bf16_requested=request.bf16 and request.recipe.bf16,
            bf16_supported=bf16_supported,
        )
        set_seed(request.seed)

        source = request.model_source or request.recipe.backbone_model_id
        tokenizer_source = request.tokenizer_source or source
        cache_dir = None if request.hf_cache is None else str(request.hf_cache)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            cache_dir=cache_dir,
            use_fast=True,
        )
        backbone = AutoModel.from_pretrained(source, cache_dir=cache_dir)
        if getattr(backbone.config, "model_type", None) != "xlm-roberta":
            raise ValueError("Phase3 backbone must be an XLM-R-compatible model")
        if request.gradient_checkpointing:
            backbone.gradient_checkpointing_enable()

        view = request.data_view
        needs_model_head = (
            request.recipe.auxiliary_weight > 0
            or request.recipe.consistency_weight > 0
        )
        model_labels = (
            tuple(
                view.id_to_train_visible_model[index]
                for index in range(len(view.id_to_train_visible_model))
            )
            if needs_model_head
            else ()
        )
        family_labels = tuple(
            view.id_to_family[index] for index in range(len(view.id_to_family))
        )
        model = HierarchicalAttributionModel(
            backbone=backbone,
            recipe=request.recipe,
            family_labels=family_labels,
            model_labels=model_labels,
        )

        train_dataset = Phase3AttributionDataset(
            samples=view.train_samples,
            data_view=view,
            tokenizer=tokenizer,
            max_length=request.recipe.max_length,
            window_mode=request.recipe.window_mode,
        )
        validation_dataset = Phase3AttributionDataset(
            samples=view.validation_samples,
            data_view=view,
            tokenizer=tokenizer,
            max_length=request.recipe.max_length,
            window_mode=request.recipe.window_mode,
        )
        if request.preflight_only:
            return _run_preflight(
                request,
                model,
                train_dataset,
                tokenizer,
                family_labels,
                model_labels,
            )

        checkpoint_dir = output_dir / "checkpoints"
        training_arguments = _training_arguments(
            request, checkpoint_dir, effective_bf16
        )
        model_to_family_ids = tuple(
            view.model_to_family_id[view.id_to_train_visible_model[index]]
            for index in range(len(view.id_to_train_visible_model))
        )
        callbacks = [_RecipeCheckpointCallback(request.recipe.recipe_hash)]
        if request.max_steps is None and not request.smoke_test:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=(
                        request.recipe.early_stopping_patience
                    )
                )
            )
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_arguments,
            "train_dataset": train_dataset,
            "eval_dataset": validation_dataset,
            "data_collator": default_data_collator,
            "compute_metrics": _compute_family_metrics,
            "callbacks": callbacks,
            "recipe": request.recipe,
            "train_samples": view.train_samples,
            "model_to_family_ids": model_to_family_ids,
            "seed": request.seed,
        }
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = Phase3Trainer(
            **trainer_kwargs,
        )
        trainer.train(
            resume_from_checkpoint=(
                None
                if request.resume_from_checkpoint is None
                else str(request.resume_from_checkpoint)
            )
        )
        prediction = trainer.predict(validation_dataset)
        validation_logits = np.asarray(prediction.predictions, dtype=np.float32)
        if validation_logits.ndim == 3 and validation_logits.shape[0] == 1:
            validation_logits = validation_logits[0]
        if validation_logits.ndim != 2 or validation_logits.shape[1] != len(
            family_labels
        ):
            raise RuntimeError(
                "Phase3 validation logits have an unexpected shape: "
                f"{validation_logits.shape!r}"
            )
        if not np.isfinite(validation_logits).all():
            raise FloatingPointError("non-finite Phase3 validation logits")
        validation_probabilities = _softmax(validation_logits)
        validation_labels = np.asarray(
            [view.family_to_id[row.family_id] for row in view.validation_samples],
            dtype=np.int64,
        )

        model_dir = save_phase3_model(model, output_dir / "model")
        tokenizer_dir = output_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
        write_resolved_recipe(output_dir / "resolved_recipe.json", request.recipe)
        _atomic_write_json(
            output_dir / "label_maps.json",
            {
                "schema_version": 1,
                "family_labels": list(family_labels),
                "train_visible_model_labels": list(model_labels),
            },
        )
        history = tuple(dict(row) for row in trainer.state.log_history)
        loss_history = tuple(dict(row) for row in trainer.phase3_loss_history)
        _atomic_write_json(output_dir / "trainer_history.json", list(history))
        _atomic_write_json(
            output_dir / "loss_components.json", list(loss_history)
        )
        np.savez_compressed(
            output_dir / "validation_outputs.npz",
            logits=validation_logits,
            probabilities=validation_probabilities,
            labels=validation_labels,
        )
        best_checkpoint = trainer.state.best_model_checkpoint
        _atomic_write_json(
            output_dir / "training_result.json",
            {
                "schema_version": 1,
                "recipe_hash": request.recipe.recipe_hash,
                "seed": request.seed,
                "view_id": view.view_id,
                "best_checkpoint": best_checkpoint,
                "validation_sample_count": len(validation_labels),
            },
        )
        _atomic_write_json(
            output_dir / "COMPLETED.json",
            {
                "schema_version": 1,
                "status": "complete",
                "recipe_hash": request.recipe.recipe_hash,
            },
        )
        return Phase3TrainingResult(
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            validation_logits=validation_logits,
            validation_probabilities=validation_probabilities,
            validation_labels=validation_labels,
            trainer_history=history,
            loss_components=loss_history,
            best_checkpoint=best_checkpoint,
        )
    except Exception as exc:
        _atomic_write_json(
            output_dir / "FAILED.json",
            {
                "schema_version": 1,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def _training_arguments(
    request: Phase3TrainingRequest,
    checkpoint_dir: Path,
    effective_bf16: bool,
):
    max_steps = request.max_steps
    if max_steps is None and request.smoke_test:
        max_steps = 2
    kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "overwrite_output_dir": False,
        "num_train_epochs": request.recipe.epochs,
        "learning_rate": request.recipe.learning_rate,
        "weight_decay": request.recipe.weight_decay,
        "warmup_ratio": request.recipe.warmup_ratio,
        "per_device_train_batch_size": (
            request.recipe.per_device_train_batch_size
        ),
        "per_device_eval_batch_size": max(
            1, request.recipe.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": (
            request.recipe.gradient_accumulation_steps
        ),
        "bf16": effective_bf16,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": request.recipe.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": request.seed,
        "data_seed": request.seed,
        "dataloader_num_workers": 0,
        "label_names": [
            "family_labels",
            "model_labels",
            "contrastive_family_labels",
            "contrastive_group_ids",
            "contrastive_model_ids",
        ],
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"
    else:  # Transformers 4.39 compatibility for existing server environments.
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def _compute_family_metrics(prediction) -> dict[str, float]:
    logits = np.asarray(prediction.predictions)
    if isinstance(prediction.label_ids, tuple):
        labels = np.asarray(prediction.label_ids[0])
    else:
        labels = np.asarray(prediction.label_ids)
    predicted = np.argmax(logits, axis=-1)
    family_count = logits.shape[-1]
    per_family_f1: list[float] = []
    supports: list[int] = []
    for family_id in range(family_count):
        true_positive = int(
            np.sum((labels == family_id) & (predicted == family_id))
        )
        false_positive = int(
            np.sum((labels != family_id) & (predicted == family_id))
        )
        false_negative = int(
            np.sum((labels == family_id) & (predicted != family_id))
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_family_f1.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
        supports.append(int(np.sum(labels == family_id)))
    support_total = max(sum(supports), 1)
    return {
        "accuracy": float(np.mean(labels == predicted)),
        "macro_f1": float(np.mean(per_family_f1)),
        "weighted_f1": float(
            sum(value * support for value, support in zip(per_family_f1, supports))
            / support_total
        ),
    }


def _run_preflight(
    request,
    model,
    dataset,
    tokenizer,
    family_labels,
    model_labels,
) -> Phase3TrainingResult:
    batch_size = min(request.recipe.per_device_train_batch_size, len(dataset))
    if batch_size < 1:
        raise ValueError("Phase3 preflight requires nonempty training data")
    collated = default_data_collator([dataset[index] for index in range(batch_size)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model_inputs = {
        key: value.to(device)
        for key, value in collated.items()
        if key in {"input_ids", "attention_mask", "window_mask"}
    }
    with torch.no_grad():
        output = model(**model_inputs)
    if not bool(torch.isfinite(output.family_logits).all().item()):
        raise FloatingPointError("non-finite logits during Phase3 preflight")
    model_dir = save_phase3_model(model, request.output_dir / "model")
    tokenizer_dir = request.output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    logits = output.family_logits.detach().float().cpu().numpy()
    probabilities = _softmax(logits)
    labels = np.asarray(
        [
            request.data_view.family_to_id[
                request.data_view.train_samples[index].family_id
            ]
            for index in range(batch_size)
        ]
    )
    _atomic_write_json(
        request.output_dir / "COMPLETED.json",
        {
            "schema_version": 1,
            "status": "preflight_complete",
            "recipe_hash": request.recipe.recipe_hash,
        },
    )
    return Phase3TrainingResult(
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        validation_logits=logits,
        validation_probabilities=probabilities,
        validation_labels=labels,
        trainer_history=(),
        loss_components=(),
        best_checkpoint=None,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
    temporary.replace(path)


def _require_runtime() -> None:
    if _RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Phase3 training requires PyTorch, Transformers, and Accelerate"
        ) from _RUNTIME_IMPORT_ERROR
