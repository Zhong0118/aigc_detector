"""Unified Hugging Face training engine for neural family attribution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import WeightedRandomSampler

# Transformers 4.39 eagerly imports an installed PEFT from Trainer. If that
# optional package targets a newer Transformers release, plain Trainer would
# otherwise become unusable even though this engine does not use PEFT.
from transformers.utils import import_utils as _transformers_import_utils

if _transformers_import_utils.is_peft_available():
    try:
        importlib.import_module("peft")
    except ImportError:
        _transformers_import_utils._peft_available = False

from accelerate import Accelerator, DataLoaderConfiguration
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import transformers.trainer as _transformers_trainer

from research.text_attribution.neural_config import NeuralExperimentConfig
from research.text_attribution.neural_data import (
    NeuralAttributionDataset,
    build_family_label_maps,
    model_balanced_weights,
)
from research.text_attribution.training_data import AttributionSample


if "dispatch_batches" not in inspect.signature(Accelerator.__init__).parameters:

    class _LegacyKwargCompatibleAccelerator(Accelerator):
        """Translate Transformers 4.39 dataloader kwargs for Accelerate 1.x."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            legacy_names = (
                "split_batches",
                "dispatch_batches",
                "even_batches",
                "use_seedable_sampler",
            )
            if any(name in kwargs for name in legacy_names):
                dataloader_config = DataLoaderConfiguration(
                    split_batches=kwargs.pop("split_batches", False),
                    dispatch_batches=kwargs.pop("dispatch_batches", None),
                    even_batches=kwargs.pop("even_batches", True),
                    use_seedable_sampler=kwargs.pop(
                        "use_seedable_sampler",
                        False,
                    ),
                )
                kwargs["dataloader_config"] = dataloader_config
            super().__init__(*args, **kwargs)

    _transformers_trainer.Accelerator = _LegacyKwargCompatibleAccelerator


@dataclass(frozen=True)
class NeuralTrainingRequest:
    """Inputs required for one neural-family attribution experiment."""

    experiment: NeuralExperimentConfig
    train_samples: tuple[AttributionSample, ...]
    validation_samples: tuple[AttributionSample, ...]
    family_labels: tuple[str, ...]
    output_dir: Path
    seed: int
    hf_cache: Path | None
    bf16: bool
    smoke_test: bool
    resume_from_checkpoint: Path | None


@dataclass(frozen=True)
class NeuralTrainingResult:
    """Saved artifacts and validation outputs from a completed experiment."""

    model_dir: Path
    tokenizer_dir: Path
    validation_logits: np.ndarray
    validation_probabilities: np.ndarray
    trainer_history: tuple[Mapping[str, Any], ...]
    best_checkpoint: str | None


class ModelBalancedTrainer(Trainer):
    """Trainer whose training sampler gives every concrete model equal mass."""

    def __init__(
        self,
        *args: Any,
        train_samples: tuple[AttributionSample, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_balanced_train_samples = (
            None if train_samples is None else tuple(train_samples)
        )
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("ModelBalancedTrainer requires a train dataset")
        return super().get_train_dataloader()

    def _get_train_sampler(self):
        if self.train_dataset is None:
            raise ValueError("ModelBalancedTrainer requires a train dataset")
        if self._model_balanced_train_samples is None:
            raise ValueError(
                "ModelBalancedTrainer requires train_samples for model balancing"
            )
        if len(self._model_balanced_train_samples) != len(self.train_dataset):
            raise ValueError(
                "train_samples and train dataset must contain the same number "
                "of rows"
            )

        weights = torch.as_tensor(
            model_balanced_weights(self._model_balanced_train_samples),
            dtype=torch.double,
        )
        generator = torch.Generator()
        data_seed = self.args.data_seed
        generator.manual_seed(self.args.seed if data_seed is None else data_seed)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert a finite logits matrix to stable row-normalized probabilities."""

    values = np.asarray(logits)
    if values.ndim != 2:
        raise ValueError("logits must be a two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError("logits must contain only finite values")

    floating_values = values.astype(np.float64, copy=False)
    shifted = floating_values - floating_values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def train_neural_experiment(
    request: NeuralTrainingRequest,
) -> NeuralTrainingResult:
    """Train, validate, and save one configured family-attribution model."""

    family_to_id, id_to_family = _validate_request(request)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    if request.hf_cache is not None:
        request.hf_cache.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        request.experiment.model_id,
        cache_dir=request.hf_cache,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        request.experiment.model_id,
        cache_dir=request.hf_cache,
        num_labels=len(request.family_labels),
        id2label=id_to_family,
        label2id=family_to_id,
    )

    train_dataset = NeuralAttributionDataset(
        request.train_samples,
        tokenizer,
        family_to_id,
        request.experiment.max_length,
    )
    validation_dataset = NeuralAttributionDataset(
        request.validation_samples,
        tokenizer,
        family_to_id,
        request.experiment.max_length,
    )
    checkpoint_dir = request.output_dir / "checkpoints"
    training_arguments = _build_training_arguments(request, checkpoint_dir)
    callbacks = []
    if not request.smoke_test:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=request.experiment.early_stopping_patience
            )
        )

    trainer = ModelBalancedTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        train_samples=request.train_samples,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )
    resume_checkpoint = (
        None
        if request.resume_from_checkpoint is None
        else str(request.resume_from_checkpoint)
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    prediction_output = trainer.predict(validation_dataset)
    raw_predictions = prediction_output.predictions
    if isinstance(raw_predictions, tuple):
        raw_predictions = raw_predictions[0]
    validation_logits = np.asarray(raw_predictions).copy()
    validation_probabilities = logits_to_probabilities(validation_logits)

    model_dir = request.output_dir / "model"
    tokenizer_dir = request.output_dir / "tokenizer"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(tokenizer_dir)

    history = tuple(
        _json_safe_mapping(row) for row in trainer.state.log_history
    )
    return NeuralTrainingResult(
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        validation_logits=validation_logits,
        validation_probabilities=validation_probabilities,
        trainer_history=history,
        best_checkpoint=trainer.state.best_model_checkpoint,
    )


def _validate_request(
    request: NeuralTrainingRequest,
) -> tuple[dict[str, int], dict[int, str]]:
    if not isinstance(request, NeuralTrainingRequest):
        raise TypeError("request must be a NeuralTrainingRequest")
    if not isinstance(request.experiment, NeuralExperimentConfig):
        raise TypeError("experiment must be a NeuralExperimentConfig")
    if not isinstance(request.train_samples, tuple) or not request.train_samples:
        raise ValueError("train_samples must be a non-empty tuple")
    if (
        not isinstance(request.validation_samples, tuple)
        or not request.validation_samples
    ):
        raise ValueError("validation_samples must be a non-empty tuple")
    if not isinstance(request.family_labels, tuple):
        raise ValueError("family_labels must be a tuple")

    family_to_id, id_to_family = build_family_label_maps(request.family_labels)
    expected_families = set(request.family_labels)
    _validate_split_families(
        "train_samples",
        request.train_samples,
        expected_families,
    )
    _validate_split_families(
        "validation_samples",
        request.validation_samples,
        expected_families,
    )

    if not isinstance(request.output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if request.output_dir.exists() and not request.output_dir.is_dir():
        raise ValueError("output_dir must not be an existing file")
    if request.hf_cache is not None:
        if not isinstance(request.hf_cache, Path):
            raise TypeError("hf_cache must be a Path or None")
        if request.hf_cache.exists() and not request.hf_cache.is_dir():
            raise ValueError("hf_cache must not be an existing file")
    if not isinstance(request.seed, int) or isinstance(request.seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(request.bf16, bool):
        raise ValueError("bf16 must be a boolean")
    if not isinstance(request.smoke_test, bool):
        raise ValueError("smoke_test must be a boolean")
    if request.resume_from_checkpoint is not None:
        if not isinstance(request.resume_from_checkpoint, Path):
            raise TypeError("resume_from_checkpoint must be a Path or None")
        if not request.resume_from_checkpoint.exists():
            raise ValueError("resume_from_checkpoint must exist")
    return family_to_id, id_to_family


def _validate_split_families(
    name: str,
    samples: tuple[AttributionSample, ...],
    expected_families: set[str],
) -> None:
    if any(not isinstance(sample, AttributionSample) for sample in samples):
        raise TypeError(f"{name} must contain only AttributionSample values")
    actual_families = {sample.family_id for sample in samples}
    if actual_families != expected_families:
        missing = sorted(expected_families - actual_families)
        unknown = sorted(actual_families - expected_families)
        details = []
        if missing:
            details.append(f"missing families {missing}")
        if unknown:
            details.append(f"unknown families {unknown}")
        raise ValueError(f"{name} family coverage mismatch: {'; '.join(details)}")


def _build_training_arguments(
    request: NeuralTrainingRequest,
    checkpoint_dir: Path,
) -> TrainingArguments:
    smoke = request.smoke_test
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        evaluation_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        load_best_model_at_end=not smoke,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        remove_unused_columns=True,
        learning_rate=request.experiment.learning_rate,
        weight_decay=request.experiment.weight_decay,
        warmup_ratio=request.experiment.warmup_ratio,
        num_train_epochs=request.experiment.epochs,
        per_device_train_batch_size=(
            request.experiment.per_device_train_batch_size
        ),
        per_device_eval_batch_size=(
            request.experiment.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            request.experiment.gradient_accumulation_steps
        ),
        label_smoothing_factor=request.experiment.label_smoothing_factor,
        bf16=False if smoke else request.bf16,
        save_total_limit=request.experiment.save_total_limit,
        seed=request.seed,
        data_seed=request.seed,
        max_steps=2 if smoke else -1,
        logging_strategy="steps",
        logging_steps=1,
        disable_tqdm=True,
    )


def _compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    predictions = prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted_labels = np.asarray(predictions).argmax(axis=1)
    true_labels = np.asarray(prediction.label_ids)
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def _json_safe_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in row.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe_value(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
