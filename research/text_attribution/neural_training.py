"""Unified Hugging Face training engine for neural family attribution."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from packaging.version import InvalidVersion, Version
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import WeightedRandomSampler

import transformers
from transformers.utils import import_utils as _transformers_import_utils

_transformers_version = transformers.__version__
_peft_available_at_import = _transformers_import_utils._peft_available


def _is_transformers_439(version: str) -> bool:
    try:
        release = Version(version).release
    except InvalidVersion:
        return False
    return release[:2] == (4, 39)


def _needs_legacy_accelerator_adapter(
    transformers_version: str,
    accelerator_parameters: Collection[str],
) -> bool:
    """Identify only the Transformers 4.39 / Accelerate 1.x API mismatch."""

    return (
        _is_transformers_439(transformers_version)
        and "dispatch_batches" not in accelerator_parameters
    )


_peft_439_import_incompatible = False
if _is_transformers_439(_transformers_version) and _peft_available_at_import:
    try:
        importlib.import_module("peft")
    except ImportError:
        _peft_439_import_incompatible = True

from accelerate import Accelerator, DataLoaderConfiguration
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    TrainingArguments,
)


@contextmanager
def _temporary_peft_compatibility():
    if not _peft_439_import_incompatible:
        yield
        return
    original_peft_available = _transformers_import_utils._peft_available
    _transformers_import_utils._peft_available = False
    try:
        yield
    finally:
        _transformers_import_utils._peft_available = original_peft_available


with _temporary_peft_compatibility():
    from transformers import Trainer
    import transformers.trainer as _transformers_trainer

from research.text_attribution.neural_config import NeuralExperimentConfig
from research.text_attribution.neural_data import (
    NeuralAttributionDataset,
    build_family_label_maps,
    model_balanced_weights,
)
from research.text_attribution.training_data import AttributionSample


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


@contextmanager
def _temporary_trainer_compatibility():
    needs_accelerator_adapter = _needs_legacy_accelerator_adapter(
        _transformers_version,
        inspect.signature(Accelerator.__init__).parameters,
    )
    if not _peft_439_import_incompatible and not needs_accelerator_adapter:
        yield
        return
    original_accelerator = _transformers_trainer.Accelerator
    original_peft_available = _transformers_import_utils._peft_available
    if _peft_439_import_incompatible:
        _transformers_import_utils._peft_available = False
    if needs_accelerator_adapter:
        _transformers_trainer.Accelerator = _LegacyKwargCompatibleAccelerator
    try:
        yield
    finally:
        if needs_accelerator_adapter:
            _transformers_trainer.Accelerator = original_accelerator
        if _peft_439_import_incompatible:
            _transformers_import_utils._peft_available = original_peft_available


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
        with _temporary_trainer_compatibility():
            super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("ModelBalancedTrainer requires a train dataset")
        with _temporary_peft_compatibility():
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        with _temporary_peft_compatibility():
            return super().get_eval_dataloader(eval_dataset)

    def train(self, *args: Any, **kwargs: Any):
        with _temporary_peft_compatibility():
            return super().train(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any):
        with _temporary_peft_compatibility():
            return super().predict(*args, **kwargs)

    def save_model(self, *args: Any, **kwargs: Any) -> None:
        with _temporary_peft_compatibility():
            super().save_model(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        selected_dataset = (
            self.train_dataset if train_dataset is None else train_dataset
        )
        if selected_dataset is None:
            raise ValueError("ModelBalancedTrainer requires a train dataset")
        if self._model_balanced_train_samples is None:
            raise ValueError(
                "ModelBalancedTrainer requires train_samples for model balancing"
            )
        if len(self._model_balanced_train_samples) != len(selected_dataset):
            raise ValueError(
                "train_samples and train dataset must contain the same number "
                "of rows"
            )

        weights = torch.as_tensor(
            model_balanced_weights(self._model_balanced_train_samples),
            dtype=torch.double,
        )
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
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

    transformers.set_seed(request.seed)
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

    history = tuple(
        _json_safe_mapping(row) for row in trainer.state.log_history
    )
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=request.output_dir,
            prefix=".neural-artifacts-",
        )
    )
    try:
        staged_model_dir = staging_dir / "model"
        staged_tokenizer_dir = staging_dir / "tokenizer"
        trainer.save_model(str(staged_model_dir))
        tokenizer.save_pretrained(staged_tokenizer_dir)
        _write_json_atomically(
            staging_dir / "trainer_history.json",
            list(history),
        )
        _publish_staged_artifacts(staging_dir, request.output_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    model_dir = request.output_dir / "model"
    tokenizer_dir = request.output_dir / "tokenizer"
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
    _validate_experiment(request.experiment)
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
    _ensure_final_artifacts_absent(request.output_dir)
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
        if not request.resume_from_checkpoint.is_dir():
            raise ValueError(
                "resume_from_checkpoint must be an existing checkpoint directory"
            )
        trainer_state = request.resume_from_checkpoint / "trainer_state.json"
        if not trainer_state.is_file():
            raise ValueError(
                "resume_from_checkpoint must contain trainer_state.json"
            )
    return family_to_id, id_to_family


def _ensure_final_artifacts_absent(output_dir: Path) -> None:
    final_names = (
        "model",
        "tokenizer",
        "trainer_history.json",
        "COMPLETED.json",
    )
    for name in final_names:
        if (output_dir / name).exists():
            raise ValueError(f"final artifact already exists: {output_dir / name}")


def _validate_experiment(experiment: NeuralExperimentConfig) -> None:
    if not isinstance(experiment.model_id, str) or not experiment.model_id.strip():
        raise ValueError("experiment model_id must be a non-empty string")

    positive_integer_fields = (
        "max_length",
        "per_device_train_batch_size",
        "epochs",
        "gradient_accumulation_steps",
        "early_stopping_patience",
        "save_total_limit",
    )
    for field in positive_integer_fields:
        value = getattr(experiment, field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"experiment {field} must be a positive integer")

    learning_rate = _finite_number(experiment, "learning_rate")
    if learning_rate <= 0:
        raise ValueError("experiment learning_rate must be positive")
    weight_decay = _finite_number(experiment, "weight_decay")
    if weight_decay < 0:
        raise ValueError("experiment weight_decay must be non-negative")
    for field in ("warmup_ratio", "label_smoothing_factor"):
        value = _finite_number(experiment, field)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"experiment {field} must be between 0 and 1")
    if type(experiment.bf16) is not bool:
        raise ValueError("experiment bf16 must be a bool")


def _finite_number(
    experiment: NeuralExperimentConfig,
    field: str,
) -> float:
    value = getattr(experiment, field)
    if type(value) not in (int, float):
        raise ValueError(f"experiment {field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"experiment {field} must be a finite number")
    return normalized


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
    arguments: dict[str, Any] = dict(
        output_dir=str(checkpoint_dir),
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
        use_cpu=smoke,
        save_total_limit=request.experiment.save_total_limit,
        seed=request.seed,
        data_seed=request.seed,
        max_steps=2 if smoke else -1,
        logging_strategy="steps",
        logging_steps=1,
        disable_tqdm=True,
    )
    strategy_parameter = _evaluation_strategy_parameter(
        inspect.signature(TrainingArguments.__init__).parameters
    )
    arguments[strategy_parameter] = "no" if smoke else "epoch"
    return TrainingArguments(**arguments)


def _evaluation_strategy_parameter(
    training_argument_parameters: Collection[str],
) -> str:
    if "eval_strategy" in training_argument_parameters:
        return "eval_strategy"
    if "evaluation_strategy" in training_argument_parameters:
        return "evaluation_strategy"
    raise RuntimeError(
        "TrainingArguments exposes neither eval_strategy nor "
        "evaluation_strategy"
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


def _publish_staged_artifacts(
    staging_dir: Path,
    output_dir: Path,
) -> None:
    _validate_staged_artifacts(staging_dir)
    _ensure_final_artifacts_absent(output_dir)
    publications = (
        (staging_dir / "model", output_dir / "model"),
        (staging_dir / "tokenizer", output_dir / "tokenizer"),
        (
            staging_dir / "trainer_history.json",
            output_dir / "trainer_history.json",
        ),
    )
    moved: list[tuple[Path, Path]] = []
    completed_path = output_dir / "COMPLETED.json"
    try:
        for source, destination in publications:
            os.replace(source, destination)
            moved.append((destination, source))
        _write_json_atomically(
            completed_path,
            {
                "status": "complete",
                "model_dir": "model",
                "tokenizer_dir": "tokenizer",
                "trainer_history": "trainer_history.json",
            },
        )
    except BaseException:
        completed_path.unlink(missing_ok=True)
        for destination, source in reversed(moved):
            try:
                os.replace(destination, source)
            except OSError:
                _remove_owned_artifact(destination)
        raise


def _validate_staged_artifacts(staging_dir: Path) -> None:
    model_dir = staging_dir / "model"
    if not (model_dir / "config.json").is_file():
        raise RuntimeError("staged model is missing config.json")
    model_weight_names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any((model_dir / name).is_file() for name in model_weight_names):
        raise RuntimeError("staged model is missing saved weights")

    tokenizer_dir = staging_dir / "tokenizer"
    if not (tokenizer_dir / "tokenizer_config.json").is_file():
        raise RuntimeError("staged tokenizer is missing tokenizer_config.json")
    tokenizer_asset_names = (
        "tokenizer.json",
        "vocab.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    )
    if not any(
        (tokenizer_dir / name).is_file() for name in tokenizer_asset_names
    ):
        raise RuntimeError("staged tokenizer is missing vocabulary assets")

    history_path = staging_dir / "trainer_history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("staged trainer history is not valid JSON") from error
    if not isinstance(history, list):
        raise RuntimeError("staged trainer history must be a JSON list")


def _remove_owned_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _write_json_atomically(destination: Path, payload: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
