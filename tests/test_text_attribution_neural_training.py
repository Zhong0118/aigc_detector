from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    TrainingArguments,
)

from research.text_attribution.neural_config import NeuralExperimentConfig
from research.text_attribution.neural_data import NeuralAttributionDataset
from research.text_attribution.training_data import AttributionSample


FAMILY_LABELS = tuple(f"family-{index}" for index in range(10))


def _training_module():
    try:
        return importlib.import_module("research.text_attribution.neural_training")
    except ModuleNotFoundError as error:
        pytest.fail(f"neural training module is missing: {error}")


def _experiment(model_path: Path) -> NeuralExperimentConfig:
    return NeuralExperimentConfig(
        model_id=str(model_path),
        max_length=24,
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.0,
        label_smoothing_factor=0.0,
        gradient_accumulation_steps=1,
        early_stopping_patience=1,
        save_total_limit=1,
        bf16=False,
    )


def _sample(index: int, *, split: str, model_suffix: int = 0) -> AttributionSample:
    family = FAMILY_LABELS[index % len(FAMILY_LABELS)]
    return AttributionSample(
        request_key=f"{split}-{index}",
        text=f"family sample text {index} token",
        family_id=family,
        model_id=f"{family}-model-{model_suffix}",
        prompt_id=f"prompt-{index}",
        semantic_prompt_group=f"group-{index}",
        language="en",
        task_type="explanation",
        target_length="short",
        temperature=0.2,
        repeat_index=0,
    )


def _samples(split: str) -> tuple[AttributionSample, ...]:
    return tuple(_sample(index, split=split) for index in range(10))


def _write_tiny_bert(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "family",
        "sample",
        "text",
        "token",
    ] + [str(index) for index in range(10)]
    (model_dir / "vocab.txt").write_text(
        "\n".join(vocabulary) + "\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizerFast(
        vocab_file=str(model_dir / "vocab.txt"),
        do_lower_case=True,
    )
    tokenizer.save_pretrained(model_dir)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            num_labels=len(FAMILY_LABELS),
        )
    )
    model.save_pretrained(model_dir)


def _request(
    module,
    root: Path,
    model_dir: Path,
    **overrides,
):
    values = {
        "experiment": _experiment(model_dir),
        "train_samples": _samples("train"),
        "validation_samples": _samples("validation"),
        "family_labels": FAMILY_LABELS,
        "output_dir": root / "output",
        "seed": 17,
        "hf_cache": root / "cache",
        "bf16": False,
        "smoke_test": True,
        "resume_from_checkpoint": None,
    }
    values.update(overrides)
    return module.NeuralTrainingRequest(**values)


def test_logits_to_probabilities_is_stable_and_row_normalized() -> None:
    module = _training_module()
    logits = np.asarray([[1_000.0, 1_001.0], [-1_000.0, -999.0]])

    probabilities = module.logits_to_probabilities(logits)

    assert probabilities.shape == logits.shape
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(probabilities[0], probabilities[1])


@pytest.mark.parametrize(
    "logits",
    [
        np.asarray([1.0, 2.0]),
        np.asarray([[[1.0, 2.0]]]),
        np.asarray([[1.0, np.nan]]),
        np.asarray([[1.0, np.inf]]),
    ],
)
def test_logits_to_probabilities_rejects_non_matrix_or_nonfinite_input(
    logits: np.ndarray,
) -> None:
    module = _training_module()

    with pytest.raises(ValueError):
        module.logits_to_probabilities(logits)


def test_model_balanced_trainer_uses_weighted_sampler_only_for_training(
    tmp_path: Path,
) -> None:
    module = _training_module()

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

        def pad(self, features, **kwargs):
            return {
                key: torch.tensor([feature[key] for feature in features])
                for key in features[0]
            }

    tokenizer = Tokenizer()
    train_samples = (
        _sample(0, split="train", model_suffix=0),
        _sample(0, split="train", model_suffix=0),
        _sample(1, split="train", model_suffix=1),
    )
    train_dataset = NeuralAttributionDataset(
        train_samples,
        tokenizer,
        {label: index for index, label in enumerate(FAMILY_LABELS)},
        8,
    )
    validation_dataset = NeuralAttributionDataset(
        _samples("validation"),
        tokenizer,
        {label: index for index, label in enumerate(FAMILY_LABELS)},
        8,
    )
    trainer = module.ModelBalancedTrainer(
        model=BertForSequenceClassification(
            BertConfig(
                vocab_size=32,
                hidden_size=12,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                num_labels=10,
            )
        ),
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            per_device_train_batch_size=2,
            report_to=[],
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        train_samples=train_samples,
    )

    train_sampler = trainer._get_train_sampler()
    eval_dataloader = trainer.get_eval_dataloader()

    assert isinstance(train_sampler, torch.utils.data.WeightedRandomSampler)
    np.testing.assert_allclose(
        np.asarray(train_sampler.weights),
        np.asarray([0.5, 0.5, 1.0]),
    )
    assert not isinstance(
        eval_dataloader.sampler,
        torch.utils.data.WeightedRandomSampler,
    )


def test_model_balanced_trainer_fails_clearly_without_training_data(
    tmp_path: Path,
) -> None:
    module = _training_module()
    trainer = module.ModelBalancedTrainer(
        model=BertForSequenceClassification(
            BertConfig(
                vocab_size=32,
                hidden_size=12,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                num_labels=10,
            )
        ),
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[]),
    )

    with pytest.raises(ValueError, match="train dataset"):
        trainer.get_train_dataloader()


def test_train_neural_experiment_validates_request_before_loading_model(
    tmp_path: Path,
) -> None:
    module = _training_module()
    model_dir = tmp_path / "unused-model"
    request = _request(module, tmp_path, model_dir, train_samples=())

    with patch.object(module.AutoTokenizer, "from_pretrained") as loader:
        with pytest.raises(ValueError, match="train_samples"):
            module.train_neural_experiment(request)

    loader.assert_not_called()


def test_train_neural_experiment_requires_every_split_to_cover_labels(
    tmp_path: Path,
) -> None:
    module = _training_module()
    model_dir = tmp_path / "unused-model"
    request = _request(
        module,
        tmp_path,
        model_dir,
        validation_samples=_samples("validation")[:-1],
    )

    with pytest.raises(ValueError, match="validation_samples.*family"):
        module.train_neural_experiment(request)


def test_train_neural_experiment_passes_resume_checkpoint_explicitly(
    tmp_path: Path,
) -> None:
    module = _training_module()
    model_dir = tmp_path / "tiny-model"
    _write_tiny_bert(model_dir)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    request = _request(
        module,
        tmp_path,
        model_dir,
        resume_from_checkpoint=checkpoint,
    )
    predictions = SimpleNamespace(
        predictions=np.zeros((len(request.validation_samples), len(FAMILY_LABELS)))
    )

    with (
        patch.object(module.ModelBalancedTrainer, "train") as train_mock,
        patch.object(
            module.ModelBalancedTrainer,
            "predict",
            return_value=predictions,
        ),
    ):
        module.train_neural_experiment(request)

    train_mock.assert_called_once_with(resume_from_checkpoint=str(checkpoint))


def test_smoke_training_runs_offline_and_saves_reloadable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _training_module()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    model_dir = tmp_path / "tiny-model"
    _write_tiny_bert(model_dir)
    request = _request(module, tmp_path, model_dir)

    result = module.train_neural_experiment(request)

    assert result.validation_logits.shape == (10, 10)
    assert result.validation_probabilities.shape == (10, 10)
    np.testing.assert_allclose(
        result.validation_probabilities.sum(axis=1),
        np.ones(10),
        atol=1e-6,
    )
    assert result.model_dir == request.output_dir / "model"
    assert result.tokenizer_dir == request.output_dir / "tokenizer"
    assert result.trainer_history
    assert isinstance(result.trainer_history, tuple)
    assert all(isinstance(row, dict) for row in result.trainer_history)
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(
        result.model_dir,
        local_files_only=True,
    )
    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        result.tokenizer_dir,
        local_files_only=True,
    )
    assert reloaded_model.config.num_labels == len(FAMILY_LABELS)
    assert reloaded_tokenizer.pad_token_id is not None
