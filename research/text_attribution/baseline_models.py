"""Model evaluation primitives for text-generation attribution baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .baseline_data import BaselineSample


_POOLING_VERSION = "attention-mask-mean-l2-v1"


@dataclass(frozen=True)
class PredictionRow:
    request_key: str
    method: str
    fold: int
    true_label: str
    predicted_label: str
    prompt_id: str
    semantic_prompt_group: str
    language: str
    temperature: float
    target_length: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    labels: list[str]
    predictions: list[PredictionRow]
    fold_metrics: list[dict[str, float | int]]
    oof_metrics: dict[str, float]
    confusion_matrix: list[list[float]]
    per_class: dict[str, dict[str, float | int]]


def build_sparse_pipeline(method: str, *, seed: int) -> Pipeline:
    """Build one of the two pre-registered sparse-text baselines."""

    if method == "char_lr":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=120_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if method == "tfidf_svm":
        return Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "word",
                                TfidfVectorizer(
                                    analyzer="word",
                                    ngram_range=(1, 2),
                                    min_df=2,
                                    max_features=80_000,
                                    sublinear_tf=True,
                                ),
                            ),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=2,
                                    max_features=120_000,
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported sparse method: {method}")


def evaluate_sparse_method(
    samples: Sequence[BaselineSample],
    folds: Sequence[int],
    *,
    method: str,
    seed: int = 42,
) -> EvaluationResult:
    """Fit fold-local sparse models and return out-of-fold predictions."""

    texts = [sample.text for sample in samples]
    labels = [sample.label for sample in samples]

    def predict(train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        pipeline = build_sparse_pipeline(method, seed=seed)
        pipeline.fit(
            [texts[index] for index in train_indices],
            [labels[index] for index in train_indices],
        )
        return np.asarray(
            pipeline.predict([texts[index] for index in test_indices]),
            dtype=str,
        )

    return _evaluate_fold_predictions(
        samples,
        folds,
        method=method,
        predict_fold=predict,
    )


def embedding_cache_fingerprint(
    samples: Sequence[BaselineSample],
    *,
    model_id: str,
    max_length: int,
    pooling_version: str = _POOLING_VERSION,
) -> str:
    """Hash all inputs that affect the ordered frozen-embedding matrix."""

    digest = hashlib.sha256()
    settings = {
        "model_id": model_id,
        "max_length": int(max_length),
        "pooling_version": pooling_version,
        "prefix": "passage: ",
    }
    digest.update(
        json.dumps(settings, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    for sample in samples:
        digest.update(b"\x1e")
        digest.update(sample.request_key.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(sample.text.encode("utf-8"))
    return digest.hexdigest()


def save_embedding_cache(
    path: str | Path,
    embeddings: np.ndarray,
    samples: Sequence[BaselineSample],
    *,
    model_id: str,
    max_length: int,
) -> Path:
    """Persist ordered embeddings plus enough metadata to reject stale caches."""

    cache_path = Path(path)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(samples):
        raise ValueError(
            "Embedding matrix must be two-dimensional with one row per sample"
        )
    metadata = {
        "fingerprint": embedding_cache_fingerprint(
            samples,
            model_id=model_id,
            max_length=max_length,
        ),
        "model_id": model_id,
        "max_length": int(max_length),
        "pooling_version": _POOLING_VERSION,
        "rows": len(samples),
        "dimensions": int(matrix.shape[1]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=matrix,
        request_keys=np.asarray(
            [sample.request_key for sample in samples],
            dtype=str,
        ),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return cache_path


def load_embedding_cache(
    path: str | Path,
    samples: Sequence[BaselineSample],
    *,
    model_id: str,
    max_length: int,
) -> np.ndarray:
    """Load a cache only when its inputs and row order match exactly."""

    cache_path = Path(path)
    expected_fingerprint = embedding_cache_fingerprint(
        samples,
        model_id=model_id,
        max_length=max_length,
    )
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        request_keys = payload["request_keys"].astype(str).tolist()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)

    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Embedding cache fingerprint mismatch for {cache_path}; "
            "the data, model, or max length changed"
        )
    expected_keys = [sample.request_key for sample in samples]
    if request_keys != expected_keys:
        raise ValueError(f"Embedding cache row order mismatch for {cache_path}")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(samples):
        raise ValueError(f"Embedding cache has an invalid matrix shape: {cache_path}")
    return embeddings


def load_or_extract_embeddings(
    samples: Sequence[BaselineSample],
    *,
    cache_path: str | Path,
    model_id: str,
    device: str,
    batch_size: int,
    max_length: int,
    hf_cache: str | Path | None,
) -> tuple[np.ndarray, bool]:
    """Reuse a valid cache, or extract and persist embeddings once."""

    path = Path(cache_path)
    if path.is_file():
        return (
            load_embedding_cache(
                path,
                samples,
                model_id=model_id,
                max_length=max_length,
            ),
            True,
        )
    embeddings = extract_frozen_embeddings(
        samples,
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        hf_cache=hf_cache,
    )
    save_embedding_cache(
        path,
        embeddings,
        samples,
        model_id=model_id,
        max_length=max_length,
    )
    return embeddings, False


def extract_frozen_embeddings(
    samples: Sequence[BaselineSample],
    *,
    model_id: str,
    device: str = "cuda:0",
    batch_size: int = 16,
    max_length: int = 512,
    hf_cache: str | Path | None = None,
) -> np.ndarray:
    """Encode texts with a frozen Transformer using masked mean pooling."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    import torch
    from transformers import AutoModel, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    cache_dir = str(hf_cache) if hf_cache is not None else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    model = model.to(device).eval()
    vectors: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                encoded = tokenizer(
                    [f"passage: {sample.text}" for sample in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(normalized.float().cpu().numpy())
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not vectors:
        raise ValueError("Cannot extract embeddings from an empty sample set")
    return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


def evaluate_embedding_probe(
    samples: Sequence[BaselineSample],
    folds: Sequence[int],
    embeddings: np.ndarray,
    *,
    seed: int = 42,
) -> EvaluationResult:
    """Evaluate a fold-local linear classifier on fixed dense embeddings."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(samples):
        raise ValueError(
            "Embedding matrix must be two-dimensional with one row per sample"
        )
    labels = np.asarray([sample.label for sample in samples], dtype=str)

    def predict(train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )
        classifier.fit(matrix[train_indices], labels[train_indices])
        return np.asarray(classifier.predict(matrix[test_indices]), dtype=str)

    return _evaluate_fold_predictions(
        samples,
        folds,
        method="frozen_e5",
        predict_fold=predict,
    )


def _evaluate_fold_predictions(
    samples: Sequence[BaselineSample],
    folds: Sequence[int],
    *,
    method: str,
    predict_fold: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> EvaluationResult:
    if len(samples) != len(folds):
        raise ValueError(
            f"samples/folds length mismatch: {len(samples)} != {len(folds)}"
        )
    if not samples:
        raise ValueError("At least one sample is required")

    fold_array = np.asarray(folds, dtype=int)
    unique_folds = sorted(int(value) for value in np.unique(fold_array))
    if len(unique_folds) < 2:
        raise ValueError("At least two folds are required")

    true_labels = np.asarray([sample.label for sample in samples], dtype=str)
    predicted: list[str | None] = [None] * len(samples)
    prediction_folds: list[int | None] = [None] * len(samples)
    fold_metrics: list[dict[str, float | int]] = []

    for fold in unique_folds:
        test_indices = np.flatnonzero(fold_array == fold)
        train_indices = np.flatnonzero(fold_array != fold)
        if not len(test_indices) or not len(train_indices):
            raise ValueError(f"Fold {fold} has an empty train or test partition")
        train_classes = set(true_labels[train_indices])
        if len(train_classes) < 2:
            raise ValueError(f"Fold {fold} training data has fewer than two labels")

        fold_predictions = np.asarray(
            predict_fold(train_indices, test_indices),
            dtype=str,
        )
        if len(fold_predictions) != len(test_indices):
            raise ValueError(
                f"Fold {fold} produced {len(fold_predictions)} predictions for "
                f"{len(test_indices)} samples"
            )

        for index, label in zip(test_indices, fold_predictions, strict=True):
            predicted[int(index)] = str(label)
            prediction_folds[int(index)] = fold

        fold_metrics.append(
            {
                "fold": fold,
                "samples": int(len(test_indices)),
                **_metric_triplet(true_labels[test_indices], fold_predictions),
            }
        )

    if any(label is None for label in predicted):
        raise RuntimeError("Out-of-fold evaluation left samples without predictions")

    final_predictions = np.asarray(predicted, dtype=str)
    label_order = sorted(set(true_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        final_predictions,
        labels=label_order,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(label_order)
    }
    normalized_confusion = confusion_matrix(
        true_labels,
        final_predictions,
        labels=label_order,
        normalize="true",
    )
    rows = [
        PredictionRow(
            request_key=sample.request_key,
            method=method,
            fold=int(prediction_folds[index]),
            true_label=sample.label,
            predicted_label=str(final_predictions[index]),
            prompt_id=sample.prompt_id,
            semantic_prompt_group=sample.semantic_prompt_group,
            language=sample.language,
            temperature=sample.temperature,
            target_length=sample.target_length,
        )
        for index, sample in enumerate(samples)
    ]
    return EvaluationResult(
        method=method,
        labels=label_order,
        predictions=rows,
        fold_metrics=fold_metrics,
        oof_metrics=_metric_triplet(true_labels, final_predictions),
        confusion_matrix=normalized_confusion.tolist(),
        per_class=per_class,
    )


def _metric_triplet(
    true_labels: Sequence[str] | np.ndarray,
    predicted_labels: Sequence[str] | np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
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
