"""Reproducible classical baselines for family-level text attribution."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from research.text_attribution.style_features import StylometricFeatureExtractor


EXPERIMENT_IDS = ("B0", "B1", "B2", "B3")


def build_classical_pipeline(experiment_id: str, seed: int) -> Pipeline:
    """Build one deterministic, leakage-safe classical attribution baseline."""

    if experiment_id == "B0":
        return Pipeline([("char", _char_vectorizer()), ("classifier", _logistic_regression(seed))])
    if experiment_id == "B1":
        return Pipeline(
            [
                ("style", _style_pipeline()),
                ("classifier", _logistic_regression(seed)),
            ]
        )
    if experiment_id == "B2":
        return Pipeline(
            [
                ("features", FeatureUnion([("char", _char_vectorizer()), ("style", _style_pipeline())])),
                ("classifier", _logistic_regression(seed)),
            ]
        )
    if experiment_id == "B3":
        return Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", _word_vectorizer()),
                            ("char", _char_vectorizer()),
                        ]
                    ),
                ),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(class_weight="balanced", random_state=seed),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported classical experiment: {experiment_id!r}")


def predict_family_probabilities(
    model: object, texts: Sequence[str] | np.ndarray, family_labels: Sequence[str]
) -> np.ndarray:
    """Return finite, normalized probabilities in the caller's fixed family order."""

    labels = _validated_family_labels(family_labels)
    values = _validated_texts(texts)
    try:
        classes = tuple(model.classes_)  # type: ignore[attr-defined]
    except AttributeError as error:
        raise ValueError("model must expose classes_ and predict_proba") from error

    if len(classes) != len(set(classes)) or set(classes) != set(labels):
        raise ValueError("family_labels must uniquely match model classes")
    try:
        probabilities = (
            np.asarray(model.predict_proba(values), dtype=np.float64)  # type: ignore[attr-defined]
            if values
            else np.empty((0, len(classes)))
        )
    except AttributeError as error:
        raise ValueError("model must expose classes_ and predict_proba") from error
    if probabilities.ndim != 2 or probabilities.shape != (len(values), len(classes)):
        raise ValueError("model predict_proba must return shape (texts, classes)")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("model probabilities must be finite and within [0, 1]")

    column_indexes = [classes.index(label) for label in labels]
    aligned = probabilities[:, column_indexes]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("model probability rows must have positive sums")
    return np.divide(aligned, row_sums, out=np.empty_like(aligned), where=row_sums != 0.0)


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=250_000,
        dtype=np.float32,
    )


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=150_000,
        dtype=np.float32,
    )


def _style_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("extract", StylometricFeatureExtractor(include_length_features=True)),
            ("scale", StandardScaler()),
        ]
    )


def _logistic_regression(seed: int) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
        solver="liblinear",
    )


def _validated_family_labels(family_labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(family_labels, (str, bytes)):
        raise ValueError("family_labels must be a nonempty sequence")
    labels = tuple(family_labels)
    if not labels:
        raise ValueError("family_labels must be nonempty")
    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError("family_labels must contain only nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("family_labels must be unique")
    return labels


def _validated_texts(texts: Sequence[str] | np.ndarray) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, (list, tuple, np.ndarray)):
        raise ValueError("texts must be a one-dimensional sequence of strings")
    if isinstance(texts, np.ndarray):
        if texts.ndim != 1:
            raise ValueError("texts must be a one-dimensional sequence of strings")
        values = texts.tolist()
    else:
        values = list(texts)
    if not all(isinstance(value, str) for value in values):
        raise TypeError("texts must contain only strings")
    return values
