"""Export a trusted, fitted sklearn image classifier as portable numeric data.

Use the original model's Python environment when available. This command
compares the exported softmax with that environment's predict_proba output.
It does not train a classifier or establish its accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import sklearn


def export_classifier(source: Path, destination: Path, clip_revision: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = joblib.load(source)
    model = artifact["model"]
    classes = model.classes_.tolist()
    if ("Real" not in classes or len(classes) < 3 or model.solver == "liblinear"
            or getattr(model, "multi_class", None) == "ovr"
            or artifact.get("feature_type") != "clip"):
        raise ValueError("Expected a multinomial CLIP image classifier with a Real class")
    rng = np.random.default_rng(20261015)
    features = rng.normal(size=(4, int(model.n_features_in_)))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    reference = model.predict_proba(features)
    logits = features @ model.coef_.T + model.intercept_
    portable = np.exp(logits - logits.max(axis=1, keepdims=True))
    portable /= portable.sum(axis=1, keepdims=True)
    error = float(np.max(np.abs(portable - reference)))
    if error > 1e-12:
        raise ValueError(f"Export does not match the source estimator: {error}")
    fields = ["dataset", "feature_type", "clip_model_name", "cache_dir",
              "features_used", "label_count", "sample_count", "classes"]
    metadata = {key: artifact.get(key) for key in fields}
    metadata["clip_revision"] = clip_revision
    original_version = next((getattr(w.message, "original_sklearn_version")
                             for w in caught if hasattr(w.message, "original_sklearn_version")),
                            sklearn.__version__)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    exported = {
        "format": "aigc-logistic-regression-v1", "metadata": metadata,
        "classes": classes, "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(), "solver": model.solver,
        "multi_class": getattr(model, "multi_class", None),
        "serialized_sklearn_version": original_version,
        "export_sklearn_version": sklearn.__version__,
        "source_artifact_sha256": source_hash,
        "verification": {"features": features.tolist(), "probabilities": reference.tolist()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"max_probability_difference": error, "source_artifact_sha256": source_hash}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-revision", required=True)
    args = parser.parse_args()
    print(json.dumps(export_classifier(args.input, args.output, args.clip_revision)))
