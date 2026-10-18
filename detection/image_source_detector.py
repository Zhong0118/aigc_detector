"""Local baseline using the existing eight-class image source classifier.

The portable JSON contains fitted coefficients, not a pickled sklearn object.
The binary risk score is the multiclass mass outside the Real class.
"""

import json
from pathlib import Path

import numpy as np


class ImageSourceDetector:
    def __init__(self, config: dict):
        self.config = config
        artifact = json.loads(Path(config["model_path"]).read_text(encoding="utf-8-sig"))
        if artifact.get("format") != "aigc-logistic-regression-v1":
            raise ValueError("Unsupported image classifier format")
        self.classes = list(artifact["classes"])
        self.coef = np.asarray(artifact["coef"], dtype=np.float64)
        self.intercept = np.asarray(artifact["intercept"], dtype=np.float64)
        self.metadata = dict(artifact["metadata"])
        if (len(self.classes) < 3 or artifact.get("solver") == "liblinear"
                or artifact.get("multi_class") == "ovr"
                or self.metadata.get("feature_type") != "clip"):
            raise ValueError("Expected a multinomial CLIP classifier")
        if (self.coef.ndim != 2 or self.coef.shape[0] != len(self.classes)
                or self.intercept.shape != (len(self.classes),)
                or len(set(self.classes)) != len(self.classes)
                or not np.isfinite(self.coef).all() or not np.isfinite(self.intercept).all()):
            raise ValueError("Invalid classifier dimensions or coefficients")
        self.real_index = self.classes.index("Real")
        self._extractor = None

    def predict_features(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.coef.shape[1],) or not np.isfinite(features).all():
            raise ValueError("Unexpected image feature dimensions or non-finite features")
        logits = self.coef @ features + self.intercept
        values = np.exp(logits - logits.max())
        return values / values.sum()

    def extract_features(self, path):
        from provenance.attribution import ProvenanceAttributionEngine

        if self._extractor is None:
            self._extractor = ProvenanceAttributionEngine()
        return self._extractor._extract_clip_image_features(Path(path), self.metadata, self.config)

    def detect(self, path):
        probabilities = self.predict_features(self.extract_features(path))
        real_probability = float(probabilities[self.real_index])
        score = max(0.0, min(1.0, 1.0 - real_probability))
        sources = [
            {"model": label, "probability": float(probabilities[index]) / score}
            for index, label in enumerate(self.classes)
            if label != "Real" and score > 0
        ]
        sources.sort(key=lambda item: item["probability"], reverse=True)
        top_k = sources[:max(1, int(self.config.get("top_k", 5)))]
        experimental = not bool(self.config.get("validated", False))
        return {
            "score": score,
            "score_kind": "closed_set_image_risk",
            "experimental": experimental,
            "model_scores": {} if experimental else {
                label: float(probabilities[index])
                for index, label in enumerate(self.classes) if label != "Real"
            },
            "class_scores": dict(zip(self.classes, probabilities.tolist())),
            "note": "实验模型仅覆盖训练集中的 8 个类别，分数尚需独立测试验证。",
            "attribution": {
                "status": "unvalidated" if experimental else ("ok" if top_k else "uncertain"),
                "source": "local_image_source_classifier",
                "top_k": [] if experimental else top_k,
                "confidence": top_k[0]["probability"] if top_k and not experimental else 0.0,
                "experimental_candidates": top_k if experimental else [],
                "score_kind": "source_given_ai",
                "features_used": self.metadata.get("features_used", []),
                "note": "模型尚未通过独立验证，暂不输出来源结论。" if experimental else "候选模型限于训练类别；来源分数以内容为 AI 生成为条件。",
            },
        }
