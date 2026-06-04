"""模型来源归因模块。

这个文件尝试回答：“如果内容是 AI 生成的，它可能来自哪个模型/模型家族？”

当前包含两层能力：
- 可选真实适配器：LLMDet 文本来源候选归因、UniversalAttribution 图片来源候选归因
- 研究 scaffold：未训练 MLP，保留给后续自训练归因模型

注意：
- 归因输出是“候选来源 top-k”，不是确定来源证明
- 只有 C2PA、水印、平台日志等强证据才能更接近确定来源
- 外部研究包可能依赖大模型/权重，失败时必须返回明确状态，不能拖垮主流程

MLP scaffold 仍然只是未训练模型：
- 没有训练权重时不能作为真实归因依据
- 输入 features 是简单 dict
- 输出 KNOWN_MODELS 上的 Top-K 概率

后续建议：
- 使用真实检测 API 的 generator scores 作为主来源
- 训练多模态 attribution 模型
- 按 text/image/audio/video 分别维护候选模型列表
- 保存训练数据版本和模型版本
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import importlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, List, Dict
import zipfile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class ProvenanceAttributionEngine:
    """多模态候选来源归因入口。

    这里不把研究模型硬编码进主流程，而是按 config 选择 provider。
    当前支持：
    - text: LLMDet，输出 Human/GPT-2/OPT/UniLM/LLaMA/BART/T5/Bloom/GPT-neo 等候选概率
    - image: UniversalAttribution 风格命令行，要求外部脚本输出 JSON
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """读取来源归因配置。"""
        self.config = config or {}
        self._llmdet_probability_loaded = False
        self._clip_cache: dict[str, Any] = {}

    def attribute(self, path: str | Path, modality: str) -> dict[str, Any]:
        """按模态执行候选来源归因。"""
        if not self.config.get("enabled", False):
            return self._disabled("attribution")
        path = Path(path)
        if modality == "text":
            return self._attribute_text(path)
        if modality == "image":
            return self._attribute_image(path)
        if modality == "video":
            return self._attribute_video(path)
        return {
            "status": "not_applicable",
            "source": "local_attribution",
            "provider": "none",
            "top_k": [],
            "confidence": 0.0,
            "note": f"Attribution is not configured for {modality}.",
        }

    def _attribute_text(self, path: Path) -> dict[str, Any]:
        """融合多个文本候选来源归因分支。"""
        cfg = self.config.get("text", {})
        if not cfg.get("enabled", False):
            return self._disabled("text_attribution")

        branches: list[dict[str, Any]] = []
        providers = self._provider_configs(cfg, default_provider="llmdet")
        for name, provider_cfg in providers:
            provider = str(provider_cfg.get("provider", name)).lower()
            if provider == "llmdet":
                branches.append(self._attribute_text_llmdet(path, provider_cfg))
            elif provider in {"trained_classifier", "trained_text_classifier"}:
                branches.append(self._attribute_text_trained_classifier(path, provider_cfg))
            elif provider == "openturingbench":
                try:
                    branches.append(
                        self._attribute_text_external_or_prototype(
                            path,
                            provider_cfg,
                            provider="openturingbench",
                            default_note=(
                                "OpenTuringBench is exposed here as a dataset/prototype or external-command branch; "
                                "configure prototypes_path or command before it can emit candidates."
                            ),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    branches.append(self._error("openturingbench", exc))
            elif provider in {"embedding_prototype", "text_embedding_prototype"}:
                branches.append(self._attribute_text_prototype(path, provider_cfg, provider="embedding_prototype"))
            else:
                branches.append(self._not_configured(provider, "Unsupported text attribution provider."))

        return self._fuse_branch_results(branches, modality="text", top_k=int(cfg.get("top_k", 5)))

    def _attribute_text_llmdet(self, path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        """调用 LLMDet 做文本候选来源归因。"""
        if not cfg.get("enabled", True):
            return self._disabled("llmdet")

        try:
            self._configure_huggingface_cache(cfg)
            if cfg.get("shim_unilm", True):
                self._install_unilm_compat_shim()
            llmdet = self._import_module("llmdet")
            if cfg.get("load_probability", True) and not self._llmdet_probability_loaded:
                preflight = self._preflight_llmdet_data(cfg)
                if preflight is not None:
                    return preflight
                llmdet.load_probability()
                self._llmdet_probability_loaded = True
            text = path.read_text(encoding=cfg.get("encoding", "utf-8"), errors="ignore")
            if len(text.strip()) < int(cfg.get("min_chars", 30)):
                return {
                    "status": "skipped",
                    "source": "text_attribution",
                    "provider": "llmdet",
                    "top_k": [],
                    "confidence": 0.0,
                    "note": "Text is too short for reliable LLMDet attribution.",
                }
            timeout = float(cfg.get("timeout_seconds", 45))
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llmdet.detect, text)
                raw = future.result(timeout=timeout)
            scores = raw[0] if isinstance(raw, list) and raw else raw
            top_k = self._normalize_score_map(scores, top_k=int(cfg.get("top_k", 5)))
            return {
                "status": "ok",
                "source": "llmdet",
                "provider": "llmdet",
                "top_k": top_k,
                "confidence": top_k[0]["probability"] if top_k else 0.0,
                "features_used": ["proxy_perplexity", "ngram_probability", "lightgbm_classifier"],
                "note": "LLMDet returns candidate source probabilities; it is not a definitive source proof.",
            }
        except ImportError as exc:
            return {
                "status": "dependency_missing",
                "source": "llmdet",
                "provider": "llmdet",
                "top_k": [],
                "confidence": 0.0,
                "error": str(exc),
                "install_hint": (
                    "LLMDet was installed but may require extra research dependencies. "
                    "On Windows install with PYTHONUTF8=1 and allow deprecated sklearn package; "
                    "the PyPI package also imports an unavailable `unilm` module."
                ),
            }
        except TimeoutError:
            return self._error("llmdet", RuntimeError("LLMDet attribution timed out; first-run downloads may be too slow."))
        except Exception as exc:  # noqa: BLE001 - provenance failure should not break detection.
            return self._error("llmdet", exc)

    def _attribute_image(self, path: Path) -> dict[str, Any]:
        """融合多个图片候选来源归因分支。"""
        cfg = self.config.get("image", {})
        if not cfg.get("enabled", False):
            return self._disabled("image_attribution")

        branches: list[dict[str, Any]] = []
        providers = self._provider_configs(cfg, default_provider="universal_attribution")
        for name, provider_cfg in providers:
            provider = str(provider_cfg.get("provider", name)).lower()
            if provider == "universal_attribution":
                branches.append(self._attribute_image_external(path, provider_cfg, provider="universal_attribution"))
            elif provider in {"ofaattribution", "ofa_attribution"}:
                branches.append(self._attribute_image_external(path, provider_cfg, provider="ofa_attribution"))
            elif provider in {"trained_classifier", "trained_image_classifier"}:
                branches.append(self._attribute_image_trained_classifier(path, provider_cfg))
            elif provider in {"clip_prototype", "image_embedding_prototype"}:
                branches.append(self._attribute_image_prototype(path, provider_cfg, provider="clip_prototype"))
            else:
                branches.append(self._not_configured(provider, "Unsupported image attribution provider."))

        return self._fuse_branch_results(branches, modality="image", top_k=int(cfg.get("top_k", 5)))

    def _attribute_image_external(self, path: Path, cfg: dict[str, Any], provider: str) -> dict[str, Any]:
        """调用 UniversalAttribution/OFAAttribution 风格命令行做图片候选来源归因。

        UniversalAttribution 不是标准 pip 包；不同复现仓库入口可能不同。
        因此这里采用 CLI JSON 边界：
        - command 指向脚本或可执行文件
        - 命令需要接收图片路径并输出 JSON：top_k/confidence/unknown_probability
        """
        if not cfg.get("enabled", True):
            return self._disabled(provider)

        command = cfg.get("command")
        if not command:
            return {
                "status": "not_configured",
                "source": provider,
                "provider": provider,
                "top_k": [],
                "confidence": 0.0,
                "install_hint": (
                    f"Configure provenance.attribution.image.providers.{provider}.command "
                    "to a JSON-output inference script."
                ),
            }

        try:
            args = [str(command), str(path)]
            if cfg.get("json_arg", True):
                args.append("--json")
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=float(cfg.get("timeout_seconds", 120)),
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            payload = json.loads(completed.stdout)
            top_k = self._normalize_top_k(payload.get("top_k") or payload.get("predictions") or [])
            confidence = payload.get("confidence")
            if confidence is None:
                confidence = top_k[0]["probability"] if top_k else 0.0
            return {
                "status": "ok",
                "source": provider,
                "provider": provider,
                "top_k": top_k,
                "confidence": float(confidence),
                "unknown_probability": payload.get("unknown_probability"),
                "note": "Image attribution returns candidate generator probabilities, not definitive proof.",
            }
        except Exception as exc:  # noqa: BLE001
            return self._error(provider, exc)

    def _attribute_text_trained_classifier(self, path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        """加载自训练文本来源分类器并输出候选来源 Top-K。"""
        if not cfg.get("enabled", True):
            return self._disabled("trained_classifier")
        model_path = Path(str(cfg.get("model_path", "")))
        if not model_path.exists():
            return {
                "status": "not_configured",
                "source": "trained_classifier",
                "provider": "trained_classifier",
                "top_k": [],
                "confidence": 0.0,
                "install_hint": f"Train text attribution classifier first: missing {model_path}.",
            }
        try:
            artifact = self._load_model_artifact(model_path)
            model = artifact.get("model") or artifact.get("pipeline") or artifact.get("classifier")
            if model is None:
                raise RuntimeError("Text classifier artifact must contain `model`, `pipeline`, or `classifier`.")
            text = path.read_text(encoding=cfg.get("encoding", "utf-8"), errors="ignore")
            top_k = self._predict_top_k(model, [text], top_k=int(cfg.get("top_k", 5)))
            return {
                "status": "ok",
                "source": "trained_classifier",
                "provider": "trained_classifier",
                "top_k": top_k,
                "confidence": top_k[0]["probability"] if top_k else 0.0,
                "features_used": artifact.get("features_used", ["text_pipeline"]),
                "model_path": str(model_path),
                "note": "Self-trained text source classifier result.",
            }
        except Exception as exc:  # noqa: BLE001
            return self._error("trained_classifier", exc)

    def _attribute_video(self, path: Path) -> dict[str, Any]:
        """视频归因：先抽帧，再复用图片归因并做帧级融合。

        SAGA 这类视频专用来源归因模型目前更偏研究复现；这里先实现一个
        可落地的 frame-level aggregation，等外部视频模型脚本准备好后再接入。
        """
        cfg = self.config.get("video", {})
        if not cfg.get("enabled", False):
            return self._disabled("video_attribution")

        providers = self._provider_configs(cfg, default_provider="frame_image_fusion")
        branches: list[dict[str, Any]] = []
        for name, provider_cfg in providers:
            provider = str(provider_cfg.get("provider", name)).lower()
            if provider in {"frame_image_fusion", "video_frame_fusion"}:
                branches.append(self._attribute_video_frames(path, provider_cfg))
            elif provider == "saga":
                branches.append(self._attribute_video_external(path, provider_cfg, provider="saga"))
            else:
                branches.append(self._not_configured(provider, "Unsupported video attribution provider."))
        return self._fuse_branch_results(branches, modality="video", top_k=int(cfg.get("top_k", 5)))

    def _attribute_video_frames(self, path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        """抽取视频帧，并把每帧送入图片归因分支。"""
        if not cfg.get("enabled", True):
            return self._disabled("frame_image_fusion")

        try:
            import cv2
        except ImportError as exc:
            return {
                "status": "dependency_missing",
                "source": "frame_image_fusion",
                "provider": "frame_image_fusion",
                "top_k": [],
                "confidence": 0.0,
                "error": str(exc),
                "install_hint": "Install opencv-python to enable video frame attribution.",
            }

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return self._error("frame_image_fusion", RuntimeError("Cannot open video for frame attribution."))

        max_frames = int(cfg.get("max_frames", 8))
        stride = max(1, int(cfg.get("frame_interval", 30)))
        frame_results: list[dict[str, Any]] = []
        frame_index = 0
        sampled = 0
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            while sampled < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    frame_path = Path(tmp_dir) / f"frame_{frame_index}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_result = self._attribute_image(frame_path)
                    frame_result["frame_index"] = frame_index
                    frame_results.append(frame_result)
                    sampled += 1
                frame_index += 1
        capture.release()

        if not frame_results:
            return {
                "status": "skipped",
                "source": "frame_image_fusion",
                "provider": "frame_image_fusion",
                "top_k": [],
                "confidence": 0.0,
                "note": "No frames were extracted for attribution.",
            }

        fused = self._fuse_branch_results(frame_results, modality="video_frame", top_k=int(cfg.get("top_k", 5)))
        fused.update(
            {
                "source": "frame_image_fusion",
                "provider": "frame_image_fusion",
                "frame_count": len(frame_results),
                "branches": frame_results,
                "note": "Video attribution is aggregated from sampled image-frame attribution results.",
            }
        )
        return fused

    def _attribute_video_external(self, path: Path, cfg: dict[str, Any], provider: str) -> dict[str, Any]:
        """调用视频来源归因外部脚本，例如未来复现 SAGA 后的推理入口。"""
        if not cfg.get("enabled", True):
            return self._disabled(provider)
        command = cfg.get("command")
        if not command:
            return {
                "status": "not_configured",
                "source": provider,
                "provider": provider,
                "top_k": [],
                "confidence": 0.0,
                "install_hint": (
                    "SAGA currently requires a separate research implementation/training setup. "
                    "Set command to a JSON-output inference script after preparing weights."
                ),
            }
        try:
            return self._run_external_json_command(path, cfg, provider=provider)
        except Exception as exc:  # noqa: BLE001
            return self._error(provider, exc)

    def _attribute_text_external_or_prototype(
        self,
        path: Path,
        cfg: dict[str, Any],
        provider: str,
        default_note: str,
    ) -> dict[str, Any]:
        """文本研究分支：优先外部命令，否则走 prototype 文件。"""
        if not cfg.get("enabled", True):
            return self._disabled(provider)
        if cfg.get("command"):
            try:
                return self._run_external_json_command(path, cfg, provider=provider)
            except Exception as exc:  # noqa: BLE001
                return self._error(provider, exc)
        if cfg.get("prototypes_path"):
            result = self._attribute_text_prototype(path, cfg, provider=provider)
            if result["status"] != "not_configured":
                return result
        return {
            "status": "not_configured",
            "source": provider,
            "provider": provider,
            "top_k": [],
            "confidence": 0.0,
            "note": default_note,
            "install_hint": (
                "Build a local prototype file from the OpenTuringBench dataset or set command "
                "to an OTBDetector-compatible JSON inference script."
            ),
        }

    def _attribute_text_prototype(self, path: Path, cfg: dict[str, Any], provider: str) -> dict[str, Any]:
        """基于本地文本样本库做 TF-IDF/风格相似度候选来源检索。"""
        if not cfg.get("enabled", True):
            return self._disabled(provider)
        prototypes_path = Path(str(cfg.get("prototypes_path", "")))
        if not prototypes_path.exists():
            return {
                "status": "not_configured",
                "source": provider,
                "provider": provider,
                "top_k": [],
                "confidence": 0.0,
                "install_hint": f"Create text prototype file at {prototypes_path}.",
            }

        samples = self._load_text_prototypes(prototypes_path)
        if not samples:
            return self._error(provider, RuntimeError(f"No usable text prototypes found in {prototypes_path}."))

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            return {
                "status": "dependency_missing",
                "source": provider,
                "provider": provider,
                "top_k": [],
                "confidence": 0.0,
                "error": str(exc),
                "install_hint": "Install scikit-learn to enable text embedding prototype attribution.",
            }

        query = path.read_text(encoding=cfg.get("encoding", "utf-8"), errors="ignore")
        if len(query.strip()) < int(cfg.get("min_chars", 30)):
            return {
                "status": "skipped",
                "source": provider,
                "provider": provider,
                "top_k": [],
                "confidence": 0.0,
                "note": "Text is too short for reliable prototype attribution.",
            }

        texts = [sample["text"] for sample in samples] + [query]
        vectorizer = TfidfVectorizer(
            analyzer=str(cfg.get("analyzer", "char_wb")),
            ngram_range=tuple(cfg.get("ngram_range", [3, 5])),
            max_features=int(cfg.get("max_features", 50000)),
        )
        matrix = vectorizer.fit_transform(texts)
        sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        per_model: dict[str, list[float]] = defaultdict(list)
        for sample, score in zip(samples, sims):
            per_model[sample["model"]].append(float(score))
        score_map: dict[str, float] = {}
        top_n = int(cfg.get("aggregate_top_n", 5))
        for model, values in per_model.items():
            best = sorted(values, reverse=True)[:top_n]
            score_map[model] = float(np.mean(best)) if best else 0.0

        top_k = self._normalize_probability_scores(score_map, top_k=int(cfg.get("top_k", 5)))
        return {
            "status": "ok",
            "source": provider,
            "provider": provider,
            "top_k": top_k,
            "confidence": top_k[0]["probability"] if top_k else 0.0,
            "features_used": ["char_tfidf", "prototype_similarity"],
            "prototype_count": len(samples),
            "note": "Prototype attribution compares the query with a local labeled sample library.",
        }

    def _attribute_image_prototype(self, path: Path, cfg: dict[str, Any], provider: str) -> dict[str, Any]:
        """基于本地图片样本库做候选来源检索。

        当前默认使用 pHash，相当于轻量指纹/风格原型库；如果后续要用 CLIP，
        可以把 command 指到外部 CLIP/DINO prototype 推理脚本并输出统一 JSON。
        """
        if not cfg.get("enabled", True):
            return self._disabled(provider)
        if cfg.get("command"):
            return self._run_external_json_command(path, cfg, provider=provider)

        prototypes_path = Path(str(cfg.get("prototypes_path", "")))
        if not prototypes_path.exists():
            return {
                "status": "not_configured",
                "source": provider,
                "provider": provider,
                "top_k": [],
                "confidence": 0.0,
                "install_hint": f"Create image prototype file at {prototypes_path}.",
            }

        try:
            import imagehash
        except ImportError as exc:
            return {
                "status": "dependency_missing",
                "source": provider,
                "provider": provider,
                "top_k": [],
                "confidence": 0.0,
                "error": str(exc),
                "install_hint": "Install imagehash to enable image prototype attribution.",
            }

        prototypes = self._load_image_prototypes(prototypes_path)
        if not prototypes:
            return self._error(provider, RuntimeError(f"No usable image prototypes found in {prototypes_path}."))

        query_hash = imagehash.phash(Image.open(path).convert("RGB"))
        hash_bits = int(cfg.get("hash_bits", 64))
        per_model: dict[str, list[float]] = defaultdict(list)
        for sample in prototypes:
            try:
                proto_hash = imagehash.hex_to_hash(sample["phash"])
            except Exception:  # noqa: BLE001 - 跳过损坏的原型记录。
                continue
            distance = query_hash - proto_hash
            similarity = max(0.0, 1.0 - (float(distance) / float(hash_bits)))
            per_model[sample["model"]].append(similarity)

        score_map: dict[str, float] = {}
        top_n = int(cfg.get("aggregate_top_n", 5))
        for model, values in per_model.items():
            best = sorted(values, reverse=True)[:top_n]
            score_map[model] = float(np.mean(best)) if best else 0.0

        top_k = self._normalize_probability_scores(score_map, top_k=int(cfg.get("top_k", 5)))
        return {
            "status": "ok",
            "source": provider,
            "provider": provider,
            "top_k": top_k,
            "confidence": top_k[0]["probability"] if top_k else 0.0,
            "features_used": ["phash", "prototype_similarity"],
            "prototype_count": len(prototypes),
            "note": "Image prototype attribution uses a local labeled sample library; it is not a trained source model.",
        }

    def _attribute_image_trained_classifier(self, path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        """加载自训练图片来源分类器并输出候选来源 Top-K。"""
        if not cfg.get("enabled", True):
            return self._disabled("trained_classifier")
        model_path = Path(str(cfg.get("model_path", "")))
        if not model_path.exists():
            return {
                "status": "not_configured",
                "source": "trained_classifier",
                "provider": "trained_classifier",
                "top_k": [],
                "confidence": 0.0,
                "install_hint": f"Train image attribution classifier first: missing {model_path}.",
            }
        try:
            artifact = self._load_model_artifact(model_path)
            model = artifact.get("model") or artifact.get("classifier")
            if model is None:
                raise RuntimeError("Image classifier artifact must contain `model` or `classifier`.")
            feature_type = str(artifact.get("feature_type") or cfg.get("feature_type") or "basic").lower()
            if feature_type == "clip":
                features = self._extract_clip_image_features(path, artifact, cfg)
            else:
                features = self._extract_image_features(path)
            top_k = self._predict_top_k(model, [features], top_k=int(cfg.get("top_k", 5)))
            return {
                "status": "ok",
                "source": "trained_classifier",
                "provider": "trained_classifier",
                "top_k": top_k,
                "confidence": top_k[0]["probability"] if top_k else 0.0,
                "features_used": artifact.get("features_used", [feature_type]),
                "model_path": str(model_path),
                "note": "Self-trained image source classifier result.",
            }
        except Exception as exc:  # noqa: BLE001
            return self._error("trained_classifier", exc)

    def _run_external_json_command(self, path: Path, cfg: dict[str, Any], provider: str) -> dict[str, Any]:
        """运行统一 JSON 外部命令。"""
        command = cfg.get("command")
        if not command:
            return self._not_configured(provider, "External attribution command is not configured.")
        args = [str(command), str(path)]
        if cfg.get("json_arg", True):
            args.append("--json")
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=float(cfg.get("timeout_seconds", 120)),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        payload = json.loads(completed.stdout)
        top_k = self._normalize_top_k(payload.get("top_k") or payload.get("predictions") or [])
        confidence = payload.get("confidence")
        if confidence is None:
            confidence = top_k[0]["probability"] if top_k else 0.0
        return {
            "status": "ok",
            "source": provider,
            "provider": provider,
            "top_k": top_k,
            "confidence": float(confidence),
            "raw": payload if bool(cfg.get("keep_raw", False)) else None,
        }

    def _import_module(self, name: str) -> Any:
        """导入外部研究包，单独封装便于测试 mock。"""
        return importlib.import_module(name)

    def _install_unilm_compat_shim(self) -> None:
        """为 LLMDet 的老式 `from unilm import UniLMTokenizer` 提供兼容层。

        LLMDet 的 PyPI 包没有声明可安装的 unilm 依赖，而 transformers
        可以直接加载 `microsoft/unilm-base-cased` 的 tokenizer。
        这里在导入 llmdet 前临时注入一个最小模块，避免导入阶段直接失败。
        """
        if "unilm" in sys.modules:
            return
        try:
            from transformers import AutoTokenizer
        except Exception:  # noqa: BLE001 - 让后续导入路径给出更明确错误。
            return
        shim = types.ModuleType("unilm")
        shim.UniLMTokenizer = AutoTokenizer
        sys.modules["unilm"] = shim

    def _configure_huggingface_cache(self, cfg: dict[str, Any]) -> None:
        """把 LLMDet/HuggingFace 下载缓存约束到项目目录。"""
        cache_dir = cfg.get("cache_dir") or self.config.get("cache_dir")
        if not cache_dir:
            return
        root = Path(str(cache_dir))
        hub = root / "hub"
        os.environ["HF_HOME"] = str(root)
        os.environ["HF_HUB_CACHE"] = str(hub)
        os.environ["TRANSFORMERS_CACHE"] = str(hub)

    def _preflight_llmdet_data(self, cfg: dict[str, Any]) -> dict[str, Any] | None:
        """检查 LLMDet n-gram 概率表是否完整且可读。

        这里不再用其他模型的 npz 冒充缺失文件；一旦数据不匹配，
        就显式降级，避免产生污染的来源归因结果。
        """
        cache_dir = cfg.get("cache_dir") or self.config.get("cache_dir")
        if not cache_dir:
            return None

        expected = list(
            cfg.get(
                "expected_npz",
                [
                    "gpt2",
                    "opt",
                    "unilm",
                    "llama",
                    "bart",
                    "t5",
                    "bloom",
                    "neo",
                    "vicuna",
                    "gpt2_large",
                    "opt_3b",
                ],
            )
        )
        root = Path(str(cache_dir)) / "datasets" / "downloads" / "extracted"
        npz_dirs = list(root.glob("*/npz")) if root.exists() else []
        if not npz_dirs:
            return {
                "status": "data_mismatch",
                "source": "llmdet",
                "provider": "llmdet",
                "top_k": [],
                "confidence": 0.0,
                "missing_npz": expected,
                "bad_npz": [],
                "install_hint": "Run llmdet.load_probability() once to download n-gram probability tables.",
            }

        # 选择包含文件最多的 extracted/npz，避免旧的残缺缓存干扰。
        npz_dir = max(npz_dirs, key=lambda path: len(list(path.glob("*.npz"))))
        missing = [name for name in expected if not (npz_dir / f"{name}.npz").exists()]
        bad: list[str] = []
        for name in expected:
            path = npz_dir / f"{name}.npz"
            if not path.exists():
                continue
            if not zipfile.is_zipfile(path):
                bad.append(name)

        if missing or bad:
            return {
                "status": "data_mismatch",
                "source": "llmdet",
                "provider": "llmdet",
                "top_k": [],
                "confidence": 0.0,
                "missing_npz": missing,
                "bad_npz": bad,
                "cache_npz_dir": str(npz_dir),
                "install_hint": (
                    "LLMDet PyPI package/data cache is inconsistent. "
                    "Do not replace missing probability tables with other models; "
                    "use a matching official data package or keep LLMDet as an experimental branch."
                ),
            }
        return None

    def _normalize_score_map(self, scores: Any, top_k: int = 5) -> list[dict[str, float]]:
        """把 dict 概率归一成统一 top_k 结构。"""
        if not isinstance(scores, dict):
            return []
        normalized: list[dict[str, float]] = []
        for model, probability in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True):
            normalized.append({"model": str(model), "probability": round(float(probability), 6)})
            if len(normalized) >= top_k:
                break
        return normalized

    def _normalize_top_k(self, values: Any) -> list[dict[str, float]]:
        """兼容外部 CLI 的 top_k 输出。"""
        if not isinstance(values, list):
            return []
        top_k: list[dict[str, float]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            model = item.get("model") or item.get("label") or item.get("name")
            probability = item.get("probability", item.get("score", item.get("confidence", 0.0)))
            if model is None:
                continue
            top_k.append({"model": str(model), "probability": round(float(probability), 6)})
        return top_k

    def _provider_configs(self, cfg: dict[str, Any], default_provider: str) -> list[tuple[str, dict[str, Any]]]:
        """兼容旧单 provider 配置和新 providers 多分支配置。"""
        providers = cfg.get("providers")
        if isinstance(providers, dict):
            return [
                (str(name), {**value, "provider": value.get("provider", name)})
                for name, value in providers.items()
                if isinstance(value, dict) and value.get("enabled", True)
            ]
        provider = str(cfg.get("provider", default_provider))
        return [(provider, {**cfg, "provider": provider})]

    def _fuse_branch_results(
        self,
        branches: list[dict[str, Any]],
        modality: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """把多个分支的 top_k 归一化融合成统一候选列表。"""
        score_map: dict[str, float] = defaultdict(float)
        source_map: dict[str, list[str]] = defaultdict(list)
        ok_branches: list[dict[str, Any]] = []
        ok_count = 0
        for branch in branches:
            if branch.get("status") != "ok":
                continue
            ok_count += 1
            ok_branches.append(branch)
            weight = float(branch.get("weight", 1.0))
            provider = str(branch.get("provider") or branch.get("source") or "unknown")
            for item in branch.get("top_k") or []:
                model = item.get("model")
                if not model:
                    continue
                probability = float(item.get("probability") or 0.0)
                score_map[str(model)] += probability * weight
                source_map[str(model)].append(provider)

        if len(ok_branches) == 1:
            only = ok_branches[0]
            provider = str(only.get("provider") or only.get("source") or "unknown")
            top = []
            for item in (only.get("top_k") or [])[:top_k]:
                copied = dict(item)
                copied["sources"] = [provider]
                top.append(copied)
        else:
            top = self._normalize_probability_scores(score_map, top_k=top_k)
        for item in top:
            item["sources"] = sorted(set(item.get("sources") or source_map.get(item["model"], [])))

        if top:
            status = "ok" if ok_count == len(branches) else "partial"
        elif any(branch.get("status") == "data_mismatch" for branch in branches):
            status = "data_mismatch"
        elif any(branch.get("status") == "not_configured" for branch in branches):
            status = "not_configured"
        elif any(branch.get("status") == "dependency_missing" for branch in branches):
            status = "dependency_missing"
        elif any(branch.get("status") == "error" for branch in branches):
            status = "error"
        else:
            status = "skipped"

        return {
            "status": status,
            "source": "attribution_fusion",
            "provider": "fusion",
            "modality": modality,
            "top_k": top,
            "confidence": top[0]["probability"] if top else 0.0,
            "branches": branches,
            "note": "Fusion combines available attribution branches; unavailable branches are kept as diagnostics.",
        }

    def _normalize_probability_scores(self, score_map: dict[str, float], top_k: int = 5) -> list[dict[str, float]]:
        """把任意非负分数归一为 top_k 概率。"""
        clean = {model: max(0.0, float(score)) for model, score in score_map.items()}
        total = sum(clean.values())
        if total <= 0:
            return []
        ranked = sorted(clean.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {"model": model, "probability": round(score / total, 6)}
            for model, score in ranked
        ]

    def _load_model_artifact(self, path: Path) -> dict[str, Any]:
        """加载 joblib/pickle 训练产物。"""
        try:
            import joblib

            artifact = joblib.load(path)
        except Exception:
            with path.open("rb") as handle:
                artifact = pickle.load(handle)
        if isinstance(artifact, dict):
            return artifact
        return {"model": artifact}

    def _predict_top_k(self, model: Any, inputs: list[Any], top_k: int = 5) -> list[dict[str, float]]:
        """把 sklearn 风格模型输出转换成统一 Top-K。"""
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(inputs))[0]
            labels = list(getattr(model, "classes_", range(len(probabilities))))
        elif hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(inputs))
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            logits = scores[0]
            exp = np.exp(logits - np.max(logits))
            probabilities = exp / exp.sum()
            labels = list(getattr(model, "classes_", range(len(probabilities))))
        else:
            prediction = model.predict(inputs)[0]
            return [{"model": str(prediction), "probability": 1.0}]

        pairs = sorted(
            zip(labels, probabilities),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return [
            {"model": str(label), "probability": round(float(prob), 6)}
            for label, prob in pairs
        ]

    def _extract_image_features(self, path: Path) -> list[float]:
        """提取轻量图片来源分类特征。

        这些特征不是最强图像表征，但训练和推理都很快，适合先跑通来源分类。
        后续可以替换为 CLIP/DINO embedding。
        """
        import imagehash

        image = Image.open(path).convert("RGB").resize((128, 128))
        array = np.asarray(image).astype(np.float32) / 255.0
        means = array.mean(axis=(0, 1))
        stds = array.std(axis=(0, 1))
        mins = array.min(axis=(0, 1))
        maxs = array.max(axis=(0, 1))
        gray = array.mean(axis=2)
        phash = imagehash.phash(image)
        dhash = imagehash.dhash(image)
        hash_values = [float(bit) for bit in self._hash_to_bits(str(phash)) + self._hash_to_bits(str(dhash))]
        stats = [
            float(gray.mean()),
            float(gray.std()),
            float(np.percentile(gray, 10)),
            float(np.percentile(gray, 50)),
            float(np.percentile(gray, 90)),
        ]
        return [*means.tolist(), *stds.tolist(), *mins.tolist(), *maxs.tolist(), *stats, *hash_values]

    def _extract_clip_image_features(self, path: Path, artifact: dict[str, Any], cfg: dict[str, Any]) -> list[float]:
        """使用与训练一致的 CLIP 模型提取单张图片 embedding。"""
        from transformers import CLIPImageProcessor, CLIPModel

        model_name = str(artifact.get("clip_model_name") or cfg.get("clip_model_name") or "openai/clip-vit-base-patch32")
        cache_dir = str(cfg.get("cache_dir") or artifact.get("cache_dir") or "models/huggingface")
        self._configure_huggingface_cache({"cache_dir": cache_dir})
        device = str(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        cache_key = f"{model_name}|{cache_dir}|{device}"
        if cache_key not in self._clip_cache:
            processor = CLIPImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
            model.eval()
            self._clip_cache[cache_key] = (processor, model, device)
        processor, model, device = self._clip_cache[cache_key]
        image = Image.open(path).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            features = model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].astype(float).tolist()

    def _hash_to_bits(self, hex_hash: str) -> list[int]:
        """把 imagehash 的十六进制哈希转换成 bit 列表。"""
        bits = bin(int(hex_hash, 16))[2:].zfill(len(hex_hash) * 4)
        return [int(bit) for bit in bits]

    def _load_text_prototypes(self, path: Path) -> list[dict[str, str]]:
        """读取 JSONL 文本原型库。"""
        samples: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            model = payload.get("model") or payload.get("label") or payload.get("source_model")
            if not model:
                continue
            if isinstance(payload.get("samples"), list):
                for text in payload["samples"]:
                    if isinstance(text, str) and text.strip():
                        samples.append({"model": str(model), "text": text})
            elif isinstance(payload.get("text"), str) and payload["text"].strip():
                samples.append({"model": str(model), "text": payload["text"]})
        return samples

    def _load_image_prototypes(self, path: Path) -> list[dict[str, str]]:
        """读取 JSONL 图片原型库。"""
        samples: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            model = payload.get("model") or payload.get("label") or payload.get("source_model")
            phash = payload.get("phash")
            if model and phash:
                samples.append({"model": str(model), "phash": str(phash)})
        return samples

    def _disabled(self, name: str) -> dict[str, Any]:
        """生成禁用状态。"""
        return {
            "status": "disabled",
            "source": name,
            "provider": "none",
            "top_k": [],
            "confidence": 0.0,
            "note": f"{name} is disabled in config.",
        }

    def _not_configured(self, provider: str, note: str) -> dict[str, Any]:
        """生成未配置状态。"""
        return {
            "status": "not_configured",
            "source": provider,
            "provider": provider,
            "top_k": [],
            "confidence": 0.0,
            "note": note,
        }

    def _error(self, provider: str, exc: Exception) -> dict[str, Any]:
        """生成错误状态。"""
        return {
            "status": "error",
            "source": provider,
            "provider": provider,
            "top_k": [],
            "confidence": 0.0,
            "error": str(exc),
        }


@dataclass
class AttributionResult:
    """模型归因输出结果。"""

    top_k: List[Dict[str, float]]
    confidence: float
    features_used: List[str]


class AttributionMLP(nn.Module):
    """简单 MLP 归因模型。

    输入固定长度特征向量，输出已知模型列表上的 logits。
    """

    def __init__(self, input_dim: int = 64, num_models: int = 10):
        """构建 MLP 网络结构。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_models),
        )

    def forward(self, x):
        """前向传播。"""
        return self.net(x)


KNOWN_MODELS = [
    "chatgpt-4", "chatgpt-3.5", "claude-3", "gemini-pro",
    "stable-diffusion-xl", "midjourney-v6", "dall-e-3",
    "elevenlabs", "bark", "suno",
]


class ModelAttribution:
    """模型来源归因器 scaffold。"""

    def __init__(self, device: str | None = None):
        """初始化归因模型。"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttributionMLP(
            input_dim=64, num_models=len(KNOWN_MODELS)
        ).to(self.device)
        self.model.eval()

    def attribute(self, features: Dict[str, float], top_k: int = 3) -> AttributionResult:
        """根据特征输出 Top-K 可能来源模型。

        注意：未加载训练权重时，输出没有实际可信度。
        """
        feature_vec = self._prepare_features(features)
        tensor = torch.FloatTensor(feature_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_indices = torch.topk(probs, min(top_k, len(KNOWN_MODELS))).indices
        results = []
        for idx in top_indices:
            results.append({
                "model": KNOWN_MODELS[idx.item()],
                "probability": probs[idx.item()].item(),
            })

        return AttributionResult(
            top_k=results,
            confidence=results[0]["probability"] if results else 0.0,
            features_used=list(features.keys()),
        )

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """把 dict 特征整理成固定长度 64 维向量。"""
        vec = np.zeros(64, dtype=np.float32)
        for i, (_, val) in enumerate(sorted(features.items())):
            if i >= 64:
                break
            vec[i] = val
        return vec

    def load_weights(self, path: str):
        """加载训练好的归因模型权重。"""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
