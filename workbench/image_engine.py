"""Real image adapter for the preview workbench.

Default path: D-OpenCLIP detection score (fixed 0.5) and, only when the score reaches
the threshold, A0 ResNet50 22-source closed-set candidates. Research mode adds the
registered 1%/5% calibration thresholds and frozen OFA pair-verification scores.

Fails closed: no scaffold/demo/hash scores, no silent zero, unknown is never derived
from softmax. Registered research files are imported read-only and hash-checked.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

SCHEMA_VERSION = "image_result.v1"
ROOT = Path(__file__).resolve().parents[1]

ERROR_STATUS = {
    "invalid_input": 400, "too_large": 413, "decode_failed": 422,
    "model_unavailable": 503, "artifact_mismatch": 503, "busy": 429,
    "inference_failed": 500, "report_not_found": 404,
}
DEFAULT_LIMITS = dict(max_bytes=20 * 1024 * 1024, min_side=32, max_side=8192, max_pixels=40_000_000)
FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "BMP": "image/bmp"}
MODES = {"1", "L", "LA", "P", "PA", "RGB", "RGBA", "CMYK", "YCbCr"}
DETECTION_THRESHOLD = 0.5
MAIN_ARTIFACTS = ("openclip_config", "openclip_backbone", "dopenclip_head", "dopenclip_provenance",
                  "dopenclip_metrics", "a0_model", "a0_asset_manifest")
OFA_ARTIFACTS = ("ofa_fingerprint", "ofa_denoiser", "ofa_reference_bank", "ofa_reference_manifest")
# Registered implementation files whose hashes D-OpenCLIP provenance pins.
PINNED_CODE = ("research/image_attribution/openclip_route.py", "scripts/run_image_openclip.py",
               "research/image_attribution/models.py")
CONDITIONAL_STATEMENT = "在图片属于AI生成、且来源属于已知22类的条件下，以下标签最接近模型已学习的来源。"


class WorkbenchError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable
        self.status = ERROR_STATUS[code]

    def payload(self, request_id: str) -> dict[str, Any]:
        return dict(code=self.code, message=self.message, request_id=request_id, retryable=self.retryable)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decode_image(data: bytes, limits: dict | None = None) -> tuple[Image.Image, dict[str, Any]]:
    """Validate on the server and decode exactly like the research loaders: convert('RGB')."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    if not data:
        raise WorkbenchError("invalid_input", "上传内容为空")
    if len(data) > limits["max_bytes"]:
        raise WorkbenchError("too_large", f"文件超过 {limits['max_bytes']} 字节上限")
    try:
        probe = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise WorkbenchError("invalid_input", "无法识别为图片，仅支持 JPEG/PNG/WebP/BMP") from None
    except Image.DecompressionBombError:
        raise WorkbenchError("too_large", "图片像素数超过解码安全上限") from None
    except Exception:
        raise WorkbenchError("decode_failed", "图片头部损坏，无法读取") from None
    with probe:
        fmt, (width, height), mode = probe.format, probe.size, probe.mode
        frames = int(getattr(probe, "n_frames", 1) or 1)
        if fmt not in FORMATS:
            raise WorkbenchError("invalid_input", f"不支持的图片格式 {fmt}，仅支持 JPEG/PNG/WebP/BMP")
        if frames != 1:
            raise WorkbenchError("invalid_input", f"不支持多帧或动图（{frames} 帧）")
        if min(width, height) < limits["min_side"]:
            raise WorkbenchError("invalid_input", f"图片边长小于 {limits['min_side']} 像素")
        if max(width, height) > limits["max_side"] or width * height > limits["max_pixels"]:
            raise WorkbenchError("too_large", "图片宽高或总像素超过上限")
        if mode not in MODES:
            raise WorkbenchError("invalid_input", f"不支持的像素模式 {mode}")
        try:
            probe.verify()
        except Exception:
            raise WorkbenchError("decode_failed", "图片数据损坏或被截断") from None
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise WorkbenchError("decode_failed", "图片数据损坏或被截断") from None
    with image:
        orientation = image.getexif().get(274)
        has_alpha = image.mode in {"RGBA", "LA", "PA"} or (image.mode == "P" and "transparency" in image.info)
        rgb = image.convert("RGB")
    info = dict(sha256=sha256_bytes(data), bytes=len(data), media_type=FORMATS[fmt], width=width,
                height=height, mode=mode, frames=frames, has_alpha=has_alpha,
                exif_orientation=int(orientation) if orientation else None,
                decode_policy=dict(rgb="pil_convert_rgb", alpha="discarded_by_convert" if has_alpha else "none",
                                   exif_orientation="not_applied", matches_research_loader=True))
    return rgb, info


def _probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("model returned a non-finite or out-of-range score")
    return value


class ImageModels:
    """Loads registered artifacts once; every artifact and pinned source file is hash-checked."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.root = Path(settings["research_root"])
        self.device = str(settings.get("device", "cpu"))
        self.bundle: dict[str, str] = {}
        self.ofa_ready = False
        self.ofa_error: str | None = None

    def _path(self, name: str) -> tuple[Path, str]:
        spec = self.settings["artifacts"][name]
        return self.root / spec["path"], spec["sha256"]

    def _verify(self, names) -> None:
        for name in names:
            path, expected = self._path(name)
            if not path.is_file():
                raise WorkbenchError("model_unavailable", f"缺少模型产物 {name}")
            if sha256_file(path) != expected:
                raise WorkbenchError("artifact_mismatch", f"模型产物哈希不符 {name}")
            self.bundle[name] = expected

    def _json(self, name: str) -> dict:
        return json.loads(self._path(name)[0].read_text())

    def load(self) -> None:
        import torch
        if self.device != "cpu":
            raise WorkbenchError("model_unavailable", "预览只登记了 CPU 设备策略")
        torch.set_num_threads(int(self.settings.get("torch_threads", 8)))
        self._verify(MAIN_ARTIFACTS)
        provenance = self._json("dopenclip_provenance")
        for relative in PINNED_CODE:
            if sha256_file(ROOT / relative) != provenance["code"][relative]:
                raise WorkbenchError("artifact_mismatch", f"预处理/加载代码与登记哈希不符 {relative}")
        from research.image_attribution.models import load_checkpoint, resnet_transform
        from research.image_attribution.openclip_route import DIMENSION, PREPROCESSING, load_backbone, normalized
        from scripts.run_image_openclip import load_head
        config = self._json("openclip_config")
        if (provenance["classes"] != ["real", "ai"] or provenance["preprocessing"] != PREPROCESSING
                or provenance["head_asset_sha256"] != self.bundle["dopenclip_head"]
                or provenance["asset"]["sha256"] != self.bundle["openclip_backbone"]
                or config["asset"]["sha256"] != self.bundle["openclip_backbone"]):
            raise WorkbenchError("artifact_mismatch", "D-OpenCLIP 登记信息与产物不一致")
        try:
            self.visual, self.clip_transform = load_backbone(self.root, config, "cpu")
            self.head = load_head(self._path("dopenclip_head")[0], ["real", "ai"], "cpu")
        except ValueError as exc:
            raise WorkbenchError("artifact_mismatch", f"D-OpenCLIP 加载校验失败：{exc}") from None
        self.normalized, self.dimension, self.preprocessing = normalized, DIMENSION, PREPROCESSING
        metrics = self._json("dopenclip_metrics")
        self.fixed_metrics = metrics["fixed_0_5"]
        self.calibrated = metrics["calibrated"]
        if self._json("a0_asset_manifest")["model.pth"]["sha256"] != self.bundle["a0_model"]:
            raise WorkbenchError("artifact_mismatch", "A0 资产清单与模型哈希不一致")
        model, checkpoint = load_checkpoint(self._path("a0_model")[0], "cpu")
        classes = list(checkpoint["classes"])
        if checkpoint["task"] != "attribution" or len(classes) != 22:
            raise WorkbenchError("artifact_mismatch", "A0 不是 22 类来源模型")
        self.a0, self.a0_classes, self.a0_transform = model.eval(), classes, resnet_transform()
        self.a0_label_space = "iab-v7-200-q179-22:" + digest_json(classes)[:16]
        if self.settings.get("research_ofa", True):
            try:  # research-only; its failure must not take down the default path
                self._load_ofa()
            except WorkbenchError as exc:
                self.ofa_error = exc.code
            except Exception as exc:
                self.ofa_error = type(exc).__name__

    def _load_ofa(self) -> None:
        import torch
        from research.image_attribution.models import load_ofa, ofa_features, ofa_transform, reference_score
        self._verify(OFA_ARTIFACTS)
        self.ofa_encoder, self.ofa_pair, self.ofa_denoiser = load_ofa(
            self._path("ofa_fingerprint")[0], self._path("ofa_denoiser")[0], "cpu")
        bank = torch.load(self._path("ofa_reference_bank")[0], map_location="cpu", weights_only=True)
        manifest = self._json("ofa_reference_manifest")
        if list(manifest) != bank["classes"] or any(len(manifest[c]) != len(bank["references"][c]) for c in manifest):
            raise WorkbenchError("artifact_mismatch", "OFA 参考库与参考清单不一致")
        self.ofa_classes, self.ofa_bank = bank["classes"], bank["references"]
        self.ofa_reference_counts = {c: len(manifest[c]) for c in self.ofa_classes}
        self.ofa_reference_ids_sha256 = digest_json(manifest)
        self.ofa_features, self.ofa_transform, self.reference_score = ofa_features, ofa_transform, reference_score
        self.ofa_ready = True

    # Batched forms exist so the consistency check can compare single vs batch.
    def detect(self, images: list) -> list[float]:
        import torch
        with torch.inference_mode():
            x = torch.stack([self.clip_transform(image) for image in images])
            features = self.normalized(self.visual(x), self.dimension)
            return [_probability(p) for p in self.head(features).softmax(1)[:, 1].tolist()]

    def attribute(self, images: list) -> list[list[float]]:
        import torch
        with torch.inference_mode():
            x = torch.stack([self.a0_transform(image) for image in images])
            rows = self.a0(x).softmax(1).tolist()
        return [[_probability(p) for p in row] for row in rows]

    def ofa_scores(self, images: list) -> list[list[float]]:
        import torch
        with torch.inference_mode():
            features = self.ofa_features(torch.stack([self.ofa_transform(i) for i in images]),
                                         self.ofa_encoder, self.ofa_denoiser)
            return [[_probability(self.reference_score(self.ofa_pair, self.ofa_bank[c], f))
                     for c in self.ofa_classes] for f in features]


def _ranked(classes: list[str], scores: list[float]) -> list[dict[str, Any]]:
    order = sorted(range(len(classes)), key=lambda i: (-scores[i], i))
    return [dict(rank=r + 1, label=classes[i], score=scores[i]) for r, i in enumerate(order)]


def _ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 1)


class ImageService:
    """One resident model set, one inference at a time, bounded waiting, request-id dedupe."""

    def __init__(self, settings: dict[str, Any] | None, models: ImageModels | None = None):
        self.settings = settings or {}
        self.limits = {**DEFAULT_LIMITS, **(self.settings.get("limits") or {})}
        self.models = models or (ImageModels(self.settings) if self.settings else None)
        self.state = "ready" if models else ("not_loaded" if self.models else "disabled")
        self.error: WorkbenchError | None = None
        self._gate = threading.BoundedSemaphore(int(self.settings.get("max_pending", 2)))
        self._lock = threading.Lock()
        self._recent: OrderedDict[str, dict] = OrderedDict()
        self._recent_lock = threading.Lock()

    def start(self) -> None:
        if self.state == "not_loaded":
            self.state = "loading"
            threading.Thread(target=self._load, name="image-model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            self.models.load()
            self.state = "ready"
        except WorkbenchError as exc:
            self.state, self.error = ("artifact_mismatch" if exc.code == "artifact_mismatch" else "unavailable"), exc
        except Exception as exc:  # fail closed, keep message for operators
            self.state = "unavailable"
            self.error = WorkbenchError("model_unavailable", f"模型加载失败：{type(exc).__name__}")

    def _require_ready(self) -> None:
        if self.state == "ready":
            return
        if self.state in ("loading", "not_loaded"):
            raise WorkbenchError("model_unavailable", "图片模型正在加载，请稍后重试", retryable=True)
        if self.state == "disabled":
            raise WorkbenchError("model_unavailable", "当前服务未配置图片模型；不会回退到演示或占位分数")
        raise self.error or WorkbenchError("model_unavailable", "图片模型不可用")

    def capabilities(self) -> dict[str, Any]:
        m = self.models if self.state == "ready" else None
        return dict(
            state=self.state, error=self.error.payload("capabilities") if self.error else None,
            device=self.settings.get("device", "cpu") if self.settings else None,
            limits=self.limits, formats=sorted(FORMATS.values()),
            detection=dict(model_id="D-OpenCLIP", score_kind="d_openclip_softmax_ai", calibrated_probability=False,
                           threshold=DETECTION_THRESHOLD),
            attribution=dict(model_id="A0-ResNet50-iab-v7-200-q179-e30", label_space_size=22,
                             label_space_id=m.a0_label_space if m else None,
                             classes=m.a0_classes if m else None, conditional_statement=CONDITIONAL_STATEMENT),
            unknown=dict(status="unavailable", score=None),
            research=dict(ofa_pair_verification=(
                "disabled" if not self.settings.get("research_ofa", True) else
                "ready" if m and m.ofa_ready else "unavailable"),
                ofa_note=self.settings.get("research_ofa_note")),
            model_bundle=self._bundle() if m else None)

    def _bundle(self) -> dict[str, Any]:
        components = dict(sorted(self.models.bundle.items()))
        return dict(bundle_id=digest_json(components)[:16], components=components)

    def analyze(self, data: bytes, filename: str, research: bool = False, request_id: str | None = None) -> dict:
        request_id = request_id or uuid.uuid4().hex
        with self._recent_lock:
            if request_id in self._recent:
                return dict(self._recent[request_id], deduplicated=True)
        self._require_ready()
        start = time.monotonic()
        rgb, info = decode_image(data, self.limits)
        decode_ms = _ms(start)
        if not self._gate.acquire(blocking=False):
            raise WorkbenchError("busy", "已有图片在排队分析，请稍后重试", retryable=True)
        try:
            waited = time.monotonic()
            if not self._lock.acquire(timeout=float(self.settings.get("queue_timeout_seconds", 120))):
                raise WorkbenchError("busy", "等待推理超时，请稍后重试", retryable=True)
            try:
                result = self._run(rgb, info, research, request_id, dict(decode_ms=decode_ms, queue_ms=_ms(waited)), start)
            finally:
                self._lock.release()
        finally:
            self._gate.release()
        with self._recent_lock:
            self._recent[request_id] = result
            while len(self._recent) > 64:
                self._recent.popitem(last=False)
        return result

    def _run(self, rgb, info, research, request_id, timing, start) -> dict[str, Any]:
        m, failures = self.models, []
        t = time.monotonic()
        try:
            score = m.detect([rgb])[0]
        except Exception:
            raise WorkbenchError("inference_failed", "检测模型推理失败", retryable=True) from None
        timing["detection_ms"] = _ms(t)
        reached = score >= DETECTION_THRESHOLD
        fixed = m.fixed_metrics
        detection = dict(
            status="ok", model_id="D-OpenCLIP", score=score, score_kind="d_openclip_softmax_ai",
            calibrated_probability=False,
            label="at_or_above_threshold" if reached else "below_threshold", threshold=DETECTION_THRESHOLD,
            threshold_provenance=dict(kind="registered_fixed_operating_point", role="evaluation",
                                      n=fixed["n"], negative_n=fixed["negative_count"],
                                      positive_n=fixed["positive_count"], fpr=fixed["fpr"], tpr=fixed["tpr"]))
        a0_scores, t = None, time.monotonic()
        if reached or research:
            try:
                a0_scores = m.attribute([rgb])[0]
            except Exception:
                failures.append("attribution")
        timing["attribution_ms"] = _ms(t)
        attribution = dict(model_id="A0-ResNet50-iab-v7-200-q179-e30", label_space_id=m.a0_label_space,
                           score_kind="a0_softmax_closed_set_22",
                           conditional_on=dict(detection="at_or_above_threshold", source_in_known_22=True),
                           conditional_statement=CONDITIONAL_STATEMENT, top_k=[])
        if not reached:
            attribution.update(status="not_applicable", reason="detection_below_threshold")
        elif a0_scores is None:
            attribution.update(status="error", reason="inference_failed")
        else:
            attribution.update(status="ok", top_k=_ranked(m.a0_classes, a0_scores))
        t = time.monotonic()
        items = self._research(rgb, score, a0_scores, reached, failures) if research else []
        timing["research_ms"] = _ms(t)
        timing["total_ms"] = _ms(start)
        return dict(
            schema_version=SCHEMA_VERSION, analysis_id=uuid.uuid4().hex, request_id=request_id,
            created_at=utc_now(), status="partial" if failures else "ok", mode="research" if research else "default",
            input=info, model_bundle=self._bundle(), detection=detection, attribution=attribution,
            unknown=dict(status="unavailable", score=None, reason="main_22_class_interface_has_no_accepted_unknown_rejection"),
            research=items,
            provenance=dict(device="cpu", external_calls=False, preprocessing=dict(
                detection=dict(m.preprocessing, identity="openclip-vit-l-14-quickgelu"),
                attribution=dict(identity="iab-resnet50-v1", resize=256, center_crop=224, normalization="imagenet")),
                c2pa="not_checked", watermark="not_checked", fingerprint_registry="not_used"),
            limitations=[
                "模型检测分数不是校准概率；固定阈值 0.5 来自开发集 evaluation 角色（单 seed）。",
                "来源候选只在 22 个已学习来源内比较；未知来源拒识在主接口不可用。",
                "结果来自 IAB v7-200-q179 开发协议，不是独立外部测试，holdout 未推理。",
                "未检查 C2PA 与水印；缺少 C2PA 不构成伪造证据。",
            ],
            timing=timing)

    def _research(self, rgb, score, a0_scores, reached, failures) -> list[dict[str, Any]]:
        m = self.models
        rows = []
        for target, entry in sorted(m.calibrated.items(), key=lambda kv: float(kv[0])):
            ev = entry["evaluation"]
            rows.append(dict(target_fpr=float(target), threshold=ev["threshold"], reached=score >= ev["threshold"],
                             calibration_negative_n=entry["calibration_negative_n"], calibration_fpr=entry["calibration_fpr"],
                             evaluation_n=ev["n"], evaluation_negative_n=ev["negative_count"],
                             evaluation_fpr=ev["fpr"], evaluation_tpr=ev["tpr"]))
        items = [dict(id="d_openclip_calibrated_thresholds", status="ok", score_kind="d_openclip_softmax_ai",
                      rows=rows, note="阈值只用 calibration 真实图拟合；跨域（如 held-FFHQ）不保证目标 FPR。")]
        if not reached:
            items.append(dict(id="a0_closed_set_research", status="ok" if a0_scores else "error",
                              score_kind="a0_softmax_closed_set_22", label_space_id=m.a0_label_space,
                              top_k=_ranked(m.a0_classes, a0_scores) if a0_scores else [],
                              note="检测分数低于阈值，主结论不带来源；此处仅供研究查看闭集分数。"))
        registered = self.settings.get("artifacts") or {}
        ofa = dict(id="ofa_pair_verification", score_kind="ofa_pair_match_mean_sigmoid",
                   # Bound to the registered reference bank/ID list even when scores are withheld.
                   reference_bank_sha256=m.bundle.get("ofa_reference_bank") or
                   registered.get("ofa_reference_bank", {}).get("sha256"),
                   reference_manifest_sha256=m.bundle.get("ofa_reference_manifest") or
                   registered.get("ofa_reference_manifest", {}).get("sha256"),
                   reference_ids_sha256=getattr(m, "ofa_reference_ids_sha256", None),
                   note="每类为与该来源登记参考图的配对匹配均值，不是类别概率，不与检测分数或 A0 分数混合。")
        if not m.ofa_ready:
            ofa.update(status="unavailable", scores=[],
                       reason=m.ofa_error or self.settings.get("research_ofa_note") or "disabled")
        else:
            try:
                ofa.update(status="ok", scores=_ranked(m.ofa_classes, m.ofa_scores([rgb])[0]),
                           reference_counts=m.ofa_reference_counts)
            except Exception:
                failures.append("ofa")
                ofa.update(status="error", scores=[])
        items.append(ofa)
        items.append(dict(id="registered_only", status="registered",
                          entries=["fixed_50_50_fusion", "calibrated_fusion_alpha_0.75", "open20_unknown_rejection"],
                          note="融合与 open20 拒识只展示登记结果（见实验对照页），不在上传图片上重拟合或推理。"))
        return items
