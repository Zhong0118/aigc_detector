"""Real image models for the workbench: bundle image-standard-q179-v1 (contract image-result-v1).

Standard path: det-openclip-q179-v1 (p_ai = softmax[:, 1], fixed threshold 0.5); only when
p_ai >= 0.5, attr-resnet50-q179-v1 ranks the 22 known AI sources. Unknown origin is always
unavailable/null. Research mode attaches at most one allowlisted method and never rewrites
the standard result.

Fails closed: no scaffold/demo/hash scores, no silent zero/human, no substitute weights.
Every asset, evidence file and registered source file is hash-checked; registered research
loaders are imported read-only (their training entry points are never executed).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from PIL import Image

SCHEMA_VERSION = "image-result-v1"
DECODE_POLICY = "image-upload-rgb-v1"
ROOT = Path(__file__).resolve().parents[1]

ERROR_STATUS = {
    "empty_file": 400, "decode_failed": 400, "unsupported_image_policy": 400,
    "too_large": 413, "unsupported_media_type": 415,
    "unsupported_mode": 422, "unknown_method_id": 422, "invalid_request_id": 422,
    "request_id_conflict": 409,
    "model_unavailable": 503, "artifact_mismatch": 503, "busy": 503,
    "inference_failed": 500, "report_not_found": 404,
}
DEFAULT_LIMITS = dict(max_bytes=10 * 1024 * 1024, max_pixels=16_000_000, max_side=8192)
MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png"}
OTHER_FORMATS = ((b"GIF8", "GIF"), (b"BM", "BMP"), (b"II*\x00", "TIFF"), (b"MM\x00*", "TIFF"),
                 (b"\x00\x00\x01\x00", "ICO"), (b"8BPS", "PSD"))
FIX_HINT = "请先转为方向正确的不透明 8-bit RGB JPEG/PNG 后再上传。"
CONDITIONAL_STATEMENT = "在图片属于AI生成、且来源属于已知22类的条件下，以下标签是模型的候选排序。"
UNKNOWN = dict(status="unavailable", score=None, reason_code="not_calibrated_for_22_classes",
               message="尚无经验证的通用未知来源拒识能力")
MODES = ("standard", "research")
DEPENDENCIES = ("torch", "torchvision", "open_clip_torch", "Pillow", "numpy", "safetensors")
LIMITATIONS = [
    "检测分数是模型输出的 softmax 分数，不是校准概率；固定阈值 0.5 未针对新上传图片重新标定。",
    "来源候选是条件排序：只在“图片为 AI 生成且来源属于已知 22 类”的假设下成立；未知来源拒识在标准模式不可用。",
    "所有性能数字来自 IAB v7-200-q179 单 seed、已查看过的开发角色，不是独立外部测试；holdout 未推理。",
    "跨语义域会明显增加误报：held-FFHQ 93 张真实人脸中 OpenCLIP 路线固定 0.5 误报 15 张。",
    "A0 来源模型的父 dev 参与过选模；来源标签是数据集目录名，不代表商业服务的实际版本或架构家族。",
    "未检查 C2PA 与水印；结果不是“100% 鉴定”，需结合人工复核。",
]


class WorkbenchError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable
        self.status = ERROR_STATUS[code]

    def payload(self, request_id: str | None) -> dict[str, Any]:
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
    """Same digest as research openclip_route.digest (class maps, preprocessing)."""
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ----------------------------------------------------------------------------- input policy
def sniff_format(data: bytes) -> str | None:
    """Identify by content, never by filename or Content-Type."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[4:8] == b"ftyp":
        return "HEIF/AVIF"
    for magic, name in OTHER_FORMATS:
        if data.startswith(magic):
            return name
    return None


def _policy(message: str) -> WorkbenchError:
    return WorkbenchError("unsupported_image_policy", f"{message}。{FIX_HINT}")


def decode_image(data: bytes, limits: dict | None = None) -> tuple[Image.Image, dict[str, Any]]:
    """Policy image-upload-rgb-v1: static 8-bit JPEG/PNG, RGB or L, no alpha, EXIF orientation absent/1."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    if not data:
        raise WorkbenchError("empty_file", "上传内容为空")
    if len(data) > limits["max_bytes"]:
        raise WorkbenchError("too_large", f"文件超过 {limits['max_bytes'] // (1024 * 1024)} MiB 上限")
    fmt = sniff_format(data)
    if fmt not in MEDIA_TYPES:
        if fmt:
            raise WorkbenchError("unsupported_media_type", f"不支持 {fmt}，仅接受静态 JPEG/PNG。{FIX_HINT}")
        raise WorkbenchError("decode_failed", "文件内容不是可识别的 JPEG/PNG 图片")
    if fmt == "PNG":  # PIL silently reduces 16-bit RGB to 8-bit, so read IHDR directly
        if len(data) < 33 or data[12:16] != b"IHDR":
            raise WorkbenchError("decode_failed", "PNG 头部损坏")
        bit_depth, color_type = data[24], data[25]
        if bit_depth != 8:
            raise _policy(f"不支持 {bit_depth}-bit PNG")
        if color_type not in (0, 2):
            raise _policy({3: "不支持调色板 PNG", 4: "不支持带透明通道的灰度 PNG",
                           6: "不支持带透明通道的 RGBA PNG"}.get(color_type, "不支持的 PNG 颜色类型"))
        width, height = struct.unpack(">II", data[16:24])
        _check_size(width, height, limits)
    try:
        probe = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise WorkbenchError("too_large", "图片像素数超过解码安全上限") from None
    except Exception:
        raise WorkbenchError("decode_failed", "图片头部损坏，无法读取") from None
    with probe:
        width, height = probe.size
        _check_size(width, height, limits)
        if probe.format == "MPO" or int(getattr(probe, "n_frames", 1) or 1) != 1:
            raise _policy("不支持多帧、动图或多图 JPEG")
        if probe.format != fmt:
            raise WorkbenchError("decode_failed", "文件内容与图片格式不一致")
        if probe.mode not in ("RGB", "L"):
            raise _policy(f"不支持像素模式 {probe.mode}（如 CMYK、16-bit、调色板或透明）")
        if fmt == "JPEG" and int(getattr(probe, "bits", 8)) != 8:
            raise _policy("仅支持 8-bit JPEG")
        if "transparency" in probe.info:
            raise _policy("不支持透明通道")
        try:  # PNG getexif() may read past IDAT, so truncation can surface here
            orientation = probe.getexif().get(0x0112)
        except Exception:
            raise WorkbenchError("decode_failed", "图片数据损坏或被截断") from None
        if orientation not in (None, 1):
            raise _policy(f"EXIF 方向为 {orientation}，服务不会自动旋转")
        try:
            probe.load()
            rgb = probe.convert("RGB")  # L -> RGB exactly like the research loaders
        except Exception:
            raise WorkbenchError("decode_failed", "图片数据损坏或被截断") from None
        color_mode = probe.mode
    info = dict(sha256=sha256_bytes(data), media_type=MEDIA_TYPES[fmt], width=width, height=height,
                decode_policy=DECODE_POLICY, bytes=len(data), color_mode=color_mode)
    return rgb, info


def _check_size(width: int, height: int, limits: dict) -> None:
    if width <= 0 or height <= 0:
        raise WorkbenchError("decode_failed", "图片宽高无效")
    if max(width, height) > limits["max_side"] or width * height > limits["max_pixels"]:
        raise WorkbenchError("too_large", f"图片 {width}×{height} 超过单边 {limits['max_side']} 或总像素 "
                                          f"{limits['max_pixels']:,} 上限")


# ----------------------------------------------------------------------------- models
def _probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("model returned a non-finite or out-of-range score")
    return value


def _row(values: list[float], size: int) -> list[float]:
    if len(values) != size:
        raise ValueError("score vector length does not match the class map")
    row = [_probability(v) for v in values]
    if abs(sum(row) - 1.0) > 1e-4:
        raise ValueError("softmax row does not sum to one")
    return row


def verify_asset(root: Path, spec: dict, name: str) -> Path:
    """Registered relative path inside the research tree (or its shared assets link) + size + SHA."""
    relative = Path(spec["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise WorkbenchError("artifact_mismatch", f"{name} 路径不在允许范围内")
    path = root / relative
    allowed = (root.resolve(), (root / "outputs/image_experiments/assets").resolve())
    if not any(path.resolve().is_relative_to(base) for base in allowed):
        raise WorkbenchError("artifact_mismatch", f"{name} 路径不在允许范围内")
    if not path.is_file():
        raise WorkbenchError("model_unavailable", f"缺少模型产物 {name}")
    if spec.get("bytes") is not None and path.stat().st_size != int(spec["bytes"]):
        raise WorkbenchError("artifact_mismatch", f"模型产物大小不符 {name}")
    if sha256_file(path) != spec["sha256"]:
        raise WorkbenchError("artifact_mismatch", f"模型产物哈希不符 {name}")
    return path


def _mismatch(condition: bool, message: str) -> None:
    if condition:
        raise WorkbenchError("artifact_mismatch", message)


def _ranked(classes: list[str], scores: list[float]) -> list[dict[str, Any]]:
    order = sorted(range(len(classes)), key=lambda i: (-scores[i], i))  # stable: (-score, class index)
    return [dict(rank=r + 1, source=classes[i], score=scores[i]) for r, i in enumerate(order)]


class ImageModels:
    """Loads the registered bundle once (CPU, FP32, eval) after verifying every hash."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.root = Path(settings["research_root"])
        self.registry = settings["registry"]
        self.research: dict[str, str] = {}  # method_id -> "ready" | error code
        self.loaded_assets: dict[str, str] = {}

    def _asset(self, name: str, spec: dict) -> Path:
        path = verify_asset(self.root, spec, name)
        self.loaded_assets[name] = spec["sha256"]
        return path

    def _json(self, name: str, spec: dict) -> dict:
        return json.loads(self._asset(name, spec).read_text())

    def load(self) -> None:
        import torch
        if self.settings.get("device", "cpu") != "cpu":
            raise WorkbenchError("model_unavailable", "预览只登记并验收了 CPU 设备")
        torch.set_num_threads(int(self.settings.get("torch_threads", 8)))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for relative, expected in self.registry["code_sha256"].items():
            _mismatch(sha256_file(ROOT / relative) != expected, f"登记的加载/预处理代码哈希不符 {relative}")
        self._load_detection(self.registry["detection"])
        self._load_attribution(self.registry["attribution"])
        for spec in self.registry.get("research_methods", []):
            try:  # research adapters must never take down the standard path
                if spec.get("loader") == "dna":
                    self._load_dna(spec)
                self.research[spec["id"]] = "ready"
            except WorkbenchError as exc:
                self.research[spec["id"]] = exc.code
            except Exception as exc:
                self.research[spec["id"]] = f"load_failed:{type(exc).__name__}"

    def _load_detection(self, spec: dict) -> None:
        from research.image_attribution.openclip_route import DIMENSION, PREPROCESSING, load_backbone, normalized
        from scripts.run_image_openclip import load_head
        assets, evidence = spec["assets"], spec["evidence"]
        backbone = self._asset("detection.backbone", assets["backbone"])
        head_path = self._asset("detection.head", assets["head"])
        config = self._json("detection.config", evidence["config"])
        provenance = self._json("detection.provenance", evidence["provenance"])
        metrics = self._json("detection.metrics", evidence["metrics"])
        classes = list(spec["classes"])
        _mismatch(classes != ["real", "ai"] or digest_json(classes) != spec["classes_sha256"], "检测类别顺序不符")
        _mismatch(provenance["classes"] != classes, "D-OpenCLIP 登记类别与清单不符")
        _mismatch(digest_json(PREPROCESSING) != spec["preprocessing_sha256"] or provenance["preprocessing"] != PREPROCESSING,
                  "D-OpenCLIP 预处理与登记不符")
        _mismatch(provenance["head_asset_sha256"] != assets["head"]["sha256"]
                  or provenance["asset"]["sha256"] != assets["backbone"]["sha256"]
                  or config["asset"]["sha256"] != assets["backbone"]["sha256"]
                  or (self.root / config["asset"]["path"]).resolve() != backbone.resolve(),
                  "D-OpenCLIP 登记信息与产物不一致")
        try:
            self.visual, self.clip_transform = load_backbone(self.root, config, "cpu")
            self.head = load_head(head_path, classes, "cpu")
        except ValueError as exc:
            raise WorkbenchError("artifact_mismatch", f"D-OpenCLIP 加载校验失败：{exc}") from None
        self.visual.eval()
        self.normalized, self.clip_dimension = normalized, DIMENSION
        self.fixed_metrics, self.calibrated = metrics["fixed_0_5"], metrics["calibrated"]

    def _load_attribution(self, spec: dict) -> None:
        from research.image_attribution.models import load_checkpoint, resnet_transform
        model_path = self._asset("attribution.model", spec["assets"]["model"])
        manifest = self._json("attribution.asset_manifest", spec["evidence"]["asset_manifest"])
        _mismatch(manifest["model.pth"]["sha256"] != spec["assets"]["model"]["sha256"], "A0 资产清单与模型哈希不一致")
        _mismatch(digest_json(spec["preprocessing"]) != spec["preprocessing_sha256"], "A0 预处理描述哈希不符")
        transform = resnet_transform()
        _check_resnet_transform(transform, spec["preprocessing"])
        model, checkpoint = load_checkpoint(model_path, "cpu")  # weights_only + strict inside
        classes = list(checkpoint["classes"])
        _mismatch(checkpoint.get("task") != "attribution" or classes != list(spec["classes"])
                  or digest_json(classes) != spec["classes_sha256"], "A0 类别表与登记不符")
        self.a0, self.a0_classes, self.a0_transform = model.eval(), classes, transform

    def _load_dna(self, spec: dict) -> None:
        from research.image_attribution.dna_route import DIMENSION, PREPROCESSING, load_backbone
        from scripts.run_image_dna import load_head
        self._asset(f"{spec['id']}.backbone", spec["assets"]["backbone"])
        head_path = self._asset(f"{spec['id']}.head", spec["assets"]["head"])
        config = self._json(f"{spec['id']}.config", spec["evidence"]["config"])
        provenance = self._json(f"{spec['id']}.provenance", spec["evidence"]["provenance"])
        classes = list(provenance["classes"])
        _mismatch(digest_json(classes) != spec["classes_sha256"] or classes != self.a0_classes, "DNA 类别表与登记不符")
        _mismatch(digest_json(PREPROCESSING) != spec["preprocessing_sha256"] or provenance["preprocessing"] != PREPROCESSING,
                  "DNA 预处理与登记不符")
        _mismatch(provenance["head_asset_sha256"] != spec["assets"]["head"]["sha256"]
                  or config["asset"]["sha256"] != spec["assets"]["backbone"]["sha256"], "DNA 登记信息与产物不一致")
        try:
            backbone, transform = load_backbone(self.root, config, "cpu")
            head = load_head(head_path, classes, "cpu")
        except ValueError as exc:
            raise WorkbenchError("artifact_mismatch", f"DNA 加载校验失败：{exc}") from None
        self.dna = (backbone.eval(), transform, head.eval(), DIMENSION)

    # Batched forms exist so acceptance can compare single vs batch on the real models.
    def detect(self, images: list) -> list[float]:
        import torch
        with torch.inference_mode():
            x = torch.stack([self.clip_transform(image) for image in images])
            features = self.normalized(self.visual(x).float(), self.clip_dimension)
            rows = self.head(features).softmax(1).tolist()
        return [_row(row, 2)[1] for row in rows]

    def attribute(self, images: list) -> list[list[float]]:
        import torch
        with torch.inference_mode():
            x = torch.stack([self.a0_transform(image) for image in images])
            rows = self.a0(x).softmax(1).tolist()
        return [_row(row, len(self.a0_classes)) for row in rows]

    def dna_scores(self, images: list) -> list[list[float]]:
        import torch
        backbone, transform, head, dimension = self.dna
        with torch.inference_mode():
            x = torch.stack([transform(image) for image in images])
            rows = head(self.normalized(backbone(x), dimension)).softmax(1).tolist()
        return [_row(row, len(self.a0_classes)) for row in rows]


def _check_resnet_transform(transform, spec: dict) -> None:
    from torchvision import transforms as T
    steps = list(transform.transforms)
    ok = (len(steps) == 4 and isinstance(steps[0], T.Resize) and steps[0].size == spec["resize_short_edge"]
          and steps[0].interpolation == T.InterpolationMode.BILINEAR
          and isinstance(steps[1], T.CenterCrop) and list(steps[1].size) == list(spec["center_crop"])
          and isinstance(steps[2], T.ToTensor) and isinstance(steps[3], T.Normalize)
          and list(steps[3].mean) == spec["mean"] and list(steps[3].std) == spec["std"])
    _mismatch(not ok, "A0 预处理与登记不符")


def _git_commit() -> str | None:
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--",
                                "workbench", "api", "storage"], capture_output=True, text=True, timeout=10).stdout
        return (head + ("+dirty" if dirty.strip() else "")) or None
    except Exception:
        return None


def _versions() -> dict[str, str | None]:
    out = {}
    for name in DEPENDENCIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 1)


# ----------------------------------------------------------------------------- service
class ImageService:
    """One resident model set; one inference at a time, bounded waiting, request-id dedupe."""

    def __init__(self, settings: dict[str, Any] | None, models: ImageModels | None = None,
                 acceptance: dict | None = None):
        self.settings = settings or {}
        self.registry = self.settings.get("registry") or {}
        self.registry_sha256 = digest_json(self.registry) if self.registry else None
        self.limits = {**DEFAULT_LIMITS, **(self.settings.get("limits") or {})}
        self.methods = {m["id"]: m for m in self.registry.get("research_methods", [])}
        self.models = models or (ImageModels(self.settings) if self.registry else None)
        self.acceptance = acceptance
        self.state = "ready" if models else ("not_loaded" if self.models else "disabled")
        self.error: WorkbenchError | None = None
        self.code_commit = self.settings.get("code_commit") or _git_commit()
        self.versions = _versions()
        self.load_seconds: float | None = None
        self._gate = threading.BoundedSemaphore(int(self.settings.get("max_pending", 2)))
        self._lock = threading.Lock()
        self._recent: OrderedDict[str, tuple[tuple, dict]] = OrderedDict()
        self._inflight: dict[str, tuple] = {}
        self._table_lock = threading.Lock()

    # -- lifecycle
    def start(self) -> None:
        if self.state == "not_loaded":
            self.state = "loading"
            threading.Thread(target=self._load, name="image-model-loader", daemon=True).start()

    def _load(self) -> None:
        started = time.monotonic()
        try:
            self.models.load()
            self.acceptance = self.acceptance or _read_acceptance(self.settings)
            missing = [c for c in ("detection", "attribution") if not self._accepted(self.registry[c]["model_id"],
                                                                                       self.registry[c])]
            if missing:
                raise WorkbenchError("model_unavailable", "默认模型尚未通过本设备的实际数值验收")
            self.state = "ready"
        except WorkbenchError as exc:
            self.state, self.error = ("artifact_mismatch" if exc.code == "artifact_mismatch" else "unavailable"), exc
        except Exception as exc:  # fail closed; never fall back to another model
            self.state = "unavailable"
            self.error = WorkbenchError("model_unavailable", f"模型加载失败：{type(exc).__name__}")
        self.load_seconds = round(time.monotonic() - started, 2)

    def _accepted(self, component_id: str, spec: dict) -> bool:
        """Numerical acceptance on this device must exist for exactly these assets."""
        if self.acceptance is None:
            return False
        record = (self.acceptance.get("components") or {}).get(component_id) or {}
        assets = {k: v["sha256"] for k, v in (spec.get("assets") or {}).items()}
        return (record.get("passed") is True and self.acceptance.get("device") == self.settings.get("device", "cpu")
                and record.get("assets_sha256") == assets)

    def method_state(self, method_id: str) -> str:
        spec = self.methods[method_id]
        if self.state != "ready":
            return self.state
        if (self.models.research.get(method_id) if self.models else None) != "ready":
            return "unavailable"
        if spec.get("uses_default_models"):  # same accepted models, different registered reading
            return "ready"
        return "ready" if self._accepted(method_id, spec) else "unvalidated"

    def _require_ready(self) -> None:
        if self.state == "ready":
            return
        if self.state in ("loading", "not_loaded"):
            raise WorkbenchError("model_unavailable", "图片模型正在加载，请稍后重试", retryable=True)
        if self.state == "disabled":
            raise WorkbenchError("model_unavailable", "当前服务未配置图片模型；不会回退到演示或占位分数")
        raise self.error or WorkbenchError("model_unavailable", "图片模型不可用")

    # -- read-only descriptions
    def capabilities(self) -> dict[str, Any]:
        reg = self.registry
        detection, attribution = reg.get("detection") or {}, reg.get("attribution") or {}
        return dict(
            state=self.state, error=self.error.payload(None) if self.error else None,
            bundle_id=reg.get("bundle_id"), registry_sha256=self.registry_sha256,
            inference_code_commit=self.code_commit, device=self.settings.get("device") if reg else None,
            precision="fp32", load_seconds=self.load_seconds, external_calls=False, modes=list(MODES),
            input_policy=dict(decode_policy=DECODE_POLICY, media_types=sorted(MEDIA_TYPES.values()),
                              max_bytes=self.limits["max_bytes"], max_pixels=self.limits["max_pixels"],
                              max_side=self.limits["max_side"], bit_depth=8, color_modes=["RGB", "L"],
                              rejected=["multi_frame", "alpha", "palette", "cmyk", "16_bit", "exif_orientation_not_1"]),
            defaults=[
                dict(model_id=detection.get("model_id"), task="detection", score_kind="binary_softmax_p_ai",
                     classes=detection.get("classes"), class_map_sha256=detection.get("classes_sha256"),
                     preprocessing_sha256=detection.get("preprocessing_sha256"),
                     threshold=detection.get("threshold"), threshold_id=detection.get("threshold_id"),
                     assets_sha256={k: v["sha256"] for k, v in (detection.get("assets") or {}).items()},
                     offline_validation="completed", online_integration=self.state),
                dict(model_id=attribution.get("model_id"), task="source_classification",
                     score_kind="known_ai_source_softmax", label_space="known_ai_22",
                     classes=attribution.get("classes"), class_map_sha256=attribution.get("classes_sha256"),
                     preprocessing_sha256=attribution.get("preprocessing_sha256"),
                     gate=f"detection score >= {detection.get('threshold')}",
                     assets_sha256={k: v["sha256"] for k, v in (attribution.get("assets") or {}).items()},
                     offline_validation="completed", online_integration=self.state,
                     conditional_statement=CONDITIONAL_STATEMENT)] if reg else [],
            unknown={k: UNKNOWN[k] for k in ("status", "score", "reason_code", "message")},
            research_methods=[dict(id=m["id"], label=m["label"], task=m["task"], score_kind=m["score_kind"],
                                   label_space=m.get("label_space"), offline_validation="completed",
                                   online_integration=self.method_state(m["id"]),
                                   run_allowed=self.method_state(m["id"]) == "ready", note=m.get("note"))
                              for m in self.methods.values()],
            offline_only_methods=[dict(m, offline_validation="completed", online_integration="not_implemented",
                                       run_allowed=False) for m in reg.get("offline_only_methods", [])],
            not_exposed=list(reg.get("not_exposed", [])))

    # -- analysis
    def validate_request(self, mode: str, method_id: str | None, request_id: str | None) -> str:
        if request_id is not None:
            try:
                request_id = str(uuid.UUID(str(request_id)))
            except ValueError:
                raise WorkbenchError("invalid_request_id", "request_id 必须是 UUID") from None
        if mode not in MODES:
            raise WorkbenchError("unsupported_mode", "mode 只能是 standard 或 research")
        if mode == "standard" and method_id:
            raise WorkbenchError("unknown_method_id", "standard 模式不接受 research_method_id")
        if mode == "research":
            if not method_id or method_id not in self.methods:
                raise WorkbenchError("unknown_method_id", "research 模式需要一个后端允许的研究方法 ID")
        return request_id or str(uuid.uuid4())

    def analyze(self, data: bytes, mode: str = "standard", method_id: str | None = None,
                request_id: str | None = None, persist: Callable[[dict], int] | None = None) -> tuple[dict, bool]:
        """Returns (image_result, deduplicated)."""
        request_id = self.validate_request(mode, method_id, request_id)
        key = (sha256_bytes(data), mode, method_id, self.registry_sha256)
        with self._table_lock:
            if request_id in self._recent:
                stored_key, stored = self._recent[request_id]
                if stored_key != key:
                    raise WorkbenchError("request_id_conflict", "同一 request_id 已用于不同的输入或参数")
                return stored, True
            if request_id in self._inflight:
                if self._inflight[request_id] != key:
                    raise WorkbenchError("request_id_conflict", "同一 request_id 已用于不同的输入或参数")
                raise WorkbenchError("busy", "该请求仍在处理中，请稍后用同一 request_id 查询", retryable=True)
            self._inflight[request_id] = key
        try:
            self._require_ready()
            if mode == "research" and self.method_state(method_id) != "ready":
                raise WorkbenchError("model_unavailable", "该研究方法只有离线结果，在线未接入或未通过验收")
            start = time.monotonic()
            rgb, info = decode_image(data, self.limits)
            timing = dict(decode_ms=_ms(start))
            if not self._gate.acquire(blocking=False):
                raise WorkbenchError("busy", "已有图片在分析，请稍后重试", retryable=True)
            try:
                waited = time.monotonic()
                if not self._lock.acquire(timeout=float(self.settings.get("queue_timeout_seconds", 120))):
                    raise WorkbenchError("busy", "等待推理超时，请稍后重试", retryable=True)
                try:
                    timing["queue_ms"] = _ms(waited)
                    result = self._run(rgb, info, mode, method_id, request_id, timing, start)
                finally:
                    self._lock.release()
            finally:
                self._gate.release()
            self._persist(result, persist)
            with self._table_lock:
                self._recent[request_id] = (key, result)
                while len(self._recent) > 128:
                    self._recent.popitem(last=False)
            return result, False
        finally:
            with self._table_lock:
                self._inflight.pop(request_id, None)

    @staticmethod
    def _persist(result: dict, persist: Callable[[dict], int] | None) -> None:
        if persist is None:
            result["persistence"] = dict(status="error", content_id=None, reason_code="storage_not_configured")
            result["status"] = "partial"
            return
        try:
            persist(result)  # sets persistence=ok/content_id inside the stored snapshot before commit
        except Exception:
            result["persistence"] = dict(status="error", content_id=None, reason_code="storage_failed")
            result["status"] = "partial"

    def _run(self, rgb, info, mode, method_id, request_id, timing, start) -> dict[str, Any]:
        m, reg, failed = self.models, self.registry, False
        det, att = reg["detection"], reg["attribution"]
        t = time.monotonic()
        try:
            score = m.detect([rgb])[0]
        except Exception:
            raise WorkbenchError("inference_failed", "检测模型推理失败，未产生任何结论", retryable=True) from None
        timing["detection_ms"] = _ms(t)
        threshold = float(det["threshold"])
        reached = score >= threshold
        fixed = m.fixed_metrics
        detection = dict(
            status="ok", model_id=det["model_id"], score=score, score_kind="binary_softmax_p_ai",
            label="ai" if reached else "real", threshold=threshold, threshold_id=det["threshold_id"],
            class_map_sha256=det["classes_sha256"],
            threshold_evidence=dict(role="evaluation", n=fixed["n"], real_n=fixed["negative_count"],
                                    ai_n=fixed["positive_count"], fpr=fixed["fpr"], tpr=fixed["tpr"]))
        attribution = dict(model_id=att["model_id"], label_space="known_ai_22", score_kind="known_ai_source_softmax",
                           conditional_on=dict(detection_label="ai", source_in_known_22=True),
                           conditional_statement=CONDITIONAL_STATEMENT, class_map_sha256=att["classes_sha256"],
                           top_k=[])
        a0_scores = None
        timing["attribution_ms"] = None
        if not reached:
            attribution.update(status="skipped", reason_code="below_detection_threshold")
        else:
            t = time.monotonic()
            try:
                a0_scores = m.attribute([rgb])[0]
                attribution.update(status="ok", reason_code=None, top_k=_ranked(m.a0_classes, a0_scores))
            except Exception:
                failed = True
                attribution.update(status="error", reason_code="inference_failed")
            timing["attribution_ms"] = _ms(t)
        research, timing["research_ms"] = [], None
        if mode == "research":
            t = time.monotonic()
            item = self._research(method_id, rgb, score, a0_scores)
            failed = failed or item["status"] != "ok"
            research.append(item)
            timing["research_ms"] = _ms(t)
        timing["total_ms"] = _ms(start)
        return dict(
            schema_version=SCHEMA_VERSION, request_id=request_id, created_at=utc_now(), mode=mode,
            status="partial" if failed else "ok", bundle_id=reg["bundle_id"], registry_sha256=self.registry_sha256,
            inference_code_commit=self.code_commit, input=info, detection=detection, attribution=attribution,
            unknown=dict(UNKNOWN), research=research,
            runtime=dict(device=self.settings.get("device", "cpu"), precision="fp32", tf32=False,
                         torch_threads=int(self.settings.get("torch_threads", 8)), dependencies=self.versions,
                         preprocessing=dict(detection=det["preprocessing_sha256"],
                                            attribution=att["preprocessing_sha256"]),
                         timing_ms=timing, external_calls=False),
            limitations=list(LIMITATIONS),
            persistence=dict(status="error", content_id=None, reason_code="not_attempted"))

    def _research(self, method_id: str, rgb, score: float, a0_scores) -> dict[str, Any]:
        m, spec = self.models, self.methods[method_id]
        item = dict(method_id=method_id, label=spec["label"], task=spec["task"], status="ok",
                    score_kind=spec["score_kind"], label_space=spec.get("label_space"), outputs={},
                    limitations=list(spec.get("limitations", [])))
        try:
            if spec["kind"] == "calibrated_thresholds":
                rows = []
                for target, entry in sorted(m.calibrated.items(), key=lambda kv: float(kv[0])):
                    ev = entry["evaluation"]
                    rows.append(dict(threshold_id=f"cal-fpr-{float(target):g}-v1", target_fpr=float(target),
                                     threshold=ev["threshold"], label="ai" if score >= ev["threshold"] else "real",
                                     fitted_on="calibration_real_images", calibration_real_n=entry["calibration_negative_n"],
                                     calibration_fpr=entry["calibration_fpr"], evaluation_n=ev["n"],
                                     evaluation_real_n=ev["negative_count"], evaluation_fpr=ev["fpr"],
                                     evaluation_tpr=ev["tpr"]))
                item["outputs"] = dict(p_ai=score, thresholds=rows)
            elif spec["kind"] == "a0_assumed_known_ai":
                scores = a0_scores if a0_scores is not None else m.attribute([rgb])[0]
                item["outputs"] = dict(top_k=_ranked(m.a0_classes, scores), conditional_statement=CONDITIONAL_STATEMENT,
                                       class_map_sha256=spec["classes_sha256"])
            elif spec["kind"] == "dna":
                item["outputs"] = dict(top_k=_ranked(m.a0_classes, m.dna_scores([rgb])[0]),
                                       conditional_statement=CONDITIONAL_STATEMENT,
                                       class_map_sha256=spec["classes_sha256"])
            else:
                raise ValueError("unknown research method kind")
        except Exception:
            item.update(status="error", outputs={}, reason_code="inference_failed")
        return item


def _read_acceptance(settings: dict) -> dict | None:
    spec = settings.get("acceptance_evidence")
    if not spec:
        return None
    path = ROOT / spec["path"]
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        return None
    return json.loads(path.read_text())
