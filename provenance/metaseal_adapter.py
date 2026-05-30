"""Meta Seal 家族水印检测适配器。

Meta Seal 不是一个单一 Python 包，而是一组按模态拆开的项目：
- image/video: VideoSeal / PixelSeal / ChunkySeal，Python 包通常为 `videoseal`
- audio: AudioSeal，Python 包通常为 `audioseal`
- text: TextSeal，目前更像研究框架/CLI，实际接口可能随版本变化

本文件做“真实适配边界”：
- 依赖安装后会真实调用对应 detector
- 依赖缺失时返回 dependency_missing，不再用 toy 结果冒充
- 每个模态输出统一 status/provider/result/error 字段
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class MetaSealWatermarkDetector:
    """Meta Seal 多模态水印检测入口。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """读取 watermark 配置。"""
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(self.config.get("threshold", 0.5))
        self._audio_detector: Any | None = None
        self._videoseal_model: Any | None = None

    def detect(self, path: str | Path, modality: str) -> dict[str, Any]:
        """按模态检测水印。"""
        path = Path(path)
        if modality == "audio":
            return self.detect_audio(path)
        if modality == "image":
            return self.detect_image(path)
        if modality == "video":
            return self.detect_video(path)
        if modality == "text":
            return self.detect_text(path)
        return {
            "status": "not_applicable",
            "provider": "Meta Seal",
            "result": None,
            "note": f"Unsupported watermark modality: {modality}",
        }

    def detect_audio(self, path: Path) -> dict[str, Any]:
        """调用 AudioSeal 检测音频水印。"""
        cfg = self.config.get("audio", {})
        if not cfg.get("enabled", True):
            return self._disabled("AudioSeal")

        try:
            detector = self._load_audioseal_detector(cfg)
            librosa = self._import_module("librosa")
            audio, sample_rate = librosa.load(str(path), sr=int(cfg.get("sample_rate", 16000)), mono=True)
            wav = torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)
            result, message = detector.detect_watermark(wav)
            confidence = self._to_float(result)
            return {
                "status": "ok",
                "provider": "AudioSeal",
                "result": {
                    "detected": confidence >= float(cfg.get("threshold", self.threshold)),
                    "confidence": round(confidence, 4),
                    "message_bits": self._message_to_bits(message),
                    "sample_rate": sample_rate,
                },
                "note": "AudioSeal detector executed locally.",
            }
        except ImportError as exc:
            return self._dependency_missing("AudioSeal", "audioseal", exc)
        except Exception as exc:  # noqa: BLE001 - watermark errors should not break analysis.
            return self._error("AudioSeal", exc)

    def detect_image(self, path: Path) -> dict[str, Any]:
        """调用 VideoSeal/PixelSeal 检测图片水印。"""
        cfg = self.config.get("image", {})
        if not cfg.get("enabled", True):
            return self._disabled("VideoSeal")

        try:
            model = self._load_videoseal_model(cfg)
            image = Image.open(path).convert("RGB")
            tensor = self._pil_to_tensor(image).to(self.device)
            detected = model.detect(tensor)
            confidence = self._confidence_from_videoseal(detected)
            return {
                "status": "ok",
                "provider": "VideoSeal",
                "result": {
                    "detected": confidence >= float(cfg.get("threshold", self.threshold)),
                    "confidence": round(confidence, 4),
                    "message_bits": self._videoseal_message_bits(detected),
                },
                "note": "VideoSeal image detector executed locally.",
            }
        except ImportError as exc:
            return self._dependency_missing("VideoSeal", "videoseal", exc)
        except Exception as exc:  # noqa: BLE001
            return self._error("VideoSeal", exc)

    def detect_video(self, path: Path) -> dict[str, Any]:
        """调用 VideoSeal 检测视频水印。"""
        cfg = self.config.get("video", {})
        if not cfg.get("enabled", True):
            return self._disabled("VideoSeal")

        try:
            model = self._load_videoseal_model(cfg)
            torchvision = self._import_module("torchvision")
            video, _, info = torchvision.io.read_video(str(path), pts_unit="sec")
            max_frames = int(cfg.get("max_frames", 32))
            if video.shape[0] > max_frames:
                video = video[:max_frames]
            tensor = video.permute(0, 3, 1, 2).float().to(self.device) / 255.0
            detected = model.detect(tensor, is_video=True)
            confidence = self._confidence_from_videoseal(detected)
            return {
                "status": "ok",
                "provider": "VideoSeal",
                "result": {
                    "detected": confidence >= float(cfg.get("threshold", self.threshold)),
                    "confidence": round(confidence, 4),
                    "message_bits": self._videoseal_message_bits(detected),
                    "fps": info.get("video_fps") if isinstance(info, dict) else None,
                    "frames_checked": int(tensor.shape[0]),
                },
                "note": "VideoSeal video detector executed locally.",
            }
        except ImportError as exc:
            return self._dependency_missing("VideoSeal", "videoseal/torchvision", exc)
        except Exception as exc:  # noqa: BLE001
            return self._error("VideoSeal", exc)

    def detect_text(self, path: Path) -> dict[str, Any]:
        """调用 TextSeal CLI 检测文本水印。

        TextSeal 的 Python API 仍在快速变化；这里优先支持配置 CLI 命令。
        如果未配置 command，会返回 dependency_missing/needs_command，避免假装已检测。
        """
        cfg = self.config.get("text", {})
        if not cfg.get("enabled", True):
            return self._disabled("TextSeal")

        command = cfg.get("command")
        if not command:
            return {
                "status": "dependency_missing",
                "provider": "TextSeal",
                "result": None,
                "error": "TextSeal command is not configured.",
                "install_hint": "Install/configure facebookresearch/textseal and set provenance.watermark.text.command.",
            }

        try:
            completed = subprocess.run(
                [str(command), "detect", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=float(cfg.get("timeout_seconds", 60)),
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            payload = json.loads(completed.stdout)
            confidence = float(payload.get("confidence", payload.get("score", 0.0)))
            return {
                "status": "ok",
                "provider": "TextSeal",
                "result": {
                    "detected": bool(payload.get("detected", confidence >= float(cfg.get("threshold", self.threshold)))),
                    "confidence": round(confidence, 4),
                    "raw": payload,
                },
                "note": "TextSeal command executed locally.",
            }
        except Exception as exc:  # noqa: BLE001
            return self._error("TextSeal", exc)

    def _load_audioseal_detector(self, cfg: dict[str, Any]) -> Any:
        """懒加载 AudioSeal detector。"""
        if self._audio_detector is None:
            cache_dir = cfg.get("cache_dir")
            if cache_dir:
                # AudioSeal 会在 AUDIOSEAL_CACHE_DIR 下再创建 audioseal/ 子目录。
                os.environ.setdefault("AUDIOSEAL_CACHE_DIR", str(Path(cache_dir).parent))
            audioseal = self._import_module("audioseal")
            model_name = cfg.get("model", "audioseal_detector_16bits")
            self._audio_detector = audioseal.AudioSeal.load_detector(model_name)
            if hasattr(self._audio_detector, "to"):
                self._audio_detector = self._audio_detector.to(self.device)
            if hasattr(self._audio_detector, "eval"):
                self._audio_detector.eval()
        return self._audio_detector

    def _load_videoseal_model(self, cfg: dict[str, Any]) -> Any:
        """懒加载 VideoSeal/PixelSeal 模型。"""
        if self._videoseal_model is None:
            videoseal = self._import_module("videoseal")
            model_name = cfg.get("model", self.config.get("model", "videoseal"))
            package_root = cfg.get("package_root") or self.config.get("package_root")
            if package_root:
                self._videoseal_model = self._load_videoseal_from_root(
                    videoseal,
                    model_name,
                    Path(package_root),
                )
            else:
                self._videoseal_model = videoseal.load(model_name)
            if hasattr(self._videoseal_model, "to"):
                self._videoseal_model = self._videoseal_model.to(self.device)
            if hasattr(self._videoseal_model, "eval"):
                self._videoseal_model.eval()
        return self._videoseal_model

    def _import_module(self, name: str) -> Any:
        """导入模块。

        单独封装是为了测试时可以 mock，避免加载真实大模型。
        """
        return importlib.import_module(name)

    def _load_videoseal_from_root(self, videoseal: Any, model_name: str, package_root: Path) -> Any:
        """在 VideoSeal 源码根目录下加载模型。

        VideoSeal 的部分版本会用相对路径读取 cards/configs/ckpts，
        所以需要临时切换工作目录到源码根目录。
        """
        old_cwd = Path.cwd()
        try:
            os.chdir(package_root)
            return videoseal.load(model_name)
        finally:
            os.chdir(old_cwd)

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """PIL 图片转 1xCxHxW tensor，避免额外依赖 torchvision transforms。"""
        arr = np.array(image).astype("float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    def _confidence_from_videoseal(self, detected: Any) -> float:
        """从 VideoSeal detect 输出中提取置信度。"""
        if isinstance(detected, dict):
            for key in ["score", "confidence", "probability"]:
                if key in detected:
                    return self._to_float(detected[key])
            preds = detected.get("preds")
            if preds is not None:
                tensor = torch.as_tensor(preds).float()
                if tensor.numel() == 0:
                    return 0.0
                if tensor.ndim >= 2 and tensor.shape[-1] > 1:
                    return float(torch.sigmoid(tensor[..., 1:]).mean().item())
                return float(torch.sigmoid(tensor).mean().item())
        return self._to_float(detected)

    def _videoseal_message_bits(self, detected: Any) -> list[int]:
        """从 VideoSeal 输出中提取隐藏消息 bit。"""
        if not isinstance(detected, dict) or "preds" not in detected:
            return []
        tensor = torch.as_tensor(detected["preds"]).float().flatten()
        if tensor.numel() <= 1:
            return []
        return [int(value > 0) for value in tensor[1:].tolist()]

    def _message_to_bits(self, message: Any) -> list[int]:
        """AudioSeal message tensor 转 bit 列表。"""
        if message is None:
            return []
        tensor = torch.as_tensor(message).detach().cpu().flatten()
        return [int(value > 0.5) for value in tensor.tolist()]

    def _to_float(self, value: Any) -> float:
        """把 tensor/list/scalar 归一成 float。"""
        tensor = torch.as_tensor(value).float()
        if tensor.numel() == 0:
            return 0.0
        return float(tensor.mean().item())

    def _disabled(self, provider: str) -> dict[str, Any]:
        """生成禁用状态。"""
        return {
            "status": "disabled",
            "provider": provider,
            "result": None,
            "note": f"{provider} is disabled in config.",
        }

    def _dependency_missing(self, provider: str, package: str, exc: Exception) -> dict[str, Any]:
        """生成依赖缺失状态。"""
        return {
            "status": "dependency_missing",
            "provider": provider,
            "result": None,
            "error": str(exc),
            "install_hint": f"Install/configure {package} to enable {provider} watermark detection.",
        }

    def _error(self, provider: str, exc: Exception) -> dict[str, Any]:
        """生成错误状态。"""
        return {
            "status": "error",
            "provider": provider,
            "result": None,
            "error": str(exc),
        }
