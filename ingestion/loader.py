"""内容加载与模态识别。

这个文件是 ingestion 层的入口之一，负责回答两个问题：
1. 这个文件是什么类型？text / image / audio / video
2. 这个文件如何读成后续检测器能使用的 Python 对象？

当前模态识别主要依赖文件扩展名。后续更稳的做法是增加 MIME sniffing
或文件头签名识别，避免用户改扩展名导致误判。

后续建议补充：
- 文件大小限制，避免超大视频/音频拖垮服务
- MIME 类型识别，例如 python-magic
- 文件头签名识别，例如 PNG/JPEG/MP4 magic bytes
- 编码容错，例如文本自动识别 UTF-8/GBK
- 上传文件安全检查，例如禁止路径穿越和危险扩展名
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 扩展名到模态的映射表。
# FastAPI 上传文件后，会先保存成临时文件，再通过这个表判断走哪个检测分支。
MODALITY_MAP = {
    "text": {".txt", ".md", ".csv", ".json", ".html"},
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".webm"},
}


@dataclass
class ContentItem:
    """加载后的统一内容对象。

    path: 文件路径
    modality: 内容模态，例如 text/image/audio/video
    raw_data: 已读取的原始数据，不同模态类型不同
    metadata: 预留字段，当前 metadata 在 metadata.py 单独提取
    fingerprint: 预留字段，当前 fingerprint 在 fingerprint.py 单独计算
    """

    path: Path
    modality: str
    raw_data: object = field(repr=False)
    metadata: Optional[dict] = None
    fingerprint: Optional[str] = None


def detect_modality(path: Path) -> str:
    """根据文件扩展名判断内容模态。

    返回值会影响后续路由：
    - text -> 文本检测 API/模型
    - image -> 图片检测 API/模型
    - audio -> 音频检测 API/模型
    - video -> 视频检测 API/模型

    后续预留：这里可以加入 `python-magic` 或文件头识别，
    让判断不只依赖后缀名。
    """
    suffix = path.suffix.lower()
    for modality, extensions in MODALITY_MAP.items():
        if suffix in extensions:
            return modality
    raise ValueError(f"Unsupported file type: {suffix}")


def load_content(path: str | Path) -> ContentItem:
    """读取文件内容并返回统一 ContentItem。

    这个函数会先调用 detect_modality 判断类型，再按类型读取：
    - text: 读取为字符串
    - image: 读取为 PIL.Image RGB 对象
    - audio: 用 librosa 读取为 16000Hz numpy 数组
    - video: 用 OpenCV VideoCapture 打开视频
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    modality = detect_modality(path)

    if modality == "text":
        # 文本直接读成 str，后续可以传给文本检测 API 或本地文本模型。
        raw_data = path.read_text(encoding="utf-8")
    elif modality == "image":
        # 图片统一转 RGB，避免 PNG 透明通道、灰度图等格式差异影响后续检测器。
        from PIL import Image
        raw_data = Image.open(path).convert("RGB")
    elif modality == "audio":
        # 音频统一重采样到 16kHz，这是很多语音/音频模型常用采样率。
        import librosa
        raw_data, _ = librosa.load(str(path), sr=16000)
    elif modality == "video":
        # 视频先返回 VideoCapture，真正抽帧可以交给 video detector 或后续 API adapter。
        import cv2
        raw_data = cv2.VideoCapture(str(path))
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return ContentItem(path=path, modality=modality, raw_data=raw_data)
