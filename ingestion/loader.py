from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MODALITY_MAP = {
    "text": {".txt", ".md", ".csv", ".json", ".html"},
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".webm"},
}


@dataclass
class ContentItem:
    path: Path
    modality: str
    raw_data: object = field(repr=False)
    metadata: Optional[dict] = None
    fingerprint: Optional[str] = None


def detect_modality(path: Path) -> str:
    suffix = path.suffix.lower()
    for modality, extensions in MODALITY_MAP.items():
        if suffix in extensions:
            return modality
    raise ValueError(f"Unsupported file type: {suffix}")


def load_content(path: str | Path) -> ContentItem:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    modality = detect_modality(path)

    if modality == "text":
        raw_data = path.read_text(encoding="utf-8")
    elif modality == "image":
        from PIL import Image
        raw_data = Image.open(path).convert("RGB")
    elif modality == "audio":
        import librosa
        raw_data, _ = librosa.load(str(path), sr=16000)
    elif modality == "video":
        import cv2
        raw_data = cv2.VideoCapture(str(path))
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return ContentItem(path=path, modality=modality, raw_data=raw_data)
