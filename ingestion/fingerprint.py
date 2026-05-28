import hashlib
from pathlib import Path

import numpy as np


def compute_fingerprint(path: str | Path, modality: str | None = None) -> str:
    path = Path(path)

    if modality is None:
        from .loader import detect_modality
        modality = detect_modality(path)

    if modality == "text":
        return _text_fingerprint(path)
    elif modality == "image":
        return _image_fingerprint(path)
    elif modality == "audio":
        return _audio_fingerprint(path)
    elif modality == "video":
        return _video_fingerprint(path)
    else:
        return _file_hash(path)


def _file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _text_fingerprint(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _image_fingerprint(path: Path) -> str:
    import imagehash
    from PIL import Image

    img = Image.open(path)
    phash = imagehash.phash(img)
    return str(phash)


def _audio_fingerprint(path: Path) -> str:
    import librosa

    y, sr = librosa.load(str(path), sr=16000, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfcc, axis=1)
    fingerprint_bytes = mean_mfcc.tobytes()
    return hashlib.sha256(fingerprint_bytes).hexdigest()


def _video_fingerprint(path: Path) -> str:
    import cv2
    import imagehash
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    hashes = []
    frame_count = 0

    while cap.isOpened() and len(hashes) < 10:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 30 == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hashes.append(str(imagehash.phash(pil_img)))
        frame_count += 1

    cap.release()
    combined = "|".join(hashes)
    return hashlib.sha256(combined.encode()).hexdigest()
