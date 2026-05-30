"""内容指纹计算。

fingerprint 用于“识别同一内容或相似内容”，和 AI 检测分数不是一回事。
它后续可以用于：
- 去重
- 历史记录匹配
- 指纹库检索
- 已知 AI 内容 registry lookup
- 溯源链路中的内容身份标识

不同模态的指纹策略不同：
- text: 规范化文本后哈希
- image: 感知哈希 phash
- audio: MFCC 特征哈希
- video: 抽帧 phash 后组合哈希
"""

import hashlib
from pathlib import Path

import numpy as np


def compute_fingerprint(path: str | Path, modality: str | None = None) -> str:
    """根据模态计算内容指纹。

    modality 可以由上游传入；如果没传，就调用 detect_modality 自动判断。
    返回值是字符串，最终会进入 API 返回结果和数据库 Content.fingerprint。
    """
    path = Path(path)

    if modality is None:
        # 避免文件顶部导入 loader 造成循环导入，所以这里局部导入。
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
    """计算原始文件 SHA-256。

    这是最保守的文件身份标识：只要文件任何一个字节变化，hash 就会变化。
    对未知类型或兜底场景很有用。
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        # 分块读取，避免大文件一次性读入内存。
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _text_fingerprint(path: Path) -> str:
    """计算文本内容指纹。

    会先统一小写并折叠空白字符，这样简单的换行/多空格变化不会导致完全不同的指纹。
    """
    text = path.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _image_fingerprint(path: Path) -> str:
    """计算图片感知哈希。

    phash 能在图片轻微压缩、缩放、格式变化时保持一定相似性，
    比原始 SHA-256 更适合图片相似匹配。
    """
    import imagehash
    from PIL import Image

    img = Image.open(path)
    phash = imagehash.phash(img)
    return str(phash)


def _audio_fingerprint(path: Path) -> str:
    """计算音频特征指纹。

    当前取前 30 秒音频，提取 MFCC 后对均值向量做 SHA-256。
    这是轻量占位方案，后续可以替换为更专业的 audio embedding。
    """
    import librosa

    y, sr = librosa.load(str(path), sr=16000, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfcc, axis=1)
    fingerprint_bytes = mean_mfcc.tobytes()
    return hashlib.sha256(fingerprint_bytes).hexdigest()


def _video_fingerprint(path: Path) -> str:
    """计算视频抽帧指纹。

    当前最多抽取 10 帧，每 30 帧取一次，对每帧计算图片 phash，
    再把这些帧 hash 拼起来做 SHA-256。
    """
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
            # OpenCV 读取是 BGR，需要转成 RGB 后再交给 PIL/imagehash。
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hashes.append(str(imagehash.phash(pil_img)))
        frame_count += 1

    cap.release()
    combined = "|".join(hashes)
    return hashlib.sha256(combined.encode()).hexdigest()
