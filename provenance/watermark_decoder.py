"""水印检测占位实现。

当前只有一个非常轻量的图片 LSB 检测器，用于演示 watermark 模块位置。
它不是 Meta Seal、不是 SynthID，也不是通用 AIGC 水印检测器。

后续建议：
- 新增 `metaseal_adapter.py`
- 针对 text/audio/video/image 分别接 TextSeal、AudioSeal、VideoSeal、image watermark
- 输出统一字段：detected、confidence、provider、payload/segments
"""

from typing import Optional

import numpy as np
from PIL import Image


def decode_watermark(image: Image.Image) -> Optional[dict]:
    """尝试检测图片中的简单 LSB 水印。

    当前只支持 LSB pattern 检测。真实项目里，SynthID、Meta Seal
    或其他水印都需要各自的模型/decoder。
    """
    arr = np.array(image)
    lsb_result = _detect_lsb_watermark(arr)

    if lsb_result:
        return {"method": "lsb", "detected": True, "data": lsb_result}

    return {"method": "none", "detected": False, "data": None}


def _detect_lsb_watermark(arr: np.ndarray) -> Optional[str]:
    """检查最低有效位 LSB 是否存在简单结构化信息。"""
    if arr.ndim < 3:
        return None

    lsb = arr[:, :, 0] & 1
    entropy = _compute_binary_entropy(lsb)

    # 随机 LSB 平面通常表示没有简单 LSB 水印；
    # 低熵/结构化 LSB 平面可能表示嵌入了信息。
    if entropy < 0.95:
        bits = lsb.flatten()[:64]
        header = bits_to_bytes(bits)
        try:
            return header.decode("ascii", errors="ignore")
        except Exception:
            return None

    return None


def _compute_binary_entropy(arr: np.ndarray) -> float:
    """计算二值数组熵。

    熵越接近 1，0/1 分布越随机；熵越低，越可能存在结构。
    """
    p1 = np.mean(arr)
    p0 = 1 - p1
    if p0 == 0 or p1 == 0:
        return 0.0
    return -(p0 * np.log2(p0) + p1 * np.log2(p1))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """把 0/1 bit 序列按 8 位一组转成 bytes。"""
    result = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | int(bits[i + j])
        result.append(byte_val)
    return bytes(result)
