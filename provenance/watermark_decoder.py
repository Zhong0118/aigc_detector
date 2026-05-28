from typing import Optional

import numpy as np
from PIL import Image


def decode_watermark(image: Image.Image) -> Optional[dict]:
    """Attempt to decode invisible watermarks from an image.

    Currently supports LSB-based watermark detection.
    SynthID and other proprietary watermarks require model-specific decoders.
    """
    arr = np.array(image)
    lsb_result = _detect_lsb_watermark(arr)

    if lsb_result:
        return {"method": "lsb", "detected": True, "data": lsb_result}

    return {"method": "none", "detected": False, "data": None}


def _detect_lsb_watermark(arr: np.ndarray) -> Optional[str]:
    """Check for LSB steganography patterns."""
    if arr.ndim < 3:
        return None

    lsb = arr[:, :, 0] & 1
    entropy = _compute_binary_entropy(lsb)

    # Random-looking LSB plane suggests no simple watermark
    # Structured LSB plane may indicate embedded data
    if entropy < 0.95:
        bits = lsb.flatten()[:64]
        header = bits_to_bytes(bits)
        try:
            return header.decode("ascii", errors="ignore")
        except Exception:
            return None

    return None


def _compute_binary_entropy(arr: np.ndarray) -> float:
    p1 = np.mean(arr)
    p0 = 1 - p1
    if p0 == 0 or p1 == 0:
        return 0.0
    return -(p0 * np.log2(p0) + p1 * np.log2(p1))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    result = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | int(bits[i + j])
        result.append(byte_val)
    return bytes(result)
