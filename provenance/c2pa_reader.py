"""C2PA metadata 读取占位实现。

C2PA 是内容凭证/来源链相关标准。理想情况下，我们应该使用官方
`c2patool` 或成熟库来验证 manifest、签名、assertions。

当前这个文件只是轻量 parser：
- 尝试从 JPEG APP11 / PNG caBX 中找 C2PA/JUMBF 数据
- 尝试把找到的 bytes 当 JSON 解析
- 不做完整签名验证
- 不支持完整 C2PA manifest store 解析

后续建议：
- 新增 `provenance/c2pa_tool.py`
- 用 subprocess 调用 `c2patool file --json`
- 归一化输出到 pipeline 当前使用的 JSON 格式
"""

import json
import struct
from pathlib import Path
from typing import Optional


def read_c2pa_metadata(path: str | Path) -> Optional[dict]:
    """读取 C2PA metadata。

    返回 None 表示没找到或解析失败；
    返回 dict 表示提取到 claim_generator/title/assertions/signature_info 等字段。
    """
    path = Path(path)
    raw = _extract_c2pa_box(path)
    if raw is None:
        return None

    try:
        manifest = json.loads(raw)
        return {
            "claim_generator": manifest.get("claim_generator"),
            "title": manifest.get("title"),
            "assertions": manifest.get("assertions", []),
            "signature_info": manifest.get("signature_info"),
        }
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _extract_c2pa_box(path: Path) -> Optional[bytes]:
    """尝试从 JPEG/PNG 中提取 C2PA/JUMBF 原始 bytes。"""
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return _extract_from_jpeg(path)
    elif suffix == ".png":
        return _extract_from_png(path)
    return None


def _extract_from_jpeg(path: Path) -> Optional[bytes]:
    """从 JPEG APP11 marker 中寻找 JUMBF 数据。

    这是简化实现，只用于 MVP 占位，不等价于完整 C2PA 解析器。
    """
    data = path.read_bytes()
    # C2PA 通常使用 APP11 marker (0xFFEB) 携带 JUMBF。
    marker = b"\xff\xeb"
    idx = data.find(marker)
    if idx == -1:
        return None

    length = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
    payload = data[idx + 4 : idx + 2 + length]

    jumbf_magic = b"JP\x00\x00"
    if jumbf_magic not in payload:
        return None

    jumbf_start = payload.find(jumbf_magic)
    return payload[jumbf_start:]


def _extract_from_png(path: Path) -> Optional[bytes]:
    """从 PNG caBX chunk 中寻找 C2PA 数据。

    这是简化实现，后续应由 c2patool 替代。
    """
    data = path.read_bytes()
    # PNG 中 C2PA 可能使用 caBX chunk。
    chunk_type = b"caBX"
    idx = data.find(chunk_type)
    if idx == -1:
        return None

    length = struct.unpack(">I", data[idx - 4 : idx])[0]
    return data[idx + 4 : idx + 4 + length]
