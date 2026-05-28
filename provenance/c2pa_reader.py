import json
import struct
from pathlib import Path
from typing import Optional


def read_c2pa_metadata(path: str | Path) -> Optional[dict]:
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
    """Try to find C2PA JUMBF box in JPEG/PNG files."""
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return _extract_from_jpeg(path)
    elif suffix == ".png":
        return _extract_from_png(path)
    return None


def _extract_from_jpeg(path: Path) -> Optional[bytes]:
    data = path.read_bytes()
    # C2PA uses APP11 marker (0xFFEB) with JUMBF
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
    data = path.read_bytes()
    # C2PA in PNG uses caBX chunk
    chunk_type = b"caBX"
    idx = data.find(chunk_type)
    if idx == -1:
        return None

    length = struct.unpack(">I", data[idx - 4 : idx])[0]
    return data[idx + 4 : idx + 4 + length]
