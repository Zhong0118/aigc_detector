"""C2PA metadata 读取实现。

C2PA 是内容凭证/来源链相关标准。理想情况下，我们应该使用官方
`c2patool` 或成熟库来验证 manifest、签名、assertions。

当前实现优先调用本机 `c2patool`：
- 如果 c2patool 可用：执行真实 JSON 解析，尽量读取 claim、assertions、signature_info
- 如果 c2patool 不可用：回退到轻量 JPEG/PNG 字节扫描
- 如果没有 C2PA：返回 None，让 pipeline 标记 found=False
"""

import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional


def read_c2pa_metadata(path: str | Path, tool_path: str | None = None) -> Optional[dict]:
    """读取 C2PA metadata。

    返回 None 表示没找到或解析失败；
    返回 dict 表示提取到 claim_generator/title/assertions/signature_info 等字段。
    """
    path = Path(path)
    tool_result = _read_with_c2patool(path, tool_path=tool_path)
    if tool_result is not None:
        return tool_result

    raw = _extract_c2pa_box(path)
    if raw is None:
        return None

    try:
        manifest = json.loads(raw)
        return {
            "parser": "lightweight_box_scan",
            "found": True,
            "claim_generator": manifest.get("claim_generator"),
            "title": manifest.get("title"),
            "assertions": manifest.get("assertions", []),
            "signature_info": manifest.get("signature_info"),
        }
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _read_with_c2patool(path: Path, tool_path: str | None = None) -> Optional[dict]:
    """优先调用官方 c2patool 读取 C2PA。

    c2patool 不一定安装在演示环境中，所以这里不能让缺工具导致整个分析失败。
    """
    tool = tool_path or shutil.which("c2patool")
    if not tool:
        return None

    completed = _run_c2patool_command(tool, path, use_json_flag=True)
    if completed is not None and completed.returncode != 0 and "unexpected argument" in completed.stderr:
        # c2patool 0.9.x 直接输出 JSON，不支持新版 `--json` 参数。
        completed = _run_c2patool_command(tool, path, use_json_flag=False)
    if completed is None:
        return None

    if completed.returncode != 0 and not completed.stdout.strip():
        return None

    if not completed.stdout.strip():
        return None

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None

    manifest = _active_manifest(payload)
    if manifest is None:
        return None

    return {
        "parser": "c2patool",
        "found": True,
        "active_manifest": payload.get("active_manifest"),
        "claim_generator": manifest.get("claim_generator"),
        "title": manifest.get("title"),
        "format": manifest.get("format"),
        "assertions": manifest.get("assertions", []),
        "ingredients": manifest.get("ingredients", []),
        "signature_info": manifest.get("signature_info"),
        "validation_status": payload.get("validation_status") or manifest.get("validation_status"),
        "raw": payload,
    }


def _run_c2patool_command(tool: str, path: Path, use_json_flag: bool) -> subprocess.CompletedProcess | None:
    """运行 c2patool，兼容新版 --json 和旧版默认 JSON 输出。"""
    command = [tool, str(path), "--json"] if use_json_flag else [tool, str(path)]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _active_manifest(payload: dict) -> Optional[dict]:
    """从 c2patool JSON 中提取 active manifest。"""
    manifests = payload.get("manifests")
    if isinstance(manifests, dict):
        active_key = payload.get("active_manifest")
        if active_key and isinstance(manifests.get(active_key), dict):
            return manifests[active_key]
        for manifest in manifests.values():
            if isinstance(manifest, dict):
                return manifest
    if isinstance(payload.get("manifest"), dict):
        return payload["manifest"]
    if any(key in payload for key in ["claim_generator", "assertions", "signature_info"]):
        return payload
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
