"""文件 metadata 提取。

metadata 是“文件自身属性”，不是 AI 检测结果。
它用于展示、审计、报告和后续溯源，例如：
- 文件名
- 扩展名
- 大小
- 创建/修改时间
- 图片尺寸、格式、EXIF

后续预留：
- 音频时长、采样率、声道数
- 视频时长、帧率、分辨率
- MIME 类型和文件头签名
- 文档类 metadata，例如 PDF 页数、Office 作者字段
- C2PA/EXIF/XMP 这类 provenance metadata 的统一入口
"""

import os
from pathlib import Path
from datetime import datetime


def extract_metadata(path: str | Path) -> dict:
    """提取通用文件 metadata。

    当前所有文件都会提取基础字段；如果是图片，再追加图片 metadata。
    这个结果会被 api/analysis.py 放入最终返回 JSON，并存入报告上下文。
    """
    path = Path(path)
    stat = path.stat()

    meta = {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".webp"}:
        # 图片有额外可用信息，例如尺寸、格式、EXIF。
        meta.update(_extract_image_metadata(path))

    return meta


def _extract_image_metadata(path: Path) -> dict:
    """提取图片尺寸、格式和 EXIF。

    EXIF 里可能包含拍摄设备、软件、时间等信息，
    对 AIGC 溯源有参考价值，但不能单独作为判定依据。
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(path)
        info = {"width": img.width, "height": img.height, "format": img.format}

        exif_data = img.getexif()
        if exif_data:
            exif = {}
            for tag_id, value in exif_data.items():
                # EXIF tag 是数字 ID，这里转成人类可读的名称。
                tag_name = TAGS.get(tag_id, tag_id)
                exif[tag_name] = str(value)
            info["exif"] = exif

        return info
    except Exception:
        # metadata 是辅助信息，提取失败不应该阻断主检测流程。
        return {}
