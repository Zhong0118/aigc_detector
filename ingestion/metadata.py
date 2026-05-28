import os
from pathlib import Path
from datetime import datetime


def extract_metadata(path: str | Path) -> dict:
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
        meta.update(_extract_image_metadata(path))

    return meta


def _extract_image_metadata(path: Path) -> dict:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(path)
        info = {"width": img.width, "height": img.height, "format": img.format}

        exif_data = img.getexif()
        if exif_data:
            exif = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif[tag_name] = str(value)
            info["exif"] = exif

        return info
    except Exception:
        return {}
