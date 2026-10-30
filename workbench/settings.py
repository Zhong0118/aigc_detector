"""Read the `workbench:` section of the active config (AIGC_CONFIG, default config.yaml)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def config_path() -> str:
    return os.getenv("AIGC_CONFIG", "config.yaml")


def load_config(path: str | None = None) -> dict[str, Any]:
    path = Path(path or config_path())
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def workbench_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Empty dict means the image adapter is not configured (images then fail closed)."""
    return dict((config if config is not None else load_config()).get("workbench") or {})
