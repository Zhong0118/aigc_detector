"""Select a project configuration without changing the server configuration."""

import os
from pathlib import Path


def resolve_config_path(explicit: str | None = None) -> Path:
    selected = explicit or os.environ.get("AIGC_CONFIG_PATH")
    return Path(selected) if selected else Path(__file__).resolve().parent / "config.yaml"
