"""Secret-safe configuration for text-attribution data gateways."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


GATEWAY_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "closeai": (
        "https://api.openai-proxy.org/v1",
        "CLOSEAI_API_KEY",
        "CLOSEAI_BASE_URL",
    ),
    "zhizengzeng": (
        "https://api.zhizengzeng.com/v1",
        "ZZZ_API_KEY",
        "ZZZ_BASE_URL",
    ),
}


@dataclass(frozen=True, repr=False)
class GatewayConfig:
    """Resolved gateway configuration whose representation redacts the key."""

    name: str
    base_url: str
    api_key_env: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        is_local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and not is_local:
            raise ValueError("Remote gateway base_url must use HTTPS")
        if not self.api_key.strip():
            raise ValueError(f"Missing API key in environment variable {self.api_key_env}")

    def __repr__(self) -> str:
        return (
            "GatewayConfig("
            f"name={self.name!r}, "
            f"base_url={self.base_url!r}, "
            f"api_key_env={self.api_key_env!r}, "
            "api_key=<redacted>, "
            f"timeout_seconds={self.timeout_seconds!r}"
            ")"
        )

    def authorization_headers(self) -> dict[str, str]:
        """Create authorization headers only at the HTTP request boundary."""

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def load_env_file(path: str | Path) -> dict[str, str]:
    """Read a small dotenv-style file without logging or mutating global state."""

    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"Invalid environment entry at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid environment key at line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_local_environment(path: str | Path = ".env.local") -> dict[str, str]:
    """Merge `.env.local` below real process variables; process values win."""

    return {**load_env_file(path), **os.environ}


def resolve_gateway(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> GatewayConfig:
    """Resolve one supported OpenAI-compatible gateway from environment values."""

    normalized = name.strip().lower()
    if normalized not in GATEWAY_DEFAULTS:
        raise ValueError(f"Unsupported gateway: {name}")
    default_url, key_env, url_env = GATEWAY_DEFAULTS[normalized]
    values = dict(os.environ if environ is None else environ)
    api_key = values.get(key_env, "")
    if not api_key.strip():
        raise ValueError(f"Missing API key in environment variable {key_env}")
    return GatewayConfig(
        name=normalized,
        base_url=values.get(url_env, default_url).rstrip("/"),
        api_key_env=key_env,
        api_key=api_key,
    )

