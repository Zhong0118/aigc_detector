"""Small, secret-safe CloseAI connectivity probe.

This script is intentionally independent from the attribution collector. It
checks the two official OpenAI-compatible routes and prints a JSON report:

    python -m scripts.test_closeai_connectivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any

import requests


DEFAULT_BASE_URL = "https://api.openai-proxy.org/v1"
DEFAULT_MODEL = "gpt-3.5-turbo"


def run_connectivity_probe(
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    max_tokens: int = 16,
    session: Any | None = None,
) -> dict[str, Any]:
    """Probe `/models` and `/chat/completions` without exposing the key."""

    if not api_key.strip():
        raise ValueError("CLOSEAI_API_KEY is missing")
    http = session or requests.Session()
    root = base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    models = _request_report(
        http,
        "GET",
        f"{root}/models",
        headers=headers,
        timeout=timeout,
        api_key=api_key,
        kind="models",
    )
    chat = _request_report(
        http,
        "POST",
        f"{root}/chat/completions",
        headers=headers,
        timeout=timeout,
        api_key=api_key,
        kind="chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    return {
        "base_url": root,
        "requested_model": model,
        "models": models,
        "chat": chat,
    }


def _request_report(
    session: Any,
    method: str,
    url: str,
    *,
    api_key: str,
    kind: str,
    **kwargs: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = session.request(method, url, **kwargs)
    except requests.RequestException as exc:
        return {
            "endpoint": url,
            "ok": False,
            "network_error": _redact(str(exc), api_key),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    elapsed = round(time.perf_counter() - started, 3)
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = {"raw": getattr(response, "text", "")[:500]}
    safe_payload = _redact_payload(payload, api_key)
    report: dict[str, Any] = {
        "endpoint": url,
        "ok": 200 <= response.status_code < 300,
        "status": response.status_code,
        "elapsed_seconds": elapsed,
    }
    if not report["ok"]:
        report["error"] = safe_payload
        return report

    if kind == "models":
        rows = safe_payload.get("data") if isinstance(safe_payload, dict) else None
        report["model_count"] = len(rows) if isinstance(rows, list) else None
    else:
        choices = safe_payload.get("choices") if isinstance(safe_payload, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else {}
        message = first.get("message") if isinstance(first, dict) else {}
        text = message.get("content") if isinstance(message, dict) else None
        report["response_model"] = safe_payload.get("model")
        report["text_preview"] = text[:160] if isinstance(text, str) else None
    return report


def _redact_payload(payload: Any, api_key: str) -> Any:
    rendered = json.dumps(payload, ensure_ascii=False)
    return json.loads(_redact(rendered, api_key))


def _redact(value: str, api_key: str) -> str:
    if api_key:
        value = value.replace(api_key, "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", value)


def _load_closeai_key(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"Environment file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "CLOSEAI_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value
    raise ValueError("CLOSEAI_API_KEY is missing from the environment file")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test CloseAI connectivity safely.")
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    try:
        report = run_connectivity_probe(
            api_key=_load_closeai_key(args.env_file),
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
    except ValueError as exc:
        print(json.dumps({"ok": False, "configuration_error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["chat"].get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
