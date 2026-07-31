"""OpenAI-compatible gateway boundary for controlled text generation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import requests

from .config import GatewayConfig


@dataclass(frozen=True)
class GenerationResult:
    gateway: str
    request_model: str
    response_model: str | None
    response_id: str | None
    gateway_request_id: str | None
    text: str
    finish_reason: str | None
    created: int | None
    usage: dict[str, Any]
    elapsed_seconds: float

    def to_record(self) -> dict[str, Any]:
        return {
            "gateway": self.gateway,
            "request_model": self.request_model,
            "response_model": self.response_model,
            "response_id": self.response_id,
            "gateway_request_id": self.gateway_request_id,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "created": self.created,
            "usage": self.usage,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


class OpenAICompatibleGateway:
    """Minimal gateway client with no secret-bearing response records."""

    def __init__(self, config: GatewayConfig, session: Any | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def list_models(self) -> list[str]:
        response = self._request("GET", "/models")
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("Model-list response must contain a `data` list")
        return sorted(
            {
                str(row["id"])
                for row in rows
                if isinstance(row, dict) and row.get("id")
            }
        )

    def generate(
        self,
        *,
        model: str,
        user_prompt: str,
        system_prompt: str | None,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
    ) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if top_p is not None:
            body["top_p"] = top_p
        started = time.perf_counter()
        response = self._request("POST", "/chat/completions", json=body)
        elapsed = time.perf_counter() - started
        payload = response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("Chat completion response must contain at least one choice")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        text = _normalize_content(content)
        if not text.strip():
            raise ValueError("Chat completion returned empty text")
        return GenerationResult(
            gateway=self.config.name,
            request_model=model,
            response_model=_optional_string(payload.get("model")),
            response_id=_optional_string(payload.get("id")),
            gateway_request_id=_request_id(response.headers),
            text=text,
            finish_reason=_optional_string(choice.get("finish_reason")),
            created=payload.get("created") if isinstance(payload.get("created"), int) else None,
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
            elapsed_seconds=elapsed,
        )

    def _request(self, method: str, path: str, **kwargs):
        response = self.session.request(
            method,
            f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}",
            headers=self.config.authorization_headers(),
            timeout=self.config.timeout_seconds,
            **kwargs,
        )
        response.raise_for_status()
        return response


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _request_id(headers: Any) -> str | None:
    for key in ("x-request-id", "request-id", "x-trace-id"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value:
            return str(value)
    return None
