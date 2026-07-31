from __future__ import annotations

import unittest

from scripts.test_closeai_connectivity import run_connectivity_probe


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = {"content-type": "application/json"}

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return next(self.responses)


class CloseAiConnectivityTests(unittest.TestCase):
    def test_probe_uses_official_models_and_chat_routes(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(200, {"data": [{"id": "gpt-3.5-turbo"}]}),
                _FakeResponse(
                    200,
                    {
                        "id": "chatcmpl-test",
                        "model": "gpt-3.5-turbo",
                        "choices": [{"message": {"content": "Hi"}}],
                    },
                ),
            ]
        )

        report = run_connectivity_probe(
            api_key="sk-test-secret",
            base_url="https://api.openai-proxy.org/v1",
            model="gpt-3.5-turbo",
            session=session,
        )

        self.assertEqual(
            [call["url"] for call in session.calls],
            [
                "https://api.openai-proxy.org/v1/models",
                "https://api.openai-proxy.org/v1/chat/completions",
            ],
        )
        self.assertEqual(report["models"]["status"], 200)
        self.assertEqual(report["chat"]["status"], 200)
        self.assertEqual(report["chat"]["response_model"], "gpt-3.5-turbo")
        self.assertEqual(report["chat"]["text_preview"], "Hi")

    def test_probe_redacts_api_key_from_error_payload(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(
                    401,
                    {
                        "error": {
                            "code": "invalid_credentials",
                            "message": "bad token sk-test-secret",
                        }
                    },
                ),
                _FakeResponse(
                    401,
                    {
                        "error": {
                            "code": "invalid_credentials",
                            "message": "bad token sk-test-secret",
                        }
                    },
                ),
            ]
        )

        report = run_connectivity_probe(
            api_key="sk-test-secret",
            base_url="https://api.openai-proxy.org/v1",
            model="gpt-3.5-turbo",
            session=session,
        )

        rendered = str(report)
        self.assertNotIn("sk-test-secret", rendered)
        self.assertIn("<redacted>", rendered)


if __name__ == "__main__":
    unittest.main()
