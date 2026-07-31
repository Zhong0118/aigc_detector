from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from research.text_attribution.collection import (
    CollectionParameters,
    build_request_plan,
    write_jsonl_record,
)
from research.text_attribution.config import GatewayConfig, load_env_file, resolve_gateway
from research.text_attribution.gateway import OpenAICompatibleGateway
from research.text_attribution.prompts import PILOT_PROMPTS
from scripts.collect_attribution_pilot import main as collection_main


class GatewayConfigurationTests(unittest.TestCase):
    def test_env_file_loads_allowlisted_values_without_exposing_secret_in_repr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env.local"
            env_path.write_text(
                "CLOSEAI_API_KEY=secret-closeai-value\n"
                "CLOSEAI_BASE_URL=https://relay.example/v1\n",
                encoding="utf-8",
            )

            values = load_env_file(env_path)
            config = resolve_gateway("closeai", environ=values)

        self.assertEqual(config.base_url, "https://relay.example/v1")
        self.assertEqual(config.api_key_env, "CLOSEAI_API_KEY")
        self.assertNotIn("secret-closeai-value", repr(config))
        self.assertIn("<redacted>", repr(config))

    def test_missing_key_names_the_environment_variable_but_not_a_secret(self) -> None:
        with self.assertRaisesRegex(ValueError, "ZZZ_API_KEY") as caught:
            resolve_gateway("zhizengzeng", environ={})

        self.assertNotIn("Bearer", str(caught.exception))

    def test_gateway_rejects_non_https_remote_base_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            GatewayConfig(
                name="unsafe",
                base_url="http://remote.example/v1",
                api_key_env="UNSAFE_KEY",
                api_key="secret",
            )


class PromptBankTests(unittest.TestCase):
    def test_pilot_prompt_ids_are_unique_and_cover_required_factors(self) -> None:
        ids = [prompt.prompt_id for prompt in PILOT_PROMPTS]
        languages = {prompt.language for prompt in PILOT_PROMPTS}
        tasks = {prompt.task_type for prompt in PILOT_PROMPTS}
        domains = {prompt.domain for prompt in PILOT_PROMPTS}
        lengths = {prompt.target_length for prompt in PILOT_PROMPTS}

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(languages, {"zh", "en"})
        self.assertGreaterEqual(len(tasks), 6)
        self.assertGreaterEqual(len(domains), 5)
        self.assertGreaterEqual(len(lengths), 3)
        self.assertTrue(all(prompt.user_prompt.strip() for prompt in PILOT_PROMPTS))


class FakeResponse:
    def __init__(self, payload: dict, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}
        self.status_code = 200
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


class GatewayClientTests(unittest.TestCase):
    def test_chat_completion_omits_top_p_when_sampling_value_is_none(self) -> None:
        session = FakeSession(
            FakeResponse(
                {
                    "id": "chatcmpl-test",
                    "model": "claude-haiku",
                    "choices": [
                        {
                            "message": {"content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        )
        gateway = OpenAICompatibleGateway(
            GatewayConfig(
                name="zhizengzeng",
                base_url="https://relay.example/v1",
                api_key_env="ZZZ_API_KEY",
                api_key="secret-zzz-value",
            ),
            session=session,
        )

        gateway.generate(
            model="claude-haiku",
            user_prompt="Reply OK",
            system_prompt=None,
            temperature=0.0,
            top_p=None,
            max_tokens=16,
        )

        self.assertNotIn("top_p", session.calls[0]["json"])

    def test_chat_completion_records_model_identity_without_secret_or_headers(self) -> None:
        response = FakeResponse(
            {
                "id": "chatcmpl-test",
                "created": 1785060000,
                "model": "upstream-model-version",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "A controlled response."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
            headers={"x-request-id": "gateway-request-123"},
        )
        session = FakeSession(response)
        config = GatewayConfig(
            name="closeai",
            base_url="https://relay.example/v1",
            api_key_env="CLOSEAI_API_KEY",
            api_key="secret-closeai-value",
        )
        gateway = OpenAICompatibleGateway(config, session=session)

        result = gateway.generate(
            model="requested-model-alias",
            user_prompt="Write one controlled answer.",
            system_prompt=None,
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
        )
        record = result.to_record()

        self.assertEqual(record["request_model"], "requested-model-alias")
        self.assertEqual(record["response_model"], "upstream-model-version")
        self.assertEqual(record["gateway_request_id"], "gateway-request-123")
        self.assertEqual(record["text"], "A controlled response.")
        self.assertNotIn("secret-closeai-value", json.dumps(record))
        self.assertNotIn("headers", record)
        self.assertEqual(
            session.calls[0]["json"]["messages"],
            [{"role": "user", "content": "Write one controlled answer."}],
        )

    def test_model_listing_normalizes_openai_compatible_response(self) -> None:
        session = FakeSession(
            FakeResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": "model-b", "object": "model"},
                        {"id": "model-a", "object": "model"},
                    ],
                }
            )
        )
        gateway = OpenAICompatibleGateway(
            GatewayConfig(
                name="zhizengzeng",
                base_url="https://relay.example/v1",
                api_key_env="ZZZ_API_KEY",
                api_key="secret-zzz-value",
            ),
            session=session,
        )

        models = gateway.list_models()

        self.assertEqual(models, ["model-a", "model-b"])
        self.assertEqual(session.calls[0]["method"], "GET")
        self.assertTrue(session.calls[0]["url"].endswith("/models"))


class CollectionPlanTests(unittest.TestCase):
    def test_repetitions_create_distinct_repeat_indexes(self) -> None:
        plan = build_request_plan(
            gateway_name="zhizengzeng",
            models=["model-a"],
            prompts=PILOT_PROMPTS[:1],
            parameters=[
                CollectionParameters(temperature=0.0, top_p=1.0, max_tokens=128)
            ],
            repetitions=2,
            max_requests=2,
        )

        self.assertEqual([item.repeat_index for item in plan], [0, 1])

    def test_request_plan_is_deterministic_and_enforces_hard_limit(self) -> None:
        parameters = [
            CollectionParameters(temperature=0.0, top_p=1.0, max_tokens=128),
            CollectionParameters(temperature=0.8, top_p=0.95, max_tokens=128),
        ]

        plan = build_request_plan(
            gateway_name="closeai",
            models=["model-a", "model-b"],
            prompts=PILOT_PROMPTS[:3],
            parameters=parameters,
            max_requests=5,
        )

        self.assertEqual(len(plan), 5)
        self.assertEqual(plan[0].model, "model-a")
        self.assertEqual(plan[0].prompt.prompt_id, PILOT_PROMPTS[0].prompt_id)
        self.assertEqual(plan[0].parameters.temperature, 0.0)
        self.assertEqual(plan, build_request_plan(
            gateway_name="closeai",
            models=["model-a", "model-b"],
            prompts=PILOT_PROMPTS[:3],
            parameters=parameters,
            max_requests=5,
        ))

    def test_jsonl_writer_rejects_secret_shaped_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "records.jsonl"
            with self.assertRaisesRegex(ValueError, "secret"):
                write_jsonl_record(output, {"text": "safe", "api_key": "must-not-write"})

    def test_cli_dry_run_previews_a_hard_limited_plan_without_key(self) -> None:
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = collection_main(
                [
                    "--gateway",
                    "closeai",
                    "--models",
                    "model-a,model-b",
                    "--prompt-limit",
                    "2",
                    "--max-requests",
                    "3",
                    "--dry-run",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["request_count"], 3)
        self.assertEqual(payload["gateway"], "closeai")
        self.assertNotIn("api_key", stdout.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
