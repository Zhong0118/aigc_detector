from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import requests

from research.text_attribution.gateway import GenerationResult
from research.text_attribution.production import (
    build_plan_from_config,
    request_key,
    run_collection,
)
from research.text_attribution.run_config import load_run_config
from scripts.collect_attribution_run import parse_args


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "text_attribution" / "runs" / "smoke.json"


def _result(model: str, text: str = "OK") -> GenerationResult:
    return GenerationResult(
        gateway="zhizengzeng",
        request_model=model,
        response_model=model,
        response_id="response-id",
        gateway_request_id="request-id",
        text=text,
        finish_reason="stop",
        created=1,
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        elapsed_seconds=0.01,
    )


def _http_error(status: int, message: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({"error": {"message": message}}).encode()
    error = requests.HTTPError(message, response=response)
    return error


class _FakeGateway:
    class _Config:
        api_key = "sk-super-secret"

    config = _Config()

    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[str] = []

    def generate(self, *, model: str, **kwargs):
        self.calls.append(model)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return _result(model)


class _ConcurrencyGateway(_FakeGateway):
    def __init__(self, delay_seconds: float = 0.05) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.active_by_model: dict[str, int] = {}
        self.max_active_by_model: dict[str, int] = {}

    def generate(self, *, model: str, **kwargs):
        with self._lock:
            self.calls.append(model)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            model_active = self.active_by_model.get(model, 0) + 1
            self.active_by_model[model] = model_active
            self.max_active_by_model[model] = max(
                self.max_active_by_model.get(model, 0),
                model_active,
            )
        try:
            time.sleep(self.delay_seconds)
            return _result(model)
        finally:
            with self._lock:
                self.active -= 1
                self.active_by_model[model] -= 1


class ProductionPlanTests(unittest.TestCase):
    def test_repeat_index_is_part_of_request_identity(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        first = build_plan_from_config(config)[0]
        repeated = type(first)(
            gateway_name=first.gateway_name,
            model=first.model,
            prompt=first.prompt,
            parameters=first.parameters,
            repeat_index=first.repeat_index + 1,
        )

        self.assertNotEqual(request_key(first), request_key(repeated))

    def test_completed_success_is_skipped_on_resume(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)[:2]
        gateway = _FakeGateway()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            errors = Path(temporary) / "errors.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "request_key": request_key(plan[0]),
                        "text": "usable",
                        "finish_reason": "stop",
                        "route_mismatch": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_collection(
                config,
                gateway=gateway,
                output_path=output,
                error_path=errors,
                plan=plan,
                sleep_seconds=0,
            )

        self.assertEqual(summary.skipped, 1)
        self.assertEqual(summary.written, 1)
        self.assertEqual(len(gateway.calls), 1)

    def test_truncated_record_is_retried_on_resume(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        item = build_plan_from_config(config)[0]
        gateway = _FakeGateway()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            errors = Path(temporary) / "errors.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "request_key": request_key(item),
                        "text": "partial",
                        "finish_reason": "length",
                        "route_mismatch": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_collection(
                config,
                gateway=gateway,
                output_path=output,
                error_path=errors,
                plan=[item],
                sleep_seconds=0,
            )

        self.assertEqual(summary.skipped, 0)
        self.assertEqual(summary.written, 1)
        self.assertEqual(len(gateway.calls), 1)


class ProductionCliTests(unittest.TestCase):
    def test_concurrency_and_empty_response_controls_have_safe_defaults(self) -> None:
        args = parse_args(["--config", str(SMOKE_CONFIG), "--dry-run"])

        self.assertEqual(args.max_workers, 4)
        self.assertEqual(args.empty_max_attempts, 2)
        self.assertEqual(args.empty_circuit_threshold, 2)
        self.assertEqual(args.transient_circuit_threshold, 2)


class ProductionConcurrencyTests(unittest.TestCase):
    def test_different_models_can_run_concurrently(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        full_plan = build_plan_from_config(config)
        first = full_plan[0]
        second = next(item for item in full_plan if item.model != first.model)
        gateway = _ConcurrencyGateway()

        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=[first, second],
                sleep_seconds=0,
                max_workers=2,
            )

        self.assertEqual(summary.written, 2)
        self.assertEqual(gateway.max_active, 2)

    def test_model_major_plan_is_round_robined_before_submission(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        full_plan = build_plan_from_config(config)
        selected_models: list[str] = []
        for item in full_plan:
            if item.model not in selected_models:
                selected_models.append(item.model)
            if len(selected_models) == 4:
                break
        plan = [
            item
            for model in selected_models
            for item in full_plan
            if item.model == model
        ]
        gateway = _ConcurrencyGateway(delay_seconds=0.1)

        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=plan,
                sleep_seconds=0,
                max_workers=4,
            )

        self.assertEqual(summary.written, len(plan))
        self.assertEqual(gateway.max_active, 4)

    def test_same_model_requests_are_serialized(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)[:2]
        self.assertEqual(plan[0].model, plan[1].model)
        gateway = _ConcurrencyGateway()

        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=plan,
                sleep_seconds=0,
                max_workers=2,
            )

        self.assertEqual(summary.written, 2)
        self.assertEqual(gateway.max_active_by_model[plan[0].model], 1)


class ProductionFailureTests(unittest.TestCase):
    def test_403_is_recorded_but_does_not_stop_other_models(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        full_plan = build_plan_from_config(config)
        first = full_plan[0]
        second = next(item for item in full_plan if item.model != first.model)
        plan = [first, second]
        gateway = _FakeGateway(
            [_http_error(403, "model forbidden"), _result(plan[1].model)]
        )
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=plan,
                sleep_seconds=0,
            )

        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.written, 1)
        self.assertIsNone(summary.stopped_reason)
        self.assertEqual(len(gateway.calls), 2)

    def test_401_records_redacted_error_and_stops(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)[:2]
        gateway = _FakeGateway(
            [_http_error(401, "invalid sk-super-secret")]
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            errors = Path(temporary) / "errors.jsonl"

            summary = run_collection(
                config,
                gateway=gateway,
                output_path=output,
                error_path=errors,
                plan=plan,
                sleep_seconds=0,
            )
            rendered = errors.read_text(encoding="utf-8")

        self.assertEqual(summary.stopped_reason, "authentication")
        self.assertEqual(summary.failed, 1)
        self.assertEqual(len(gateway.calls), 1)
        self.assertNotIn("sk-super-secret", rendered)
        self.assertIn("<redacted>", rendered)

    def test_429_is_retried_then_succeeds(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)[:1]
        gateway = _FakeGateway(
            [_http_error(429, "rate limited"), _result(plan[0].model)]
        )
        sleeps: list[float] = []
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=plan,
                sleep_seconds=0,
                max_attempts=2,
                retry_base_seconds=0.25,
                sleeper=sleeps.append,
            )

        self.assertEqual(summary.written, 1)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(sleeps, [0.25])

    def test_empty_completion_is_retried_then_succeeds(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)[:1]
        gateway = _FakeGateway(
            [
                ValueError("Chat completion returned empty text"),
                _result(plan[0].model),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=plan,
                sleep_seconds=0,
                max_attempts=2,
                retry_base_seconds=0,
                sleeper=lambda _: None,
            )

        self.assertEqual(summary.written, 1)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(len(gateway.calls), 2)

    def test_exhausted_500_is_recorded_and_next_request_continues(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)[:2]
        gateway = _FakeGateway(
            [
                _http_error(500, "upstream unavailable"),
                _http_error(500, "upstream unavailable"),
                _result(plan[1].model),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            errors = Path(temporary) / "errors.jsonl"

            summary = run_collection(
                config,
                gateway=gateway,
                output_path=output,
                error_path=errors,
                plan=plan,
                sleep_seconds=0,
                max_attempts=2,
                retry_base_seconds=0,
                sleeper=lambda _: None,
            )

        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.written, 1)
        self.assertIsNone(summary.stopped_reason)

    def test_404_opens_model_circuit_and_skips_remaining_requests(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)[:2]
        gateway = _FakeGateway([_http_error(404, "route missing")])

        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=plan,
                sleep_seconds=0,
                max_workers=2,
            )

        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.circuit_skipped, 1)
        self.assertEqual(len(gateway.calls), 1)

    def test_two_exhausted_empty_tasks_open_model_circuit(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        base_plan = build_plan_from_config(config)[:2]
        third = type(base_plan[0])(
            gateway_name=base_plan[0].gateway_name,
            model=base_plan[0].model,
            prompt=base_plan[0].prompt,
            parameters=base_plan[0].parameters,
            repeat_index=99,
        )
        empty = ValueError("Chat completion returned empty text")
        gateway = _FakeGateway([empty, empty, empty, empty])

        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=[*base_plan, third],
                sleep_seconds=0,
                max_workers=2,
                max_attempts=4,
                empty_max_attempts=2,
                empty_circuit_threshold=2,
                retry_base_seconds=0,
                sleeper=lambda _: None,
            )

        self.assertEqual(summary.failed, 2)
        self.assertEqual(summary.circuit_skipped, 1)
        self.assertEqual(len(gateway.calls), 4)

    def test_two_exhausted_network_tasks_open_model_circuit(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        base_plan = build_plan_from_config(config)[:2]
        third = type(base_plan[0])(
            gateway_name=base_plan[0].gateway_name,
            model=base_plan[0].model,
            prompt=base_plan[0].prompt,
            parameters=base_plan[0].parameters,
            repeat_index=99,
        )
        gateway = _FakeGateway(
            [
                requests.Timeout("read timed out"),
                requests.ConnectionError("connection reset"),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            summary = run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=Path(temporary) / "errors.jsonl",
                plan=[*base_plan, third],
                sleep_seconds=0,
                max_workers=2,
                max_attempts=1,
                transient_circuit_threshold=2,
                retry_base_seconds=0,
                sleeper=lambda _: None,
            )

        self.assertEqual(summary.failed, 2)
        self.assertEqual(summary.circuit_skipped, 1)
        self.assertEqual(len(gateway.calls), 2)

    def test_http_error_record_contains_redacted_response_excerpt(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)[:1]
        gateway = _FakeGateway(
            [_http_error(400, "invalid request for sk-super-secret")]
        )

        with tempfile.TemporaryDirectory() as temporary:
            errors = Path(temporary) / "errors.jsonl"
            run_collection(
                config,
                gateway=gateway,
                output_path=Path(temporary) / "raw.jsonl",
                error_path=errors,
                plan=plan,
                sleep_seconds=0,
            )
            record = json.loads(errors.read_text(encoding="utf-8"))

        self.assertIn("response_body_excerpt", record)
        self.assertIn("<redacted>", record["response_body_excerpt"])
        self.assertNotIn("sk-super-secret", record["response_body_excerpt"])


if __name__ == "__main__":
    unittest.main()
