from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import requests

from research.text_attribution.gateway import GenerationResult
from research.text_attribution.model_probe import (
    CandidateManifest,
    ModelCandidate,
    load_candidate_manifest,
    probe_candidates,
)
from scripts.probe_attribution_models import parse_args


def _result(model: str, text: str = "usable response") -> GenerationResult:
    return GenerationResult(
        gateway="zhizengzeng",
        request_model=model,
        response_model=model,
        response_id="response-id",
        gateway_request_id="request-id",
        text=text,
        finish_reason="stop",
        created=1,
        usage={"completion_tokens": 3},
        elapsed_seconds=0.01,
    )


def _http_error(status: int, message: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({"error": {"message": message}}).encode()
    return requests.HTTPError(message, response=response)


class _ProbeGateway:
    class _Config:
        api_key = "sk-probe-secret"

    config = _Config()

    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return _result(kwargs["model"])


class CandidateManifestTests(unittest.TestCase):
    def test_loads_flat_candidate_rows(self) -> None:
        payload = {
            "schema_version": 1,
            "gateway": "zhizengzeng",
            "candidates": [
                {
                    "family_id": "kimi",
                    "model_id": "kimi-k2",
                    "replacement_for": "kimi-k2-turbo-preview",
                    "variant_role": "fast",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = load_candidate_manifest(path)

        self.assertEqual(manifest.gateway, "zhizengzeng")
        self.assertEqual(manifest.candidates[0].model_id, "kimi-k2")


class CandidateProbeTests(unittest.TestCase):
    def test_candidate_is_eligible_only_after_zh_and_en_succeed(self) -> None:
        manifest = CandidateManifest(
            gateway="zhizengzeng",
            candidates=(
                ModelCandidate(
                    family_id="kimi",
                    model_id="kimi-k2",
                    replacement_for="kimi-k2-turbo-preview",
                    variant_role="fast",
                ),
            ),
        )
        gateway = _ProbeGateway()

        results = probe_candidates(
            manifest,
            gateway_factory=lambda: gateway,
            max_workers=1,
            max_tokens=64,
        )

        self.assertTrue(results[0]["eligible"])
        self.assertEqual(
            [probe["language"] for probe in results[0]["probes"]],
            ["zh", "en"],
        )
        self.assertTrue(all(probe["ok"] for probe in results[0]["probes"]))
        self.assertEqual(len(gateway.calls), 2)

    def test_one_failed_language_makes_candidate_ineligible(self) -> None:
        manifest = CandidateManifest(
            gateway="zhizengzeng",
            candidates=(
                ModelCandidate(
                    family_id="mimo",
                    model_id="mimo-v2-pro",
                    replacement_for="mimo-v2-flash",
                    variant_role="fast",
                ),
            ),
        )
        gateway = _ProbeGateway(
            [_result("mimo-v2-pro"), _http_error(404, "route missing")]
        )

        results = probe_candidates(
            manifest,
            gateway_factory=lambda: gateway,
            max_workers=1,
            max_tokens=64,
        )

        self.assertFalse(results[0]["eligible"])
        self.assertEqual(results[0]["probes"][1]["http_status"], 404)

    def test_probe_errors_redact_gateway_secret(self) -> None:
        manifest = CandidateManifest(
            gateway="zhizengzeng",
            candidates=(
                ModelCandidate(
                    family_id="mimo",
                    model_id="mimo-v2-pro",
                    replacement_for="mimo-v2-flash",
                    variant_role="fast",
                ),
            ),
        )
        gateway = _ProbeGateway(
            [
                _http_error(400, "invalid sk-probe-secret"),
                _http_error(400, "invalid sk-probe-secret"),
            ]
        )

        results = probe_candidates(
            manifest,
            gateway_factory=lambda: gateway,
            max_workers=1,
            max_tokens=64,
        )
        rendered = json.dumps(results)

        self.assertNotIn("sk-probe-secret", rendered)
        self.assertIn("<redacted>", rendered)


class CandidateProbeCliTests(unittest.TestCase):
    def test_cli_defaults_to_dry_safe_probe_limits(self) -> None:
        args = parse_args(["--config", "candidates.json", "--dry-run"])

        self.assertEqual(args.max_workers, 4)
        self.assertEqual(args.max_tokens, 256)
        self.assertEqual(args.timeout_seconds, 120.0)


if __name__ == "__main__":
    unittest.main()
