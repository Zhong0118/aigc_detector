from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.text_attribution.prompts import PILOT_PROMPTS
from research.text_attribution.run_config import load_model_catalog, load_run_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "text_attribution"


class AttributionManifestTests(unittest.TestCase):
    def test_catalog_contains_the_ten_approved_unique_families(self) -> None:
        models = load_model_catalog(CONFIG_ROOT / "zhizengzeng_model_catalog.json")

        self.assertEqual(len(models), 10)
        self.assertEqual(len({model.family_id for model in models}), 10)
        self.assertEqual(
            {model.model_id for model in models},
            {
                "gpt-4o-mini-2024-07-18",
                "claude-haiku-4-5-20251001",
                "gemini-2.5-flash-lite",
                "deepseek-v3.2",
                "qwen3-30b-a3b-instruct-2507",
                "glm-4.6",
                "MiniMax-M2.5",
                "kimi-k2.5",
                "doubao-seed-1-6-flash-250828",
                "mimo-v2.5",
            },
        )
        self.assertTrue(all(model.enabled for model in models))
        self.assertTrue(all(model.fixed_route for model in models))

    def test_prompt_manifest_exactly_matches_the_canonical_prompt_bank(self) -> None:
        payload = json.loads(
            (CONFIG_ROOT / "prompt_manifest.pilot-v1.json").read_text(encoding="utf-8")
        )
        rows = payload["prompts"]

        self.assertEqual(len(rows), 24)
        self.assertEqual(
            rows,
            [
                {
                    "prompt_id": prompt.prompt_id,
                    "version": prompt.version,
                    "language": prompt.language,
                    "domain": prompt.domain,
                    "task_type": prompt.task_type,
                    "target_length": prompt.target_length,
                    "user_prompt": prompt.user_prompt,
                    "system_prompt": prompt.system_prompt,
                }
                for prompt in PILOT_PROMPTS
            ],
        )


class RunConfigTests(unittest.TestCase):
    def test_sampling_can_omit_top_p_for_cross_provider_compatibility(self) -> None:
        source = CONFIG_ROOT / "runs" / "smoke.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["model_catalog"] = str(
                (CONFIG_ROOT / "zhizengzeng_model_catalog.json").resolve()
            )
            payload["prompt_manifest"] = str(
                (CONFIG_ROOT / "prompt_manifest.pilot-v1.json").resolve()
            )
            payload["sampling"][0]["top_p"] = None
            run = root / "run.json"
            run.write_text(json.dumps(payload), encoding="utf-8")

            resolved = load_run_config(run)

        self.assertIsNone(resolved.sampling[0].top_p)

    def test_smoke_resolves_to_twenty_paired_requests(self) -> None:
        resolved = load_run_config(CONFIG_ROOT / "runs" / "smoke.json")

        self.assertEqual(len(resolved.models), 10)
        self.assertEqual(len(resolved.prompts), 2)
        self.assertEqual(len(resolved.sampling), 1)
        self.assertEqual(resolved.repetitions, 1)
        self.assertEqual(resolved.request_count, 20)
        self.assertEqual(resolved.max_requests, 20)

    def test_local_pilot_resolves_to_960_total_and_96_per_model(self) -> None:
        resolved = load_run_config(CONFIG_ROOT / "runs" / "local_pilot.json")

        self.assertEqual(len(resolved.models), 10)
        self.assertEqual(len(resolved.prompts), 24)
        self.assertEqual(len(resolved.sampling), 2)
        self.assertEqual(resolved.repetitions, 2)
        self.assertEqual(resolved.request_count, 960)
        self.assertEqual(resolved.requests_per_model, 96)

    def test_disabled_server_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "disabled"):
            load_run_config(CONFIG_ROOT / "runs" / "server_10k.json")

    def test_unknown_prompt_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.json"
            prompts = root / "prompts.json"
            run = root / "run.json"
            catalog.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "family_id": "family",
                                "provider": "Provider",
                                "model_id": "model",
                                "enabled": True,
                                "fixed_route": True,
                                "roles": ["pilot"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            prompts.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "prompt_id": "known",
                                "version": "v1",
                                "language": "en",
                                "domain": "general",
                                "task_type": "qa",
                                "target_length": "short",
                                "user_prompt": "Hello",
                                "system_prompt": None,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            run.write_text(
                json.dumps(
                    {
                        "run_id": "bad-prompt",
                        "gateway": "zhizengzeng",
                        "model_catalog": "catalog.json",
                        "model_roles": ["pilot"],
                        "prompt_manifest": "prompts.json",
                        "prompt_ids": ["missing"],
                        "sampling": [
                            {"temperature": 0, "top_p": 1, "max_tokens": 16}
                        ],
                        "repetitions": 1,
                        "max_requests": 1,
                        "enabled": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown prompt_id"):
                load_run_config(run)

    def test_hard_cap_must_equal_the_complete_cartesian_product(self) -> None:
        source = CONFIG_ROOT / "runs" / "smoke.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["model_catalog"] = str(
                (CONFIG_ROOT / "zhizengzeng_model_catalog.json").resolve()
            )
            payload["prompt_manifest"] = str(
                (CONFIG_ROOT / "prompt_manifest.pilot-v1.json").resolve()
            )
            payload["max_requests"] = 19
            run = root / "run.json"
            run.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Cartesian product"):
                load_run_config(run)


if __name__ == "__main__":
    unittest.main()
