from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter, defaultdict

from research.text_attribution.production import build_plan_from_config
from research.text_attribution.run_config import (
    load_model_catalog,
    load_prompt_manifest,
    load_run_config,
)
from scripts.collect_attribution_run import build_preview, parse_args


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "text_attribution"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class V2ManifestSemanticsTests(unittest.TestCase):
    def test_catalog_allows_two_unique_variants_in_one_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog = Path(temporary) / "catalog.json"
            _write_json(
                catalog,
                {
                    "models": [
                        {
                            "family_id": "example",
                            "provider": "Example",
                            "model_id": "example-fast",
                            "variant_role": "fast",
                            "enabled": True,
                            "fixed_route": True,
                            "roles": ["v2"],
                        },
                        {
                            "family_id": "example",
                            "provider": "Example",
                            "model_id": "example-balanced",
                            "variant_role": "balanced",
                            "enabled": True,
                            "fixed_route": True,
                            "roles": ["v2"],
                        },
                    ]
                },
            )

            models = load_model_catalog(catalog)

        self.assertEqual([model.family_id for model in models], ["example", "example"])
        self.assertEqual(
            [model.variant_role for model in models], ["fast", "balanced"]
        )

    def test_catalog_rejects_duplicate_variant_role_inside_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog = Path(temporary) / "catalog.json"
            _write_json(
                catalog,
                {
                    "models": [
                        {
                            "family_id": "example",
                            "provider": "Example",
                            "model_id": "example-a",
                            "variant_role": "fast",
                            "enabled": True,
                            "fixed_route": True,
                            "roles": ["v2"],
                        },
                        {
                            "family_id": "example",
                            "provider": "Example",
                            "model_id": "example-b",
                            "variant_role": "fast",
                            "enabled": True,
                            "fixed_route": True,
                            "roles": ["v2"],
                        },
                    ]
                },
            )

            with self.assertRaisesRegex(ValueError, "family_id.*variant_role"):
                load_model_catalog(catalog)

    def test_sampling_profiles_own_repetitions_and_prompts_own_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.json"
            prompts = root / "prompts.json"
            run = root / "run.json"
            _write_json(
                catalog,
                {
                    "models": [
                        {
                            "family_id": "example",
                            "provider": "Example",
                            "model_id": "example-fast",
                            "variant_role": "fast",
                            "enabled": True,
                            "fixed_route": True,
                            "roles": ["v2"],
                        },
                        {
                            "family_id": "example",
                            "provider": "Example",
                            "model_id": "example-balanced",
                            "variant_role": "balanced",
                            "enabled": True,
                            "fixed_route": True,
                            "roles": ["v2"],
                        },
                    ]
                },
            )
            _write_json(
                prompts,
                {
                    "prompts": [
                        {
                            "prompt_id": "zh-short",
                            "semantic_group_id": "paired-001",
                            "version": "v2",
                            "language": "zh",
                            "language_origin": "paired",
                            "domain": "general",
                            "task_type": "explanation",
                            "target_length": "short",
                            "target_length_spec": "150-250 Chinese characters",
                            "format_type": "natural",
                            "system_prompt_type": "none",
                            "difficulty": "core",
                            "max_output_tokens": 512,
                            "quality_checks": ["language", "length"],
                            "user_prompt": "请简要解释。",
                            "system_prompt": None,
                        },
                        {
                            "prompt_id": "en-long",
                            "semantic_group_id": "native-en-001",
                            "version": "v2",
                            "language": "en",
                            "language_origin": "en_native",
                            "domain": "general",
                            "task_type": "analysis",
                            "target_length": "long",
                            "target_length_spec": "600-900 English words",
                            "format_type": "structured",
                            "system_prompt_type": "neutral_role",
                            "difficulty": "challenge",
                            "max_output_tokens": 3072,
                            "quality_checks": ["language", "length", "format"],
                            "user_prompt": "Analyze the issue.",
                            "system_prompt": "You are an analyst.",
                        },
                    ]
                },
            )
            _write_json(
                run,
                {
                    "schema_version": 2,
                    "run_id": "v2-test",
                    "gateway": "zhizengzeng",
                    "model_catalog": "catalog.json",
                    "model_roles": ["v2"],
                    "prompt_manifest": "prompts.json",
                    "prompt_ids": ["*"],
                    "minimum_prompt_count": 2,
                    "sampling": [
                        {
                            "sampling_id": "deterministic",
                            "temperature": 0.0,
                            "top_p": None,
                            "repetitions": 1,
                            "prompt_ids": ["zh-short"],
                        },
                        {
                            "sampling_id": "diverse",
                            "temperature": 0.8,
                            "top_p": None,
                            "repetitions": 2,
                            "prompt_ids": ["en-long"],
                        },
                    ],
                    "max_requests": 6,
                    "enabled": True,
                },
            )

            resolved = load_run_config(run)
            plan = build_plan_from_config(resolved)

        self.assertEqual(resolved.request_count, 6)
        self.assertEqual(resolved.requests_per_model, 3)
        self.assertEqual(
            [(profile.sampling_id, profile.repetitions) for profile in resolved.sampling],
            [("deterministic", 1), ("diverse", 2)],
        )
        self.assertEqual(
            sorted({item.parameters.max_tokens for item in plan}), [512, 3072]
        )
        self.assertEqual(
            {item.parameters.sampling_id for item in plan},
            {"deterministic", "diverse"},
        )


class RepositoryV2ConfigurationTests(unittest.TestCase):
    def test_v3_catalog_replaces_only_the_three_failed_routes(self) -> None:
        catalog = load_model_catalog(
            CONFIG_ROOT / "zhizengzeng_model_catalog.v3.json"
        )
        model_ids = {model.model_id for model in catalog}

        self.assertEqual(len(catalog), 20)
        self.assertEqual(len({model.family_id for model in catalog}), 10)
        self.assertTrue(
            {
                "gpt-4o-mini-2024-07-18",
                "deepseek-chat",
                "glm-4-flashx-250414",
            }.issubset(model_ids)
        )
        self.assertTrue(
            {
                "gpt-4.1-nano-2025-04-14",
                "deepseek-v4-pro",
                "glm-4.5-air",
            }.isdisjoint(model_ids)
        )

    def test_v3_replacement_batch_is_exactly_three_by_thirty(self) -> None:
        batch = load_run_config(
            CONFIG_ROOT / "runs" / "replacement_batch_90_v3.json"
        )

        self.assertEqual(batch.request_count, 90)
        self.assertEqual(batch.requests_per_model, 30)
        self.assertEqual(
            {model.model_id for model in batch.models},
            {
                "gpt-4o-mini-2024-07-18",
                "deepseek-chat",
                "glm-4-flashx-250414",
            },
        )

    def test_v3_full_logical_profile_remains_twenty_by_thirty(self) -> None:
        stage = load_run_config(CONFIG_ROOT / "runs" / "server_600_v3.json")

        self.assertEqual(stage.request_count, 600)
        self.assertEqual(stage.requests_per_model, 30)
        self.assertEqual(len(stage.models), 20)
        self.assertEqual(len(stage.prompts), 10)

    def test_replacement_candidate_probe_resolves_to_twelve_requests(self) -> None:
        probe = load_run_config(
            CONFIG_ROOT / "runs" / "replacement_candidates_probe_v3.json"
        )

        self.assertEqual(probe.request_count, 12)
        self.assertEqual(probe.requests_per_model, 2)
        self.assertEqual(
            {model.family_id for model in probe.models},
            {"deepseek", "glm"},
        )
        self.assertEqual(
            {model.model_id for model in probe.models},
            {
                "deepseek-chat",
                "deepseek-v3.2",
                "deepseek-v3.1-terminus",
                "glm-4.5-flash",
                "glm-4-air-250414",
                "glm-4-flashx-250414",
            },
        )

    def test_collection_cli_accepts_long_read_timeout(self) -> None:
        args = parse_args(
            [
                "--config",
                "run.json",
                "--timeout-seconds",
                "900",
            ]
        )

        self.assertEqual(args.timeout_seconds, 900)

    def test_repository_v2_catalog_and_smoke_resolve_to_40_requests(self) -> None:
        catalog = load_model_catalog(
            CONFIG_ROOT / "zhizengzeng_model_catalog.v2.json"
        )
        smoke = load_run_config(CONFIG_ROOT / "runs" / "smoke_v2.json")

        self.assertEqual(len(catalog), 20)
        self.assertEqual(len({model.family_id for model in catalog}), 10)
        self.assertTrue(
            all(
                {model.variant_role for model in catalog if model.family_id == family}
                == {"fast", "balanced"}
                for family in {model.family_id for model in catalog}
            )
        )
        self.assertEqual(smoke.request_count, 40)
        self.assertEqual(len(smoke.prompts), 2)

    def test_v2_preview_reports_profiles_variants_and_output_budgets(self) -> None:
        smoke = load_run_config(CONFIG_ROOT / "runs" / "smoke_v2.json")

        preview = build_preview(smoke, build_plan_from_config(smoke))

        self.assertEqual(
            preview["sampling_profiles"],
            [
                {
                    "sampling_id": "deterministic",
                    "temperature": 0.0,
                    "top_p": None,
                    "repetitions": 1,
                    "prompt_ids": ["zh-smoke-office-001"],
                },
                {
                    "sampling_id": "diverse",
                    "temperature": 0.8,
                    "top_p": None,
                    "repetitions": 1,
                    "prompt_ids": ["en-smoke-analysis-001"],
                },
            ],
        )
        self.assertEqual(preview["output_budgets"], {"1536": 20, "3072": 20})
        self.assertEqual(
            {row["variant_role"] for row in preview["models"]},
            {"fast", "balanced"},
        )

    def test_production_prompt_manifest_has_the_approved_balanced_design(self) -> None:
        prompts = load_prompt_manifest(CONFIG_ROOT / "prompt_manifest.v2.json")
        by_group: dict[str, list] = defaultdict(list)
        for prompt in prompts:
            by_group[str(prompt.semantic_group_id)].append(prompt)

        self.assertEqual(len(prompts), 100)
        self.assertEqual(len(by_group), 60)
        self.assertEqual(Counter(prompt.language for prompt in prompts), {"zh": 50, "en": 50})
        self.assertEqual(
            Counter(prompt.target_length for prompt in prompts),
            {"short": 30, "medium": 45, "long": 25},
        )
        self.assertEqual(
            Counter(prompt.format_type for prompt in prompts),
            {"natural": 50, "structured": 30, "strict": 20},
        )
        self.assertEqual(
            Counter(prompt.system_prompt_type for prompt in prompts),
            {"none": 70, "neutral_role": 20, "strong_style": 10},
        )
        self.assertEqual(
            Counter(prompt.difficulty for prompt in prompts),
            {"core": 80, "challenge": 20},
        )
        self.assertEqual(
            Counter(prompt.domain for prompt in prompts),
            {
                "science": 10,
                "health": 10,
                "technology": 10,
                "data_reasoning": 10,
                "education": 10,
                "business": 10,
                "office": 10,
                "public_history": 10,
                "culture_life": 10,
                "creative": 10,
            },
        )
        paired_groups = [
            group
            for group in by_group.values()
            if {prompt.language_origin for prompt in group} == {"paired"}
        ]
        self.assertEqual(len(paired_groups), 40)
        self.assertTrue(
            all(
                len(group) == 2
                and {prompt.language for prompt in group} == {"zh", "en"}
                for group in paired_groups
            )
        )
        self.assertTrue(
            all(
                prompt.max_output_tokens
                == {"short": 512, "medium": 1536, "long": 3072}[
                    prompt.target_length
                ]
                for prompt in prompts
            )
        )

    def test_repository_production_profile_resolves_to_6000_requests(self) -> None:
        production = load_run_config(
            CONFIG_ROOT / "runs" / "server_6k_v2.json"
        )

        self.assertEqual(len(production.models), 20)
        self.assertEqual(len(production.prompts), 100)
        self.assertEqual(
            [
                (profile.sampling_id, profile.temperature, profile.repetitions)
                for profile in production.sampling
            ],
            [
                ("deterministic", 0.0, 1),
                ("diverse", 0.8, 2),
            ],
        )
        self.assertEqual(production.request_count, 6000)
        self.assertEqual(production.requests_per_model, 300)

    def test_repository_stage_profile_resolves_to_600_requests(self) -> None:
        stage = load_run_config(
            CONFIG_ROOT / "runs" / "server_600_v2.json"
        )

        self.assertEqual(len(stage.models), 20)
        self.assertEqual(len(stage.prompts), 10)
        self.assertEqual(stage.request_count, 600)
        self.assertEqual(stage.requests_per_model, 30)


if __name__ == "__main__":
    unittest.main()
