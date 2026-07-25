from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import scripts.bootstrap_text_resources as bootstrap_text_resources
from detection.text_resource_adapters import (
    TextSample,
    adapt_hc3_rows,
    adapt_m4_rows,
    adapt_raid_rows,
    adapt_tnews_rows,
    normalize_source_domain,
)
from detection.text_resources import (
    CORE_DATASETS,
    MULTIDOMAIN_DATASETS,
    DatasetSpec,
    TextResourceManager,
    conformal_high_threshold,
)
from scripts.bootstrap_text_resources import (
    _balanced_human_samples,
    _partition_assignment,
    build_resource_stats,
    build_threshold_artifact,
)


class DiskBackedRows:
    """Test double that permits sequential iteration but rejects size probing."""

    def __init__(self, path: Path, rows=None):
        self._path = path
        if rows is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

    def __iter__(self):
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    def __len__(self):
        raise AssertionError("rows must not be materialized as a Python list")


class TextResourceAdapterTests(unittest.TestCase):
    def test_text_sample_is_immutable_and_builds_group_key(self) -> None:
        sample = TextSample(
            text="A human answer.",
            label="human",
            language="en",
            domain="general_qa",
            source="hc3_en",
            source_id="question-1:human:0",
        )

        self.assertEqual(sample.group_key("short"), "en:general_qa:short")
        with self.assertRaises(FrozenInstanceError):
            sample.text = "changed"  # type: ignore[misc]

    def test_text_sample_defaults_to_train_with_no_generator(self) -> None:
        sample = TextSample(
            text="A default sample.",
            label="human",
            language="en",
            domain="general_qa",
            source="fixture",
            source_id="fixture-1",
        )

        self.assertEqual(sample.split, "train")
        self.assertIsNone(sample.generator)
        self.assertIsNone(sample.partition_id)
        self.assertEqual(sample.partition_key, sample.source_id)

    def test_text_sample_validates_runtime_strings_and_length_bucket(self) -> None:
        values = {
            "text": "Valid text.",
            "label": "human",
            "language": "en",
            "domain": "general_qa",
            "source": "fixture",
            "source_id": "fixture-1",
            "split": "train",
        }
        for field_name in ("text", "source", "source_id", "split"):
            with self.subTest(field=field_name, kind="type"):
                invalid = {**values, field_name: None}
                with self.assertRaisesRegex(TypeError, field_name):
                    TextSample(**invalid)
            with self.subTest(field=field_name, kind="blank"):
                invalid = {**values, field_name: " "}
                with self.assertRaisesRegex(ValueError, field_name):
                    TextSample(**invalid)

        for generator in ("", "NaN"):
            with self.subTest(generator=generator):
                with self.assertRaisesRegex(ValueError, "generator"):
                    TextSample(**values, generator=generator)
        for generator in (float("nan"), True, [], {"model": "gpt"}):
            with self.subTest(generator=repr(generator)):
                with self.assertRaisesRegex(TypeError, "generator"):
                    TextSample(**values, generator=generator)

        sample = TextSample(**values)
        with self.assertRaisesRegex(ValueError, "length_bucket"):
            sample.group_key("medium")
        with self.assertRaisesRegex(TypeError, "length_bucket"):
            sample.group_key(None)

    def test_reddit_normalizes_to_general_qa(self) -> None:
        self.assertEqual(normalize_source_domain("reddit"), "general_qa")

    def test_source_domains_normalize_to_the_four_canonical_domains(self) -> None:
        expected = {
            "books": "literature",
            "Poetry": "literature",
            "peer_read": "academic_technical",
            "PeerRead": "academic_technical",
            "arXiv": "academic_technical",
            "abstracts": "academic_technical",
            "code": "academic_technical",
            "medicine": "academic_technical",
            "news": "official_news",
            "wikipedia": "general_qa",
            "wiki-how": "general_qa",
            "reddit_eli5": "general_qa",
            "reviews": "general_qa",
            "recipes": "general_qa",
            "unmapped-source": "general_qa",
        }

        self.assertEqual(
            {source: normalize_source_domain(source) for source in expected},
            expected,
        )

    def test_hc3_flattens_human_and_ai_answers_with_shared_question_prefix(self) -> None:
        samples = list(
            adapt_hc3_rows(
                [
                    {
                        "id": "question-7",
                        "human_answers": ["Human one.", "Human two."],
                        "chatgpt_answers": ["AI one.", "AI two."],
                    }
                ],
                source_name="reddit_eli5",
                language="en",
                split="validation",
            )
        )

        self.assertEqual([sample.label for sample in samples], ["human", "human", "ai", "ai"])
        self.assertEqual([sample.generator for sample in samples], [None, None, "chatgpt", "chatgpt"])
        self.assertEqual({sample.language for sample in samples}, {"en"})
        self.assertEqual({sample.domain for sample in samples}, {"general_qa"})
        self.assertEqual({sample.split for sample in samples}, {"validation"})
        self.assertEqual(len({sample.source_id for sample in samples}), 4)
        self.assertEqual(
            {sample.source_id.rsplit(":", 2)[0] for sample in samples},
            {"reddit_eli5:validation:question-7"},
        )
        self.assertEqual(
            {sample.partition_id for sample in samples},
            {"reddit_eli5:validation:question-7"},
        )
        self.assertEqual(len({sample.partition_key for sample in samples}), 1)
        self.assertEqual(len({sample.identity_key for sample in samples}), 4)

    def test_invalid_row_ids_use_stable_content_identity(self) -> None:
        invalid_ids = (None, " ", float("nan"), True, [], {"nested": "id"})
        source_ids = []
        for invalid_id in invalid_ids:
            with self.subTest(identifier=repr(invalid_id)):
                sample = list(
                    adapt_tnews_rows(
                        [{"id": invalid_id, "sentence": "同一条无有效编号的新闻。"}],
                        source_name="tnews",
                    )
                )[0]
                self.assertNotIn("nan", sample.source_id.casefold())
                source_ids.append(sample.source_id)

        self.assertEqual(len(set(source_ids)), 1)

    def test_idless_row_identity_is_stable_under_reordering(self) -> None:
        rows = [
            {"sentence": "第一条无编号新闻。"},
            {"sentence": "第二条无编号新闻。"},
        ]

        original = {
            sample.text: sample.source_id
            for sample in adapt_tnews_rows(rows, source_name="tnews", split="train")
        }
        reordered = {
            sample.text: sample.source_id
            for sample in adapt_tnews_rows(reversed(rows), source_name="tnews", split="train")
        }

        self.assertEqual(original, reordered)

    def test_source_identity_includes_split_to_avoid_collisions(self) -> None:
        row = {"id": "shared-id", "sentence": "同一条新闻。"}

        train = list(adapt_tnews_rows([row], source_name="tnews", split="train"))[0]
        validation = list(
            adapt_tnews_rows([row], source_name="tnews", split="validation")
        )[0]

        self.assertNotEqual(train.source_id, validation.source_id)
        self.assertNotEqual(train.identity_key, validation.identity_key)
        self.assertIn(":train:", train.source_id)
        self.assertIn(":validation:", validation.source_id)

    def test_tnews_reads_sentence_and_text_as_chinese_official_news(self) -> None:
        samples = list(
            adapt_tnews_rows(
                [
                    {"id": "a", "sentence": "政府发布年度工作公告。"},
                    {"id": "b", "text": "有关部门公布最新通知。"},
                ],
                source_name="tnews",
                split="validation",
            )
        )

        self.assertEqual([sample.text for sample in samples], ["政府发布年度工作公告。", "有关部门公布最新通知。"])
        self.assertEqual({sample.label for sample in samples}, {"human"})
        self.assertEqual({sample.language for sample in samples}, {"zh"})
        self.assertEqual({sample.domain for sample in samples}, {"official_news"})
        self.assertEqual({sample.split for sample in samples}, {"validation"})

    def test_m4_maps_labels_sources_and_generators(self) -> None:
        samples = list(
            adapt_m4_rows(
                [
                    {"id": "human-1", "text": "A reviewed manuscript.", "label": 0, "source": "peerread"},
                    {
                        "id": "ai-1",
                        "text": "A generated encyclopedia entry.",
                        "label": 1,
                        "source": "wikipedia",
                        "model": "gpt-4",
                    },
                ],
                source_name="m4_multilingual",
                language="en",
            )
        )

        self.assertEqual([sample.label for sample in samples], ["human", "ai"])
        self.assertEqual([sample.domain for sample in samples], ["academic_technical", "general_qa"])
        self.assertEqual([sample.generator for sample in samples], [None, "gpt-4"])
        self.assertEqual({sample.source for sample in samples}, {"m4_multilingual"})

    def test_m4_does_not_stringify_invalid_generator_metadata(self) -> None:
        for model in (float("nan"), True, [], {"name": "gpt"}):
            with self.subTest(model=repr(model)):
                sample = list(
                    adapt_m4_rows(
                        [{"text": "Generated text.", "label": 1, "model": model}],
                        source_name="m4_multilingual",
                        language="en",
                    )
                )[0]
                self.assertIsNone(sample.generator)

    def test_raid_maps_human_and_ai_and_excludes_adversarial_rows(self) -> None:
        samples = list(
            adapt_raid_rows(
                [
                    {
                        "id": "human-1",
                        "generation": "A newspaper report.",
                        "model": "human",
                        "domain": "news",
                        "attack": None,
                    },
                    {
                        "id": "ai-1",
                        "generation": "A generated poem.",
                        "model": "gpt2",
                        "domain": "poetry",
                        "attack": "none",
                    },
                    {
                        "id": "attacked-1",
                        "generation": "An attacked sample.",
                        "model": "gpt2",
                        "domain": "reddit",
                        "attack": "homoglyph",
                    },
                ],
                source_name="raid_no_adversarial",
                language="en",
            )
        )

        self.assertEqual([sample.label for sample in samples], ["human", "ai"])
        self.assertEqual([sample.domain for sample in samples], ["official_news", "literature"])
        self.assertEqual([sample.generator for sample in samples], [None, "gpt2"])

    def test_raid_rejects_malformed_model_metadata(self) -> None:
        malformed_models = (
            ("missing", {}),
            ("blank", {"model": " "}),
            ("nan", {"model": float("nan")}),
            ("bool", {"model": False}),
            ("list", {"model": []}),
            ("mapping", {"model": {"name": "gpt2"}}),
        )
        for name, metadata in malformed_models:
            with self.subTest(model=name):
                row = {
                    "id": f"bad-{name}",
                    "generation": "This row must not be emitted.",
                    "domain": "reddit",
                    "attack": None,
                    **metadata,
                }
                with self.assertRaisesRegex(ValueError, "RAID model.*bad-"):
                    list(
                        adapt_raid_rows(
                            [row],
                            source_name="raid_no_adversarial",
                            language="en",
                        )
                    )

    def test_adapters_reject_blank_text_unsupported_languages_and_unknown_labels(self) -> None:
        self.assertEqual(
            list(adapt_hc3_rows([{"human_answers": [" ", ""]}], "hc3_en", "en")),
            [],
        )
        self.assertEqual(
            list(adapt_tnews_rows([{"sentence": "  "}], source_name="tnews")),
            [],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            list(adapt_hc3_rows([], "hc3_fr", "fr"))
        with self.assertRaisesRegex(ValueError, "Unsupported M4 label"):
            list(adapt_m4_rows([{"text": "Valid text.", "label": 2}], "m4", "en"))


class TextResourceManagerTests(unittest.TestCase):
    def test_dataset_spec_rejects_unsafe_portable_resource_names(self) -> None:
        unsafe_names = (
            "",
            ".",
            "..",
            "../escape",
            "nested/name",
            r"nested\name",
            "C:drive",
            r"C:\drive",
            r"\\server\share",
            "bad:name",
            "bad*name",
            "trailing.",
            "CON",
            "CON.txt",
            "payload.txt:stream",
        )
        for name in unsafe_names:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                "dataset name",
            ):
                DatasetSpec(
                    name=name,
                    url="https://example.invalid/source.zip",
                    license="MIT",
                    domain="literature",
                    source="https://example.invalid/project",
                )

        safe = DatasetSpec(
            name="safe_resource-1.0",
            url="https://example.invalid/source.zip",
            license="MIT",
            domain="literature",
            source="https://example.invalid/project",
        )
        self.assertEqual(safe.name, "safe_resource-1.0")

    def test_dataset_spec_rejects_unsafe_explicit_filenames(self) -> None:
        unsafe_filenames = (
            "payload.txt:stream",
            "CON",
            "CON.jsonl",
            "trailing.",
            "trailing ",
            "/absolute.jsonl",
            "nested/payload.jsonl",
            r"nested\payload.jsonl",
        )
        for filename in unsafe_filenames:
            with self.subTest(filename=filename), self.assertRaisesRegex(
                ValueError,
                "dataset filename",
            ):
                DatasetSpec(
                    name="safe_resource",
                    url="https://example.invalid/source.jsonl",
                    license="MIT",
                    domain="literature",
                    source="https://example.invalid/project",
                    download_kind="file",
                    filename=filename,
                )

        safe = DatasetSpec(
            name="safe_resource",
            url="https://example.invalid/source.jsonl",
            license="MIT",
            domain="literature",
            source="https://example.invalid/project",
            download_kind="file",
            filename="payload-1.0.jsonl",
        )
        self.assertEqual(safe.filename, "payload-1.0.jsonl")

    def test_dataset_spec_rejects_unsafe_url_derived_filenames(self) -> None:
        unsafe_urls = (
            "https://example.invalid/payload.txt:stream",
            "https://example.invalid/CON",
            "https://example.invalid/CON.jsonl",
            "https://example.invalid/trailing.",
            "https://example.invalid/nested%2Fpayload.jsonl",
        )
        for url in unsafe_urls:
            with self.subTest(url=url), self.assertRaisesRegex(
                ValueError,
                "dataset filename",
            ):
                DatasetSpec(
                    name="safe_resource",
                    url=url,
                    license="MIT",
                    domain="literature",
                    source="https://example.invalid/project",
                    download_kind="file",
                )

    def test_resource_profiles_register_official_multidomain_sources(self) -> None:
        core_names = {spec.name for spec in CORE_DATASETS}
        multidomain = {spec.name: spec for spec in MULTIDOMAIN_DATASETS}

        self.assertEqual(core_names, {"chinese_poetry", "chinese_classical_corpus", "csl"})
        self.assertEqual(
            set(multidomain),
            core_names
            | {"hc3_en", "hc3_zh", "tnews", "m4_multilingual", "raid_no_adversarial"},
        )
        self.assertEqual(multidomain["hc3_en"].dataset_id, "Hello-SimpleAI/HC3")
        self.assertEqual(multidomain["hc3_zh"].dataset_id, "Hello-SimpleAI/HC3-Chinese")
        self.assertEqual(multidomain["hc3_en"].dataset_config, "all")
        self.assertEqual(multidomain["hc3_zh"].dataset_config, "all")
        self.assertEqual(multidomain["tnews"].dataset_id, "clue")
        self.assertEqual(multidomain["tnews"].dataset_config, "tnews")
        self.assertEqual(multidomain["tnews"].license, "unknown")
        self.assertEqual(multidomain["m4_multilingual"].dataset_id, "d0rj/SemEval2024-task8")
        self.assertEqual(
            multidomain["m4_multilingual"].dataset_config,
            "subtaskA_multilingual",
        )
        self.assertEqual(multidomain["m4_multilingual"].splits, ("train", "dev"))
        self.assertEqual(multidomain["m4_multilingual"].license, "Apache-2.0")
        self.assertEqual(
            multidomain["m4_multilingual"].attribution_url,
            "https://github.com/mbzuai-nlp/SemEval2024-task8",
        )
        self.assertEqual(
            multidomain["raid_no_adversarial"].url,
            "https://dataset.raid-bench.xyz/train_none.csv",
        )
        self.assertEqual(multidomain["raid_no_adversarial"].source_kind, "raid_csv")
        self.assertEqual(multidomain["raid_no_adversarial"].license, "MIT")

    def test_profile_aliases_select_expected_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = TextResourceManager(project_root=tmp, profile="core")
            multidomain = TextResourceManager(project_root=tmp, profile="multidomain")
            all_resources = TextResourceManager(project_root=tmp, profile="all")
            documented = TextResourceManager(
                project_root=tmp,
                dataset_profile="multidomain",
            )

        self.assertEqual(core.dataset_specs, CORE_DATASETS)
        self.assertEqual(multidomain.dataset_specs, MULTIDOMAIN_DATASETS)
        self.assertEqual(all_resources.dataset_specs, MULTIDOMAIN_DATASETS)
        self.assertEqual(documented.dataset_specs, MULTIDOMAIN_DATASETS)
        self.assertEqual(documented.dataset_profile, "multidomain")
        self.assertEqual(documented.profile, "multidomain")
        with self.assertRaisesRegex(ValueError, "profile"):
            TextResourceManager(project_root=".", profile="unknown")
        with self.assertRaisesRegex(ValueError, "conflicting.*profile"):
            TextResourceManager(
                project_root=".",
                dataset_profile="multidomain",
                profile="core",
            )

    def test_default_resources_stay_inside_project_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TextResourceManager(project_root=root)

            status = manager.status()

        self.assertEqual(Path(status["model_dir"]), root / "models" / "huggingface")
        self.assertEqual(Path(status["data_dir"]), root / "data" / "text_false_positive" / "corpora")
        self.assertEqual(
            Path(status["normalized_dir"]),
            root / "data" / "text_false_positive" / "normalized",
        )
        self.assertEqual(Path(status["artifact_path"]), root / "artifacts" / "text_fp_controller" / "thresholds.json")
        self.assertEqual(
            {item["name"] for item in status["datasets"]},
            {"chinese_poetry", "chinese_classical_corpus", "csl"},
        )
        classical = next(
            item for item in status["datasets"] if item["name"] == "chinese_classical_corpus"
        )
        self.assertEqual(classical["license"], "CC0-1.0")
        self.assertEqual(classical["calibration_buckets"], ("long",))
        self.assertIn("source_ready", classical)
        self.assertIn("normalized_ready", classical)

    def test_hugging_face_adapter_dispatch_uses_disk_backed_project_cache(self) -> None:
        loader_calls = []

        def loader(dataset_id, dataset_config=None, **kwargs):
            loader_calls.append((dataset_id, dataset_config, kwargs))
            return DiskBackedRows(
                Path(kwargs["cache_dir"]) / f"fixture-{kwargs['split']}.arrow",
                [
                    {
                        "id": "question-1",
                        "human_answers": ["A human response."],
                        "chatgpt_answers": ["An AI response."],
                    }
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TextResourceManager(
                project_root=root,
                profile="multidomain",
                dataset_loader=loader,
            )
            samples = list(manager.iter_source_samples("hc3_en"))
            zh_samples = list(manager.iter_source_samples("hc3_zh"))

        self.assertEqual([sample.label for sample in samples], ["human", "ai"])
        self.assertEqual([sample.language for sample in samples], ["en", "en"])
        self.assertEqual([sample.language for sample in zh_samples], ["zh", "zh"])
        self.assertEqual(
            [call[0:2] for call in loader_calls],
            [
                ("Hello-SimpleAI/HC3", "all"),
                ("Hello-SimpleAI/HC3-Chinese", "all"),
            ],
        )
        self.assertEqual(loader_calls[0][2]["split"], "train")
        self.assertIs(loader_calls[0][2]["keep_in_memory"], False)
        self.assertNotIn("streaming", loader_calls[0][2])
        self.assertEqual(
            Path(loader_calls[0][2]["cache_dir"]),
            root / "data" / "text_false_positive" / "corpora" / "hc3_en" / "hf_cache",
        )

    def test_m4_dispatch_keeps_supported_languages_and_skips_others(self) -> None:
        def loader(*args, **kwargs):
            rows = [
                {
                    "id": "english",
                    "text": "A human encyclopedia article.",
                    "label": 0,
                    "source": "wikipedia",
                },
                {
                    "id": "chinese",
                    "text": "这是一段机器生成的中文内容。",
                    "label": 1,
                    "source": "Chinese",
                    "model": "chatgpt",
                },
                {
                    "id": "arabic",
                    "text": "نص عربي",
                    "label": 0,
                    "source": "Arabic",
                },
            ]
            return DiskBackedRows(
                Path(kwargs["cache_dir"]) / f"fixture-{kwargs['split']}.arrow",
                rows if kwargs["split"] == "train" else [],
            )

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                profile="multidomain",
                dataset_loader=loader,
            )
            samples = list(manager.iter_samples("m4_multilingual"))

        self.assertEqual([sample.language for sample in samples], ["en", "zh"])
        self.assertEqual([sample.label for sample in samples], ["human", "ai"])

    def test_hf_ensure_scans_every_configured_split_before_publishing_manifest(self) -> None:
        loader_calls = []

        def loader(*args, **kwargs):
            loader_calls.append(kwargs)
            cache = Path(kwargs["cache_dir"])
            (cache / "downloads").mkdir(parents=True, exist_ok=True)
            split = kwargs["split"]
            (cache / "downloads" / "ignored.part").write_bytes(b"partial")
            (cache / "downloads" / "ignored.lock").write_bytes(b"lock")
            rows = {
                "train": [{"id": "train-1", "sentence": "训练新闻样本。"}],
                "validation": [{"id": "validation-1", "sentence": "验证新闻样本。"}],
            }
            return DiskBackedRows(cache / "downloads" / f"{split}.arrow", rows[split])

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                dataset_profile="multidomain",
                dataset_loader=loader,
            )
            spec = manager.resource_spec("tnews")

            manager.ensure_dataset(spec)
            marker = json.loads(
                (manager.dataset_dir / spec.name / ".source.complete.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertTrue(manager.is_source_ready(spec))

        self.assertEqual(
            [call["split"] for call in loader_calls],
            ["train", "validation"],
        )
        self.assertTrue(all(call["keep_in_memory"] is False for call in loader_calls))
        self.assertTrue(all("streaming" not in call for call in loader_calls))
        self.assertEqual(marker["validated_splits"], ["train", "validation"])
        self.assertEqual(set(marker["split_evidence"]), {"train", "validation"})
        self.assertEqual(marker["split_evidence"]["train"]["raw_rows"], 1)
        self.assertEqual(marker["split_evidence"]["validation"]["adapted_samples"], 1)
        self.assertEqual(marker["expected_labels"], ["human"])
        self.assertEqual(
            marker["split_evidence"]["train"]["label_counts"],
            {"human": 1, "ai": 0},
        )
        self.assertEqual(len(marker["scan_sha256"]), 64)
        self.assertEqual(marker["preparation_kind"], "hf_disk_cache_v1")
        self.assertEqual(marker["cache_evidence"]["kind"], "file_inventory_sha256_v1")
        cache_files = marker["cache_evidence"]["files"]
        self.assertEqual(
            [item["path"] for item in cache_files],
            ["downloads/train.arrow", "downloads/validation.arrow"],
        )
        self.assertTrue(all(item["byte_size"] > 0 for item in cache_files))
        self.assertTrue(all(len(item["content_sha256"]) == 64 for item in cache_files))
        self.assertEqual(len(marker["cache_evidence"]["inventory_sha256"]), 64)

    def test_hf_readiness_detects_deleted_or_same_size_corrupted_cache_evidence(self) -> None:
        def loader(*args, **kwargs):
            cache = Path(kwargs["cache_dir"])
            cache.mkdir(parents=True, exist_ok=True)
            rows = {
                "train": [{"id": "train-1", "sentence": "训练新闻样本。"}],
                "validation": [{"id": "validation-1", "sentence": "验证新闻样本。"}],
            }
            return DiskBackedRows(
                cache / f"{kwargs['split']}.arrow",
                rows[kwargs["split"]],
            )

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                dataset_profile="multidomain",
                dataset_loader=loader,
            )
            spec = manager.resource_spec("tnews")
            manager.ensure_dataset(spec)
            cache_file = manager.dataset_dir / spec.name / "hf_cache" / "train.arrow"

            self.assertTrue(manager.is_source_ready(spec))
            original_stat = cache_file.stat()
            original = cache_file.read_bytes()
            cache_file.unlink()
            self.assertFalse(manager.is_source_ready(spec))
            cache_file.write_bytes(original)
            os.utime(
                cache_file,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            self.assertTrue(manager.is_source_ready(spec))
            cache_file.write_bytes(b"x" * len(original))
            os.utime(
                cache_file,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            self.assertTrue(manager.is_source_ready(spec))
            self.assertFalse(manager.is_source_ready(spec, deep=True))

    def test_hf_incomplete_or_corrupt_split_manifest_is_not_ready(self) -> None:
        def incomplete_loader(*args, **kwargs):
            if kwargs["split"] == "validation":
                raise KeyError("validation")
            return DiskBackedRows(
                Path(kwargs["cache_dir"]) / "train.arrow",
                [{"id": "train-1", "sentence": "只有训练集。"}],
            )

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                dataset_profile="multidomain",
                dataset_loader=incomplete_loader,
            )
            spec = manager.resource_spec("tnews")
            marker_path = manager.dataset_dir / spec.name / ".source.complete.json"

            with self.assertRaisesRegex(RuntimeError, "validation"):
                manager.ensure_dataset(spec)
            self.assertFalse(marker_path.exists())
            self.assertFalse(manager.is_source_ready(spec))

            corrupt = {
                "preparation_kind": "hf_disk_cache_v1",
                "source_spec_fingerprint": spec.fingerprint,
                "validated_splits": ["train"],
                "split_evidence": {
                    "train": {
                        "raw_rows": 1,
                        "adapted_samples": 1,
                        "content_sha256": "0" * 64,
                    }
                },
                "scan_sha256": "0" * 64,
            }
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(json.dumps(corrupt), encoding="utf-8")
            self.assertFalse(manager.is_source_ready(spec))

    def test_hf_prepared_splits_are_reused_sequentially_in_offline_mode(self) -> None:
        loader_calls = []
        rows_by_split = {
            "train": [{"id": "train-1", "sentence": "训练新闻样本。"}],
            "validation": [{"id": "validation-1", "sentence": "验证新闻样本。"}],
        }

        def loader(*args, **kwargs):
            loader_calls.append(kwargs.copy())
            cache = Path(kwargs["cache_dir"])
            cache.mkdir(parents=True, exist_ok=True)
            cache_file = cache / f"{kwargs['split']}.arrow"
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                if not cache_file.is_file():
                    raise RuntimeError("offline cache is missing")
            else:
                return DiskBackedRows(cache_file, rows_by_split[kwargs["split"]])
            return DiskBackedRows(cache_file)

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                dataset_profile="multidomain",
                dataset_loader=loader,
            )
            spec = manager.resource_spec("tnews")
            manager.ensure_dataset(spec)

            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                samples = list(manager.iter_source_samples(spec))

        self.assertEqual([sample.text for sample in samples], ["训练新闻样本。", "验证新闻样本。"])
        self.assertEqual(
            [call["split"] for call in loader_calls],
            ["train", "validation", "train", "validation"],
        )
        self.assertTrue(all(call["keep_in_memory"] is False for call in loader_calls))
        self.assertTrue(all("streaming" not in call for call in loader_calls))

    def test_raid_csv_is_processed_sequentially_without_bulk_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, profile="multidomain")
            csv_path = manager.dataset_dir / "raid_no_adversarial" / "train_none.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text(
                "id,generation,model,domain,attack\n"
                "human-1,A human report.,human,news,none\n"
                "ai-1,A generated review.,gpt2,reviews,none\n"
                "attacked,A modified sample.,gpt2,reviews,homoglyph\n",
                encoding="utf-8",
            )

            sample_iterator = manager.iter_samples("raid_no_adversarial")
            self.assertFalse(isinstance(sample_iterator, list))
            samples = list(sample_iterator)

        self.assertEqual([sample.label for sample in samples], ["human", "ai"])
        self.assertEqual(
            [sample.domain for sample in samples],
            ["official_news", "general_qa"],
        )

    def test_normalized_samples_and_fingerprint_marker_are_published_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, profile="multidomain")
            sample = TextSample(
                text="A stable fixture sample.",
                label="human",
                language="en",
                domain="general_qa",
                source="hc3_en",
                source_id="hc3_en:train:fixture:human:0",
            )
            output = manager.write_normalized_samples("hc3_en", [sample])
            marker = manager.normalized_marker_path("hc3_en")
            payload = json.loads(marker.read_text(encoding="utf-8"))
            serialized = output.read_text(encoding="utf-8").strip()

            self.assertTrue(manager.is_normalized_ready("hc3_en"))
            self.assertFalse(output.with_suffix(output.suffix + ".part").exists())
            self.assertEqual(payload["sample_count"], 1)
            self.assertEqual(len(payload["content_sha256"]), 64)
            self.assertEqual(
                payload["source_spec_fingerprint"],
                manager.resource_spec("hc3_en").fingerprint,
            )
            self.assertEqual(json.loads(serialized)["source_id"], sample.source_id)
            self.assertEqual(
                serialized,
                json.dumps(
                    json.loads(serialized),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

    def test_empty_normalized_output_is_rejected_and_never_marked_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, profile="multidomain")

            with self.assertRaisesRegex(ValueError, "at least one"):
                manager.write_normalized_samples("hc3_en", [])

            self.assertFalse(manager.normalized_path("hc3_en").exists())
            self.assertFalse(manager.normalized_marker_path("hc3_en").exists())
            self.assertFalse(manager.is_normalized_ready("hc3_en"))

    def test_normalized_readiness_rejects_nonpositive_marker_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, profile="multidomain")
            sample = TextSample(
                text="A normalized fixture.",
                label="human",
                language="en",
                domain="general_qa",
                source="hc3_en",
                source_id="hc3_en:train:fixture:human:0",
            )
            manager.write_normalized_samples("hc3_en", [sample])
            marker = manager.normalized_marker_path("hc3_en")
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["sample_count"] = 0
            marker.write_text(json.dumps(payload), encoding="utf-8")

            self.assertFalse(manager.is_normalized_ready("hc3_en"))

    def test_stale_normalized_fingerprint_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = DatasetSpec(
                name="fixture",
                url="https://example.invalid/v1.csv",
                license="CC0-1.0",
                domain="general_qa",
                source="https://example.invalid/project",
                source_kind="raid_csv",
                adapter="raid",
                filename="fixture.csv",
                language="en",
            )
            manager = TextResourceManager(project_root=root, dataset_specs=(original,))
            manager.write_normalized_samples(
                original,
                [
                    TextSample(
                        text="A fixture sample.",
                        label="human",
                        language="en",
                        domain="general_qa",
                        source="fixture",
                        source_id="fixture:train:1",
                    )
                ],
            )
            changed = DatasetSpec(
                name="fixture",
                url="https://example.invalid/v2.csv",
                license="CC0-1.0",
                domain="general_qa",
                source="https://example.invalid/project",
                source_kind="raid_csv",
                adapter="raid",
                filename="fixture.csv",
                language="en",
            )
            changed_manager = TextResourceManager(project_root=root, dataset_specs=(changed,))

            self.assertFalse(changed_manager.is_normalized_ready(changed))

    def test_failed_marker_publish_cannot_validate_replaced_normalized_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, profile="multidomain")
            original = TextSample(
                text="Original normalized content.",
                label="human",
                language="en",
                domain="general_qa",
                source="hc3_en",
                source_id="hc3_en:train:original:human:0",
            )
            replacement = replace(
                original,
                text="Replacement normalized content.",
                source_id="hc3_en:train:replacement:human:0",
            )
            output = manager.write_normalized_samples("hc3_en", [original])
            old_marker = manager.normalized_marker_path("hc3_en").read_bytes()

            with patch.object(
                manager,
                "_write_json_atomic",
                side_effect=OSError("marker publish failed"),
            ):
                with self.assertRaisesRegex(OSError, "marker publish failed"):
                    manager.write_normalized_samples("hc3_en", [replacement])

            self.assertIn("Replacement", output.read_text(encoding="utf-8"))
            self.assertEqual(
                manager.normalized_marker_path("hc3_en").read_bytes(),
                old_marker,
            )
            self.assertFalse(manager.is_normalized_ready("hc3_en"))

    def test_stale_source_fingerprint_and_missing_payload_are_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, profile="multidomain")
            spec = manager.resource_spec("raid_no_adversarial")
            source = manager.dataset_dir / spec.name
            source.mkdir(parents=True)
            marker = source / ".source.complete.json"
            marker.write_text(
                json.dumps({"source_spec_fingerprint": "stale"}),
                encoding="utf-8",
            )

            self.assertFalse(manager.is_source_ready(spec))
            marker.write_text(
                json.dumps({"source_spec_fingerprint": spec.fingerprint}),
                encoding="utf-8",
            )
            self.assertFalse(manager.is_source_ready(spec))

            (source / "train_none.csv").write_text(
                "id,generation,model,domain,attack\n",
                encoding="utf-8",
            )
            self.assertFalse(manager.is_source_ready(spec))

    def test_arbitrary_hf_cache_files_without_scan_manifest_are_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, profile="multidomain")
            spec = manager.resource_spec("hc3_en")
            source = manager.dataset_dir / spec.name
            cache = source / "hf_cache"
            cache.mkdir(parents=True)
            (source / ".source.complete.json").write_text(
                json.dumps({"source_spec_fingerprint": spec.fingerprint}),
                encoding="utf-8",
            )

            self.assertFalse(manager.is_source_ready(spec))
            (cache / "prepared.arrow").write_bytes(b"fixture")
            self.assertFalse(manager.is_source_ready(spec))

    def test_legacy_marker_must_match_current_spec_before_compatibility_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TextResourceManager(project_root=root)
            spec = manager.resource_spec("chinese_poetry")
            target = manager.dataset_dir / spec.name
            target.mkdir(parents=True)
            (target / "poems.json").write_text(
                json.dumps(
                    [{"title": "诗", "paragraphs": ["床前明月光疑是地上霜"]}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            legacy_marker = {
                "name": spec.name,
                "url": spec.url,
                "license": spec.license,
                "domain": spec.domain,
                "source": spec.source,
                "download_kind": spec.download_kind,
                "filename": spec.filename,
                "calibration_buckets": list(spec.calibration_buckets),
            }
            (target / ".complete.json").write_text(
                json.dumps(legacy_marker),
                encoding="utf-8",
            )

            self.assertTrue(manager.is_source_ready(spec))
            for changes in (
                {"url": "https://example.invalid/new-poetry.zip"},
                {"language": "en"},
                {"adapter": "hc3"},
                {"required": False},
            ):
                with self.subTest(changes=changes):
                    changed = replace(spec, **changes)
                    changed_manager = TextResourceManager(
                        project_root=root,
                        dataset_specs=(changed,),
                    )
                    self.assertFalse(changed_manager.is_source_ready(changed))

            incomplete_marker = dict(legacy_marker)
            incomplete_marker.pop("calibration_buckets")
            (target / ".complete.json").write_text(
                json.dumps(incomplete_marker),
                encoding="utf-8",
            )
            self.assertFalse(manager.is_source_ready(spec))
            (target / ".complete.json").write_text(
                json.dumps(legacy_marker),
                encoding="utf-8",
            )

            with patch.object(manager, "_download_and_extract") as download:
                manager.ensure_dataset(spec)
            migrated = json.loads(
                (target / ".complete.json").read_text(encoding="utf-8")
            )
            self.assertEqual(download.call_count, 0)
            self.assertEqual(
                migrated["source_spec_fingerprint"],
                spec.fingerprint,
            )
            self.assertEqual(migrated["preparation_kind"], "legacy_zip_tree_v1")
            self.assertTrue(manager.is_source_ready(spec))

    def test_zip_refresh_replaces_old_tree_without_stale_files(self) -> None:
        spec = DatasetSpec(
            name="zip_fixture",
            url="https://example.invalid/archive.zip",
            license="MIT",
            domain="literature",
            source="https://example.invalid/project",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, dataset_specs=(spec,))
            target = manager.dataset_dir / spec.name
            target.mkdir(parents=True)
            (target / "obsolete.txt").write_text("old", encoding="utf-8")

            def write_fixture_archive(url, destination, **_kwargs):
                with zipfile.ZipFile(destination, "w") as bundle:
                    bundle.writestr(
                        "prepared/new.json",
                        json.dumps(
                            [{"content": "authenticated replacement text"}]
                        ),
                    )

            with patch.object(
                manager,
                "_download_to_path",
                side_effect=write_fixture_archive,
            ):
                prepared = manager.ensure_dataset(spec)

            new_text = (prepared / "prepared" / "new.json").read_text(encoding="utf-8")
            obsolete_exists = (prepared / "obsolete.txt").exists()
            staging_exists = (prepared.parent / ".zip_fixture.staging").exists()
            backup_exists = (prepared.parent / ".zip_fixture.backup").exists()
            marker = json.loads(
                (prepared / ".complete.json").read_text(encoding="utf-8")
            )
            source_ready = manager.is_source_ready(spec)
            (prepared / "prepared" / "new.json").write_text("bad", encoding="utf-8")
            changed_ready = manager.is_source_ready(spec)
            (prepared / "prepared" / "new.json").unlink()
            deleted_ready = manager.is_source_ready(spec)

        self.assertIn("authenticated replacement text", new_text)
        self.assertFalse(obsolete_exists)
        self.assertFalse(staging_exists)
        self.assertFalse(backup_exists)
        self.assertEqual(marker["preparation_kind"], "legacy_zip_tree_v1")
        self.assertEqual(marker["files"][0]["path"], "prepared/new.json")
        self.assertGreater(marker["files"][0]["byte_size"], 3)
        self.assertEqual(len(marker["files"][0]["content_sha256"]), 64)
        self.assertIsInstance(marker["files"][0]["mtime_ns"], int)
        self.assertTrue(source_ready)
        self.assertFalse(changed_ready)
        self.assertFalse(deleted_ready)

    def test_zip_marker_failure_restores_the_previous_resource_tree(self) -> None:
        spec = DatasetSpec(
            name="zip_transaction",
            url="https://example.invalid/archive.zip",
            license="MIT",
            domain="literature",
            source="https://example.invalid/project",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, dataset_specs=(spec,))
            target = manager.dataset_dir / spec.name
            target.mkdir(parents=True)
            (target / "old.txt").write_text("old", encoding="utf-8")

            def write_fixture_archive(url, destination, **_kwargs):
                with zipfile.ZipFile(destination, "w") as bundle:
                    bundle.writestr(
                        "new.json",
                        json.dumps([{"content": "authenticated new text"}]),
                    )

            original_write = manager._write_json_atomic

            def fail_staged_marker(path, payload, **kwargs):
                if path.name == ".complete.json" and ".staging" in path.parent.name:
                    raise OSError("marker publish failed")
                return original_write(path, payload, **kwargs)

            with patch.object(
                manager,
                "_download_to_path",
                side_effect=write_fixture_archive,
            ), patch.object(
                manager,
                "_write_json_atomic",
                side_effect=fail_staged_marker,
            ):
                with self.assertRaisesRegex(OSError, "marker publish failed"):
                    manager.ensure_dataset(spec)

            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse((target / "new.json").exists())
            self.assertFalse((manager.dataset_dir / ".zip_transaction.staging").exists())

    def test_zip_retry_preserves_the_only_recoverable_backup_until_staging_is_ready(
        self,
    ) -> None:
        spec = DatasetSpec(
            name="zip_recovery",
            url="https://example.invalid/archive.zip",
            license="MIT",
            domain="literature",
            source="https://example.invalid/project",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, dataset_specs=(spec,))
            target = manager.dataset_dir / spec.name
            backup = manager.dataset_dir / ".zip_recovery.backup"
            backup.mkdir(parents=True)
            (backup / "recoverable.txt").write_text("recoverable", encoding="utf-8")

            with patch.object(
                manager,
                "_download_to_path",
                side_effect=OSError("download failed"),
            ):
                with self.assertRaisesRegex(OSError, "download failed"):
                    manager.ensure_dataset(spec)

            self.assertEqual(
                (target / "recoverable.txt").read_text(encoding="utf-8"),
                "recoverable",
            )

    def test_dataset_download_is_idempotent_after_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            spec = manager.dataset_specs[0]

            def prepare_fixture(_url, destination, **_kwargs):
                with zipfile.ZipFile(destination, "w") as bundle:
                    bundle.writestr(
                        "poems.json",
                        json.dumps(
                            [{"title": "诗", "paragraphs": ["认证诗歌正文内容"]}],
                            ensure_ascii=False,
                        ),
                    )

            with patch.object(
                manager,
                "_download_to_path",
                side_effect=prepare_fixture,
            ) as download:
                first = manager.ensure_dataset(spec)
                second = manager.ensure_dataset(spec)

            self.assertEqual(download.call_count, 1)
            self.assertEqual(first, second)
            self.assertTrue((first / ".complete.json").exists())

    def test_single_file_dataset_download_is_atomic_and_idempotent(self) -> None:
        class Response:
            status_code = 200
            payload = b'{"content":"classical text"}\n'
            headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                yield self.payload

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            spec = next(
                item for item in manager.dataset_specs if item.name == "chinese_classical_corpus"
            )
            with patch("detection.text_resources.requests.get", return_value=Response()) as request:
                first = manager.ensure_dataset(spec)
                second = manager.ensure_dataset(spec)

            downloaded = first / "corpus.jsonl"
            downloaded_bytes = downloaded.read_bytes()
            source_ready = manager.is_source_ready(spec)
            downloaded.write_bytes(b"x" * len(downloaded_bytes))
            changed_ready = manager.is_source_ready(spec)

        self.assertEqual(first, second)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(downloaded_bytes, b'{"content":"classical text"}\n')
        self.assertTrue(source_ready)
        self.assertFalse(changed_ready)

    def test_fast_readiness_uses_metadata_and_deep_verification_rehashes(self) -> None:
        class Response:
            status_code = 200
            headers = {"Content-Length": "39"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield b'{"content":"authenticated human text"}\n'

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            spec = manager.resource_spec("chinese_classical_corpus")
            with patch(
                "detection.text_resources.requests.get",
                return_value=Response(),
            ):
                target = manager.ensure_dataset(spec)
            source = target / "corpus.jsonl"
            original_stat = source.stat()
            original = source.read_bytes()

            with patch.object(
                manager,
                "_file_sha256",
                side_effect=AssertionError("fast readiness must not hash"),
            ):
                self.assertTrue(manager.is_source_ready(spec))
                self.assertEqual(manager.ensure_dataset(spec), target)

            source.write_bytes(b"x" * len(original))
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            self.assertTrue(manager.is_source_ready(spec))
            self.assertFalse(manager.is_source_ready(spec, deep=True))
            self.assertFalse(manager.verify_source(spec))

    def test_legacy_file_rejects_truncated_jsonl_before_marker_publication(self) -> None:
        class Response:
            status_code = 200
            payload = (
                b'{"content":"valid authenticated text"}\n'
                b'{"content":'
            )
            headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield self.payload

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp)
            spec = manager.resource_spec("chinese_classical_corpus")
            marker = manager.dataset_dir / spec.name / ".complete.json"
            with patch(
                "detection.text_resources.requests.get",
                return_value=Response(),
            ):
                with self.assertRaisesRegex(RuntimeError, "Malformed authenticated"):
                    manager.ensure_dataset(spec)

            self.assertFalse(marker.exists())
            self.assertFalse(manager.is_source_ready(spec))

    def test_authenticated_legacy_file_iteration_surfaces_malformed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp)
            spec = manager.resource_spec("chinese_classical_corpus")
            source = manager.dataset_dir / spec.name
            source.mkdir(parents=True)
            (source / spec.filename).write_text(
                '{"content":"valid authenticated text"}\n{"content":',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Malformed authenticated"):
                list(manager.iter_human_texts(spec.name))

    def test_direct_download_retries_and_leaves_no_partial_file(self) -> None:
        class Response:
            status_code = 200
            payload = (
                b"id,generation,model,domain,attack\n"
                b"human-1,A human report.,human,news,none\n"
                b"ai-1,A generated review.,gpt2,reviews,none\n"
            )
            headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield self.payload

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                profile="multidomain",
                download_retries=3,
            )
            spec = manager.resource_spec("raid_no_adversarial")
            with patch(
                "detection.text_resources.requests.get",
                side_effect=[
                    requests.ConnectTimeout("first"),
                    requests.ConnectTimeout("second"),
                    Response(),
                ],
            ) as request:
                target = manager.ensure_dataset(spec)

            destination = target / "train_none.csv"
            partial = destination.with_suffix(destination.suffix + ".part")
            downloaded_text = destination.read_text(encoding="utf-8")
            partial_exists = partial.exists()
            marker = json.loads(
                (target / ".source.complete.json").read_text(encoding="utf-8")
            )
            source_ready = manager.is_source_ready(spec)

        self.assertEqual(request.call_count, 3)
        self.assertIn("human-1,A human report.", downloaded_text)
        self.assertIn("ai-1,A generated review.", downloaded_text)
        self.assertFalse(partial_exists)
        self.assertTrue(source_ready)
        self.assertEqual(marker["byte_size"], len(downloaded_text.encode("utf-8")))
        self.assertEqual(
            marker["content_sha256"],
            hashlib.sha256(downloaded_text.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(marker["row_count"], 2)
        self.assertEqual(marker["label_counts"], {"human": 1, "ai": 1})

    def test_download_resumes_partial_content_with_range_and_backoff(self) -> None:
        class Response:
            def __init__(self, status_code, headers, chunks, error=None):
                self.status_code = status_code
                self.headers = headers
                self._chunks = chunks
                self._error = error

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield from self._chunks
                if self._error is not None:
                    raise self._error

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, download_retries=2)
            destination = manager.dataset_dir / "resume.bin"
            destination.parent.mkdir(parents=True)
            responses = [
                Response(
                    200,
                    {"Content-Length": "6", "ETag": '"fixture-v1"'},
                    [b"abc"],
                    requests.ConnectTimeout("interrupted"),
                ),
                Response(
                    206,
                    {
                        "Content-Length": "3",
                        "Content-Range": "bytes 3-5/6",
                        "ETag": '"fixture-v1"',
                    },
                    [b"def"],
                ),
            ]

            with patch(
                "detection.text_resources.requests.get",
                side_effect=responses,
            ) as request, patch("detection.text_resources.time.sleep") as sleep:
                manager._download_to_path("https://example.invalid/file", destination)

            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertFalse(destination.with_suffix(".bin.part").exists())
            self.assertEqual(
                request.call_args_list[1].kwargs["headers"],
                {
                    "Range": "bytes=3-",
                    "If-Range": '"fixture-v1"',
                },
            )
            sleep.assert_called_once_with(1.0)

    def test_download_restarts_when_server_ignores_range(self) -> None:
        class Response:
            status_code = 200
            headers = {"Content-Length": "3"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield b"new"

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp)
            destination = manager.dataset_dir / "restart.bin"
            destination.parent.mkdir(parents=True)
            destination.with_suffix(".bin.part").write_bytes(b"old")

            with patch(
                "detection.text_resources.requests.get",
                return_value=Response(),
            ):
                manager._download_to_path("https://example.invalid/file", destination)

            self.assertEqual(destination.read_bytes(), b"new")

    def test_raid_readiness_detects_file_corruption_and_ignores_part_debris(self) -> None:
        class Response:
            status_code = 200
            payload = (
                b"id,generation,model,domain,attack\n"
                b"1,A human sample.,human,news,none\n"
                b"2,A generated sample.,gpt2,reviews,none\n"
            )
            headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield self.payload

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                dataset_profile="multidomain",
            )
            spec = manager.resource_spec("raid_no_adversarial")
            with patch(
                "detection.text_resources.requests.get",
                return_value=Response(),
            ):
                target = manager.ensure_dataset(spec)

            data_path = target / "train_none.csv"
            self.assertTrue(manager.is_source_ready(spec))
            marker_path = target / ".source.complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            corrupt_counts = dict(marker)
            corrupt_counts["row_count"] = marker["row_count"] + 1
            marker_path.write_text(json.dumps(corrupt_counts), encoding="utf-8")
            self.assertFalse(manager.is_source_ready(spec))
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            self.assertTrue(manager.is_source_ready(spec))
            data_path.write_bytes(data_path.read_bytes() + b"corruption")
            (target / "train_none.csv.part").write_bytes(b"temporary")
            self.assertFalse(manager.is_source_ready(spec))

    def test_raid_header_only_or_missing_fields_never_publishes_marker(self) -> None:
        payloads = {
            "header_only": b"id,generation,model,domain,attack\n",
            "missing_model": (
                b"id,generation,domain,attack\n"
                b"1,A human sample.,news,none\n"
            ),
        }

        for name, payload in payloads.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                class Response:
                    status_code = 200
                    headers = {"Content-Length": str(len(payload))}

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def raise_for_status(self) -> None:
                        return None

                    def iter_content(self, chunk_size: int):
                        yield payload

                manager = TextResourceManager(
                    project_root=tmp,
                    dataset_profile="multidomain",
                )
                spec = manager.resource_spec("raid_no_adversarial")
                marker = manager.dataset_dir / spec.name / ".source.complete.json"
                with patch(
                    "detection.text_resources.requests.get",
                    return_value=Response(),
                ):
                    with self.assertRaisesRegex(RuntimeError, "RAID"):
                        manager.ensure_dataset(spec)

                self.assertFalse(marker.exists())
                self.assertFalse(manager.is_source_ready(spec))

    def test_iter_human_texts_reads_poetry_and_csl_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            poetry = manager.dataset_dir / "chinese_poetry"
            csl = manager.dataset_dir / "csl"
            poetry.mkdir(parents=True)
            csl.mkdir(parents=True)
            (poetry / "poems.json").write_text(
                json.dumps([{"title": "静夜思", "paragraphs": ["床前明月光", "疑是地上霜"]}], ensure_ascii=False),
                encoding="utf-8",
            )
            (csl / "train.jsonl").write_text(
                json.dumps({"text_a": "本文研究一种可靠的文本检测方法。"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            literature = list(manager.iter_human_texts("chinese_poetry", limit=5))
            academic = list(manager.iter_human_texts("csl", limit=5))

        self.assertEqual(literature, ["静夜思。床前明月光\n疑是地上霜"])
        self.assertEqual(academic, ["本文研究一种可靠的文本检测方法。"])

    def test_iter_human_texts_reads_official_csl_headerless_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            csl = manager.dataset_dir / "csl" / "CSL-master" / "benchmark" / "ts"
            csl.mkdir(parents=True)
            (csl / "train.tsv").write_text(
                "to title\t本文提出一种面向专业文本的稳健检测方法。\t稳健检测研究\n",
                encoding="utf-8",
            )

            academic = list(manager.iter_human_texts("csl", limit=5))

        self.assertEqual(academic, ["本文提出一种面向专业文本的稳健检测方法。"])

    def test_iter_human_texts_ignores_untracked_files_for_legacy_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            spec = manager.resource_spec("chinese_classical_corpus")
            source = manager.dataset_dir / spec.name
            source.mkdir(parents=True)
            (source / spec.filename).write_text(
                json.dumps(
                    {"content": "这是清单认证的古籍正文内容。"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (source / "stale.jsonl").write_text(
                json.dumps(
                    {"content": "这是未被清单认证的旧文件内容。"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            texts = list(manager.iter_human_texts(spec.name))

        self.assertEqual(texts, ["这是清单认证的古籍正文内容。"])

    def test_safe_zip_extraction_rejects_duplicate_encrypted_and_oversized_members(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                zip_max_members=1,
                zip_max_total_uncompressed=16,
                zip_max_file_uncompressed=12,
                zip_max_compression_ratio=2.0,
            )
            archive = Path(tmp) / "fixture.zip"
            target = manager.dataset_dir / "zip_target"
            target.mkdir(parents=True)

            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("same/file.txt", "a")
                bundle.writestr(r"same\file.txt", "b")
            with zipfile.ZipFile(archive) as bundle:
                with self.assertRaisesRegex(ValueError, "member count|Duplicate"):
                    manager._safe_extract(bundle, target)

            encrypted = zipfile.ZipInfo("secret.txt")
            encrypted.flag_bits |= 0x1
            encrypted.file_size = 1
            encrypted.compress_size = 1

            class EncryptedBundle:
                def infolist(self):
                    return [encrypted]

            with self.assertRaisesRegex(ValueError, "encrypted"):
                manager._safe_extract(EncryptedBundle(), target)

            manager.zip_max_members = 10
            with zipfile.ZipFile(
                archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as bundle:
                bundle.writestr("large.txt", "0" * 20)
            with zipfile.ZipFile(archive) as bundle:
                with self.assertRaisesRegex(
                    ValueError,
                    "per-file|total|compression ratio",
                ):
                    manager._safe_extract(bundle, target)

    def test_resume_rejects_unknown_total_and_never_publishes_partial(self) -> None:
        class Response:
            def __init__(self, status_code, headers, chunks, error=None):
                self.status_code = status_code
                self.headers = headers
                self._chunks = chunks
                self._error = error

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield from self._chunks
                if self._error is not None:
                    raise self._error

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(
                project_root=tmp,
                download_retries=2,
            )
            destination = manager.dataset_dir / "unknown-total.bin"
            destination.parent.mkdir(parents=True)
            responses = [
                Response(
                    200,
                    {"Content-Length": "5", "ETag": '"fixture-v1"'},
                    [b"abc"],
                    requests.ConnectTimeout("interrupted"),
                ),
                Response(
                    206,
                    {
                        "Content-Length": "2",
                        "Content-Range": "bytes 3-4/*",
                        "ETag": '"fixture-v1"',
                    },
                    [b"de"],
                ),
            ]

            with patch(
                "detection.text_resources.requests.get",
                side_effect=responses,
            ), patch("detection.text_resources.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "total"):
                    manager._download_to_path(
                        "https://example.invalid/unknown-total",
                        destination,
                    )

            self.assertFalse(destination.exists())

    def test_http_416_finalizes_only_an_exact_complete_partial(self) -> None:
        class Response:
            def __init__(self, status_code, headers, chunks=(), error=None):
                self.status_code = status_code
                self.headers = headers
                self._chunks = chunks
                self._error = error

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield from self._chunks
                if self._error is not None:
                    raise self._error

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, download_retries=2)
            destination = manager.dataset_dir / "complete.bin"
            destination.parent.mkdir(parents=True)
            responses = [
                Response(
                    200,
                    {
                        "Content-Length": "3",
                        "ETag": '"fixture-v1"',
                    },
                    [b"abc"],
                    requests.ConnectTimeout("response ended late"),
                ),
                Response(
                    416,
                    {
                        "Content-Range": "bytes */3",
                        "ETag": '"fixture-v1"',
                    },
                ),
            ]

            with patch(
                "detection.text_resources.requests.get",
                side_effect=responses,
            ), patch("detection.text_resources.time.sleep"):
                manager._download_to_path(
                    "https://example.invalid/complete",
                    destination,
                )

            self.assertEqual(destination.read_bytes(), b"abc")
            self.assertFalse(
                destination.with_name(destination.name + ".part").exists()
            )
            self.assertFalse(
                destination.with_name(destination.name + ".part.json").exists()
            )

    def test_http_416_with_changed_validator_discards_partial_and_restarts(self) -> None:
        class Response:
            def __init__(self, status_code, headers, chunks=(), error=None):
                self.status_code = status_code
                self.headers = headers
                self._chunks = chunks
                self._error = error

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield from self._chunks
                if self._error is not None:
                    raise self._error

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, download_retries=3)
            destination = manager.dataset_dir / "changed.bin"
            destination.parent.mkdir(parents=True)
            responses = [
                Response(
                    200,
                    {"Content-Length": "3", "ETag": '"fixture-v1"'},
                    [b"old"],
                    requests.ConnectTimeout("response ended late"),
                ),
                Response(
                    416,
                    {"Content-Range": "bytes */3", "ETag": '"fixture-v2"'},
                ),
                Response(
                    200,
                    {"Content-Length": "3", "ETag": '"fixture-v2"'},
                    [b"new"],
                ),
            ]

            with patch(
                "detection.text_resources.requests.get",
                side_effect=responses,
            ) as request, patch("detection.text_resources.time.sleep"):
                manager._download_to_path(
                    "https://example.invalid/changed",
                    destination,
                )

            self.assertEqual(destination.read_bytes(), b"new")
            self.assertEqual(request.call_count, 3)
            self.assertEqual(request.call_args_list[2].kwargs["headers"], {})

    def test_http_416_without_a_numeric_total_discards_before_retry(self) -> None:
        class Response:
            def __init__(self, status_code, headers, chunks=(), error=None):
                self.status_code = status_code
                self.headers = headers
                self._chunks = chunks
                self._error = error

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield from self._chunks
                if self._error is not None:
                    raise self._error

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, download_retries=3)
            destination = manager.dataset_dir / "invalid-416.bin"
            destination.parent.mkdir(parents=True)
            responses = [
                Response(
                    200,
                    {"Content-Length": "3", "ETag": '"fixture-v1"'},
                    [b"old"],
                    requests.ConnectTimeout("response ended late"),
                ),
                Response(
                    416,
                    {"Content-Range": "bytes */*", "ETag": '"fixture-v1"'},
                ),
                Response(
                    200,
                    {"Content-Length": "3", "ETag": '"fixture-v2"'},
                    [b"new"],
                ),
            ]

            with patch(
                "detection.text_resources.requests.get",
                side_effect=responses,
            ) as request, patch("detection.text_resources.time.sleep"):
                manager._download_to_path(
                    "https://example.invalid/invalid-416",
                    destination,
                )

            self.assertEqual(destination.read_bytes(), b"new")
            self.assertEqual(request.call_args_list[2].kwargs["headers"], {})

    def test_resume_sidecar_uses_if_range_and_restarts_for_a_new_source(self) -> None:
        class Response:
            def __init__(self, status_code, headers, chunks, error=None):
                self.status_code = status_code
                self.headers = headers
                self._chunks = chunks
                self._error = error

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield from self._chunks
                if self._error is not None:
                    raise self._error

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp, download_retries=1)
            destination = manager.dataset_dir / "sidecar.bin"
            destination.parent.mkdir(parents=True)
            interrupted = Response(
                200,
                {"Content-Length": "6", "ETag": '"fixture-v1"'},
                [b"abc"],
                requests.ConnectTimeout("interrupted"),
            )
            with patch(
                "detection.text_resources.requests.get",
                return_value=interrupted,
            ):
                with self.assertRaises(requests.ConnectTimeout):
                    manager._download_to_path(
                        "https://example.invalid/source-a",
                        destination,
                        source_fingerprint="spec-a",
                    )

            sidecar = destination.with_name(destination.name + ".part.json")
            self.assertTrue(sidecar.is_file())

            resumed = Response(
                206,
                {
                    "Content-Length": "3",
                    "Content-Range": "bytes 3-5/6",
                    "ETag": '"fixture-v1"',
                },
                [b"def"],
            )
            with patch(
                "detection.text_resources.requests.get",
                return_value=resumed,
            ) as request:
                manager._download_to_path(
                    "https://example.invalid/source-a",
                    destination,
                    source_fingerprint="spec-a",
                )

            self.assertEqual(
                request.call_args.kwargs["headers"],
                {
                    "Range": "bytes=3-",
                    "If-Range": '"fixture-v1"',
                },
            )
            self.assertEqual(destination.read_bytes(), b"abcdef")

            destination.unlink()
            interrupted = Response(
                200,
                {"Content-Length": "6", "ETag": '"fixture-v1"'},
                [b"old"],
                requests.ConnectTimeout("interrupted"),
            )
            with patch(
                "detection.text_resources.requests.get",
                return_value=interrupted,
            ):
                with self.assertRaises(requests.ConnectTimeout):
                    manager._download_to_path(
                        "https://example.invalid/source-a",
                        destination,
                        source_fingerprint="spec-a",
                    )

            replacement = Response(
                200,
                {"Content-Length": "3", "ETag": '"fixture-v2"'},
                [b"new"],
            )
            with patch(
                "detection.text_resources.requests.get",
                return_value=replacement,
            ) as request:
                manager._download_to_path(
                    "https://example.invalid/source-b",
                    destination,
                    source_fingerprint="spec-b",
                )

            self.assertEqual(request.call_args.kwargs["headers"], {})
            self.assertEqual(destination.read_bytes(), b"new")

    def test_download_limit_is_enforced_before_and_during_streaming(self) -> None:
        class Response:
            status_code = 200

            def __init__(self, headers, chunks):
                self.headers = headers
                self._chunks = chunks

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield from self._chunks

        cases = (
            Response({"Content-Length": "4"}, [b"data"]),
            Response({}, [b"toolarge"]),
        )
        for response in cases:
            with self.subTest(headers=response.headers), tempfile.TemporaryDirectory() as tmp:
                manager = TextResourceManager(
                    project_root=tmp,
                    download_retries=1,
                    max_download_bytes=3,
                )
                destination = manager.dataset_dir / "bounded.bin"
                destination.parent.mkdir(parents=True)
                with patch(
                    "detection.text_resources.requests.get",
                    return_value=response,
                ):
                    with self.assertRaisesRegex(RuntimeError, "size limit|total"):
                        manager._download_to_path(
                            "https://example.invalid/bounded",
                            destination,
                        )
                self.assertFalse(destination.exists())

    def test_zip_rejects_windows_unsafe_components_and_hierarchy_collisions(self) -> None:
        unsafe_names = (
            "dir/file.txt:stream",
            "dir/CON",
            "dir/con.txt",
            "dir/trailing.",
            "dir/trailing ",
            "dir//empty.txt",
            "dir/./dot.txt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp)
            archive = Path(tmp) / "unsafe.zip"
            target = manager.dataset_dir / "unsafe"
            for name in unsafe_names:
                with self.subTest(name=name):
                    with zipfile.ZipFile(archive, "w") as bundle:
                        bundle.writestr(name, "content")
                    with zipfile.ZipFile(archive) as bundle:
                        with self.assertRaisesRegex(ValueError, "Unsafe"):
                            manager._safe_extract(bundle, target)

            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("Folder", "file")
                bundle.writestr("folder/child.txt", "child")
            with zipfile.ZipFile(archive) as bundle:
                with self.assertRaisesRegex(ValueError, "collision"):
                    manager._safe_extract(bundle, target)

    def test_zip_manifest_authenticates_parseable_inputs_and_iteration_ignores_extras(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=tmp)
            spec = manager.resource_spec("chinese_poetry")

            def valid_archive(_url, destination, **_kwargs):
                with zipfile.ZipFile(destination, "w") as bundle:
                    bundle.writestr(
                        "poems.json",
                        json.dumps(
                            [{"title": "诗", "paragraphs": ["认证正文内容"]}],
                            ensure_ascii=False,
                        ),
                    )

            with patch.object(
                manager,
                "_download_to_path",
                side_effect=valid_archive,
            ):
                target = manager.ensure_dataset(spec)

            marker = json.loads(
                (target / ".complete.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["corpus_files"], ["poems.json"])
            self.assertEqual(marker["usable_text_count"], 1)

            (target / "untracked.json").write_text(
                json.dumps(
                    [{"title": "旧文件", "paragraphs": ["未认证旧正文内容"]}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                list(manager.iter_human_texts(spec.name)),
                ["诗。认证正文内容"],
            )

            marker["corpus_files"].append("untracked.json")
            (target / ".complete.json").write_text(
                json.dumps(marker),
                encoding="utf-8",
            )
            self.assertFalse(manager.is_source_ready(spec))
            self.assertEqual(list(manager.iter_human_texts(spec.name)), [])

            invalid_manager = TextResourceManager(
                project_root=Path(tmp) / "invalid",
            )
            invalid_spec = invalid_manager.resource_spec("chinese_poetry")

            def malformed_archive(_url, destination, **_kwargs):
                with zipfile.ZipFile(destination, "w") as bundle:
                    bundle.writestr("broken.json", '{"paragraphs":')

            with patch.object(
                invalid_manager,
                "_download_to_path",
                side_effect=malformed_archive,
            ):
                with self.assertRaisesRegex(RuntimeError, "Malformed"):
                    invalid_manager.ensure_dataset(invalid_spec)
            self.assertFalse(
                (
                    invalid_manager.dataset_dir
                    / invalid_spec.name
                    / ".complete.json"
                ).exists()
            )

    def test_file_resource_update_is_transactional_for_legacy_and_raid(self) -> None:
        fixtures = (
            (
                "core",
                "chinese_classical_corpus",
                b'{"content":"last known valid classical text"}\n',
                b'{"content":',
            ),
            (
                "multidomain",
                "raid_no_adversarial",
                (
                    b"id,generation,model,domain,attack\n"
                    b"1,A human sample.,human,news,none\n"
                    b"2,A generated sample.,gpt2,reviews,none\n"
                ),
                b"id,generation,model,domain,attack\n",
            ),
        )

        for profile, resource_name, valid, malformed in fixtures:
            with self.subTest(resource=resource_name), tempfile.TemporaryDirectory() as tmp:
                manager = TextResourceManager(project_root=tmp, profile=profile)
                spec = manager.resource_spec(resource_name)

                class Response:
                    status_code = 200

                    def __init__(self, payload):
                        self.payload = payload
                        self.headers = {"Content-Length": str(len(payload))}

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def raise_for_status(self) -> None:
                        return None

                    def iter_content(self, chunk_size: int):
                        yield self.payload

                with patch(
                    "detection.text_resources.requests.get",
                    return_value=Response(valid),
                ):
                    target = manager.ensure_dataset(spec)
                filename = spec.filename or "train_none.csv"
                original_marker = manager._source_marker_path(spec).read_bytes()
                self.assertEqual((target / filename).read_bytes(), valid)

                changed_spec = replace(
                    spec,
                    url=spec.url + "?revision=broken",
                )
                changed_manager = TextResourceManager(
                    project_root=tmp,
                    dataset_specs=(changed_spec,),
                )
                with patch(
                    "detection.text_resources.requests.get",
                    return_value=Response(malformed),
                ):
                    with self.assertRaises(RuntimeError):
                        changed_manager.ensure_dataset(changed_spec)

                self.assertEqual((target / filename).read_bytes(), valid)
                self.assertEqual(
                    manager._source_marker_path(spec).read_bytes(),
                    original_marker,
                )
                self.assertTrue(manager.is_source_ready(spec))

    def test_hf_requires_adapted_samples_and_expected_label_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            def empty_loader(*args, **kwargs):
                return DiskBackedRows(
                    Path(kwargs["cache_dir"]) / f"{kwargs['split']}.arrow",
                    [{"id": "empty", "sentence": ""}],
                )

            manager = TextResourceManager(
                project_root=Path(tmp) / "empty",
                profile="multidomain",
                dataset_loader=empty_loader,
            )
            spec = manager.resource_spec("tnews")
            with self.assertRaisesRegex(RuntimeError, "adapted"):
                manager.ensure_dataset(spec)
            self.assertFalse(manager.is_source_ready(spec))

            def human_only_hc3_loader(*args, **kwargs):
                return DiskBackedRows(
                    Path(kwargs["cache_dir"]) / f"{kwargs['split']}.arrow",
                    [
                        {
                            "id": "hc3-human-only",
                            "question": "Question?",
                            "human_answers": ["A human answer."],
                            "chatgpt_answers": [],
                        }
                    ],
                )

            manager = TextResourceManager(
                project_root=Path(tmp) / "labels",
                profile="multidomain",
                dataset_loader=human_only_hc3_loader,
            )
            spec = manager.resource_spec("hc3_en")
            with self.assertRaisesRegex(RuntimeError, "human and AI|labels"):
                manager.ensure_dataset(spec)
            self.assertFalse(manager.is_source_ready(spec))

            def paired_hc3_loader(*args, **kwargs):
                return DiskBackedRows(
                    Path(kwargs["cache_dir"]) / f"{kwargs['split']}.arrow",
                    [
                        {
                            "id": "hc3-pair",
                            "question": "Question?",
                            "human_answers": ["A human answer."],
                            "chatgpt_answers": ["An AI answer."],
                        }
                    ],
                )

            manager = TextResourceManager(
                project_root=Path(tmp) / "paired",
                profile="multidomain",
                dataset_loader=paired_hc3_loader,
            )
            spec = manager.resource_spec("hc3_en")
            target = manager.ensure_dataset(spec)
            marker_path = target / ".source.complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["expected_labels"], ["human", "ai"])
            self.assertEqual(
                marker["split_evidence"][spec.splits[0]]["label_counts"],
                {"human": 1, "ai": 1},
            )
            self.assertTrue(manager.is_source_ready(spec))

            marker["split_evidence"][spec.splits[0]]["label_counts"]["ai"] = 0
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            self.assertFalse(manager.is_source_ready(spec))

    def test_conformal_threshold_uses_finite_sample_order_statistic(self) -> None:
        scores = [index / 100 for index in range(100)]

        threshold = conformal_high_threshold(scores, alpha=0.02)

        self.assertEqual(threshold, 0.98)

    def test_conformal_threshold_is_infinite_when_sample_is_too_small(self) -> None:
        self.assertEqual(conformal_high_threshold([0.2, 0.4, 0.6], alpha=0.02), float("inf"))

    def test_short_documents_are_not_concatenated_into_fake_long_samples(self) -> None:
        spec = DatasetSpec(
            name="short_only",
            url="https://example.invalid/short-only.zip",
            license="CC0-1.0",
            domain="literature",
            source="https://example.invalid/short-only",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            source = manager.dataset_dir / spec.name
            source.mkdir(parents=True)
            (source / "items.jsonl").write_text(
                "\n".join(
                    json.dumps({"content": f"短篇原文{index}，保持独立。"}, ensure_ascii=False)
                    for index in range(12)
                ),
                encoding="utf-8",
            )

            samples = list(
                _balanced_human_samples(
                    manager,
                    spec.name,
                    sample_limit=3,
                    short_text_chars=20,
                    long_target_chars=30,
                )
            )

        self.assertFalse([text for text, bucket in samples if bucket == "long"])

    def test_one_source_document_contributes_at_most_one_calibration_window(self) -> None:
        spec = DatasetSpec(
            name="one_long_document",
            url="https://example.invalid/one-long-document.zip",
            license="CC0-1.0",
            domain="literature",
            source="https://example.invalid/one-long-document",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            source = manager.dataset_dir / spec.name
            source.mkdir(parents=True)
            (source / "items.jsonl").write_text(
                json.dumps({"content": "天地玄黄，宇宙洪荒。" * 80}, ensure_ascii=False),
                encoding="utf-8",
            )

            samples = list(
                _balanced_human_samples(
                    manager,
                    spec.name,
                    sample_limit=10,
                    short_text_chars=20,
                    long_target_chars=40,
                    long_max_chars=60,
                )
            )

        self.assertEqual(len([item for item in samples if item[1] == "long"]), 1)

    def test_build_threshold_artifact_scores_mature_human_corpora(self) -> None:
        class Result:
            def __init__(self, score: float) -> None:
                self.score = score

        class Detector:
            def detect(self, text: str) -> Result:
                return Result(0.7 if "本文" in text else 0.8)

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            poetry = manager.dataset_dir / "chinese_poetry"
            classical = manager.dataset_dir / "chinese_classical_corpus"
            csl = manager.dataset_dir / "csl"
            poetry.mkdir(parents=True)
            classical.mkdir(parents=True)
            csl.mkdir(parents=True)
            (poetry / "poems.json").write_text(
                json.dumps(
                    [
                        {"title": f"诗{index}", "paragraphs": [f"床前明月光第{index}句"]}
                        for index in range(12)
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (classical / "corpus.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {"content": f"古籍卷{index}。" + "天地玄黄，宇宙洪荒。" * 5},
                        ensure_ascii=False,
                    )
                    for index in range(12)
                ),
                encoding="utf-8",
            )
            (csl / "train.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "text_a": (
                                f"本文研究第{index}种可靠方法。"
                                if index % 2 == 0
                                else f"本文研究第{index}种可靠方法。" + "实验结果表明方法有效。" * 4
                            )
                        },
                        ensure_ascii=False,
                    )
                    for index in range(12)
                ),
                encoding="utf-8",
            )

            payload = build_threshold_artifact(
                manager=manager,
                detector=Detector(),
                sample_limit=4,
                validation_limit=1,
                minimum_group_samples=2,
                minimum_validation_samples=1,
                alpha=0.2,
                short_text_chars=20,
                long_target_chars=30,
                high_threshold_floors={"literature": 0.9},
                low_thresholds={"literature": 0.35},
            )

            saved = json.loads(manager.artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, saved)
        self.assertEqual(saved["calibration_kind"], "human_only_split_conformal")
        self.assertIn("zh:literature:short", saved["groups"])
        self.assertIn("zh:literature:long", saved["groups"])
        self.assertIn("zh:academic_technical:long", saved["groups"])
        self.assertIn("zh:academic_technical:short", saved["groups"])
        self.assertEqual(saved["groups"]["zh:literature:short"]["high_threshold"], 0.9)
        self.assertEqual(saved["groups"]["zh:literature:short"]["low_threshold"], 0.35)
        self.assertEqual(saved["groups"]["zh:literature:short"]["threshold_floor"], 0.9)
        literature = saved["groups"]["zh:literature:long"]
        self.assertIn("calibration_quantiles", literature)
        self.assertLessEqual(literature["validation_samples"], 1)
        self.assertGreaterEqual(literature["validation_false_positives"], 0)
        if literature["validation_samples"] == 0:
            self.assertIsNone(literature["validation_fpr"])
            self.assertIn("insufficient_human_validation", literature["warnings"])
        else:
            self.assertEqual(literature["validation_fpr"], 0.0)
        self.assertIn("validation_fpr_interval_95", literature)
        self.assertIn("reliability_status", literature)
        self.assertEqual(saved["sampling"]["seed"], 20260722)

    def test_calibration_reserves_holdout_documents_when_source_is_smaller_than_limits(self) -> None:
        class Detector:
            def detect(self, text: str):
                return type("Result", (), {"score": 0.2})()

        spec = DatasetSpec(
            name="limited_classical",
            url="https://example.invalid/limited-classical.zip",
            license="CC0-1.0",
            domain="literature",
            source="https://example.invalid/limited-classical",
            calibration_buckets=("long",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            source = manager.dataset_dir / spec.name
            source.mkdir(parents=True)
            (source / "items.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {"content": f"古籍卷{index}。" + "天地玄黄，宇宙洪荒。" * 8},
                        ensure_ascii=False,
                    )
                    for index in range(120)
                ),
                encoding="utf-8",
            )

            payload = build_threshold_artifact(
                manager=manager,
                detector=Detector(),
                sample_limit=100,
                validation_limit=50,
                minimum_group_samples=20,
                minimum_validation_samples=20,
                alpha=0.1,
                short_text_chars=20,
                long_target_chars=40,
                long_max_chars=60,
            )

        calibration_count = payload["source_length_buckets"][spec.name]["long"]
        validation_count = payload["validation_source_length_buckets"][spec.name]["long"]
        self.assertEqual(calibration_count + validation_count, 120)
        self.assertGreater(calibration_count, 0)
        self.assertGreater(validation_count, 0)

    def test_build_threshold_artifact_rejects_an_empty_required_source(self) -> None:
        class Detector:
            def detect(self, text: str):
                return type("Result", (), {"score": 0.2})()

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            poetry = manager.dataset_dir / "chinese_poetry"
            classical = manager.dataset_dir / "chinese_classical_corpus"
            poetry.mkdir(parents=True)
            classical.mkdir(parents=True)
            (poetry / "poems.json").write_text(
                json.dumps(
                    [{"title": f"诗{index}", "paragraphs": [f"床前明月光第{index}句"]} for index in range(4)],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (classical / "corpus.jsonl").write_text(
                "\n".join(
                    json.dumps({"content": f"古籍卷{index}。" + "天地玄黄，宇宙洪荒。" * 4}, ensure_ascii=False)
                    for index in range(4)
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "csl"):
                build_threshold_artifact(
                    manager=manager,
                    detector=Detector(),
                    sample_limit=3,
                    validation_limit=1,
                    minimum_group_samples=1,
                    minimum_validation_samples=1,
                    alpha=0.2,
                    short_text_chars=20,
                    long_target_chars=21,
                )

    def test_artifact_v2_uses_human_calibration_and_reports_held_out_ai_metrics(self) -> None:
        class Detector:
            def detect(self, text: str):
                score = float(text.rsplit("=", 1)[1])
                return type("Result", (), {"score": score})()

        spec = DatasetSpec(
            name="paired_fixture",
            url="https://example.invalid/paired.zip",
            license="CC0-1.0",
            domain="general_qa",
            source="https://example.invalid/paired",
            language="en",
        )
        seed = 17
        calibration_ids = [
            f"cal-{index}"
            for index in range(100)
            if _partition_assignment(f"cal-{index}", seed, 0.5) == "calibration"
        ][:4]
        validation_ids = [
            f"val-{index}"
            for index in range(100)
            if _partition_assignment(f"val-{index}", seed, 0.5) == "validation"
        ][:3]
        samples = [
            TextSample(
                text=f"Human calibration sample number {index} score={score}",
                label="human",
                language="en",
                domain="general_qa",
                source=spec.name,
                source_id=f"{partition}:human",
                partition_id=partition,
            )
            for index, (partition, score) in enumerate(
                zip(calibration_ids, (0.1, 0.2, 0.3, 0.4), strict=True)
            )
        ]
        for index, (partition, human_score, ai_score) in enumerate(
            zip(validation_ids, (0.1, 0.8, 0.2), (0.9, 0.7, 0.3), strict=True)
        ):
            samples.extend(
                [
                    TextSample(
                        text=f"Human validation sample number {index} score={human_score}",
                        label="human",
                        language="en",
                        domain="general_qa",
                        source=spec.name,
                        source_id=f"{partition}:human",
                        partition_id=partition,
                    ),
                    TextSample(
                        text=f"Artificial validation sample number {index} score={ai_score}",
                        label="ai",
                        language="en",
                        domain="general_qa",
                        source=spec.name,
                        source_id=f"{partition}:ai",
                        partition_id=partition,
                        generator="fixture-model",
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            with patch.object(manager, "iter_source_samples", return_value=iter(samples)):
                payload = build_threshold_artifact(
                    manager,
                    Detector(),
                    human_calibration_limit=10,
                    human_validation_limit=10,
                    ai_validation_limit=10,
                    validation_fraction=0.5,
                    minimum_group_samples=3,
                    minimum_validation_samples=2,
                    minimum_ai_validation_samples=2,
                    alpha=0.25,
                    sampling_seed=seed,
                    min_effective_chars=10,
                    short_text_chars=200,
                    high_threshold_floors={"default": 0.0},
                )

        group = payload["groups"]["en:general_qa:short"]
        self.assertEqual(payload["artifact_schema_version"], 2)
        self.assertEqual(payload["decision_operator"], "score > high_threshold")
        self.assertEqual(group["high_threshold"], 0.4)
        self.assertEqual(group["human_calibration_samples"], 4)
        self.assertEqual(group["human_validation_samples"], 3)
        self.assertEqual(group["ai_validation_samples"], 3)
        self.assertEqual(group["validation_false_positives"], 1)
        self.assertEqual(group["ai_true_positives"], 2)
        self.assertAlmostEqual(group["validation_fpr"], 1 / 3, places=6)
        self.assertAlmostEqual(group["ai_recall"], 2 / 3, places=6)
        self.assertAlmostEqual(group["ai_fnr"], 1 / 3, places=6)
        self.assertIsNotNone(group["validation_auroc"])
        self.assertIn("ai_recall_interval_95", group)
        self.assertEqual(payload["sampling"]["partition_key"], "TextSample.partition_key")

    def test_ai_score_changes_do_not_change_fitted_threshold(self) -> None:
        class Detector:
            def __init__(self, ai_score: float) -> None:
                self.ai_score = ai_score

            def detect(self, text: str):
                score = self.ai_score if text.startswith("AI") else float(text.rsplit("=", 1)[1])
                return type("Result", (), {"score": score})()

        spec = DatasetSpec(
            name="human_only_fit",
            url="https://example.invalid/human-only.zip",
            license="CC0-1.0",
            domain="general_qa",
            source="https://example.invalid/human-only",
            language="en",
        )
        seed = 29
        cal_ids = [
            f"c-{index}"
            for index in range(200)
            if _partition_assignment(f"c-{index}", seed, 0.5) == "calibration"
        ][:4]
        val_ids = [
            f"v-{index}"
            for index in range(200)
            if _partition_assignment(f"v-{index}", seed, 0.5) == "validation"
        ][:2]
        samples = [
            TextSample(
                text=f"Human fitted text score={score}",
                label="human",
                language="en",
                domain="general_qa",
                source=spec.name,
                source_id=partition,
                partition_id=partition,
            )
            for partition, score in zip(cal_ids, (0.1, 0.2, 0.3, 0.4), strict=True)
        ]
        for partition in val_ids:
            samples.extend(
                [
                    TextSample(
                        text="Human held out text score=0.2",
                        label="human",
                        language="en",
                        domain="general_qa",
                        source=spec.name,
                        source_id=f"{partition}:h",
                        partition_id=partition,
                    ),
                    TextSample(
                        text="AI held out text",
                        label="ai",
                        language="en",
                        domain="general_qa",
                        source=spec.name,
                        source_id=f"{partition}:a",
                        partition_id=partition,
                        generator="fixture",
                    ),
                ]
            )

        thresholds = []
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            for ai_score in (0.01, 0.99):
                with patch.object(manager, "iter_source_samples", return_value=iter(samples)):
                    result = build_threshold_artifact(
                        manager,
                        Detector(ai_score),
                        human_calibration_limit=10,
                        human_validation_limit=10,
                        ai_validation_limit=10,
                        validation_fraction=0.5,
                        minimum_group_samples=3,
                        minimum_validation_samples=1,
                        minimum_ai_validation_samples=1,
                        alpha=0.25,
                        sampling_seed=seed,
                        min_effective_chars=5,
                        high_threshold_floors={"default": 0.0},
                    )
                thresholds.append(result["groups"]["en:general_qa:short"]["high_threshold"])
        self.assertEqual(thresholds, [0.4, 0.4])

    def test_artifact_records_selected_identity_keys_for_reproducibility(self) -> None:
        class Detector:
            def detect(self, text: str):
                return type("Result", (), {"score": 0.2})()

        spec = DatasetSpec(
            name="identity_evidence",
            url="https://example.invalid/identity.zip",
            license="CC0-1.0",
            domain="general_qa",
            source="https://example.invalid/identity",
            language="en",
        )
        samples = [
            TextSample(
                text=f"Stable calibration identity sample {index}",
                label="human",
                language="en",
                domain="general_qa",
                source=spec.name,
                source_id=f"identity-{index}",
            )
            for index in range(6)
        ]
        expected_keys = sorted(sample.identity_key for sample in samples)

        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            with patch.object(manager, "iter_source_samples", return_value=iter(samples)):
                payload = build_threshold_artifact(
                    manager,
                    Detector(),
                    human_calibration_limit=10,
                    human_validation_limit=0,
                    ai_validation_limit=0,
                    validation_fraction=0,
                    minimum_group_samples=3,
                    minimum_validation_samples=0,
                    minimum_ai_validation_samples=0,
                    alpha=0.25,
                    sampling_seed=79,
                    min_effective_chars=5,
                    high_threshold_floors={"default": 0.0},
                )

        recorded = payload["sampling"]["selected_identity_keys"]
        self.assertEqual(
            recorded["en:general_qa:short"]["human_calibration"],
            expected_keys,
        )

    def test_holdout_fpr_warning_requires_wilson_lower_bound_above_target(self) -> None:
        group = bootstrap_text_resources._group_artifact(
            human_calibration=[0.1, 0.2, 0.3, 0.4],
            human_validation=[0.9, *([0.1] * 19)],
            ai_validation=[0.9, 0.8],
            threshold=0.4,
            threshold_floor=0.0,
            low_threshold=0.2,
            alpha=0.02,
            minimum_validation_samples=1,
            minimum_ai_validation_samples=1,
            minimum_ai_recall=0.5,
        )

        self.assertGreater(group["validation_fpr"], group["target_fpr"])
        self.assertLessEqual(
            group["validation_fpr_interval_95"][0],
            group["target_fpr"],
        )
        self.assertNotIn("holdout_fpr_above_target", group["warnings"])

    def test_partition_assignment_keeps_paired_samples_together(self) -> None:
        for partition in ("question-1", "question-2", "question-3"):
            human = TextSample(
                text="Human paired answer.",
                label="human",
                language="en",
                domain="general_qa",
                source="pair_source",
                source_id=f"{partition}:human",
                partition_id=partition,
            )
            ai = replace(
                human,
                text="AI paired answer.",
                label="ai",
                source_id=f"{partition}:ai",
                generator="fixture",
            )
            self.assertEqual(human.partition_key, ai.partition_key)
            self.assertEqual(
                _partition_assignment(human.partition_key, 123, 0.2),
                _partition_assignment(ai.partition_key, 123, 0.2),
            )

    def test_detector_failures_are_grouped_and_old_artifact_survives_insufficient_evidence(self) -> None:
        class BrokenDetector:
            def detect(self, text: str):
                if "broken" in text:
                    raise TimeoutError("fixture timeout")
                return type("Result", (), {"score": 0.2})()

        spec = DatasetSpec(
            name="failure_fixture",
            url="https://example.invalid/failure.zip",
            license="CC0-1.0",
            domain="general_qa",
            source="https://example.invalid/failure",
            language="en",
        )
        seed = 41
        calibration_ids = [
            f"failure-{index}"
            for index in range(100)
            if _partition_assignment(f"failure-{index}", seed, 0.5) == "calibration"
        ][:2]
        samples = [
            TextSample(
                text=f"{'broken' if index == 0 else 'usable'} human calibration text",
                label="human",
                language="en",
                domain="general_qa",
                source=spec.name,
                source_id=partition,
                partition_id=partition,
            )
            for index, partition in enumerate(calibration_ids)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            manager.artifact_path.parent.mkdir(parents=True)
            original = b'{"existing":"artifact"}\n'
            manager.artifact_path.write_bytes(original)
            with patch.object(manager, "iter_source_samples", return_value=iter(samples)):
                with self.assertRaisesRegex(RuntimeError, "calibration"):
                    build_threshold_artifact(
                        manager,
                        BrokenDetector(),
                        human_calibration_limit=10,
                        human_validation_limit=0,
                        ai_validation_limit=0,
                        validation_fraction=0.5,
                        minimum_group_samples=2,
                        minimum_validation_samples=0,
                        minimum_ai_validation_samples=0,
                        alpha=0.25,
                        sampling_seed=seed,
                        min_effective_chars=5,
                        high_threshold_floors={"default": 0.0},
                    )
            self.assertEqual(manager.artifact_path.read_bytes(), original)

    def test_selection_is_order_independent_and_excludes_too_short_samples(self) -> None:
        class Detector:
            def detect(self, text: str):
                return type("Result", (), {"score": 0.2})()

        spec = DatasetSpec(
            name="ordered_fixture",
            url="https://example.invalid/ordered.zip",
            license="CC0-1.0",
            domain="general_qa",
            source="https://example.invalid/ordered",
            language="en",
        )
        samples = [
            TextSample(
                text=f"Stable human calibration sample number {index}",
                label="human",
                language="en",
                domain="general_qa",
                source=spec.name,
                source_id=f"ordered-{index}",
            )
            for index in range(20)
        ]
        samples.append(
            TextSample(
                text="tiny",
                label="human",
                language="en",
                domain="general_qa",
                source=spec.name,
                source_id="too-short",
            )
        )
        normalized_outputs = []
        payloads = []
        for ordered in (samples, list(reversed(samples))):
            with tempfile.TemporaryDirectory() as tmp:
                manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
                with patch.object(manager, "iter_source_samples", return_value=iter(ordered)):
                    payload = build_threshold_artifact(
                        manager,
                        Detector(),
                        human_calibration_limit=5,
                        human_validation_limit=0,
                        ai_validation_limit=0,
                        validation_fraction=0,
                        minimum_group_samples=3,
                        minimum_validation_samples=0,
                        minimum_ai_validation_samples=0,
                        alpha=0.25,
                        sampling_seed=71,
                        min_effective_chars=10,
                        high_threshold_floors={"default": 0.0},
                    )
                normalized_outputs.append(
                    manager.normalized_path(spec).read_text(encoding="utf-8")
                )
                payloads.append(payload)

        self.assertEqual(normalized_outputs[0], normalized_outputs[1])
        self.assertEqual(payloads[0]["groups"], payloads[1]["groups"])
        self.assertEqual(
            payloads[0]["sampling"]["selected_group_counts"],
            payloads[1]["sampling"]["selected_group_counts"],
        )
        self.assertEqual(payloads[0]["sampling"]["excluded_short_total"], 1)

    def test_successful_artifact_records_detector_exception_classes(self) -> None:
        class Detector:
            def detect(self, text: str):
                if "timeout" in text:
                    raise TimeoutError("fixture")
                return type("Result", (), {"score": 0.2})()

        spec = DatasetSpec(
            name="recorded_failure",
            url="https://example.invalid/recorded-failure.zip",
            license="CC0-1.0",
            domain="general_qa",
            source="https://example.invalid/recorded-failure",
            language="en",
        )
        samples = [
            TextSample(
                text=f"{'timeout' if index == 0 else 'usable'} calibration sample {index}",
                label="human",
                language="en",
                domain="general_qa",
                source=spec.name,
                source_id=f"recorded-{index}",
            )
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            with patch.object(manager, "iter_source_samples", return_value=iter(samples)):
                payload = build_threshold_artifact(
                    manager,
                    Detector(),
                    human_calibration_limit=6,
                    human_validation_limit=0,
                    ai_validation_limit=0,
                    validation_fraction=0,
                    minimum_group_samples=4,
                    minimum_validation_samples=0,
                    minimum_ai_validation_samples=0,
                    alpha=0.25,
                    sampling_seed=73,
                    min_effective_chars=5,
                    high_threshold_floors={"default": 0.0},
                )

        self.assertEqual(payload["detector_failures"]["total"], 1)
        self.assertEqual(
            payload["detector_failures"]["by_exception_class"],
            {"TimeoutError": 1},
        )
        self.assertEqual(
            payload["detector_failures"]["by_role"]["human_calibration"],
            {"TimeoutError": 1},
        )

    def test_artifact_replace_failure_preserves_existing_bytes(self) -> None:
        class Detector:
            def detect(self, text: str):
                return type("Result", (), {"score": 0.2})()

        spec = DatasetSpec(
            name="atomic_fixture",
            url="https://example.invalid/atomic.zip",
            license="CC0-1.0",
            domain="general_qa",
            source="https://example.invalid/atomic",
            language="en",
        )
        seed = 53
        partitions = [
            f"atomic-{index}"
            for index in range(100)
            if _partition_assignment(f"atomic-{index}", seed, 0.5) == "calibration"
        ][:3]
        samples = [
            TextSample(
                text=f"Atomic human calibration text {index}",
                label="human",
                language="en",
                domain="general_qa",
                source=spec.name,
                source_id=partition,
                partition_id=partition,
            )
            for index, partition in enumerate(partitions)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            manager.artifact_path.parent.mkdir(parents=True)
            original = b'{"stable":"old"}'
            manager.artifact_path.write_bytes(original)
            with (
                patch.object(manager, "iter_source_samples", return_value=iter(samples)),
                patch(
                    "scripts.bootstrap_text_resources.os.replace",
                    side_effect=OSError("replace failed"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    build_threshold_artifact(
                        manager,
                        Detector(),
                        human_calibration_limit=10,
                        human_validation_limit=0,
                        ai_validation_limit=0,
                        minimum_group_samples=2,
                        minimum_validation_samples=0,
                        minimum_ai_validation_samples=0,
                        alpha=0.25,
                        validation_fraction=0.5,
                        sampling_seed=seed,
                        min_effective_chars=5,
                        high_threshold_floors={"default": 0.0},
                    )
            self.assertEqual(manager.artifact_path.read_bytes(), original)
            self.assertFalse(manager.artifact_path.with_suffix(".json.tmp").exists())

    def test_resource_stats_reads_normalized_samples_without_detector(self) -> None:
        spec = DatasetSpec(
            name="stats_fixture",
            url="https://example.invalid/stats.zip",
            license="CC0-1.0",
            domain="official_news",
            source="https://example.invalid/stats",
            language="en",
        )
        sample = TextSample(
            text="Prepared human news sample.",
            label="human",
            language="en",
            domain="official_news",
            source=spec.name,
            source_id="news-1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp), dataset_specs=(spec,))
            manager.write_normalized_samples(spec, [sample])
            stats = build_resource_stats(manager, short_text_chars=200)

        self.assertEqual(stats["sources"][spec.name]["normalized_samples"], 1)
        self.assertEqual(
            stats["sources"][spec.name]["counts"]["human:en:official_news:short"],
            1,
        )
        self.assertFalse(stats["sources"][spec.name]["source_ready"])

    def test_stats_cli_wires_dataset_profile_without_building_detector(self) -> None:
        manager = MagicMock()
        manager.dataset_profile = "core"
        with (
            patch.object(
                bootstrap_text_resources,
                "TextResourceManager",
                return_value=manager,
            ) as manager_factory,
            patch.object(
                bootstrap_text_resources,
                "build_resource_stats",
                return_value={"sources": {}},
            ) as stats_builder,
            patch.object(
                bootstrap_text_resources,
                "_build_detector",
                side_effect=AssertionError("stats must not build a detector"),
            ),
        ):
            result = bootstrap_text_resources.main(
                ["--profile", "stats", "--dataset-profile", "core"]
            )

        self.assertEqual(result, 0)
        manager_factory.assert_called_once()
        self.assertEqual(
            manager_factory.call_args.kwargs["dataset_profile"],
            "core",
        )
        stats_builder.assert_called_once()
        manager.ensure_datasets.assert_not_called()
        manager.ensure_models.assert_not_called()

    def test_legacy_sample_limit_flags_remain_parseable(self) -> None:
        args = bootstrap_text_resources.parse_args(
            ["--sample-limit", "11", "--validation-limit", "7"]
        )
        self.assertEqual(args.sample_limit, 11)
        self.assertEqual(args.validation_limit, 7)
        self.assertIsNone(args.human_calibration_limit)
        self.assertIsNone(args.human_validation_limit)
        self.assertIsNone(args.ai_validation_limit)


if __name__ == "__main__":
    unittest.main()
