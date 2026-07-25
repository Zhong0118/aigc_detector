from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from detection.text_resource_adapters import (
    TextSample,
    adapt_hc3_rows,
    adapt_m4_rows,
    adapt_raid_rows,
    adapt_tnews_rows,
    normalize_source_domain,
)
from detection.text_resources import DatasetSpec, TextResourceManager, conformal_high_threshold
from scripts.bootstrap_text_resources import _balanced_human_samples, build_threshold_artifact


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
    def test_default_resources_stay_inside_project_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TextResourceManager(project_root=root)

            status = manager.status()

        self.assertEqual(Path(status["model_dir"]), root / "models" / "huggingface")
        self.assertEqual(Path(status["data_dir"]), root / "data" / "text_false_positive" / "corpora")
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

    def test_dataset_download_is_idempotent_after_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TextResourceManager(project_root=Path(tmp))
            spec = manager.dataset_specs[0]

            with patch.object(manager, "_download_and_extract") as download:
                first = manager.ensure_dataset(spec)
                second = manager.ensure_dataset(spec)

            self.assertEqual(download.call_count, 1)
            self.assertEqual(first, second)
            self.assertTrue((first / ".complete.json").exists())

    def test_single_file_dataset_download_is_atomic_and_idempotent(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                yield b'{"content":"classical text"}\n'

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

        self.assertEqual(first, second)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(downloaded_bytes, b'{"content":"classical text"}\n')

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
        self.assertEqual(literature["validation_samples"], 1)
        self.assertEqual(literature["validation_false_positives"], 0)
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

        self.assertEqual(payload["source_length_buckets"][spec.name]["long"], 96)
        self.assertEqual(payload["validation_source_length_buckets"][spec.name]["long"], 24)

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


if __name__ == "__main__":
    unittest.main()
