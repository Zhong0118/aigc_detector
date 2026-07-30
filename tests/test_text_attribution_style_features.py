from __future__ import annotations

import unittest

import numpy as np

from research.text_attribution.style_features import StylometricFeatureExtractor


class StylometricFeatureExtractorTests(unittest.TestCase):
    def test_exposes_the_exact_public_feature_name_contract(self) -> None:
        self.assertEqual(
            StylometricFeatureExtractor().get_feature_names_out().tolist(),
            [
                "char_count_log1p",
                "word_count_log1p",
                "sentence_count_log1p",
                "paragraph_count_log1p",
                "mean_sentence_chars_log1p",
                "std_sentence_chars_log1p",
                "mean_paragraph_chars_log1p",
                "cjk_ratio",
                "latin_ratio",
                "digit_ratio",
                "whitespace_ratio",
                "newline_ratio",
                "comma_per_1000",
                "period_per_1000",
                "colon_per_1000",
                "semicolon_per_1000",
                "question_per_1000",
                "exclamation_per_1000",
                "quote_per_1000",
                "parenthesis_per_1000",
                "dash_per_1000",
                "markdown_heading_ratio",
                "bullet_line_ratio",
                "numbered_line_ratio",
                "indented_line_ratio",
                "bold_marker_per_1000",
                "code_fence_per_1000",
                "type_token_ratio",
                "repeated_bigram_ratio",
                "zh_de_ratio",
                "zh_shi_ratio",
                "zh_because_ratio",
                "zh_however_ratio",
                "en_the_ratio",
                "en_and_ratio",
                "en_because_ratio",
                "en_however_ratio",
                "starts_with_transition",
                "ends_with_summary",
            ],
        )
        self.assertEqual(
            StylometricFeatureExtractor(include_length_features=False)
            .get_feature_names_out()
            .tolist(),
            [
                "cjk_ratio",
                "latin_ratio",
                "digit_ratio",
                "whitespace_ratio",
                "newline_ratio",
                "comma_per_1000",
                "period_per_1000",
                "colon_per_1000",
                "semicolon_per_1000",
                "question_per_1000",
                "exclamation_per_1000",
                "quote_per_1000",
                "parenthesis_per_1000",
                "dash_per_1000",
                "markdown_heading_ratio",
                "bullet_line_ratio",
                "numbered_line_ratio",
                "indented_line_ratio",
                "bold_marker_per_1000",
                "code_fence_per_1000",
                "type_token_ratio",
                "repeated_bigram_ratio",
                "zh_de_ratio",
                "zh_shi_ratio",
                "zh_because_ratio",
                "zh_however_ratio",
                "en_the_ratio",
                "en_and_ratio",
                "en_because_ratio",
                "en_however_ratio",
                "starts_with_transition",
                "ends_with_summary",
            ],
        )

    def test_extracts_finite_multilingual_markdown_features(self) -> None:
        extractor = StylometricFeatureExtractor()
        chinese = "因此，以下是说明。\n\n# 标题\n\n- 第一项\n- 第二项\n\n这是结论。"
        english = "Introduction.\n\nThis is a plain English paragraph, however it ends here."

        matrix = extractor.fit_transform([chinese, english])
        names = extractor.get_feature_names_out()
        features = dict(zip(names, matrix[0], strict=True))

        self.assertEqual(matrix.shape, (2, len(names)))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertGreater(features["markdown_heading_ratio"], 0.0)
        self.assertGreater(features["bullet_line_ratio"], 0.0)
        self.assertGreater(features["cjk_ratio"], matrix[1, list(names).index("cjk_ratio")])
        self.assertEqual(features["starts_with_transition"], 1.0)
        self.assertEqual(features["ends_with_summary"], 1.0)

    def test_can_disable_length_features_without_removing_style_features(self) -> None:
        extractor = StylometricFeatureExtractor(include_length_features=False)
        names = extractor.fit(["One, two."]).get_feature_names_out()

        self.assertNotIn("char_count_log1p", names)
        self.assertIn("comma_per_1000", names)

    def test_handles_empty_whitespace_and_punctuation_only_texts(self) -> None:
        extractor = StylometricFeatureExtractor()
        matrix = extractor.fit_transform(["", " \n\t ", "?!...，，，"])

        self.assertEqual(matrix.shape, (3, len(extractor.get_feature_names_out())))
        self.assertTrue(np.isfinite(matrix).all())

    def test_is_deterministic_and_exposes_unique_ordered_names(self) -> None:
        extractor = StylometricFeatureExtractor()
        texts = ["However, this is a test.", "因为这是一个测试，所以总结。"]

        first = extractor.fit_transform(texts)
        second = extractor.transform(texts)
        names = extractor.get_feature_names_out()

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(names), len(set(names.tolist())))
        self.assertEqual(names[:3].tolist(), ["char_count_log1p", "word_count_log1p", "sentence_count_log1p"])
        self.assertLess(names.tolist().index("comma_per_1000"), names.tolist().index("markdown_heading_ratio"))

    def test_rejects_scalar_and_non_string_inputs(self) -> None:
        extractor = StylometricFeatureExtractor()

        with self.assertRaisesRegex(ValueError, "sequence"):
            extractor.fit_transform("not a text sequence")
        with self.assertRaisesRegex(TypeError, "strings"):
            extractor.fit_transform(["valid", 7])


if __name__ == "__main__":
    unittest.main()
