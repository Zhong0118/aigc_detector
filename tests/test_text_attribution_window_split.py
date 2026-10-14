from research.text_attribution.training_data import AttributionSample
from research.text_attribution.window_split import (
    expand_samples_to_windows,
    split_token_ids,
)


class _CharTokenizer:
    num_special_tokens_to_add = staticmethod(lambda pair=False: 2)

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": list(range(len(text)))}

    def decode(self, token_ids, skip_special_tokens=True):
        # Tests only care that each window is a contiguous slice of the original.
        start = token_ids[0] if token_ids else 0
        return "x" * len(token_ids) if token_ids else ""


def _sample(text: str, key: str = "doc-1") -> AttributionSample:
    return AttributionSample(
        request_key=key,
        text=text,
        family_id="openai",
        model_id="gpt-5",
        prompt_id="p1",
        semantic_prompt_group="g1",
        language="zh",
        task_type="qa",
        target_length="long",
        temperature=0.0,
        repeat_index=0,
    )


def test_split_token_ids_covers_without_gap() -> None:
    spans = split_token_ids(list(range(10)), content_length=4, stride=4)
    assert spans == [(0, 4), (4, 8), (8, 10)]


def test_short_document_stays_one_window() -> None:
    rows, audit = expand_samples_to_windows(
        [_sample("short")],
        _CharTokenizer(),
        max_length=8,
    )
    assert len(rows) == 1
    assert rows[0].family_id == "openai"
    assert rows[0].model_id == "gpt-5"
    assert rows[0].semantic_prompt_group == "g1"
    assert rows[0].request_key == "doc-1::w0"
    assert audit["window_rows"] == 1
    assert audit["multi_window_documents"] == 0


def test_long_document_copies_labels_onto_each_window() -> None:
    rows, audit = expand_samples_to_windows(
        [_sample("abcdefghij", key="doc-9")],
        _CharTokenizer(),
        max_length=6,  # 2 special + 4 content
    )
    assert audit["source_documents"] == 1
    assert audit["window_rows"] == 3
    assert [row.request_key for row in rows] == [
        "doc-9::w0",
        "doc-9::w1",
        "doc-9::w2",
    ]
    assert {row.family_id for row in rows} == {"openai"}
    assert {row.prompt_id for row in rows} == {"p1"}


def test_tokenizer_limit_is_restored_after_window_tokenization_error() -> None:
    class BrokenTokenizer(_CharTokenizer):
        model_max_length = 512
        def __call__(self, *args, **kwargs):
            raise RuntimeError("tokenization failed")

    tokenizer = BrokenTokenizer()
    import pytest
    with pytest.raises(RuntimeError, match="tokenization failed"):
        expand_samples_to_windows([_sample("text")], tokenizer, max_length=8)
    assert tokenizer.model_max_length == 512
