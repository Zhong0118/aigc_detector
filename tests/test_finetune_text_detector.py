from __future__ import annotations

import importlib
import json

import pytest


def _module():
    return importlib.import_module("scripts.finetune_text_detector")


def _row(text, label=0, split="train", group="group-a", identity="document-a"):
    return {
        "text": text,
        "label": label,
        "split": split,
        "group": group,
        "identity": identity,
        "source": "fixture",
    }


@pytest.mark.parametrize(
    "second, match",
    [
        (_row("另一篇文档", split="validation", identity="document-b"), "group"),
        (_row("相同  文本", split="test", group="group-b", identity="document-b"), "text"),
        (_row("另一篇文档", split="calibration", group="group-b"), "identity"),
    ],
)
def test_loader_rejects_cross_split_leakage(tmp_path, second, match):
    # Removing any of the group/text/identity checks must make this test fail.
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in [_row("相同 文本"), second]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        _module().load_dataset_rows(data)


def test_loader_preserves_held_out_splits_without_selecting_them(tmp_path):
    rows = [
        _row("训练文档", group="train"),
        _row("验证文档", label=1, split="validation", group="val", identity="val"),
        _row("校准文档", split="calibration", group="cal", identity="cal"),
        _row("测试文档", label=1, split="test", group="test", identity="test"),
    ]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    loaded = _module().load_dataset_rows(data)
    assert [row["split"] for row in loaded] == ["train", "validation", "calibration", "test"]
    selected = _module().select_split_rows(loaded, "train", max_per_label=10, seed=17)
    assert [row["text"] for row in selected] == ["训练文档"]
    with pytest.raises(ValueError, match="train.*validation"):
        _module().select_split_rows(loaded, "test", max_per_label=10, seed=17)


def test_source_stratified_selection_keeps_small_modern_source():
    rows = [
        dict(_row(f"文档 {index}", identity=f"id-{index}", group=f"g-{index}"), source="legacy")
        for index in range(20)
    ]
    rows.append(dict(_row("新模型文档", identity="modern", group="modern"), source="modern"))
    selected = _module().select_split_rows(rows, "train", max_per_label=4, seed=17)
    assert len(selected) == 4
    assert {row["source"] for row in selected} == {"legacy", "modern"}
    assert selected == _module().select_split_rows(rows, "train", max_per_label=4, seed=17)


def test_source_balanced_weights_equalize_labels_sources_and_document_mass():
    rows = [
        {"label": 0, "source": "legacy"}, {"label": 0, "source": "legacy"},
        {"label": 0, "source": "long_reports"}, {"label": 1, "source": "legacy"},
        {"label": 1, "source": "long_reports"}, {"label": 1, "source": "long_reports"},
    ]
    # Each pair of values for one document must sum to its single-window peer.
    weights = _module().training_sample_weights(rows, [2, 1, 2, 1, 2, 1], balance_sources=True)
    assert weights == pytest.approx([0.125, 0.125, 0.25, 0.25, 0.25, 0.5, 0.125, 0.125, 0.25])
    assert sum(weights[:5]) == pytest.approx(1.0)
    assert sum(weights[5:]) == pytest.approx(1.0)
    assert sum(weights[:3]) == pytest.approx(0.5)
    assert sum(weights[3:5]) == pytest.approx(0.5)
    assert weights[5] == pytest.approx(0.5)
    assert sum(weights[6:]) == pytest.approx(0.5)


def test_default_weights_preserve_class_balanced_document_sampling():
    rows = [{"label": 0, "source": "legacy"}, {"label": 0, "source": "legacy"},
            {"label": 0, "source": "long_reports"}, {"label": 1, "source": "long_reports"}]
    weights = _module().training_sample_weights(rows, [2, 1, 1, 2])
    assert weights == pytest.approx([1 / 6, 1 / 6, 1 / 3, 1 / 3, 0.5, 0.5])
    arguments = ["--data", "fixture.jsonl", "--base-model", "fixture", "--output", "output"]
    assert _module().parse_args(arguments).balance_sources is False
    assert _module().parse_args(arguments + ["--balance-sources"]).balance_sources is True


@pytest.fixture
def tiny_bert(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n人\n智\n。\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab), model_max_length=8)
    model = transformers.BertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=8, hidden_size=12, num_hidden_layers=6,
            num_attention_heads=3, intermediate_size=20,
            max_position_embeddings=8, num_labels=2,
        )
    )
    return tokenizer, model


def test_tokenizer_overflow_keeps_tail_and_training_uses_both_ends(tiny_bert):
    tokenizer, _ = tiny_bert
    windows = _module().document_windows("人" * 12 + "智" * 2, tokenizer, max_length=8)
    assert [window["token_weight"] for window in windows] == [6, 6, 2]
    token_ids = [token for window in windows for token in window["features"]["input_ids"][1:-1]]
    assert token_ids == [5] * 12 + [6] * 2
    selected = _module().evenly_spaced_windows(windows, limit=2)
    assert selected[0]["features"]["input_ids"] == [2] + [5] * 6 + [3]
    assert selected[1]["features"]["input_ids"] == [2, 6, 6, 3]


def test_freezing_keeps_only_last_four_layers_pooler_and_head_trainable(tiny_bert):
    torch = pytest.importorskip("torch")
    _, model = tiny_bert
    _module().freeze_bert_layers(model, last_layers=4)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert "bert.embeddings.word_embeddings.weight" not in trainable
    assert "bert.encoder.layer.1.attention.self.query.weight" not in trainable
    assert "bert.encoder.layer.2.attention.self.query.weight" in trainable
    assert "bert.encoder.layer.5.output.dense.weight" in trainable
    assert "bert.pooler.dense.weight" in trainable
    assert "classifier.weight" in trainable
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    output = model(input_ids=torch.tensor([[2, 5, 6, 3]]), labels=torch.tensor([1]))
    output.loss.backward()
    optimizer.step()
    assert torch.equal(before["bert.embeddings.word_embeddings.weight"], model.bert.embeddings.word_embeddings.weight)
    assert not torch.equal(before["classifier.weight"], model.classifier.weight)


def test_batched_document_scores_match_runtime_token_and_character_weights(tiny_bert):
    torch = pytest.importorskip("torch")
    from detection.text_detector import RoutedTextAigcDetector

    tokenizer, model = tiny_bert
    model.eval()
    rows = [_row("人" * 12 + "智" * 2), _row("人。智。人。智。", identity="other", group="other")]
    detector = RoutedTextAigcDetector(models={"zh_short": "fixture", "zh_long": "fixture"}, device="cpu", max_length=8)
    detector._loaded = {"zh_short": (tokenizer, model), "zh_long": (tokenizer, model)}
    expected = [detector.detect(row["text"]).score for row in rows]
    actual = _module().score_documents(model, tokenizer, rows, device="cpu", batch_size=3, max_length=8)
    assert actual == pytest.approx(expected, abs=0.0001)
    assert model.training is False


def test_tiny_offline_training_saves_reloadable_binary_checkpoint(tiny_bert, tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("sklearn")
    tokenizer, model = tiny_bert
    base = tmp_path / "base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    rows = []
    for index, (split, label, text) in enumerate([
        ("train", 0, "人人人"), ("train", 1, "智智智"),
        ("validation", 0, "人人"), ("validation", 1, "智智"),
        ("calibration", 0, "人"), ("test", 1, "智"),
    ]):
        rows.append(_row(text, label=label, split=split, group=f"group-{index}", identity=f"id-{index}"))
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "trained"
    _module().main([
        "--data", str(data), "--base-model", str(base), "--output", str(output),
        "--device", "cpu", "--epochs", "1", "--batch-size", "2", "--eval-batch-size", "2",
    ])
    checkpoint = transformers.AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True)
    assert checkpoint.config.id2label == {0: "human", 1: "ai"}
    metadata = json.loads((output / "training_metadata.json").read_text())
    assert metadata["selected_train"]["documents"] == 2
    assert metadata["selected_validation"]["documents"] == 2
    assert metadata["best_epoch"] == 1
    assert metadata["model_sha256"]["model.safetensors"]
    assert len(metadata["epochs"]) == 1
    assert metadata["epochs"][0]["validation_auroc"] in (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="output must be empty"):
        _module().main([
            "--data", str(data), "--base-model", str(base), "--output", str(output),
            "--device", "cpu", "--epochs", "1",
        ])


def test_freezing_supports_deberta_classifier_and_pooler():
    pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    config = transformers.DebertaV2Config(vocab_size=16, hidden_size=8, num_hidden_layers=3,
                                         num_attention_heads=2, intermediate_size=16, max_position_embeddings=16)
    model = transformers.DebertaV2ForSequenceClassification(config)
    _module().freeze_bert_layers(model, last_layers=1)
    assert not model.deberta.embeddings.word_embeddings.weight.requires_grad
    assert not model.deberta.encoder.layer[0].attention.self.query_proj.weight.requires_grad
    assert model.deberta.encoder.layer[-1].attention.self.query_proj.weight.requires_grad
    assert model.classifier.weight.requires_grad
    assert model.pooler.dense.weight.requires_grad
