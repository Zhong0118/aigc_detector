from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
from unittest.mock import patch

from research.text_attribution.training_data import AttributionSample
from research.text_attribution.window_split import expand_samples_to_windows

import numpy as np
import pytest

from research.text_attribution import phase3_config as config
from research.text_attribution import phase3_training as training

CONFIG = Path(__file__).parents[1] / "configs/text_attribution/training/family_phase3_enhancement.v1.json"


def test_legacy_window_recipe_cannot_bypass_hash(tmp_path):
    recipe = config.resolve_phase3_recipe(config.load_phase3_config(CONFIG), "A0")
    path = tmp_path / "resolved_recipe.json"
    config.write_resolved_recipe(path, replace(recipe, recipe_id="A0-window-split", stage_id="A0-window-split"))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="hash"):
        config.load_resolved_recipe(path, verify_recipe_hash=False)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="hash"):
        training.train_phase3_experiment(
            training.Phase3TrainingRequest(
                recipe=replace(recipe, recipe_id="A0-window-split", stage_id="A0-window-split"),
                data_view=None,
                output_dir=tmp_path / "no-training",
                seed=42,
            )
        )
    assert not (tmp_path / "no-training").exists()


def test_window_profile_hash_roundtrip_and_stride_identity(tmp_path):
    base = config.resolve_phase3_recipe(config.load_phase3_config(CONFIG), "A0")
    profile = config.apply_window_split_profile(base, stride=510)
    changed = config.apply_window_split_profile(base, stride=255)
    assert profile.recipe_hash != base.recipe_hash
    assert profile.recipe_hash != changed.recipe_hash
    assert profile.recipe_hash != config.apply_window_split_profile(replace(base, max_length=256), stride=254).recipe_hash
    path = tmp_path / "resolved_recipe.json"
    config.write_resolved_recipe(path, profile)
    assert config.load_resolved_recipe(path) == profile
    assert profile.parent_recipe_id == "A0"


def test_checkpoint_metrics_average_probabilities_by_document():
    # The long document's head is wrong; its two tails reverse the decision.
    # Mean logits stays wrong, so this also detects averaging the wrong quantity.
    logits = np.array([[12., 0.], [0., 3.], [0., 3.], [4., 0.]])
    prediction = SimpleNamespace(predictions=logits, label_ids=np.array([1, 1, 1, 0]))
    metrics = training._compute_window_document_metrics(prediction, window_counts=(3, 1))
    assert metrics["macro_f1"] == 1.0
    probabilities = training._window_document_probabilities(logits, (3, 1))
    expected = training._softmax(logits)
    np.testing.assert_allclose(probabilities, [expected[:3].mean(axis=0), expected[3]])


def test_validation_expansion_preserves_document_order_and_all_tails():
    from research.text_attribution import window_split

    class Tokenizer:
        model_max_length = 512
        num_special_tokens_to_add = staticmethod(lambda pair=False: 2)
        def __call__(self, text, **kwargs):
            assert text != "DO NOT FORWARD TEST"
            return {"input_ids": [ord(c) for c in text]}
        def decode(self, ids, **kwargs):
            return "".join(chr(i) for i in ids)

    def sample(text, key):
        return AttributionSample(key, text, "a", "a-fast", "p", "g", "en", "qa", "long", 0., 0)
    samples = (sample("abcdefghij", "doc::w0"), sample("z", "other"))
    windows, counts = window_split.expand_validation_windows(samples, Tokenizer(), max_length=6)
    assert counts == (3, 1)
    assert [row.text for row in windows] == ["abcd", "efgh", "ij", "z"]


def test_window_runner_passes_only_validation_windows_and_hash_bound_recipe(tmp_path, monkeypatch):
    from scripts import run_family_attribution_window_split as runner
    from research.text_attribution.phase3_data import Phase3DataView
    import transformers

    class Tokenizer:
        model_max_length = 512
        num_special_tokens_to_add = staticmethod(lambda pair=False: 2)
        def __call__(self, text, **kwargs):
            assert "TEST SECRET" not in text
            return {"input_ids": [ord(c) for c in text]}
        def decode(self, ids, **kwargs):
            return "".join(chr(i) for i in ids)

    def sample(text, key):
        return AttributionSample(key, text, "a", "a-fast", "p", "g", "en", "qa", "long", 0., 0)
    train = (sample("x" * 520, "train"),)
    validation = (sample("y" * 1100, "val"),)
    test = (sample("TEST SECRET", "test"),)
    view = Phase3DataView("closed", train, validation, {"a": 0}, {0: "a"}, {"a-fast": 0}, {0: "a-fast"}, {"a-fast": 0})
    monkeypatch.setattr(runner, "load_training_dataset", lambda *args: SimpleNamespace(samples=train + validation + test))
    monkeypatch.setattr(runner, "load_variant_catalog", lambda *args: None)
    monkeypatch.setattr(runner, "load_split_manifest", lambda *args: SimpleNamespace(dataset_fingerprint="d" * 64))
    monkeypatch.setattr(runner, "apply_split_manifest", lambda *args: {"train": train, "validation": validation, "test": test})
    monkeypatch.setattr(runner, "build_phase3_data_view", lambda *args, **kwargs: view)
    monkeypatch.setattr(runner, "_local_xlmr_source", lambda *args: tmp_path)
    monkeypatch.setattr(runner, "_sha256_file", lambda *args: "f" * 64)
    monkeypatch.setattr(runner, "_build_validation_only_audit", lambda *args: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())

    class Captured(Exception):
        pass

    def inspect_request(request):
        assert request.validation_window_counts == (3,)
        assert [len(row.text) for row in request.validation_evaluation_samples] == [510, 510, 80]
        assert request.data_view.validation_samples == validation
        assert all("TEST SECRET" not in row.text for row in request.data_view.train_samples)
        assert request.recipe.recipe_id == "A0-window-split"
        path = tmp_path / "check-recipe.json"
        config.write_resolved_recipe(path, request.recipe)
        assert config.load_resolved_recipe(path) == request.recipe
        raise Captured

    monkeypatch.setattr(training, "train_phase3_experiment", inspect_request)
    with pytest.raises(Captured):
        runner.main(["--input", str(tmp_path / "input"), "--split-manifest", str(tmp_path / "split"),
                     "--output-dir", str(tmp_path / "run"), "--smoke-test", "--device", "cpu"])
