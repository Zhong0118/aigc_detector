import importlib

import numpy as np
import pytest


def implementation():
    try:
        return importlib.import_module('scripts.run_image_score_fusion')
    except ModuleNotFoundError:
        pytest.fail('score fusion implementation is missing')


def test_calibration_prefers_more_accurate_first_model():
    module = implementation()
    good = np.array([[.95, .05], [.05, .95]])
    bad = good[:, ::-1]
    alpha, losses = module.fit_weight(np.array([0, 1]), good, bad)
    assert alpha == 1.
    assert len(losses) == 21


def test_calibration_can_prefer_second_model():
    module = implementation()
    good = np.array([[.95, .05], [.05, .95]])
    alpha, _ = module.fit_weight(np.array([0, 1]), good[:, ::-1], good)
    assert alpha == 0.


def test_equal_models_prefer_unchanged_resnet_endpoint():
    module = implementation()
    scores = np.array([[.8, .2], [.2, .8]])
    alpha, _ = module.fit_weight(np.array([0, 1]), scores, scores)
    assert alpha == 1.


def test_mixing_preserves_probability_and_endpoints():
    module = implementation()
    first = np.array([[.8, .2], [.3, .7]])
    second = first[:, ::-1]
    assert np.allclose(module.mix_scores(first, second, .25), .25*first + .75*second)
    assert np.allclose(module.mix_scores(first, second, 1), first)
    assert np.allclose(module.mix_scores(first, second, 0), second)


@pytest.mark.parametrize('scores', [np.array([[2., -1.]]), np.array([[.3, .3]]),
                                   np.array([[np.nan, .5]])])
def test_invalid_probabilities_are_rejected(scores):
    module = implementation()
    with pytest.raises(ValueError):
        module.fit_weight(np.array([0]), scores, scores)


def test_mismatched_samples_and_out_of_range_weight_are_rejected():
    module = implementation()
    first = np.array([[.8, .2], [.3, .7]])
    with pytest.raises(ValueError):
        module.mix_scores(first, first[:1], .5)
    with pytest.raises(ValueError):
        module.mix_scores(first, first, 1.1)
    with pytest.raises(ValueError):
        module.fit_weight(np.array([0]), first, first)
