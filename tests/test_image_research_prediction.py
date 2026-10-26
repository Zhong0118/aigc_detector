import pytest
import json
import hashlib
from scripts.predict_image_research import task_output, verify_completed_run


def test_detection_has_no_source_probability():
    result = task_output('detection', ['real', 'ai'], [.9, .1])
    assert result['label'] == 'real'
    assert result['score_ai'] == .1
    assert 'top_k' not in result
    assert result['source_scores_used'] is False


def test_source_prediction_cannot_claim_known_or_ai():
    result = task_output('attribution', ['a', 'b'], [.9, .1])
    assert result['unknown']['decision'] is None
    assert result['conditional_on_ai'] is True
    assert 'score_ai' not in result
    with pytest.raises(ValueError):
        task_output('attribution', ['real', 'a'], [.9, .1])
    with pytest.raises(ValueError):
        task_output('detection', ['a', 'b'], [.9, .1])


def test_no_nan_or_invalid_probability_fallback():
    for values in ([float('nan'), .1], [-.1, 1.1], [.4, .4]):
        with pytest.raises(ValueError):
            task_output('detection', ['real', 'ai'], values)


def test_only_completed_and_hash_verified_heads_are_accepted(tmp_path):
    (tmp_path / 'head.pth').write_bytes(b'fixture-head')
    (tmp_path / 'provenance.json').write_text(json.dumps(dict(head_sha256=hashlib.sha256(b'fixture-head').hexdigest())))
    for status in ('running', 'failed'):
        (tmp_path / 'status.json').write_text(json.dumps(dict(status=status, exit_code=1)))
        with pytest.raises(ValueError, match='not completed'):
            verify_completed_run(tmp_path)
    (tmp_path / 'status.json').write_text(json.dumps(dict(status='completed', exit_code=0)))
    verify_completed_run(tmp_path)
    (tmp_path / 'head.pth').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='head hash'):
        verify_completed_run(tmp_path)
