from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from research.text_attribution.phase4_config import (
    canonical_config_hash,
    load_phase4_fast_track_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/text_attribution/training/family_phase4_fast_track.v1.json"
)


def test_loads_frozen_fast_track_contract() -> None:
    config = load_phase4_fast_track_config(CONFIG_PATH)
    assert config.seeds == (42, 52, 62, 12, 22, 32, 72, 82)
    assert config.views == (
        "closed",
        "fast-to-balanced",
        "balanced-to-fast",
    )
    assert config.production_seed == 42
    assert config.bootstrap_resamples == 2000
    assert len(canonical_config_hash(config)) == 64
    with pytest.raises(FrozenInstanceError):
        config.production_seed = 52  # type: ignore[misc]


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown.*unexpected"):
        load_phase4_fast_track_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("production_seed", 99, "production_seed"),
        ("cross_fit_folds", 4, "cross_fit_folds"),
        ("weight_step", 0.3, "weight_step"),
        ("branches", ["A4", "B0"], "branches"),
    ),
)
def test_invalid_contract_values_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / f"bad-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_phase4_fast_track_config(path)
