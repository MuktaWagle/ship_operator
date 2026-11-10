"""Tests for the scenario_source option (real / synthetic / mixed scenarios).

The helper logic needs only numpy. The EnvConfig tests need torch (they import the training package)
and are skipped without it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("synthetic_under_test", ROOT / "training" / "synthetic_scenarios.py")
syn = importlib.util.module_from_spec(_spec)
sys.modules["synthetic_under_test"] = syn
_spec.loader.exec_module(syn)


def test_parse_spec_accepts_pairs_and_rejects_garbage():
    assert syn.parse_spec("400:1500, 250:1000") == [(400.0, 1500.0), (250.0, 1000.0)]
    for bad in ("", "400", "a:b", "400:1500:7"):
        with pytest.raises(ValueError):
            syn.parse_spec(bad)


def test_uses_synthetic_by_source():
    assert not any(syn.uses_synthetic("real", s) for s in range(10))
    assert all(syn.uses_synthetic("synthetic", s) for s in range(10))
    assert [syn.uses_synthetic("mixed", s) for s in range(6)] == [True, False, True, False, True, False]
    with pytest.raises(ValueError):
        syn.uses_synthetic("fake", 0)


def test_settings_take_turns():
    spec = "400:1500,250:1000"
    assert [syn.setting_for("synthetic", spec, s) for s in range(4)] == [(400, 1500), (250, 1000)] * 2
    # mixed only uses even seeds for synthetic, so it must still alternate across those
    assert [syn.setting_for("mixed", spec, s) for s in (0, 2, 4, 6)] == [(400, 1500), (250, 1000)] * 2


def test_generated_scenarios_have_the_simulator_format_and_unique_ids():
    seen = set()
    for seed in (0, 1, 10_000, 20_000):
        trajectories, timestamps, ids = syn.make_crossing_scenario(10, seed, spread_s=250.0, offset_m=1000.0)
        assert len(trajectories) == len(timestamps) == len(ids) == 10
        for trajectory, stamps in zip(trajectories, timestamps):
            assert trajectory.ndim == 2 and trajectory.shape[1] == 4 and len(trajectory) >= 2
            assert len(stamps) == len(trajectory) and stamps[0] == 0 and (stamps[1:] - stamps[:-1] == 10).all()
        assert not seen & set(ids), "ship ids must differ between seeds (they define the 'ship set')"
        seen |= set(ids)
    assert min(seen) >= 10_000  # never clashes with real ship ids (<= 5341)


def test_envconfig_defaults_roundtrip_and_validation():
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT))
    from training.scenarios import EnvConfig

    config = EnvConfig()
    assert config.scenario_source == "real"  # old behaviour stays the default
    assert EnvConfig(**config.to_dict()) == config
    old = config.to_dict()
    old.pop("scenario_source")
    old.pop("synthetic_spec")
    assert EnvConfig(**old).scenario_source == "real"  # saved configs from before this option still load
    with pytest.raises(ValueError):
        EnvConfig(scenario_source="fake")
    with pytest.raises(ValueError):
        EnvConfig(synthetic_spec="oops")
