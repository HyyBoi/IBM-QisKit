from floodcut.config import ScenarioConfig, Weights
from floodcut.objective import build_problem
from floodcut.results_io import find_matching_instance, load_json, save_json, settings_match
from floodcut.scenario import generate_scenario


def _settings(seed=1, w=Weights(), k=3, tau=0.5):
    return build_problem(generate_scenario(ScenarioConfig(seed=seed)), k, w, tau).settings_dict()


def test_settings_match_and_mismatch():
    a = _settings()
    ok, bad = settings_match(a, _settings())
    assert ok and bad == []
    ok, bad = settings_match(a, _settings(w=Weights(1, 0.6, 1)))
    assert not ok and bad == ["weights"]
    ok, bad = settings_match(a, _settings(seed=2))
    assert not ok and "scenario_fingerprint" in bad and "scenario_seed" in bad
    ok, bad = settings_match(a, _settings(tau=0.45))
    assert bad == ["tau"]


def test_roundtrip_and_find(tmp_path):
    a = _settings()
    bench = {"instances": [{"settings": a, "payload": 1}]}
    p = save_json(tmp_path / "b.json", bench)
    loaded = load_json(p)
    assert find_matching_instance(loaded, _settings())["payload"] == 1
    assert find_matching_instance(loaded, _settings(k=4)) is None
