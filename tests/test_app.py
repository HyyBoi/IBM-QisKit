"""Headless UI checks with Streamlit's AppTest: replay matching, mismatch refusal, live run."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
BENCH = ROOT / "results" / "benchmark_full.json"

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
needs_bench = pytest.mark.skipif(not BENCH.exists(), reason="run `python -m floodcut bench --preset full` first")


def _text(at):
    parts = []
    for coll in (at.markdown, at.caption, at.error, at.warning, at.info):
        parts += [e.value for e in coll]
    return "\n".join(str(p) for p in parts)


@needs_bench
def test_replay_shows_saved_run_and_collapse_warning():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="mode").set_value("Replay saved benchmark").run()
    at.selectbox(key="init").set_value("basis").run()
    at.selectbox(key="p").set_value(1).run()
    assert not at.exception
    txt = _text(at)
    assert "Replay of saved benchmark" in txt and "fingerprint verified" in txt
    assert "Saved results do not match" not in txt
    # the saved basis-start run for the default scenario collapses onto one plan (max prob ~1)
    assert any("collapsed" in str(w.value) for w in at.warning)


@needs_bench
def test_replay_refuses_mismatched_settings():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="mode").set_value("Replay saved benchmark").run()
    at.slider(key="wF").set_value(0.6).run()
    assert not at.exception
    txt = _text(at)
    assert "Saved results do not match the current settings" in txt
    assert "Replay of saved benchmark" not in txt
    at.button[0].click().run()  # "Switch to a live run with these settings"
    assert not at.exception
    assert at.radio(key="mode").value == "Live quantum run"


def test_live_run_uses_shared_solver():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.number_input(key="maxiter_1").set_value(20).run()
    at.select_slider(key="shots").set_value(512).run()
    at.button[0].click().run()  # Run live
    assert not at.exception
    txt = _text(at)
    assert "Live run (this session)" in txt
    assert "P(seeing the optimum within m shots)" in txt
