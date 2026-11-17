import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import export_testvectors
from tracker.dataset import track_to_arr
from tracker.kalman import ConstantVelocityKF
from tracker.simulate import RadarConfig, TargetConfig, simulate_track

ROOT = Path(__file__).resolve().parent.parent


def test_the_vectors_describe_a_track_with_misses_and_both_kalman_settings():
    v = export_testvectors.build_vectors()
    assert len(v["truth"]) == 120 and not all(v["detected"]) and v["detected"][0]
    assert [p is None for p in v["xy"]] == [not d for d in v["detected"]]
    assert set(v["kf"]) == {"0.1", "100"} and np.array(v["features"]).shape == (120, 19)
    assert np.array(v["gru"]).shape == (120, 2)


def test_the_vectors_are_what_python_computes_for_that_track():
    v = export_testvectors.build_vectors()
    radar = RadarConfig(sigma_r=80.0, sigma_theta=0.008, p_detect=0.8)
    track = simulate_track(np.random.default_rng(export_testvectors.SEED), "random", radar, TargetConfig())
    np.testing.assert_allclose(v["kf"]["100"], ConstantVelocityKF(q=100.0).run(track), atol=1e-3)
    np.testing.assert_allclose(v["features"], track_to_arr(track, hybrid=True)[0], atol=1e-5)


def test_the_gru_in_the_vectors_is_not_just_the_base_position():
    v = export_testvectors.build_vectors()
    assert np.abs(np.array(v["gru"]) - np.array(v["base"])).max() > 5.0


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_javascript_port_passes_its_own_checks_against_these_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(export_testvectors, "WEB", ROOT / "web")
    export_testvectors.main()                                   # refresh the vectors from the current Python code
    r = subprocess.run(["node", str(ROOT / "web" / "test_js.mjs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 failed" in r.stdout
