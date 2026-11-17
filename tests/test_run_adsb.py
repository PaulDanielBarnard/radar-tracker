import csv
import sys

import numpy as np
import pytest

import run_adsb
from tracker.kalmannet import KalmanNet
from tracker.neural import GRUTracker

LAT0, LON0, R = 30.0, -97.0, 6_371_000.0


def write_flight(path, n=500, radius=6000.0, speed=200.0):
    """A synthetic aircraft circling at constant speed, as an OpenSky-style CSV."""
    w = speed / radius
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["time", "icao24", "lat", "lon", "velocity", "heading", "onground"])
        for k in range(n):
            x, y = radius * np.sin(w * k), radius * (1 - np.cos(w * k))
            out.writerow([1_000_000 + k, "aaa111", LAT0 + np.degrees(y / R),
                          LON0 + np.degrees(x / (R * np.cos(np.radians(LAT0)))), speed, 0.0, False])
    return path


def test_every_method_gets_one_score_per_repeat_and_the_filters_beat_the_raw_radar(tmp_path):
    path = write_flight(tmp_path / "f.csv")
    results, n_tracks = run_adsb.evaluate_adsb(path, repeats=3)
    assert n_tracks == 4                                                       # 500 s of flight, windows of 120, stride 120
    assert set(results) == {"raw radar", "KF q=0.1", "KF q=1", "KF q=10", "KF q=100", "IMM"}
    assert all(len(v) == 3 and np.all(np.isfinite(v)) for v in results.values())
    assert results["KF q=100"].mean() < results["raw radar"].mean()
    assert results["IMM"].mean() < results["raw radar"].mean()


def test_the_learned_models_are_added_only_when_given(tmp_path):
    path = write_flight(tmp_path / "f.csv")
    results, _ = run_adsb.evaluate_adsb(path, repeats=2, model=GRUTracker(), knet=KalmanNet(hidden=8))
    assert "GRU" in results and "KalmanNet" in results
    assert all(len(v) == 2 for v in results.values())


def test_repeats_change_the_radar_placement_but_not_the_flight(tmp_path):
    path = write_flight(tmp_path / "f.csv")
    a, _ = run_adsb.evaluate_adsb(path, repeats=3)
    b, _ = run_adsb.evaluate_adsb(path, repeats=3)
    np.testing.assert_array_equal(a["IMM"], b["IMM"])                          # reproducible
    assert len(set(np.round(a["IMM"], 3))) == 3                                # but each repeat differs


def test_max_tracks_limits_the_windows(tmp_path):
    path = write_flight(tmp_path / "f.csv")
    _, n = run_adsb.evaluate_adsb(path, repeats=2, max_tracks=2)
    assert n == 2


def test_a_file_without_usable_flying_is_a_clear_error(tmp_path):
    path = write_flight(tmp_path / "f.csv", n=60)                              # only 60 s: no 120-scan window
    with pytest.raises(ValueError, match="No usable flight segments"):
        run_adsb.evaluate_adsb(path, repeats=2)


def test_a_missing_file_explains_where_to_get_one(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_adsb.py", "--csv", str(tmp_path / "nope.csv")])
    with pytest.raises(SystemExit) as e:
        run_adsb.main()
    msg = str(e.value)
    assert "nope.csv" in msg and "icao24" in msg and "opensky-network.org" in msg


def test_the_script_runs_end_to_end_on_a_small_file(tmp_path, monkeypatch, capsys):
    path = write_flight(tmp_path / "f.csv")
    monkeypatch.setattr(sys, "argv", ["run_adsb.py", "--csv", str(path), "--repeats", "3",
                                      "--model", "no_such_model", "--kalmannet", "no_such_net"])
    run_adsb.main()
    out = capsys.readouterr().out
    assert "1 aircraft" in out and "KF q=100" in out and "IMM" in out and "IMM - KF q=" in out
    assert "radar luck only" in out


def test_the_noise_presets_match_the_scenarios():
    assert run_adsb.RADARS["noisy"].sigma_r == 150.0 and run_adsb.RADARS["missed"].p_detect == 0.7


def test_paired_rows_say_which_method_is_better_with_the_right_sign():
    res = {"IMM": np.array([90.0, 91.0, 92.0, 93.0]), "KF q=100": np.array([100.0, 102.0, 101.0, 103.0]),
           "GRU": np.array([100.0, 101.0, 100.5, 101.5])}
    rows = {(a, b): (d, v) for a, b, d, h, v in run_adsb.paired_rows(res, "KF q=100")}
    assert rows[("IMM", "KF q=100")][0] < -5 and rows[("IMM", "KF q=100")][1] == "IMM better"
    assert rows[("GRU", "IMM")][0] > 5 and rows[("GRU", "IMM")][1] == "IMM better"
    assert ("KalmanNet", "IMM") not in rows                                    # only methods that exist
    same = {"IMM": np.array([1.0, 3.0, 2.0, 4.0]), "KF q=1": np.array([2.0, 2.0, 3.0, 3.0])}
    assert run_adsb.paired_rows(same, "KF q=1")[0][4] == "no clear difference"
