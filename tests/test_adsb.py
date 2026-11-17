import csv
from pathlib import Path

import numpy as np
import pytest

from tracker.adsb import (Segment, load_opensky_csv, load_tracks, place_radar, segments_to_tracks,
                          split_flight, to_local_xy, truth_from_positions, windows)
from tracker.kalman import ConstantVelocityKF
from tracker.neural import GRUTracker, gru_estimates
from tracker.simulate import RadarConfig

LAT0, LON0 = 30.0, -97.0
R = 6_371_000.0


def latlon(x, y):
    """Metres east/north of (LAT0, LON0) -> degrees (the inverse of to_local_xy)."""
    return LAT0 + np.degrees(np.asarray(y) / R), LON0 + np.degrees(np.asarray(x) / (R * np.cos(np.radians(LAT0))))


def write_csv(path, rows, header=("time", "icao24", "lat", "lon", "velocity", "heading", "onground")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def straight_rows(icao, t, speed=200.0, heading_deg=90.0, x0=0.0, y0=0.0, onground=False):
    th = np.radians(90.0 - heading_deg)                    # compass heading -> maths angle
    out = []
    for ti in t:
        x, y = x0 + speed * ti * np.cos(th), y0 + speed * ti * np.sin(th)
        lat, lon = latlon(x, y)
        out.append((ti + 1_000_000, icao, lat, lon, speed, heading_deg, onground))
    return out


def test_the_projection_gives_metres_east_and_north():
    xy = to_local_xy([LAT0 + 1.0, LAT0], [LON0, LON0 + 1.0], LAT0, LON0)
    assert xy[0] == pytest.approx([0.0, 111_195.0], rel=1e-4)             # one degree north
    assert xy[1] == pytest.approx([111_195.0 * np.cos(np.radians(LAT0)), 0.0], rel=1e-4)
    x, y = 12_345.0, -23_456.0
    np.testing.assert_allclose(to_local_xy(*latlon(x, y), LAT0, LON0), [x, y], atol=1e-6)


def test_the_loader_reads_flights_separately_and_skips_ground_and_empty_rows(tmp_path):
    rows = straight_rows("aaa111", range(0, 30)) + straight_rows("bbb222", range(0, 20), x0=5000.0)
    rows += [(1_000_050, "aaa111", "", "", "", "", False)]                  # no position
    rows += straight_rows("ccc333", range(0, 10), onground=True)           # taxiing: ignored
    rows += [rows[3]]                                                       # repeated time: ignored
    flights = {f.icao24: f for f in load_opensky_csv(write_csv(tmp_path / "f.csv", rows))}
    assert set(flights) == {"aaa111", "bbb222"}
    assert len(flights["aaa111"].t) == 30 and len(flights["bbb222"].t) == 20
    assert np.all(np.diff(flights["aaa111"].t) > 0)
    assert flights["aaa111"].velocity[0] == 200.0 and flights["aaa111"].heading[0] == 90.0


def test_the_loader_accepts_other_header_names_and_missing_optional_columns(tmp_path):
    rows = [(1000 + i, "AbC123", 30.0 + i * 1e-3, -97.0) for i in range(5)]
    path = write_csv(tmp_path / "f.csv", rows, header=("timestamp", "ICAO24", "Latitude", "Longitude"))
    (f,) = load_opensky_csv(path)
    assert f.icao24 == "abc123" and len(f.t) == 5 and np.isnan(f.velocity).all() and np.isnan(f.heading).all()


def test_a_csv_without_the_required_columns_is_rejected_with_a_helpful_message(tmp_path):
    path = write_csv(tmp_path / "f.csv", [(1, 2, 3)], header=("a", "b", "c"))
    with pytest.raises(ValueError, match="time.*icao24.*lat.*lon"):
        load_opensky_csv(path)


def make_flight(tmp_path, times, **kw):
    (f,) = load_opensky_csv(write_csv(tmp_path / "f.csv", straight_rows("aaa111", times, **kw)))
    return f


def test_resampling_puts_irregular_reports_on_an_exact_one_second_grid_along_the_true_line(tmp_path):
    times = np.arange(0, 400, 3.0) + 0.4                                    # every 3 s, off the grid
    (seg,) = split_flight(make_flight(tmp_path, times), max_gap=5.0)
    assert np.all(np.diff(seg.t) == 1.0)
    speed = np.linalg.norm(np.diff(seg.xy, axis=0), axis=1)
    np.testing.assert_allclose(speed, 200.0, rtol=1e-3)                    # straight east at 200 m/s
    np.testing.assert_allclose(seg.xy[:, 1], seg.xy[0, 1], atol=0.5)        # no drift north
    assert len(seg.t) >= 396


def test_a_gap_longer_than_five_seconds_splits_the_flight(tmp_path):
    times = np.concatenate([np.arange(0, 200), np.arange(210, 450)])         # 10 s hole
    segs = split_flight(make_flight(tmp_path, times))
    assert len(segs) == 2
    assert len(segs[0].t) == 200 and len(segs[1].t) == 240
    assert split_flight(make_flight(tmp_path, np.concatenate([np.arange(0, 200), np.arange(204, 450)]))).__len__() == 1


def test_short_stretches_and_slow_ones_are_dropped(tmp_path):
    times = np.concatenate([np.arange(0, 100), np.arange(120, 400)])         # first piece only 100 s
    assert [len(s.t) for s in split_flight(make_flight(tmp_path, times))] == [280]
    assert split_flight(make_flight(tmp_path, np.arange(0, 300), speed=10.0)) == []   # taxiing speed


def test_an_impossible_position_jump_splits_the_flight(tmp_path):
    rows = straight_rows("aaa111", range(0, 200)) + straight_rows("aaa111", range(200, 400), y0=80_000.0)
    (f,) = load_opensky_csv(write_csv(tmp_path / "f.csv", rows))
    assert len(split_flight(f)) == 2                                          # 80 km in one second is a glitch


def test_windows_have_the_requested_length_and_stride():
    seg = Segment("x", np.arange(300.0), np.zeros((300, 2)))
    w = windows(seg, n_steps=120, stride=60)
    assert [len(x.t) for x in w] == [120, 120, 120, 120]
    assert w[1].t[0] == 60.0 and windows(Segment("x", np.arange(100.0), np.zeros((100, 2)))) == []


def test_placing_a_window_keeps_its_shape_and_puts_it_near_the_radar():
    rng = np.random.default_rng(0)
    xy = np.stack([np.arange(120) * 200.0, np.zeros(120)], axis=1)           # 24 km long
    for _ in range(20):
        placed = place_radar(xy, rng)
        assert placed is not None
        r = np.hypot(placed[:, 0], placed[:, 1])
        assert r.min() >= 5_000.0 and r.max() <= 70_000.0
        c = np.hypot(*placed.mean(axis=0))
        assert 20_000.0 <= c <= 60_000.0
        np.testing.assert_allclose(np.diff(placed, axis=0), np.diff(xy, axis=0), atol=1e-6)   # pure shift
    assert place_radar(np.stack([np.arange(120) * 2000.0, np.zeros(120)], axis=1), rng) is None  # 240 km: can't fit


def test_truth_velocity_comes_from_the_positions():
    xy = np.stack([np.arange(50) * 150.0, np.arange(50) * -80.0], axis=1)
    truth = truth_from_positions(xy)
    assert truth.shape == (50, 4)
    np.testing.assert_allclose(truth[:, 2], 150.0)
    np.testing.assert_allclose(truth[:, 3], -80.0)


def circle_rows(icao, n=400, radius=4000.0, speed=200.0):
    w = speed / radius                                                        # rad/s
    return [(1_000_000 + k, icao, *latlon(radius * np.sin(w * k), radius * (1 - np.cos(w * k))),
             speed, 0.0, False) for k in range(n)]


def build_tracks(tmp_path, rows, radar=RadarConfig(p_detect=1.0), seed=0, **kw):
    segs = [s for f in load_opensky_csv(write_csv(tmp_path / "f.csv", rows)) for s in split_flight(f)]
    return segments_to_tracks(segs, radar, np.random.default_rng(seed), **kw)


def test_tracks_have_the_simulators_shapes_and_real_truth(tmp_path):
    tracks = build_tracks(tmp_path, straight_rows("aaa111", range(0, 400)) + circle_rows("bbb222"))
    assert len(tracks) == 2 * 5                                               # (400 - 120) // 60 + 1 windows each
    for t in tracks:
        assert t.truth.shape == (120, 4) and t.meas_xy.shape == (120, 2) and t.detected.shape == (120,)
        assert t.detected[0]
    straight, circle = tracks[0], tracks[5]
    np.testing.assert_allclose(np.hypot(straight.truth[:, 2], straight.truth[:, 3]), 200.0, rtol=1e-3)
    centre = circle.truth[:, :2].mean(axis=0)
    np.testing.assert_allclose(np.hypot(*(circle.truth[:, :2] - centre).T).std(), 0.0, atol=500.0)
    assert np.hypot(circle.truth[60, 2], circle.truth[60, 3]) == pytest.approx(200.0, rel=0.02)


def test_the_radar_noise_and_misses_are_the_configured_ones(tmp_path):
    radar = RadarConfig(p_detect=0.7)
    tracks = build_tracks(tmp_path, straight_rows("aaa111", range(0, 1200)), radar=radar, seed=3, stride=20)
    assert len(tracks) >= 50
    det = np.concatenate([t.detected for t in tracks])
    assert abs(det.mean() - 0.7) < 0.05
    dr = np.concatenate([t.meas_polar[t.detected, 0] - np.hypot(t.truth[t.detected, 0], t.truth[t.detected, 1])
                         for t in tracks])
    assert abs(dr.std() / radar.sigma_r - 1) < 0.1 and abs(dr.mean()) < 5.0


def test_the_conversion_is_reproducible_from_the_seed(tmp_path):
    rows = straight_rows("aaa111", range(0, 400))
    a, b, c = (build_tracks(tmp_path, rows, seed=s) for s in (1, 1, 2))
    np.testing.assert_array_equal(a[0].truth, b[0].truth)
    np.testing.assert_array_equal(a[0].meas_xy, b[0].meas_xy)
    assert not np.array_equal(a[0].truth, c[0].truth)                          # a different placement


def test_real_flight_tracks_work_with_the_kalman_filter_and_the_gru(tmp_path):
    tracks = build_tracks(tmp_path, circle_rows("bbb222", n=500), radar=RadarConfig(p_detect=0.9))
    est = ConstantVelocityKF(q=100.0).run(tracks[0])
    err = np.linalg.norm(est[10:, :2] - tracks[0].truth[10:, :2], axis=1)
    assert est.shape == (120, 4) and err.mean() < 300.0                        # it really follows the flight
    assert gru_estimates(GRUTracker(), tracks).shape == (len(tracks), 120, 2)


def test_load_tracks_goes_from_a_file_to_tracks_in_one_call(tmp_path):
    path = write_csv(tmp_path / "f.csv", straight_rows("aaa111", range(0, 400)))
    tracks = load_tracks(path, seed=0, max_tracks=3)
    assert len(tracks) == 3 and tracks[0].truth.shape == (120, 4)


SAMPLE = Path(__file__).resolve().parent.parent / "data" / "opensky_sample.csv"


@pytest.mark.skipif(not SAMPLE.exists(), reason="no local ADS-B sample in data/")
def test_the_real_sample_file_loads_if_it_is_present():
    tracks = load_tracks(SAMPLE, seed=0, max_tracks=10)
    assert len(tracks) > 0 and all(np.isfinite(t.truth).all() for t in tracks)
    assert all(50 < np.median(np.hypot(t.truth[:, 2], t.truth[:, 3])) < 400 for t in tracks)   # a climb is slower than cruise
