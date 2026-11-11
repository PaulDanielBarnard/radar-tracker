import numpy as np

from tracker.kalman import ConstantVelocityKF
from tracker.metrics import position_error
from tracker.simulate import RadarConfig, TargetConfig, simulate_batch, simulate_track


def test_track_shapes_and_first_detection():
    rng = np.random.default_rng(0)
    t = simulate_track(rng, "random", RadarConfig(p_detect=0.5), TargetConfig(n_steps=50))
    assert t.truth.shape == (50, 4)
    assert t.meas_xy.shape == (50, 2)
    assert t.detected[0]
    assert np.isnan(t.meas_xy[~t.detected]).all()


def test_simulation_is_reproducible():
    a = simulate_batch(3, "random", RadarConfig(), TargetConfig(), seed=1)
    b = simulate_batch(3, "random", RadarConfig(), TargetConfig(), seed=1)
    assert np.allclose(a[0].truth, b[0].truth)
    assert np.allclose(a[2].meas_xy, b[2].meas_xy, equal_nan=True)


def test_kalman_beats_raw_radar_on_straight_flight():
    tracks = simulate_batch(100, "straight", RadarConfig(), TargetConfig(), seed=3)
    kf = ConstantVelocityKF(q=0.1)
    kf_err = np.mean([position_error(kf.run(t), t.truth)[10:].mean() for t in tracks])
    raw_err = np.mean([np.linalg.norm(t.meas_xy - t.truth[:, :2], axis=1)[10:].mean() for t in tracks])
    assert kf_err < raw_err


def test_kalman_coasts_through_missed_detections():
    tracks = simulate_batch(20, "straight", RadarConfig(p_detect=0.6), TargetConfig(), seed=5)
    kf = ConstantVelocityKF(q=0.1)
    for t in tracks:
        est = kf.run(t)
        assert np.isfinite(est).all()
