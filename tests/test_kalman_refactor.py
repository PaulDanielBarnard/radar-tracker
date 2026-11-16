"""The Kalman filter was split into reusable functions; its output must not change at all."""
import numpy as np
import pytest

from tracker.kalman import (ConstantVelocityKF, H_POS, cv_matrices, initial_state,
                            polar_to_cart_cov, predict, update)
from tracker.simulate import RadarConfig, TargetConfig, simulate_batch


def reference_run(q, init_speed_std, track):
    """A frozen copy of the filter loop as it was before the refactor."""
    radar = track.radar
    dt = radar.dt
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    Q = q * np.array([[dt**3 / 3, 0, dt**2 / 2, 0],
                      [0, dt**3 / 3, 0, dt**2 / 2],
                      [dt**2 / 2, 0, dt, 0],
                      [0, dt**2 / 2, 0, dt]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    r0, th0 = track.meas_polar[0]
    c, s = np.cos(th0), np.sin(th0)
    J = np.array([[c, -r0 * s], [s, r0 * c]])
    R0 = J @ np.diag([radar.sigma_r ** 2, radar.sigma_theta ** 2]) @ J.T
    x = np.array([track.meas_xy[0, 0], track.meas_xy[0, 1], 0.0, 0.0])
    P = np.zeros((4, 4))
    P[:2, :2] = R0
    P[2:, 2:] = np.eye(2) * init_speed_std ** 2
    T = len(track.detected)
    est = np.zeros((T, 4)); est[0] = x
    Ps = np.zeros((T, 4, 4)); Ps[0] = P
    nis = np.full(T, np.nan)
    for k in range(1, T):
        x = F @ x
        P = F @ P @ F.T + Q
        if track.detected[k]:
            r, th = track.meas_polar[k]
            c, s = np.cos(th), np.sin(th)
            J = np.array([[c, -r * s], [s, r * c]])
            R = J @ np.diag([radar.sigma_r ** 2, radar.sigma_theta ** 2]) @ J.T
            z = track.meas_xy[k]
            y = z - H @ x
            S = H @ P @ H.T + R
            nis[k] = y @ np.linalg.solve(S, y)
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P
        est[k] = x
        Ps[k] = P
    return est, Ps, nis


@pytest.mark.parametrize("q", [0.1, 10.0, 100.0])
@pytest.mark.parametrize("mode,p_detect", [("straight", 1.0), ("random", 0.7), ("turn", 0.5)])
def test_run_is_bit_identical_to_the_original_loop(q, mode, p_detect):
    tracks = simulate_batch(6, mode, RadarConfig(p_detect=p_detect), TargetConfig(), seed=11)
    kf = ConstantVelocityKF(q=q)
    for t in tracks:
        ref_est, ref_P, ref_nis = reference_run(q, kf.init_speed_std, t)
        est, diag = kf.run(t, return_diag=True)
        np.testing.assert_array_equal(est, ref_est)
        np.testing.assert_array_equal(diag["P"], ref_P)
        np.testing.assert_array_equal(diag["nis"], ref_nis)       # NaN positions match too
        np.testing.assert_array_equal(kf.run(t), ref_est)         # default return unchanged


def test_predict_with_identity_motion_and_no_noise_changes_nothing():
    x = np.array([10.0, 20.0, 1.0, 2.0])
    P = np.diag([4.0, 4.0, 1.0, 1.0])
    x2, P2 = predict(x, P, np.eye(4), np.zeros((4, 4)))
    np.testing.assert_array_equal(x2, x)
    np.testing.assert_array_equal(P2, P)


def test_predict_moves_position_by_velocity_and_grows_uncertainty():
    F, Q = cv_matrices(q=10.0, dt=2.0)
    x, P = predict(np.array([0.0, 0.0, 3.0, -1.0]), np.diag([100.0, 100.0, 4.0, 4.0]), F, Q)
    np.testing.assert_allclose(x, [6.0, -2.0, 3.0, -1.0])
    assert P[0, 0] > 100.0 + 4.0 * 2.0 ** 2          # more than the pure velocity-spread growth


def test_update_with_a_perfect_prediction_keeps_the_state_and_shrinks_uncertainty():
    x = np.array([100.0, 200.0, 5.0, 5.0])
    P = np.diag([50.0, 50.0, 4.0, 4.0])
    R = np.diag([25.0, 25.0])
    x2, P2, y, S, K = update(x, P, np.array([100.0, 200.0]), R)
    np.testing.assert_allclose(x2, x)                # zero innovation: nothing to correct
    np.testing.assert_allclose(y, [0.0, 0.0])
    assert P2[0, 0] < P[0, 0]                        # we learned something
    assert K.shape == (4, 2) and S.shape == (2, 2)


def test_update_gain_trusts_the_plot_more_when_the_prediction_is_uncertain():
    R = np.diag([100.0, 100.0])
    _, _, _, _, k_certain = update(np.zeros(4), np.diag([1.0, 1.0, 1.0, 1.0]), np.array([5.0, 5.0]), R)
    _, _, _, _, k_unsure = update(np.zeros(4), np.diag([1e5, 1e5, 1.0, 1.0]), np.array([5.0, 5.0]), R)
    assert k_unsure[0, 0] > k_certain[0, 0]
    assert 0.0 < k_unsure[0, 0] < 1.0


def test_initial_state_starts_at_the_first_plot_with_zero_velocity():
    R0 = np.diag([30.0 ** 2, 80.0 ** 2])
    x, P = initial_state(np.array([1000.0, -2000.0]), R0, init_speed_std=200.0)
    np.testing.assert_array_equal(x, [1000.0, -2000.0, 0.0, 0.0])
    np.testing.assert_array_equal(P[:2, :2], R0)
    assert P[2, 2] == P[3, 3] == 200.0 ** 2


def test_polar_covariance_is_long_across_the_line_of_sight_at_range():
    cov = polar_to_cart_cov(r=35_000.0, theta=0.0, sigma_r=50.0, sigma_theta=0.005)
    assert np.sqrt(cov[0, 0]) == pytest.approx(50.0)           # along the line of sight (x here)
    assert np.sqrt(cov[1, 1]) == pytest.approx(175.0)          # sideways: 35 km x 0.005 rad
    np.testing.assert_allclose(H_POS @ np.eye(4)[:, :2], np.eye(2))
