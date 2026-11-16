import numpy as np
import pytest

from tracker.imm import IMMFilter, _combine, _mix, default_transition
from tracker.kalman import ConstantVelocityKF
from tracker.metrics import evaluate, rmse_over_tracks
from tracker.scenarios import BASE_RADAR, TARGET
from tracker.simulate import RadarConfig, TargetConfig, simulate_batch


def test_a_single_model_imm_is_exactly_the_plain_kalman_filter():
    tracks = simulate_batch(5, "random", RadarConfig(p_detect=0.6), TargetConfig(), seed=3)
    for t in tracks:
        np.testing.assert_array_equal(IMMFilter(qs=(10.0,)).run(t), ConstantVelocityKF(q=10.0).run(t))


def test_mode_probabilities_are_valid_even_with_missed_scans():
    t = simulate_batch(1, "turn", RadarConfig(p_detect=0.6), TargetConfig(), seed=5)[0]
    _, diag = IMMFilter(qs=(1.0, 100.0)).run(t, return_diag=True)
    mu = diag["mu"]
    assert mu.shape == (len(t.detected), 2)
    np.testing.assert_allclose(mu.sum(axis=1), 1.0)
    assert ((mu >= 0) & (mu <= 1)).all()
    np.testing.assert_allclose(mu[0], [0.5, 0.5])             # the starting belief


def test_diagnostics_have_the_right_shape_and_nan_pattern():
    t = simulate_batch(1, "random", RadarConfig(p_detect=0.7), TargetConfig(), seed=2)[0]
    est, diag = IMMFilter().run(t, return_diag=True)
    T = len(t.detected)
    assert est.shape == (T, 4) and diag["P"].shape == (T, 4, 4)
    assert all(np.allclose(p, p.T, atol=1e-6) and np.linalg.eigvalsh(p).min() > 0 for p in diag["P"])
    nis = diag["nis"]
    assert np.isnan(nis[0])
    assert np.array_equal(np.isnan(nis[1:]), ~t.detected[1:])  # undefined exactly when there is no plot
    assert np.isfinite(nis[1:][t.detected[1:]]).all()
    np.testing.assert_array_equal(IMMFilter().run(t), est)     # default return unchanged


def test_the_imm_believes_the_calm_model_on_straight_flight_and_the_agile_one_in_turns():
    imm = IMMFilter(qs=(1.0, 100.0))

    def mean_agile_probability(mode):
        tracks = simulate_batch(30, mode, BASE_RADAR, TARGET, seed=77)
        return np.mean([imm.run(t, return_diag=True)[1]["mu"][10:, 1].mean() for t in tracks])

    assert mean_agile_probability("straight") < 0.5          # measured about 0.31
    assert mean_agile_probability("turn") > 0.5              # measured about 0.77


def test_the_imm_beats_a_badly_matched_filter_in_both_situations():
    imm = IMMFilter(qs=(1.0, 100.0))
    turning = simulate_batch(30, "turn", BASE_RADAR, TARGET, seed=77)
    straight = simulate_batch(30, "straight", BASE_RADAR, TARGET, seed=77)
    assert rmse_over_tracks(evaluate(turning, imm)) < rmse_over_tracks(evaluate(turning, ConstantVelocityKF(q=1.0)))
    assert rmse_over_tracks(evaluate(straight, imm)) < rmse_over_tracks(evaluate(straight, ConstantVelocityKF(q=100.0)))


def test_with_no_switching_the_imm_follows_the_calm_model_while_it_fits():
    # identity transition + all starting weight on the calm model: on straight flight the calm
    # model keeps winning the evidence, so the IMM output is that model's output
    imm = IMMFilter(qs=(1.0, 100.0), pi=np.eye(2), mu0=[1.0, 0.0])
    for t in simulate_batch(5, "straight", RadarConfig(p_detect=1.0), TargetConfig(), seed=9):
        np.testing.assert_allclose(imm.run(t), ConstantVelocityKF(q=1.0).run(t), atol=1e-6)


def test_evidence_accumulates_so_the_agile_model_takes_over_when_the_target_maneuvers():
    # same setup on a maneuvering track: even from a 1e-12 head start the agile model's
    # probability climbs to ~1, because forbidding switching does not stop evidence adding up
    imm = IMMFilter(qs=(1.0, 100.0), pi=np.eye(2), mu0=[1.0, 0.0])
    t = simulate_batch(1, "random", RadarConfig(p_detect=0.8), TargetConfig(), seed=9)[0]
    _, diag = imm.run(t, return_diag=True)
    assert diag["mu"][10, 1] < 0.01
    assert diag["mu"][-1, 1] > 0.99


def test_while_coasting_the_mode_probabilities_just_follow_the_transition_matrix():
    # no plots after scan 0 means no evidence, so mu[k] must equal pi^T mu[k-1]
    t = simulate_batch(1, "straight", RadarConfig(p_detect=0.0), TargetConfig(), seed=1)[0]
    assert t.detected[0] and not t.detected[1:].any()
    imm = IMMFilter(qs=(1.0, 100.0), mu0=[0.9, 0.1])
    _, diag = imm.run(t, return_diag=True)
    mu = diag["mu"]
    np.testing.assert_allclose(mu[1], [0.86, 0.14])                    # 0.95*0.9+0.05*0.1, ...
    np.testing.assert_allclose(mu[2], imm.pi.T @ mu[1])
    assert abs(mu[-1, 0] - 0.5) < 0.05                                 # drifts to the stationary 50/50


def test_combining_models_adds_the_spread_between_their_estimates():
    xs = [np.array([0.0, 0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0, 0.0])]
    Ps = [np.eye(4), np.eye(4)]
    x, P = _combine(xs, Ps, [0.5, 0.5])
    np.testing.assert_allclose(x, [1.0, 0.0, 0.0, 0.0])
    assert P[0, 0] == pytest.approx(2.0)          # 1 (own variance) + 1 (models disagree by +/-1)
    assert P[1, 1] == pytest.approx(1.0)          # no disagreement along y


def test_mixing_with_no_switching_changes_nothing():
    xs = [np.array([0.0, 0.0, 1.0, 1.0]), np.array([5.0, 5.0, 2.0, 2.0])]
    Ps = [np.eye(4), 3.0 * np.eye(4)]
    x0, P0, c = _mix(xs, Ps, np.array([0.3, 0.7]), np.eye(2))
    for j in range(2):
        np.testing.assert_allclose(x0[j], xs[j])
        np.testing.assert_allclose(P0[j], Ps[j])
    np.testing.assert_allclose(c, [0.3, 0.7])


def test_mixing_with_even_switching_blends_both_models_equally():
    xs = [np.array([0.0, 0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0, 0.0])]
    Ps = [np.eye(4), np.eye(4)]
    x0, P0, c = _mix(xs, Ps, np.array([0.5, 0.5]), np.full((2, 2), 0.5))
    for j in range(2):
        np.testing.assert_allclose(x0[j], [2.0, 0.0, 0.0, 0.0])        # both start at the midpoint
        assert P0[j][0, 0] == pytest.approx(1.0 + 4.0)                 # own variance + (2 m)^2 spread
    np.testing.assert_allclose(c, [0.5, 0.5])


def test_default_transition_matrix_stays_with_the_stated_probability():
    pi = default_transition(3, stay=0.9)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.diag(pi), 0.9)
    np.testing.assert_allclose(default_transition(1), [[1.0]])


def test_invalid_transition_matrices_and_probabilities_are_rejected():
    with pytest.raises(ValueError):
        IMMFilter(qs=(1.0, 100.0), pi=[[0.9, 0.2], [0.1, 0.9]])      # first row sums to 1.1
    with pytest.raises(ValueError):
        IMMFilter(qs=(1.0, 100.0), pi=np.eye(3))                      # wrong shape
    with pytest.raises(ValueError):
        IMMFilter(qs=(1.0, 100.0), mu0=[0.7, 0.7])                    # does not sum to 1
