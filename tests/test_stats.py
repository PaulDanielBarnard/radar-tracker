import numpy as np

from tracker.kalman import ConstantVelocityKF
from tracker.simulate import RadarConfig, TargetConfig, simulate_batch
from tracker.stats import chi2_bounds, mean_ci, nees, paired_diff_ci


def test_mean_ci_of_a_constant_has_zero_width():
    mean, half = mean_ci(np.full(10, 7.0))
    assert mean == 7.0
    assert half == 0.0


def test_mean_ci_matches_a_hand_calculation():
    # values 1..5: mean 3, sem = 1.5811/sqrt(5), t(0.975, 4 dof) = 2.7764 -> half-width 1.9632
    mean, half = mean_ci([1, 2, 3, 4, 5])
    assert mean == 3.0
    assert abs(half - 1.9632) < 1e-3


def test_mean_ci_of_one_value_has_unknown_width():
    mean, half = mean_ci([5.0])
    assert mean == 5.0
    assert np.isnan(half)


def test_paired_diff_detects_a_clear_shift():
    rng = np.random.default_rng(0)
    b = rng.normal(100, 5, 10)
    a = b + 1 + rng.normal(0, 0.1, 10)
    diff, half = paired_diff_ci(a, b)
    assert abs(diff - 1) < 0.2
    assert abs(diff) > half          # interval excludes 0


def test_paired_diff_does_not_invent_a_difference_from_noise():
    rng = np.random.default_rng(0)
    b = rng.normal(100, 5, 10)
    a = b + rng.normal(0, 3, 10)
    diff, half = paired_diff_ci(a, b)
    assert abs(diff) < half          # interval includes 0


def test_nees_averages_the_state_dimension_for_a_consistent_filter():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    P = A @ A.T + 4 * np.eye(4)                       # a random positive-definite covariance
    e = rng.multivariate_normal(np.zeros(4), P, size=20000)
    values = nees(e, np.zeros_like(e), np.repeat(P[None], len(e), axis=0))
    assert abs(values.mean() - 4) < 0.2               # within 5% of the dimension


def test_nees_flags_an_overconfident_filter():
    rng = np.random.default_rng(2)
    P_true = 100.0 * np.eye(4)
    e = rng.multivariate_normal(np.zeros(4), P_true, size=5000)
    P_claimed = np.repeat((1.0 * np.eye(4))[None], len(e), axis=0)   # claims 10x smaller errors
    assert nees(e, np.zeros_like(e), P_claimed).mean() > 100


def test_chi2_bounds_match_known_values():
    lo, hi = chi2_bounds(4, 200)
    assert abs(lo - 3.62) < 0.02
    assert abs(hi - 4.40) < 0.02


def test_chi2_bounds_tighten_with_more_samples():
    lo1, hi1 = chi2_bounds(4, 50)
    lo2, hi2 = chi2_bounds(4, 500)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_kalman_default_return_is_unchanged_and_diag_is_consistent():
    tracks = simulate_batch(5, "random", RadarConfig(p_detect=0.6), TargetConfig(), seed=0)
    kf = ConstantVelocityKF(q=10.0)
    for t in tracks:
        est = kf.run(t)
        assert est.shape == (len(t.detected), 4)
        est2, diag = kf.run(t, return_diag=True)
        assert np.array_equal(est, est2)              # diagnostics do not change the filter
        assert diag["P"].shape == (len(t.detected), 4, 4)
        nis = diag["nis"]
        assert np.isnan(nis[0])
        assert np.array_equal(np.isnan(nis[1:]), ~t.detected[1:])   # NaN exactly on missed scans
        assert np.isfinite(nis[1:][t.detected[1:]]).all()
