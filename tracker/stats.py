"""Statistics helpers for comparing trackers: confidence intervals and filter consistency."""
import numpy as np
from scipy import stats


def mean_ci(values, level: float = 0.95):
    """Mean of `values` and the half-width of its confidence interval (t-distribution).

    Returns (mean, half_width): the interval is mean +/- half_width. Needs at least two
    values; with one value the half-width is NaN because the spread is unknown.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan")
    sem = values.std(ddof=1) / np.sqrt(n)                    # standard error of the mean
    return mean, float(stats.t.ppf((1 + level) / 2, n - 1) * sem)


def paired_diff_ci(a, b, level: float = 0.95):
    """Confidence interval of the mean difference a - b, for paired measurements.

    `a[i]` and `b[i]` must come from the same seed / same tracks. Pairing removes the
    track-to-track variation, so small real differences become visible. If the interval
    (mean_diff +/- half_width) excludes 0, the difference is statistically clear.
    """
    return mean_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), level)


def nees(truth: np.ndarray, est: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalized estimation error squared at every step of one track.

    truth, est: (T, 4) full states (x, y, vx, vy); P: (T, 4, 4) the filter's covariance.
    NEES_k = e^T P^-1 e with e = truth - est. For an honest filter its average over many
    tracks equals the state dimension (4).
    """
    e = np.asarray(truth, dtype=float) - np.asarray(est, dtype=float)
    return np.array([e[k] @ np.linalg.solve(P[k], e[k]) for k in range(len(e))])


def chi2_bounds(dof: int, n: int, level: float = 0.95):
    """Range the AVERAGE of n independent chi-squared(dof) values falls in, with prob `level`.

    For a consistent filter, the sum of NEES over n independent tracks is chi-squared with
    dof * n degrees of freedom, so the average has these bounds. Above the upper bound the
    filter is overconfident; below the lower bound it is too cautious.
    """
    alpha = 1 - level
    lo, hi = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], dof * n) / n
    return float(lo), float(hi)
