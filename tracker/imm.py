"""Interacting Multiple Model (IMM) filter.

One Kalman filter cannot be right for both straight flight (wants a small q) and turns
(wants a large q). An IMM runs several filters at once, each with its own motion model,
and keeps a running belief (the mode probabilities) about which one fits right now.

Each scan has four steps:
  1. mix      every filter restarts from a blend of all filters' last estimates,
              weighted by how likely it is that the target switched from model i to j
  2. filter   each model runs its own Kalman predict and update
  3. score    each model is scored by how well it predicted the new plot
  4. combine  mode probabilities are updated, and the output is the probability-weighted
              average of the filters' states
"""
import numpy as np

from .kalman import (H_POS, ct_matrices, cv_matrices, initial_state, polar_to_cart_cov,
                     predict, update)
from .simulate import Track

_PROB_FLOOR = 1e-12     # keeps a mode probability from collapsing to exactly 0 (division by 0)

# The model set chosen by a sweep on validation seeds (3000 and up, never the test seeds):
# a calm and an agile straight-flight model plus a left and a right coordinated turn at
# 0.05 rad/s, the simulator's maximum turn rate (real trackers also know their aircraft's limits).
TUNED_MODELS = [("cv", 1.0), ("cv", 100.0), ("ct", 0.05, 10.0), ("ct", -0.05, 10.0)]

# Probability of staying in the same model from one scan to the next, also chosen on the
# validation seeds (tried 0.90, 0.95, 0.98; score 1.017, 0.982, 0.960; lower is better).
TUNED_STAY = 0.98


def default_transition(n_models: int, stay: float = 0.95) -> np.ndarray:
    """Transition matrix: stay in the same model with probability `stay`, else switch evenly."""
    if n_models == 1:
        return np.ones((1, 1))
    pi = np.full((n_models, n_models), (1.0 - stay) / (n_models - 1))
    np.fill_diagonal(pi, stay)
    return pi


def _mix(xs, Ps, mu, pi):
    """Step 1: mixed starting state and covariance for every model, plus predicted probabilities."""
    c = pi.T @ mu                                          # c_j = sum_i pi[i, j] mu_i
    mu_ij = pi * mu[:, None] / c[None, :]                  # P(was model i | now model j)
    M = len(mu)
    x0 = [sum(mu_ij[i, j] * xs[i] for i in range(M)) for j in range(M)]
    P0 = []
    for j in range(M):
        P0.append(sum(mu_ij[i, j] * (Ps[i] + np.outer(xs[i] - x0[j], xs[i] - x0[j]))
                      for i in range(M)))
    return x0, P0, c


def _combine(xs, Ps, weights):
    """Probability-weighted mean and covariance (the covariance includes the spread between models)."""
    x = sum(w * xj for w, xj in zip(weights, xs))
    P = sum(w * (Pj + np.outer(xj - x, xj - x)) for w, xj, Pj in zip(weights, xs, Ps))
    return x, P


def _log_likelihood(y, S):
    """log of the Gaussian density of innovation y with covariance S (2-D measurement)."""
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * (y @ np.linalg.solve(S, y) + logdet) - np.log(2.0 * np.pi)


class IMMFilter:
    """IMM filter over constant-velocity models that differ only in their process noise q.

    qs         process-noise intensity of each constant-velocity model, e.g. (1, 100): a calm
               one and an agile one
    pi         model transition matrix (rows sum to 1); default: stay with probability 0.95
    mu0        starting mode probabilities; default: equal
    models     optional list of model specs that replaces `qs`:
                 ("cv", q)           constant velocity with process noise q
                 ("ct", omega, q)    coordinated turn at omega rad/s (+ left, - right)
    """

    def __init__(self, qs=(1.0, 100.0), pi=None, mu0=None, init_speed_std: float = 200.0,
                 models=None):
        if models is None:
            models = [("cv", float(q)) for q in qs]
        self.models = [self._check_spec(spec) for spec in models]
        self.qs = tuple(spec[-1] for spec in self.models)      # process noise of every model
        M = len(self.models)
        self.pi = default_transition(M) if pi is None else np.asarray(pi, dtype=float)
        if self.pi.shape != (M, M) or not np.allclose(self.pi.sum(axis=1), 1.0) or (self.pi < 0).any():
            raise ValueError(f"pi must be a {M}x{M} matrix of probabilities with rows summing to 1")
        self.mu0 = np.full(M, 1.0 / M) if mu0 is None else np.asarray(mu0, dtype=float)
        if self.mu0.shape != (M,) or (self.mu0 < 0).any() or not np.isclose(self.mu0.sum(), 1.0):
            raise ValueError(f"mu0 must be {M} probabilities summing to 1")
        self.init_speed_std = init_speed_std

    @staticmethod
    def _check_spec(spec):
        kind = spec[0]
        if kind == "cv" and len(spec) == 2:
            return ("cv", float(spec[1]))
        if kind == "ct" and len(spec) == 3:
            return ("ct", float(spec[1]), float(spec[2]))
        raise ValueError(f"model spec must be ('cv', q) or ('ct', omega, q), got {spec!r}")

    def _model_matrices(self, dt: float):
        """(F, Q) of every model."""
        return [cv_matrices(spec[1], dt) if spec[0] == "cv" else ct_matrices(spec[1], spec[2], dt)
                for spec in self.models]

    def run(self, track: Track, return_diag: bool = False):
        """Filter one track; returns (T, 4) state estimates.

        With return_diag=True returns (estimates, diag):
          "P"   (T, 4, 4)  combined covariance after each scan
          "mu"  (T, M)     mode probabilities after each scan (row 0 is the starting belief)
          "nis" (T,)       innovation statistic of the combined prediction, y^T S^-1 y;
                           NaN at scan 0 and on missed scans
        """
        radar = track.radar
        models = self._model_matrices(radar.dt)
        M = len(models)

        r0, th0 = track.meas_polar[0]
        R0 = polar_to_cart_cov(r0, th0, radar.sigma_r, radar.sigma_theta)
        x_init, P_init = initial_state(track.meas_xy[0], R0, self.init_speed_std)
        xs = [x_init.copy() for _ in range(M)]
        Ps = [P_init.copy() for _ in range(M)]
        mu = np.maximum(self.mu0, _PROB_FLOOR)
        mu = mu / mu.sum()

        T = len(track.detected)
        est = np.zeros((T, 4)); est[0] = x_init
        P_out = np.zeros((T, 4, 4)); P_out[0] = P_init
        mus = np.zeros((T, M)); mus[0] = mu
        nis = np.full(T, np.nan)

        for k in range(1, T):
            xs, Ps, c = _mix(xs, Ps, mu, self.pi)                       # 1. mix
            pred = [predict(x, P, F, Q) for (x, P), (F, Q) in zip(zip(xs, Ps), models)]
            xs, Ps = [p[0] for p in pred], [p[1] for p in pred]

            if track.detected[k]:
                r, th = track.meas_polar[k]
                R = polar_to_cart_cov(r, th, radar.sigma_r, radar.sigma_theta)
                z = track.meas_xy[k]

                # innovation of the combined (mixture) prediction, for the consistency check
                x_bar, P_bar = _combine(xs, Ps, c)
                y_bar = z - H_POS @ x_bar
                nis[k] = y_bar @ np.linalg.solve(H_POS @ P_bar @ H_POS.T + R, y_bar)

                logw = np.empty(M)
                for j in range(M):                                      # 2. filter, 3. score
                    xs[j], Ps[j], y, S, _ = update(xs[j], Ps[j], z, R)
                    logw[j] = np.log(c[j]) + _log_likelihood(y, S)
                w = np.exp(logw - logw.max())                           # 4. new mode probabilities
                mu = w / w.sum()
            else:
                mu = c                                                  # coast: no plot, no evidence
            mu = np.maximum(mu, _PROB_FLOOR)
            mu = mu / mu.sum()

            est[k], P_out[k] = _combine(xs, Ps, mu)                     # 4. combine
            mus[k] = mu

        if return_diag:
            return est, {"P": P_out, "mu": mus, "nis": nis}
        return est


def default_imm() -> IMMFilter:
    """The IMM configuration used by the evaluation scripts."""
    return IMMFilter(models=TUNED_MODELS, pi=default_transition(len(TUNED_MODELS), TUNED_STAY))
