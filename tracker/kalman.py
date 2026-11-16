"""Constant-velocity Kalman filter for radar tracks (the classic baseline).

The two Kalman steps are plain functions (`predict`, `update`) so other trackers
(the IMM filter, the multi-target tracker) can reuse them.
"""
import numpy as np

from .simulate import Track

# The radar measures position only: it picks x and y out of [x, y, vx, vy].
H_POS = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)


def polar_to_cart_cov(r: float, theta: float, sigma_r: float, sigma_theta: float) -> np.ndarray:
    """Covariance of the (x, y) position measurement derived from range/bearing noise."""
    c, s = np.cos(theta), np.sin(theta)
    J = np.array([[c, -r * s],
                  [s,  r * c]])
    return J @ np.diag([sigma_r ** 2, sigma_theta ** 2]) @ J.T


_polar_to_cart_cov = polar_to_cart_cov   # old private name, kept so existing imports work


def cv_matrices(q: float, dt: float):
    """Constant-velocity motion model: transition F and process noise Q (intensity q)."""
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    Q = q * np.array([[dt**3 / 3, 0, dt**2 / 2, 0],
                      [0, dt**3 / 3, 0, dt**2 / 2],
                      [dt**2 / 2, 0, dt, 0],
                      [0, dt**2 / 2, 0, dt]])
    return F, Q


def initial_state(z_xy, R0, init_speed_std: float):
    """Start a track at its first plot: velocity 0 with a large velocity uncertainty."""
    x = np.array([z_xy[0], z_xy[1], 0.0, 0.0])
    P = np.zeros((4, 4))
    P[:2, :2] = R0
    P[2:, 2:] = np.eye(2) * init_speed_std ** 2
    return x, P


def predict(x, P, F, Q):
    """Kalman predict step: move the state forward and grow the uncertainty."""
    return F @ x, F @ P @ F.T + Q


def update(x, P, z, R, H=H_POS):
    """Kalman update step with a plot z (measurement covariance R).

    Returns (x, P, y, S, K): the new state and covariance, plus the innovation y, its
    covariance S and the gain K (handy for consistency checks and for learned filters).
    """
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H) @ P
    return x, P, y, S, K


class ConstantVelocityKF:
    """State [x, y, vx, vy]; measures converted position (x, y).

    q is the process-noise intensity: how much unmodelled acceleration we expect.
    Small q = smooth but slow to follow turns. Large q = agile but noisy.
    """

    def __init__(self, q: float = 1.0, init_speed_std: float = 200.0):
        self.q = q
        self.init_speed_std = init_speed_std

    def _matrices(self, dt: float):
        return cv_matrices(self.q, dt)

    def run(self, track: Track, return_diag: bool = False):
        """Filter one track; returns (T, 4) state estimates.

        With return_diag=True it returns (estimates, diag) where diag holds the filter's
        own uncertainty, used to check whether the filter is consistent:
          "P"   (T, 4, 4)  covariance after each step (predict-only on a missed scan)
          "nis" (T,)       innovation statistic y^T S^-1 y at each detected scan;
                           NaN at scan 0 and on missed scans (no innovation exists)
        """
        radar = track.radar
        F, Q = self._matrices(radar.dt)

        r0, th0 = track.meas_polar[0]
        R0 = polar_to_cart_cov(r0, th0, radar.sigma_r, radar.sigma_theta)
        x, P = initial_state(track.meas_xy[0], R0, self.init_speed_std)

        T = len(track.detected)
        est = np.zeros((T, 4))
        est[0] = x
        Ps = np.zeros((T, 4, 4))
        Ps[0] = P
        nis = np.full(T, np.nan)
        for k in range(1, T):
            x, P = predict(x, P, F, Q)
            # update (skipped on a missed detection: we just coast on the prediction)
            if track.detected[k]:
                r, th = track.meas_polar[k]
                R = polar_to_cart_cov(r, th, radar.sigma_r, radar.sigma_theta)
                x, P, y, S, _ = update(x, P, track.meas_xy[k], R)
                nis[k] = y @ np.linalg.solve(S, y)
            est[k] = x
            Ps[k] = P
        if return_diag:
            return est, {"P": Ps, "nis": nis}
        return est
