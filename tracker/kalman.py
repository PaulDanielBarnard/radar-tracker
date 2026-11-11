"""Constant-velocity Kalman filter for radar tracks (the classic baseline)."""
import numpy as np

from .simulate import Track


def _polar_to_cart_cov(r: float, theta: float, sigma_r: float, sigma_theta: float) -> np.ndarray:
    """Covariance of the (x, y) position measurement derived from range/bearing noise."""
    c, s = np.cos(theta), np.sin(theta)
    J = np.array([[c, -r * s],
                  [s,  r * c]])
    return J @ np.diag([sigma_r ** 2, sigma_theta ** 2]) @ J.T


class ConstantVelocityKF:
    """State [x, y, vx, vy]; measures converted position (x, y).

    q is the process-noise intensity: how much unmodelled acceleration we expect.
    Small q = smooth but slow to follow turns. Large q = agile but noisy.
    """

    def __init__(self, q: float = 1.0, init_speed_std: float = 200.0):
        self.q = q
        self.init_speed_std = init_speed_std

    def _matrices(self, dt: float):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        Q = self.q * np.array([[dt**3 / 3, 0, dt**2 / 2, 0],
                               [0, dt**3 / 3, 0, dt**2 / 2],
                               [dt**2 / 2, 0, dt, 0],
                               [0, dt**2 / 2, 0, dt]])
        return F, Q

    def run(self, track: Track) -> np.ndarray:
        """Filter one track; returns (T, 4) state estimates."""
        radar = track.radar
        F, Q = self._matrices(radar.dt)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)

        r0, th0 = track.meas_polar[0]
        R0 = _polar_to_cart_cov(r0, th0, radar.sigma_r, radar.sigma_theta)
        x = np.array([track.meas_xy[0, 0], track.meas_xy[0, 1], 0.0, 0.0])
        P = np.zeros((4, 4))
        P[:2, :2] = R0
        P[2:, 2:] = np.eye(2) * self.init_speed_std ** 2

        est = np.zeros((len(track.detected), 4))
        est[0] = x
        for k in range(1, len(track.detected)):
            # predict
            x = F @ x
            P = F @ P @ F.T + Q
            # update (skipped on a missed detection: we just coast on the prediction)
            if track.detected[k]:
                r, th = track.meas_polar[k]
                R = _polar_to_cart_cov(r, th, radar.sigma_r, radar.sigma_theta)
                z = track.meas_xy[k]
                y = z - H @ x
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(4) - K @ H) @ P
            est[k] = x
        return est
