"""Simulate a maneuvering air target seen through a noisy 2D radar.

The radar sits at the origin and measures range and bearing once per scan,
with Gaussian noise and an optional probability of missing the target.
All distances are in metres, angles in radians, time in seconds.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RadarConfig:
    sigma_r: float = 50.0          # range noise std (m)
    sigma_theta: float = 0.005     # bearing noise std (rad, ~0.29 deg)
    p_detect: float = 1.0          # probability the radar sees the target each scan
    dt: float = 1.0                # time between scans (s)


@dataclass
class TargetConfig:
    n_steps: int = 120
    speed_range: tuple = (150.0, 300.0)         # m/s
    start_range: tuple = (20_000.0, 50_000.0)   # m from radar
    max_turn_rate: float = 0.05                 # rad/s (~2.9 deg/s)
    max_accel: float = 5.0                      # m/s^2
    segment_steps: tuple = (8, 25)              # how long each maneuver lasts (scans)


@dataclass
class Track:
    truth: np.ndarray       # (T, 4)  x, y, vx, vy
    meas_polar: np.ndarray  # (T, 2)  range, bearing   (NaN where missed)
    meas_xy: np.ndarray     # (T, 2)  converted to x, y (NaN where missed)
    detected: np.ndarray    # (T,)    bool
    radar: RadarConfig = field(repr=False, default=None)


def _simulate_truth(rng, cfg: TargetConfig, dt: float, mode: str) -> np.ndarray:
    """Generate ground-truth motion. mode: 'straight', 'turn' or 'random'."""
    bearing0 = rng.uniform(-np.pi, np.pi)
    r0 = rng.uniform(*cfg.start_range)
    pos = np.array([r0 * np.cos(bearing0), r0 * np.sin(bearing0)])
    speed = rng.uniform(*cfg.speed_range)
    heading = rng.uniform(-np.pi, np.pi)

    turn_rate, accel = 0.0, 0.0
    steps_left = 0
    if mode == "turn":
        turn_rate = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0) * cfg.max_turn_rate

    truth = np.zeros((cfg.n_steps, 4))
    for k in range(cfg.n_steps):
        truth[k] = [pos[0], pos[1], speed * np.cos(heading), speed * np.sin(heading)]

        if mode == "random" and steps_left == 0:
            steps_left = int(rng.integers(*cfg.segment_steps))
            if rng.random() < 0.3:  # sometimes fly straight for a while
                turn_rate, accel = 0.0, 0.0
            else:
                turn_rate = rng.uniform(-1, 1) * cfg.max_turn_rate
                accel = rng.uniform(-1, 1) * cfg.max_accel
        steps_left = max(steps_left - 1, 0)

        new_speed = float(np.clip(speed + accel * dt, 80.0, 400.0))
        mid_heading = heading + 0.5 * turn_rate * dt
        mid_speed = 0.5 * (speed + new_speed)
        pos = pos + mid_speed * dt * np.array([np.cos(mid_heading), np.sin(mid_heading)])
        heading += turn_rate * dt
        speed = new_speed
    return truth


def simulate_track(rng, mode: str = "random",
                   radar: RadarConfig = RadarConfig(),
                   target: TargetConfig = TargetConfig()) -> Track:
    truth = _simulate_truth(rng, target, radar.dt, mode)
    x, y = truth[:, 0], truth[:, 1]
    r_true = np.hypot(x, y)
    th_true = np.arctan2(y, x)

    r_meas = r_true + rng.normal(0.0, radar.sigma_r, len(r_true))
    th_meas = th_true + rng.normal(0.0, radar.sigma_theta, len(th_true))
    detected = rng.random(len(r_true)) < radar.p_detect
    detected[0] = True  # always start with a detection so trackers can initialise

    meas_polar = np.stack([r_meas, th_meas], axis=1)
    meas_polar[~detected] = np.nan
    meas_xy = np.stack([r_meas * np.cos(th_meas), r_meas * np.sin(th_meas)], axis=1)
    meas_xy[~detected] = np.nan
    return Track(truth, meas_polar, meas_xy, detected, radar)


def simulate_batch(n_tracks: int, mode: str, radar: RadarConfig, target: TargetConfig,
                   seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [simulate_track(rng, mode, radar, target) for _ in range(n_tracks)]


def forward_fill(track: Track) -> np.ndarray:
    """Latest radar plot at every scan. Missed scans reuse the last real plot."""
    last = track.meas_xy[0].copy()  # scan 0 is always detected
    filled = np.zeros((len(track.detected), 2))
    for k in range(len(track.detected)):
        if track.detected[k]:
            last = track.meas_xy[k]
        filled[k] = last
    return filled
