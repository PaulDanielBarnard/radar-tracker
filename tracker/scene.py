"""Multi-target scenes: several aircraft and false alarms seen by one radar.

A single-target `Track` hands a tracker one plot per scan. A `Scene` is harder and closer to
real life: each scan returns an unordered pile of plots, some from aircraft and some false
alarms (clutter), and the tracker must work out which plot belongs to which aircraft.

Every plot carries its true source (an aircraft number, or -1 for clutter). That label is for
scoring only; a tracker must never look at it.
"""
from dataclasses import dataclass

import numpy as np

from .simulate import RadarConfig, TargetConfig, _simulate_truth


@dataclass
class ScanPlots:
    """The plots returned by one radar scan, in random order."""
    polar: np.ndarray    # (m, 2)  range, bearing
    xy: np.ndarray       # (m, 2)  the same plots converted to x, y
    source: np.ndarray   # (m,)    aircraft number the plot came from, or -1 for a false alarm


@dataclass
class Scene:
    truths: np.ndarray   # (n_targets, T, 4)  x, y, vx, vy; NaN while the target does not exist
    alive: np.ndarray    # (n_targets, T)     bool: is the target flying at this scan?
    scans: list          # T ScanPlots, one per scan
    radar: RadarConfig

    @property
    def n_targets(self) -> int:
        return self.truths.shape[0]

    @property
    def n_scans(self) -> int:
        return len(self.scans)

    def truth_at(self, k: int):
        """(ids, xy) of the aircraft that exist at scan k."""
        ids = np.flatnonzero(self.alive[:, k])
        return ids, self.truths[ids, k, :2]


def _empty_plots() -> ScanPlots:
    return ScanPlots(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=int))


def simulate_scene(rng, n_targets: int, radar: RadarConfig = RadarConfig(),
                   target: TargetConfig = TargetConfig(), clutter_rate: float = 0.0,
                   region_radius: float = 60_000.0, mode: str = "random",
                   birth_death: bool = False) -> Scene:
    """Simulate `n_targets` aircraft and Poisson(`clutter_rate`) false alarms per scan.

    Each aircraft is seen with probability radar.p_detect per scan, with the usual range and
    bearing noise. False alarms are spread uniformly over the disc of radius `region_radius`
    around the radar. With birth_death=True every aircraft appears during the first third of
    the scene and disappears during the last third, so trackers must start and end tracks.
    """
    T = target.n_steps
    truths = np.full((n_targets, T, 4), np.nan)
    alive = np.ones((n_targets, T), dtype=bool)
    for i in range(n_targets):
        full = _simulate_truth(rng, target, radar.dt, mode)
        if birth_death:
            born = int(rng.integers(0, T // 3 + 1))
            died = int(rng.integers(2 * T // 3, T + 1))
            alive[i, :born] = False
            alive[i, died:] = False
        truths[i, alive[i]] = full[alive[i]]

    scans = []
    for k in range(T):
        polar, source = [], []
        for i in np.flatnonzero(alive[:, k]):
            if rng.random() < radar.p_detect:
                x, y = truths[i, k, 0], truths[i, k, 1]
                r = np.hypot(x, y) + rng.normal(0.0, radar.sigma_r)
                th = np.arctan2(y, x) + rng.normal(0.0, radar.sigma_theta)
                polar.append((r, th))
                source.append(int(i))
        for _ in range(int(rng.poisson(clutter_rate))):
            r = region_radius * np.sqrt(rng.random())               # uniform over the disc
            th = rng.uniform(-np.pi, np.pi)
            polar.append((r, th))
            source.append(-1)
        if not polar:
            scans.append(_empty_plots())
            continue
        polar = np.array(polar)
        source = np.array(source, dtype=int)
        order = rng.permutation(len(polar))                         # the order carries no information
        polar, source = polar[order], source[order]
        xy = np.stack([polar[:, 0] * np.cos(polar[:, 1]), polar[:, 0] * np.sin(polar[:, 1])], axis=1)
        scans.append(ScanPlots(polar, xy, source))
    return Scene(truths, alive, scans, radar)


def simulate_scene_batch(n_scenes: int, n_targets: int, radar: RadarConfig, target: TargetConfig,
                         clutter_rate: float = 0.0, seed: int = 0, **kwargs) -> list:
    rng = np.random.default_rng(seed)
    return [simulate_scene(rng, n_targets, radar, target, clutter_rate, **kwargs)
            for _ in range(n_scenes)]
