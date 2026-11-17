"""Multi-target tracking: start, follow and end one Kalman track per aircraft.

Every scan the radar gives an unordered pile of plots (aircraft and false alarms). The
`MultiTargetTracker` turns that pile into a list of tracks:

  1. predict   every existing track moves forward one scan
  2. associate gating + Hungarian assignment decide which plot belongs to which track
  3. update    a matched track is corrected by its plot; an unmatched one coasts (a miss)
  4. initiate  every plot nobody claimed, and no track's gate covers, starts a TENTATIVE track
  5. manage    a tentative track becomes CONFIRMED after M hits in N scans, and is dropped
               if it cannot get there; a confirmed track is dropped after too many misses

Tentative tracks are what keep false alarms out of the answer: a single false plot starts a
tentative track, but it almost never gets M hits in N scans, so it is never reported.
"""
from dataclasses import dataclass, field

import numpy as np

from scipy.stats import chi2

from .association import GATE_99, assign
from .kalman import H_POS, cv_matrices, initial_state, predict, update

MERGE_99 = float(chi2.ppf(0.99, 4))     # two 4-state estimates this close are the same aircraft

TENTATIVE = "tentative"
CONFIRMED = "confirmed"


@dataclass
class TrackState:
    """One track inside the manager (and what `step` hands back for confirmed ones)."""
    id: int
    x: np.ndarray                     # (4,) x, y, vx, vy
    P: np.ndarray                     # (4, 4)
    status: str = TENTATIVE
    hits: list = field(default_factory=list)   # recent hit (True) / miss (False) per scan
    misses: int = 0                   # consecutive misses
    age: int = 1                      # scans since the track was started

    def copy(self) -> "TrackState":
        return TrackState(self.id, self.x.copy(), self.P.copy(), self.status,
                          list(self.hits), self.misses, self.age)


class MultiTargetTracker:
    """Track an unknown number of aircraft among false alarms.

    q            process-noise intensity of every track's constant-velocity filter
    m_of_n       (M, N): confirm a track after M hits within its first N scans
    max_misses   delete a confirmed track after this many consecutive missed scans
    gate         squared Mahalanobis distance a plot must be under to belong to a track
    dt           time between scans (s)
    init_speed_std  velocity uncertainty given to a brand-new track (m/s)
    """

    def __init__(self, q: float = 100.0, m_of_n=(3, 5), max_misses: int = 5,
                 gate: float = GATE_99, dt: float = 1.0, init_speed_std: float = 200.0):
        self.q = q
        self.m, self.n = m_of_n
        if not 1 <= self.m <= self.n:
            raise ValueError("m_of_n must satisfy 1 <= M <= N")
        self.max_misses = max_misses
        self.gate = gate
        self.dt = dt
        self.init_speed_std = init_speed_std
        self.F, self.Q = cv_matrices(q, dt)
        self.tracks: list = []        # confirmed and tentative
        self._next_id = 0

    # -- helpers ---------------------------------------------------------------------------

    def _cost(self, plots_xy, plots_R):
        """(n_tracks, n_plots) squared Mahalanobis distances, with each plot's own noise R."""
        cost = np.zeros((len(self.tracks), len(plots_xy)))
        for i, t in enumerate(self.tracks):
            for j in range(len(plots_xy)):
                S = H_POS @ t.P @ H_POS.T + plots_R[j]
                d = plots_xy[j] - H_POS @ t.x
                cost[i, j] = d @ np.linalg.solve(S, d)
        return cost

    def _start(self, z, R):
        x, P = initial_state(z, R, self.init_speed_std)
        track = TrackState(self._next_id, x, P, hits=[True])
        self._next_id += 1
        if self.m <= 1:                # M = 1: a single plot is enough to confirm
            track.status = CONFIRMED
        self.tracks.append(track)

    def _keep(self, t: TrackState) -> bool:
        """Should this track survive the scan that was just processed?"""
        if t.status == CONFIRMED:
            return t.misses < self.max_misses
        window = t.hits[-self.n:]
        return window.count(False) <= self.n - self.m   # still able to reach M hits in N

    def _remove_duplicates(self):
        """Drop a track that duplicates another one (same position AND velocity).

        An unlucky plot just outside a track's gate starts a second track on the same aircraft;
        the two then share its plots and neither is reliable. Keep the better one: confirmed
        before tentative, then the older one. Comparing velocity too keeps two different
        aircraft that merely cross each other apart.
        """
        order = sorted(self.tracks, key=lambda t: (t.status != CONFIRMED, t.id))
        kept = []
        for t in order:
            for u in kept:
                d = t.x - u.x
                if d @ np.linalg.solve(t.P + u.P, d) < MERGE_99:
                    break
            else:
                kept.append(t)
        keep_ids = {t.id for t in kept}
        self.tracks = [t for t in self.tracks if t.id in keep_ids]

    # -- one scan --------------------------------------------------------------------------

    def step(self, plots_xy, plots_R) -> list:
        """Process one scan and return the confirmed tracks (copies; safe to keep).

        plots_xy (m, 2) plot positions; plots_R (m, 2, 2) their measurement covariances
        (from kalman.polar_to_cart_cov). Plot order carries no meaning and the true source
        of a plot is never needed.
        """
        plots_xy = np.asarray(plots_xy, dtype=float).reshape(-1, 2)
        plots_R = np.asarray(plots_R, dtype=float).reshape(-1, 2, 2)

        for t in self.tracks:                                    # 1. predict
            t.x, t.P = predict(t.x, t.P, self.F, self.Q)
            t.age += 1

        cost = self._cost(plots_xy, plots_R)                     # 2. associate
        matches, free_tracks, free_plots = assign(cost, self.gate)

        for i, j in matches:                                     # 3. update / coast
            t = self.tracks[i]
            t.x, t.P, *_ = update(t.x, t.P, plots_xy[j], plots_R[j])
            t.hits.append(True)
            t.misses = 0
        for i in free_tracks:
            t = self.tracks[i]
            t.hits.append(False)
            t.misses += 1

        for t in self.tracks:                                    # 5. manage (confirm)
            t.hits = t.hits[-self.n:]
            if t.status == TENTATIVE and t.hits.count(True) >= self.m:
                t.status = CONFIRMED
        self.tracks = [t for t in self.tracks if self._keep(t)]  # 5. manage (delete)
        self._remove_duplicates()

        for j in free_plots:                                     # 4. initiate
            if len(cost) and cost[:, j].min() <= self.gate:      # inside a track's gate: it is
                continue                                         # that track's aircraft, not a new one
            self._start(plots_xy[j], plots_R[j])

        return [t.copy() for t in self.tracks if t.status == CONFIRMED]

    def confirmed(self) -> list:
        return [t.copy() for t in self.tracks if t.status == CONFIRMED]


def scan_covariances(plots, radar) -> np.ndarray:
    """(m, 2, 2) measurement covariances of a scan's plots, from their range and bearing."""
    from .kalman import polar_to_cart_cov
    return np.array([polar_to_cart_cov(r, th, radar.sigma_r, radar.sigma_theta)
                     for r, th in plots.polar]).reshape(-1, 2, 2)


def run_scene(tracker: MultiTargetTracker, scene) -> list:
    """Feed a whole Scene to the tracker; returns, per scan, the list of confirmed tracks."""
    return [tracker.step(plots.xy, scan_covariances(plots, scene.radar)) for plots in scene.scans]
