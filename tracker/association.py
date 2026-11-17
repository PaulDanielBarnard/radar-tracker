"""Data association: deciding which plot belongs to which track.

With several aircraft and false alarms, each scan gives a pile of plots and a set of tracks
that each predict where their aircraft should be. Two steps decide who gets what:

  gating      a plot is only a candidate for a track if it is statistically close to the
              track's prediction: its squared Mahalanobis distance is below a threshold
  assignment  among the candidates, pick the one-to-one matching with the smallest TOTAL
              cost (the Hungarian algorithm), which beats letting each track grab its
              nearest plot greedily
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

# Squared Mahalanobis distance that a true plot stays under 99% of the time (2 measurements).
GATE_99 = float(chi2.ppf(0.99, 2))      # about 9.21

_BIG = 1e9                              # stands in for "forbidden" inside the assignment solver


def mahalanobis_sq(pred_xy, S, plots_xy) -> np.ndarray:
    """Squared Mahalanobis distance of every plot from a track's predicted position.

    pred_xy (2,), S (2, 2) the innovation covariance, plots_xy (m, 2). Returns (m,).
    A distance of d^2 means the plot is sqrt(d^2) standard deviations from the prediction.
    """
    plots_xy = np.asarray(plots_xy, dtype=float).reshape(-1, 2)
    diff = plots_xy - np.asarray(pred_xy, dtype=float)
    return np.einsum("ij,ij->i", diff, np.linalg.solve(S, diff.T).T)


def gate(pred_xy, S, plots_xy, threshold: float = GATE_99):
    """Which plots could belong to this track? Returns (mask, d2): inside the gate, and distances."""
    d2 = mahalanobis_sq(pred_xy, S, plots_xy)
    return d2 <= threshold, d2


def cost_matrix(pred_xys, Ss, plots_xy) -> np.ndarray:
    """(n_tracks, n_plots) matrix of squared Mahalanobis distances (no gating applied)."""
    plots_xy = np.asarray(plots_xy, dtype=float).reshape(-1, 2)
    if len(pred_xys) == 0:
        return np.zeros((0, len(plots_xy)))
    return np.stack([mahalanobis_sq(p, S, plots_xy) for p, S in zip(pred_xys, Ss)])


def _split(matches, n_tracks, n_plots):
    used_t = {t for t, _ in matches}
    used_p = {p for _, p in matches}
    return (sorted(matches),
            [t for t in range(n_tracks) if t not in used_t],
            [p for p in range(n_plots) if p not in used_p])


def assign(cost, threshold: float = GATE_99):
    """Optimal one-to-one matching (Hungarian algorithm) of tracks to plots.

    cost       (n_tracks, n_plots) squared Mahalanobis distances
    threshold  pairs above it are outside the gate and may never be matched
    Returns (matches, unmatched_tracks, unmatched_plots), matches as (track, plot) pairs.
    The matching minimises the total cost over all allowed pairs.
    """
    cost = np.asarray(cost, dtype=float)
    n_tracks, n_plots = cost.shape
    if n_tracks == 0 or n_plots == 0:
        return _split([], n_tracks, n_plots)
    allowed = cost <= threshold
    rows, cols = linear_sum_assignment(np.where(allowed, cost, _BIG))
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c]]
    return _split(matches, n_tracks, n_plots)


def greedy_assign(cost, threshold: float = GATE_99):
    """Nearest-neighbour matching: repeatedly take the cheapest allowed pair left.

    Simple, but a locally good choice can force a terrible one later; `assign` avoids that.
    Kept for comparison and for the tests.
    """
    cost = np.asarray(cost, dtype=float)
    n_tracks, n_plots = cost.shape
    work = np.where(cost <= threshold, cost, np.inf)
    matches = []
    while work.size and np.isfinite(work).any():
        r, c = np.unravel_index(np.argmin(work), work.shape)
        matches.append((int(r), int(c)))
        work[r, :] = np.inf
        work[:, c] = np.inf
    return _split(matches, n_tracks, n_plots)


def associate(pred_xys, Ss, plots_xy, threshold: float = GATE_99):
    """Gate and assign in one call. Returns (matches, unmatched_tracks, unmatched_plots)."""
    return assign(cost_matrix(pred_xys, Ss, plots_xy), threshold)


def total_cost(cost, matches) -> float:
    return float(sum(np.asarray(cost)[t, p] for t, p in matches))
