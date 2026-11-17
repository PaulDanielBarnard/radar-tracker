"""OSPA distance: one number for "how wrong is this set of tracks?".

Tracking several aircraft means comparing two SETS of positions: the true ones and the
tracker's. A single RMSE cannot do that, because the sets may differ in size and have no
natural order. The OSPA (optimal sub-pattern assignment) distance handles both:

  1. pair truths with tracks one-to-one so that the total distance is smallest
  2. a pair further apart than the cutoff c counts as exactly c (it is just "wrong")
  3. every unpaired item (a missed aircraft or a ghost track) also costs c
  4. average over the larger set size n and take the p-th root

With c = 1000 m, a perfect tracker scores 0 and one that finds nothing scores 1000.
The result splits into a localisation part (how far the paired tracks are off) and a
cardinality part (the cost of getting the NUMBER of aircraft wrong):
total^p = localisation^p + cardinality^p.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def ospa(truth_xy, track_xy, c: float = 1000.0, p: float = 2.0):
    """OSPA distance between two point sets. Returns (total, localisation, cardinality).

    truth_xy (m, 2) and track_xy (n, 2) positions in metres; either may be empty.
    """
    X = np.asarray(truth_xy, dtype=float).reshape(-1, 2)
    Y = np.asarray(track_xy, dtype=float).reshape(-1, 2)
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0, 0.0, 0.0
    big = max(m, n)
    if min(m, n) == 0:                               # one set empty: every item costs c
        return float(c), 0.0, float(c)
    d = np.minimum(np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2), c) ** p
    rows, cols = linear_sum_assignment(d)
    loc_sum = d[rows, cols].sum()
    card_sum = (c ** p) * (big - min(m, n))
    loc = (loc_sum / big) ** (1.0 / p)
    card = (card_sum / big) ** (1.0 / p)
    total = ((loc_sum + card_sum) / big) ** (1.0 / p)
    return float(total), float(loc), float(card)
