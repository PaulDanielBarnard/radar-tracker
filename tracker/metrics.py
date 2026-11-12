"""Error metrics for comparing trackers."""
import numpy as np

from .simulate import forward_fill

BURN_IN = 10  # initial scans ignored while a tracker settles


def position_error(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Euclidean position error per time step. Both arrays are (T, >=2)."""
    return np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)


def rmse_over_tracks(errors: np.ndarray, burn_in: int = BURN_IN) -> float:
    """errors: (n_tracks, T). RMSE over all tracks after an initial settling period."""
    return float(np.sqrt(np.mean(errors[:, burn_in:] ** 2)))


def evaluate(tracks, kf) -> np.ndarray:
    """(n_tracks, T) position errors of a filter run over every track."""
    return np.stack([position_error(kf.run(t), t.truth) for t in tracks])


def raw_measurement_error(tracks) -> np.ndarray:
    """Baseline-of-baselines: just use the latest radar plot, no filtering."""
    return np.stack([position_error(forward_fill(t), t.truth) for t in tracks])
