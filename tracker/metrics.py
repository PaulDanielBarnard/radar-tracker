"""Error metrics for comparing trackers."""
import numpy as np


def position_error(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Euclidean position error per time step. Both arrays are (T, >=2)."""
    return np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)


def rmse_over_tracks(errors: np.ndarray, burn_in: int = 10) -> float:
    """errors: (n_tracks, T). RMSE over all tracks after an initial settling period."""
    return float(np.sqrt(np.mean(errors[:, burn_in:] ** 2)))
