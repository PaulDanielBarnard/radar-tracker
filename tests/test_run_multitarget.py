import numpy as np
import pytest

from run_multitarget import evaluate_cell, score_scene
from tracker.multitarget import TrackState
from tracker.scene import ScanPlots, Scene
from tracker.simulate import RadarConfig


def fake_scene(truth_xy_per_scan):
    """A scene with one fixed truth position set (list of (x, y)) at every scan."""
    n, T = len(truth_xy_per_scan), 20
    truths = np.zeros((n, T, 4))
    for i, p in enumerate(truth_xy_per_scan):
        truths[i, :, :2] = p
    empty = ScanPlots(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=int))
    return Scene(truths, np.ones((n, T), dtype=bool), [empty] * T, RadarConfig())


def track(xy):
    return TrackState(0, np.array([xy[0], xy[1], 0.0, 0.0]), np.eye(4))


def test_a_perfect_tracker_scores_zero_and_finding_nothing_scores_the_cutoff():
    scene = fake_scene([(0.0, 0.0), (5000.0, 0.0)])
    perfect = [[track((0.0, 0.0)), track((5000.0, 0.0))]] * 20
    assert score_scene(scene, perfect, burn_in=5) == pytest.approx([0, 0, 0, 2, 2])
    nothing = [[]] * 20
    assert score_scene(scene, nothing, burn_in=5) == pytest.approx([1000, 0, 1000, 2, 0])


def test_the_burn_in_scans_are_ignored():
    scene = fake_scene([(0.0, 0.0)])
    confirmed = [[]] * 10 + [[track((0.0, 0.0))]] * 10               # nothing found for 10 scans
    assert score_scene(scene, confirmed, burn_in=10)[0] == 0.0
    assert score_scene(scene, confirmed, burn_in=0)[0] > 0.0


def test_a_cell_returns_one_row_per_scene_with_sensible_numbers():
    res = evaluate_cell(1, 0.0, n_scenes=2, seed=5)
    assert res.shape == (2, 5)
    assert np.all(res[:, 3] == 1.0)                                    # one true aircraft
    assert np.all(res[:, 0] < 1000.0) and np.all(res[:, 4] > 0.5)      # and it is being tracked
    np.testing.assert_allclose(res, evaluate_cell(1, 0.0, n_scenes=2, seed=5))   # reproducible
