import itertools

import numpy as np
import pytest

from tracker.association import (GATE_99, assign, associate, cost_matrix, gate, greedy_assign,
                                 mahalanobis_sq, total_cost)


def test_the_99_percent_gate_is_the_chi_squared_value_for_two_dimensions():
    assert GATE_99 == pytest.approx(9.2103, abs=1e-3)


def test_mahalanobis_distance_counts_standard_deviations():
    S = np.diag([4.0, 9.0])                                  # sigmas 2 and 3
    d2 = mahalanobis_sq([0.0, 0.0], S, [[2.0, 0.0], [0.0, 3.0], [2.0, 3.0], [0.0, 0.0]])
    np.testing.assert_allclose(d2, [1.0, 1.0, 2.0, 0.0])


def test_mahalanobis_distance_accounts_for_correlation():
    S = np.array([[4.0, 3.0], [3.0, 4.0]])                   # errors in x and y tend to move together
    plots = np.array([[1.0, 1.0], [1.0, -1.0]])
    d2 = mahalanobis_sq([0.0, 0.0], S, plots)
    expected = [p @ np.linalg.inv(S) @ p for p in plots]
    np.testing.assert_allclose(d2, expected)
    assert d2[0] < d2[1]                                     # along the correlation is "closer"


def test_the_gate_keeps_nearby_plots_and_excludes_far_ones():
    S = np.eye(2) * 100.0                                    # sigma 10 m
    plots = np.array([[5.0, 5.0], [28.0, 0.0], [31.0, 0.0], [500.0, 500.0]])
    mask, d2 = gate([0.0, 0.0], S, plots)                    # 99% gate = 3.03 sigma = 30.3 m
    assert mask.tolist() == [True, True, False, False]
    assert d2[1] < GATE_99 < d2[2]


def test_clear_pairs_are_matched_and_nothing_is_left_over():
    cost = np.array([[0.5, 40.0], [30.0, 0.7]])
    matches, free_tracks, free_plots = assign(cost)
    assert matches == [(0, 0), (1, 1)] and free_tracks == [] and free_plots == []


def test_extra_plots_and_extra_tracks_stay_unmatched():
    matches, free_tracks, free_plots = assign(np.array([[1.0, 2.0, 50.0], [60.0, 1.5, 70.0]]))
    assert matches == [(0, 0), (1, 1)] and free_tracks == [] and free_plots == [2]
    matches, free_tracks, free_plots = assign(np.array([[1.0], [2.0], [3.0]]))
    assert len(matches) == 1 and len(free_tracks) == 2 and free_plots == []


def test_a_track_with_every_plot_outside_the_gate_gets_nothing():
    cost = np.array([[1.0, 2.0], [500.0, 800.0]])            # track 1 has nothing nearby
    matches, free_tracks, free_plots = assign(cost)
    assert matches == [(0, 0)] and free_tracks == [1] and free_plots == [1]


def test_empty_inputs_are_handled():
    assert assign(np.zeros((0, 3))) == ([], [], [0, 1, 2])
    assert assign(np.zeros((2, 0))) == ([], [0, 1], [])
    assert associate([], [], [[1.0, 2.0]]) == ([], [], [0])


def test_hungarian_beats_greedy_when_a_cheap_pair_forces_an_awful_one():
    # greedy grabs the cheapest pair (track 0 with plot 0, cost 1) and is then forced to pair
    # track 1 with plot 1 at cost 8; the best overall is the two cross pairs, total 2 + 2
    cost = np.array([[1.0, 2.0], [2.0, 8.0]])
    greedy, _, _ = greedy_assign(cost, threshold=100.0)
    best, _, _ = assign(cost, threshold=100.0)
    assert greedy == [(0, 0), (1, 1)] and total_cost(cost, greedy) == 9.0
    assert best == [(0, 1), (1, 0)] and total_cost(cost, best) == 4.0


def test_greedy_takes_the_cheapest_pair_first():
    # cheapest-first picks (0, 0) at cost 1 and then (1, 1); most-expensive-first would pick the
    # cross pair (0, 1) at cost 10 and end with (1, 0), so the result tells the two orders apart
    greedy, free_tracks, free_plots = greedy_assign(np.array([[1.0, 10.0], [2.0, 3.0]]), threshold=100.0)
    assert greedy == [(0, 0), (1, 1)] and free_tracks == [] and free_plots == []


def test_greedy_respects_the_gate_and_leaves_the_rest_unmatched():
    greedy, free_tracks, free_plots = greedy_assign(np.array([[1.0, 500.0], [400.0, 300.0]]))
    assert greedy == [(0, 0)] and free_tracks == [1] and free_plots == [1]


def test_the_assignment_is_optimal_against_brute_force_on_random_problems():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        cost = rng.uniform(0.0, 9.0, size=(n, m))            # everything inside the gate
        matches, _, _ = assign(cost)
        k = min(n, m)
        best = min(sum(cost[r, c] for r, c in zip(rows, cols))
                   for rows in itertools.combinations(range(n), k)
                   for cols in itertools.permutations(range(m), k))
        assert total_cost(cost, matches) == pytest.approx(best)
        assert len(matches) == k
        assert len({t for t, _ in matches}) == len(matches) == len({p for _, p in matches})


def test_associate_matches_each_track_to_its_own_plot_and_leaves_clutter_alone():
    preds = [np.array([0.0, 0.0]), np.array([1000.0, 0.0]), np.array([0.0, 1000.0])]
    S = [np.eye(2) * 400.0] * 3                              # sigma 20 m
    plots = np.array([[1003.0, 4.0], [-6.0, 2.0], [30_000.0, 30_000.0], [3.0, 1008.0]])
    matches, free_tracks, free_plots = associate(preds, S, plots)
    assert matches == [(0, 1), (1, 0), (2, 3)]
    assert free_tracks == [] and free_plots == [2]           # the far-away plot is clutter


def test_cost_matrix_has_one_row_per_track_and_one_column_per_plot():
    cost = cost_matrix([np.zeros(2), np.ones(2)], [np.eye(2)] * 2, [[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    assert cost.shape == (2, 3)
    assert cost[0, 1] == pytest.approx(25.0) and cost[1, 2] == pytest.approx(0.0)
