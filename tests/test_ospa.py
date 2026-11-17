import itertools

import numpy as np
import pytest

from tracker.ospa import ospa


def test_identical_sets_score_zero_whatever_the_order():
    X = np.array([[0.0, 0.0], [500.0, 100.0], [-300.0, 2000.0]])
    assert ospa(X, X[::-1]) == (0.0, 0.0, 0.0)


def test_two_empty_sets_score_zero_and_one_empty_set_scores_the_cutoff():
    assert ospa(np.zeros((0, 2)), np.zeros((0, 2))) == (0.0, 0.0, 0.0)
    assert ospa(np.zeros((0, 2)), [[1.0, 2.0]]) == (1000.0, 0.0, 1000.0)      # a ghost
    assert ospa([[1.0, 2.0], [3.0, 4.0]], np.zeros((0, 2)), c=500.0) == (500.0, 0.0, 500.0)


def test_a_pure_position_error_is_the_distance_and_has_no_cardinality_part():
    total, loc, card = ospa([[0.0, 0.0]], [[30.0, 40.0]])                     # 50 m off
    assert (total, loc, card) == pytest.approx((50.0, 50.0, 0.0))


def test_a_far_off_track_counts_as_the_cutoff_not_its_distance():
    assert ospa([[0.0, 0.0]], [[9000.0, 0.0]])[0] == pytest.approx(1000.0)


def test_one_missing_aircraft_hand_case():
    # truths at 0 and 5000, one track 100 m from the first: n = 2
    # total = sqrt((100^2 + 1000^2) / 2); loc = sqrt(100^2 / 2); card = sqrt(1000^2 / 2)
    total, loc, card = ospa([[0.0, 0.0], [5000.0, 0.0]], [[100.0, 0.0]])
    assert total == pytest.approx(np.sqrt((100.0 ** 2 + 1000.0 ** 2) / 2))
    assert loc == pytest.approx(np.sqrt(100.0 ** 2 / 2))
    assert card == pytest.approx(np.sqrt(1000.0 ** 2 / 2))


def test_one_ghost_track_costs_the_same_as_one_missing_aircraft():
    truth, ghost = [[0.0, 0.0]], [[100.0, 0.0], [8000.0, 0.0]]
    assert ospa(truth, ghost) == pytest.approx(ospa(ghost, truth))            # symmetric
    assert ospa(truth, ghost)[0] == pytest.approx(np.sqrt((100.0 ** 2 + 1000.0 ** 2) / 2))


def test_the_pieces_add_up_as_total_p_equals_loc_p_plus_card_p():
    rng = np.random.default_rng(0)
    for p in (1.0, 2.0, 3.0):
        X, Y = rng.uniform(-2000, 2000, (3, 2)), rng.uniform(-2000, 2000, (5, 2))
        total, loc, card = ospa(X, Y, c=800.0, p=p)
        assert total ** p == pytest.approx(loc ** p + card ** p)


def test_the_pairing_is_the_best_one_not_the_first_one():
    # listed in the "wrong" order: pairing by index would give 400 and 100, the best pairing 100 and 200
    X = np.array([[0.0, 0.0], [200.0, 0.0]])
    Y = np.array([[400.0, 0.0], [100.0, 0.0]])
    best = min(np.sqrt(np.mean([np.linalg.norm(X[i] - Y[j]) ** 2 for i, j in enumerate(perm)]))
               for perm in itertools.permutations(range(2)))
    assert ospa(X, Y)[0] == pytest.approx(best)
    assert best == pytest.approx(np.sqrt((100.0 ** 2 + 200.0 ** 2) / 2))


def test_the_order_p_changes_how_much_big_errors_matter():
    X, Y = [[0.0, 0.0], [5000.0, 0.0]], [[100.0, 0.0], [5800.0, 0.0]]
    assert ospa(X, Y, p=1.0)[0] == pytest.approx((100.0 + 800.0) / 2)
    assert ospa(X, Y, p=2.0)[0] == pytest.approx(np.sqrt((100.0 ** 2 + 800.0 ** 2) / 2))
