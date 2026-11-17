import numpy as np
import pytest
import torch

from tracker.dataset import tracks_to_tensors
from tracker.gru_numpy import forward, gru_layer, weights_from_torch
from tracker.neural import GRUTracker, gru_estimates
from tracker.scenarios import SCENARIOS, TARGET
from tracker.simulate import simulate_batch


def random_model(hybrid, layers, hidden=12, seed=0):
    """A model whose head is NOT zero (a fresh model outputs exactly the base, which would test nothing)."""
    torch.manual_seed(seed)
    from tracker.dataset import n_features
    m = GRUTracker(n_feat=n_features(hybrid), hidden=hidden, layers=layers, hybrid=hybrid)
    with torch.no_grad():
        m.head.weight.normal_(0, 0.5)
        m.head.bias.normal_(0, 0.2)
    return m.eval()


def tracks(n=3):
    mode, radar = SCENARIOS["maneuvers"]
    return simulate_batch(n, mode, radar, TARGET, seed=5)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("layers", [1, 2])
def test_numpy_forward_equals_pytorch(hybrid, layers):
    model = random_model(hybrid, layers)
    ts = tracks()
    expected = gru_estimates(model, ts)
    feats, base, _ = tracks_to_tensors(ts, hybrid=hybrid)
    got = forward(weights_from_torch(model), feats.numpy(), base.numpy())
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-2)       # metres; float32 vs float64 rounding only


def test_the_output_really_differs_from_the_base_so_the_test_is_not_vacuous():
    model = random_model(True, 2)
    feats, base, _ = tracks_to_tensors(tracks(), hybrid=True)
    out = forward(weights_from_torch(model), feats.numpy(), base.numpy())
    assert np.abs(out - base.numpy()).max() > 5.0


def test_a_single_track_gives_the_same_answer_as_a_batch():
    model = random_model(False, 1)
    feats, base, _ = tracks_to_tensors(tracks(2), hybrid=False)
    w = weights_from_torch(model)
    batch = forward(w, feats.numpy(), base.numpy())
    np.testing.assert_allclose(forward(w, feats[1].numpy(), base[1].numpy()), batch[1])


def test_gate_formulas_on_a_hand_computed_one_unit_cell():
    # hidden=1, x=[1]: all weights/biases chosen so every gate is easy to compute by hand
    w_ih, w_hh = np.array([[0.0], [0.0], [2.0]]), np.zeros((3, 1))
    b_ih, b_hh = np.zeros(3), np.zeros(3)
    out = gru_layer(np.array([[1.0], [1.0]]), w_ih, w_hh, b_ih, b_hh)
    n = np.tanh(2.0)                              # r = z = 0.5, h0 = 0
    h1 = 0.5 * n
    h2 = 0.5 * n + 0.5 * h1                       # candidate n is the same (W_hn h = 0)
    np.testing.assert_allclose(out[:, 0], [h1, h2])


def test_the_reset_gate_multiplies_only_the_recurrent_part_of_the_candidate():
    # PyTorch applies r to (W_hn h + b_hn), b_hn included. Make that the only non-zero term.
    w_ih, w_hh = np.zeros((3, 1)), np.zeros((3, 1))
    b_ih, b_hh = np.zeros(3), np.array([0.0, 0.0, 3.0])
    out = gru_layer(np.array([[0.0]]), w_ih, w_hh, b_ih, b_hh)
    np.testing.assert_allclose(out[0, 0], 0.5 * np.tanh(0.5 * 3.0))   # (1-z) * tanh(r * b_hn), r = z = 0.5
