import numpy as np
import torch

import train_neural
from tracker.dataset import N_FEATURES, N_IMM_FEATURES, n_features, track_to_arr, tracks_to_tensors
from tracker.imm import default_imm
from tracker.neural import GRUTracker, gru_estimates, load_model
from tracker.scenarios import SCENARIOS, TARGET
from tracker.simulate import simulate_batch


def tracks(n=2):
    mode, radar = SCENARIOS["maneuvers"]
    return simulate_batch(n, mode, radar, TARGET, seed=3)


def test_feature_counts_for_plain_hybrid_and_imm():
    assert n_features(False) == N_FEATURES == 11
    assert n_features(True) == 19
    assert n_features("imm") == 19 + N_IMM_FEATURES == 27


def test_the_imm_columns_are_the_imm_estimate_and_mode_probabilities():
    t = tracks(1)[0]
    feat, base, _ = track_to_arr(t, hybrid="imm")
    est, diag = default_imm().run(t, return_diag=True)
    assert feat.shape == (120, 27)
    np.testing.assert_allclose(base, est[:, :2])                       # the network corrects the IMM
    np.testing.assert_allclose(feat[:, 19 + 4:19 + 8], diag["mu"])     # last four columns = mode probabilities
    np.testing.assert_allclose(feat[:, 19 + 4:19 + 8].sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(feat[:, 19 + 2:19 + 4], est[:, 2:4] / 300.0)


def test_the_kalman_hybrid_features_are_unchanged_by_the_new_option():
    t = tracks(1)[0]
    plain, hybrid, imm = (track_to_arr(t, hybrid=h)[0] for h in (False, True, "imm"))
    np.testing.assert_array_equal(hybrid[:, :11], plain)
    np.testing.assert_array_equal(imm[:, :19], hybrid)


def test_an_untrained_imm_model_outputs_exactly_the_imm_estimate():
    model = GRUTracker(n_feat=n_features("imm"), hidden=8, layers=1, hybrid="imm")
    ts = tracks(1)
    out = gru_estimates(model, ts)[0]
    est = default_imm().run(ts[0])[:, :2]
    np.testing.assert_allclose(out, est, atol=0.01)


def test_the_imm_flag_trains_and_saves_a_model_that_reloads(tmp_path, monkeypatch):
    monkeypatch.setattr(train_neural, "OUT_DIR", tmp_path)
    monkeypatch.setattr(train_neural, "checkpoint", lambda n: tmp_path / f"{n}.pt")
    train_neural.main(["--n", "4", "--epochs", "1", "--hidden", "8", "--layers", "1", "--name", "immtiny",
                       "--imm", "--no-randomize"])
    model = load_model(tmp_path / "immtiny.pt")
    assert model.config["hybrid"] == "imm" and model.config["n_feat"] == 27
    assert train_neural.build_parser().parse_args([]).imm is False
