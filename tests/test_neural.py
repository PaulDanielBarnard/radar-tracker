import numpy as np
import torch

from tracker.dataset import N_FEATURES, n_features, tracks_to_tensors
from tracker.neural import GRUTracker, load_model, save_model
from tracker.simulate import RadarConfig,TargetConfig,simulate_batch

def test_tensors_have_no_nans_even_with_missed_scans():
    tracks = simulate_batch(5, "random", RadarConfig(p_detect=0.5), TargetConfig(), seed=0)
    feats, base, truth = tracks_to_tensors(tracks)
    assert feats.shape == (5, 120, N_FEATURES)
    assert base.shape == truth.shape == (5, 120, 2)
    assert not torch.isnan(feats).any()
    assert not torch.isnan(base).any()


def test_untrained_model_outputs_the_raw_plot():
    torch.manual_seed(0)
    tracks = simulate_batch(3, "straight", RadarConfig(), TargetConfig(), seed=0)
    feats, base, _ = tracks_to_tensors(tracks)
    pred = GRUTracker()(feats, base)
    assert pred.shape == base.shape
    assert torch.allclose(pred, base)   # zero-initialised head -> no correction yet


def test_saved_model_reloads_with_identical_outputs(tmp_path):
    torch.manual_seed(0)
    model = GRUTracker(hidden=16, layers=2)
    for p in model.parameters():           # make the head non-zero so the check means something
        torch.nn.init.normal_(p, std=0.1)
    tracks = simulate_batch(2, "random", RadarConfig(), TargetConfig(), seed=0)
    feats, base, _ = tracks_to_tensors(tracks)
    save_model(model, tmp_path / "m.pt")
    reloaded = load_model(tmp_path / "m.pt")
    assert reloaded.config == model.config
    assert torch.allclose(model(feats, base), reloaded(feats, base))


def test_hybrid_features_start_from_the_kalman_estimate():
    tracks = simulate_batch(2, "random", RadarConfig(p_detect=0.7), TargetConfig(), seed=0)
    feats, base, truth = tracks_to_tensors(tracks, hybrid=True)
    assert feats.shape == (2, 120, n_features(True))
    assert not torch.isnan(feats).any()
    _, plain_base, _ = tracks_to_tensors(tracks)
    assert not torch.allclose(base, plain_base)          # base is the Kalman position
    model = GRUTracker(n_feat=n_features(True), hybrid=True)
    assert torch.allclose(model(feats, base), base)       # zero head -> exactly the Kalman estimate
