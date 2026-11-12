import numpy as np
import torch

from tracker.dataset import tracks_to_tensors
from tracker.neural import GRUTracker
from tracker.simulate import RadarConfig,TargetConfig,simulate_batch

def test_tensors_have_no_nans_even_with_missed_scans():
    tracks = simulate_batch(5, "random", RadarConfig(p_detect=0.5), TargetConfig(), seed=0)
    feats, base, truth = tracks_to_tensors(tracks)
    assert feats.shape == (5, 120, 5)
    assert base.shape == truth.shape == (5, 120, 2)
    assert not torch.isnan(feats).any()
    assert not torch.isnan(base).any()


def test_untrained_model_stays_near_the_raw_plot():
    tracks = simulate_batch(3, "straight", RadarConfig(), TargetConfig(), seed=0)
    feats, base, _ = tracks_to_tensors(tracks)
    pred = GRUTracker()(feats, base)
    assert pred.shape == base.shape
    assert (pred - base).abs().max() < 1000   # correction starts small (under 1 km)