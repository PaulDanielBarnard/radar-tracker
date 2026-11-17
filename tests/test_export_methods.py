import numpy as np
import torch

import export_demo
from tracker.dataset import n_features
from tracker.imm import default_imm
from tracker.kalmannet import KalmanNet
from tracker.neural import GRUTracker
from tracker.scenarios import SCENARIOS, TARGET
from tracker.simulate import simulate_batch


def tracks(n=2):
    mode, radar = SCENARIOS["turning"]
    return simulate_batch(n, mode, radar, TARGET, seed=4)


def test_without_extra_models_only_the_imm_is_exported_and_it_is_the_imm():
    ts = tracks()
    out = export_demo.extra_estimates(ts)
    assert set(out) == {"imm"} and out["imm"].shape == (2, 120, 2)
    np.testing.assert_allclose(out["imm"][1], default_imm().run(ts[1])[:, :2])


def test_the_learned_gain_and_the_imm_fed_gru_are_exported_when_given():
    ts = tracks()
    torch.manual_seed(0)
    knet = KalmanNet(hidden=8, layers=1).eval()
    gru_imm = GRUTracker(n_feat=n_features("imm"), hidden=8, layers=1, hybrid="imm").eval()
    out = export_demo.extra_estimates(ts, knet=knet, gru_imm=gru_imm)
    assert set(out) == {"imm", "knet", "gruimm"}
    assert all(v.shape == (2, 120, 2) for v in out.values())
    # an untrained GRU fed the IMM outputs exactly the IMM estimate (its correction starts at zero)
    np.testing.assert_allclose(out["gruimm"], out["imm"], atol=0.01)
    assert np.abs(out["knet"] - out["imm"]).max() > 1.0        # the learned-gain filter is a different tracker
