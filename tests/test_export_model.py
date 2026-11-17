import json
import sys

import numpy as np
import pytest
import torch

import export_model
from tracker.dataset import tracks_to_tensors
from tracker.gru_numpy import forward
from tracker.neural import GRUTracker, gru_estimates, save_model
from tracker.scenarios import SCENARIOS, TARGET
from tracker.simulate import simulate_batch


def random_model(hybrid, layers, hidden=12, seed=0):
    """A model with a non-zero head (a fresh model outputs exactly the base, which would test nothing)."""
    from tracker.dataset import n_features
    torch.manual_seed(seed)
    m = GRUTracker(n_feat=n_features(hybrid), hidden=hidden, layers=layers, hybrid=hybrid)
    with torch.no_grad():
        m.head.weight.normal_(0, 0.5)
        m.head.bias.normal_(0, 0.2)
    return m.eval()


def load_js(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("window.GRU_MODEL = ") and text.rstrip().endswith(";")
    return json.loads(text[len("window.GRU_MODEL = "):].rstrip().rstrip(";"))


def as_weights(payload):
    return {"config": payload["config"],
            "layers": [{k: np.array(v) for k, v in layer.items()} for layer in payload["layers"]],
            "head": {k: np.array(v) for k, v in payload["head"].items()}}


@pytest.mark.parametrize("hybrid,layers", [(False, 1), (True, 2)])
def test_the_exported_file_rebuilds_a_model_with_the_same_outputs(tmp_path, hybrid, layers):
    model = random_model(hybrid, layers, hidden=16)
    out = tmp_path / "model.js"
    size = export_model.write_model_js(model, out)
    assert size == out.stat().st_size
    payload = load_js(out)
    mode, radar = SCENARIOS["turning"]
    tracks = simulate_batch(3, mode, radar, TARGET, seed=9)
    feats, base, _ = tracks_to_tensors(tracks, hybrid=hybrid)
    got = forward(as_weights(payload), feats.numpy(), base.numpy())
    np.testing.assert_allclose(got, gru_estimates(model, tracks), atol=0.2)   # 5 significant digits -> sub-metre


def test_config_scales_and_shapes_are_exported(tmp_path):
    model = random_model(True, 2, hidden=8)
    export_model.write_model_js(model, tmp_path / "m.js")
    p = load_js(tmp_path / "m.js")
    assert p["config"] == {"n_feat": 19, "hidden": 8, "layers": 2, "hybrid": True}
    assert p["scales"]["OUT_SCALE"] == 1000.0 and p["scales"]["HYBRID_QS"] == [0.1, 100.0]
    assert len(p["layers"]) == 2 and np.array(p["layers"][0]["w_ih"]).shape == (24, 19)
    assert np.array(p["layers"][1]["w_ih"]).shape == (24, 8) and np.array(p["head"]["w"]).shape == (2, 8)


def test_numbers_are_rounded_to_five_significant_digits():
    assert export_model._round(np.array([0.123456789, -1234.56789, 1e-7])) == [0.12346, -1234.6, 1e-07]


def test_the_default_hidden_128_two_layer_model_fits_the_size_budget(tmp_path):
    from tracker.dataset import n_features
    torch.manual_seed(0)
    model = GRUTracker(n_feat=n_features(True), hidden=128, layers=2, hybrid=True)
    size = export_model.write_model_js(model, tmp_path / "m.js")
    assert size < export_model.MAX_MODEL_MB * 1024 * 1024


def test_an_oversized_file_gets_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_model, "MAX_MODEL_MB", 0)
    export_model.write_model_js(random_model(False, 1), tmp_path / "m.js")
    assert "warning" in capsys.readouterr().out


def test_the_command_line_writes_model_js_and_complains_about_a_missing_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_model, "WEB", tmp_path)
    monkeypatch.setattr(export_model, "checkpoint", lambda n: tmp_path / f"{n}.pt")
    monkeypatch.setattr(sys, "argv", ["export_model.py", "--model", "nothing"])
    with pytest.raises(SystemExit, match="train_neural.py"):
        export_model.main()
    save_model(random_model(False, 1), tmp_path / "ok.pt")
    monkeypatch.setattr(sys, "argv", ["export_model.py", "--model", "ok"])
    export_model.main()
    assert (tmp_path / "model.js").exists() and "model.js" in capsys.readouterr().out
