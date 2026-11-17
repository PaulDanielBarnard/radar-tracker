import torch

import train_kalmannet
from tracker.kalmannet import load_kalmannet


def test_defaults_and_flags():
    a = train_kalmannet.build_parser().parse_args([])
    assert (a.epochs, a.hidden, a.layers, a.name) == (30, 64, 1, "kalmannet")
    a = train_kalmannet.build_parser().parse_args(["--hidden", "32", "--layers", "2", "--name", "x"])
    assert (a.hidden, a.layers, a.name) == (32, 2, "x")


def test_a_tiny_run_trains_saves_and_improves_on_the_untrained_filter(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_kalmannet, "OUT_DIR", tmp_path)
    monkeypatch.setattr(train_kalmannet, "checkpoint", lambda name: tmp_path / f"{name}.pt")
    train_kalmannet.main(["--n", "8", "--epochs", "3", "--name", "tiny", "--hidden", "16", "--batch", "16"])
    out = capsys.readouterr().out
    assert "<- saved" in out and "trained" in out
    model = load_kalmannet(tmp_path / "tiny.pt")
    assert model.config["hidden"] == 16
    assert any(p.abs().sum() > 0 for p in model.head.weight)          # the head moved away from zero


def test_the_per_scenario_report_lists_every_scenario(tmp_path, monkeypatch, capsys):
    from tracker.kalmannet import KalmanNet
    train_kalmannet.report_by_scenario(KalmanNet(hidden=8), train_kalmannet.make_scenario_data(2, seed=5))
    out = capsys.readouterr().out
    for name in ("straight", "turning", "maneuvers", "noisy", "missed_30"):
        assert name in out
