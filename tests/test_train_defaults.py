import train_neural
from tracker.neural import load_model


def test_the_bare_command_trains_the_recommended_model():
    a = train_neural.build_parser().parse_args([])
    assert (a.hidden, a.layers, a.epochs) == (128, 2, 50)
    assert a.hybrid is True and a.randomize is True


def test_the_old_flags_still_work_and_can_be_turned_off():
    p = train_neural.build_parser()
    a = p.parse_args(["--hybrid", "--randomize", "--hidden", "128", "--layers", "2", "--epochs", "50"])
    assert (a.hybrid, a.randomize, a.hidden, a.layers, a.epochs) == (True, True, 128, 2, 50)
    a = p.parse_args(["--no-hybrid", "--no-randomize", "--hidden", "64", "--layers", "1", "--epochs", "30"])
    assert (a.hybrid, a.randomize, a.hidden, a.layers, a.epochs) == (False, False, 64, 1, 30)


def test_a_tiny_bare_run_saves_a_hybrid_128_by_2_model_and_the_plain_model_still_trains(tmp_path, monkeypatch):
    monkeypatch.setattr(train_neural, "OUT_DIR", tmp_path)
    monkeypatch.setattr(train_neural, "checkpoint", lambda name: tmp_path / f"{name}.pt")
    train_neural.main(["--n", "8", "--epochs", "1", "--name", "bare"])
    cfg = load_model(tmp_path / "bare.pt").config
    assert cfg["hidden"] == 128 and cfg["layers"] == 2 and cfg["hybrid"] is True

    train_neural.main(["--n", "8", "--epochs", "1", "--name", "plain", "--no-hybrid", "--no-randomize",
                       "--hidden", "64", "--layers", "1"])
    cfg = load_model(tmp_path / "plain.pt").config
    assert cfg["hidden"] == 64 and cfg["layers"] == 1 and cfg["hybrid"] is False
