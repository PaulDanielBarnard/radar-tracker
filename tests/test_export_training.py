import json
import sys

import numpy as np
import pytest

import export_demo
import train_neural
from tracker.kalman import ConstantVelocityKF
from tracker.metrics import evaluate, rmse_over_tracks
from tracker.scenarios import SCENARIOS, TARGET
from tracker.simulate import simulate_batch


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """A tiny real training recording (8 tracks per scenario, 2 epochs)."""
    d = tmp_path_factory.mktemp("rec")
    original_dir, original_checkpoint = train_neural.OUT_DIR, train_neural.checkpoint
    train_neural.OUT_DIR = d
    train_neural.checkpoint = lambda n: d / f"{n}.pt"
    try:
        train_neural.main(["--n", "8", "--epochs", "2", "--hidden", "16", "--layers", "1",
                           "--name", "t", "--record", str(d / "run.json")])
    finally:
        train_neural.OUT_DIR, train_neural.checkpoint = original_dir, original_checkpoint
    return d / "run.json"


def load_js(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("window.TRAINING = ") and text.rstrip().endswith(";")
    return json.loads(text[len("window.TRAINING = "):].rstrip().rstrip(";"))


def test_the_writer_produces_a_script_the_page_can_load(recording, tmp_path):
    out = tmp_path / "training.js"
    size = export_demo.write_training_js(recording, out)
    t = load_js(out)
    assert size == out.stat().st_size and size < export_demo.MAX_TRAINING_KB * 1024
    assert t["scenarios"] == list(SCENARIOS) and t["nEpochs"] == 2 and len(t["epochs"]) == 3
    assert t["bestEpoch"] in (1, 2) and t["burnIn"] == 10
    e = t["epochs"][1]
    assert set(e) == {"epoch", "trainLoss", "valLoss", "valRmse", "saved", "rmse", "est"}
    assert set(e["rmse"]) == set(SCENARIOS) and set(e["est"]) == set(SCENARIOS)
    assert all(len(e["est"][n]) == TARGET.n_steps for n in SCENARIOS)
    assert set(t["demo"]["turning"]) == {"truth", "meas", "base"}
    assert "args" not in t and t["epochs"][0]["trainLoss"] is None


def test_reference_lines_are_the_raw_radar_and_kalman_on_the_validation_tracks():
    ref = export_demo.reference_rmse(3)
    assert set(ref) == set(SCENARIOS)
    mode, radar = SCENARIOS["turning"]
    tracks = simulate_batch(3, mode, radar, TARGET, seed=export_demo.VAL_SEED + 1)
    expected = rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=100.0)))
    assert ref["turning"]["kf"]["100"] == pytest.approx(expected, abs=0.06)
    assert ref["turning"]["bestKf"]["q"] == min(ref["turning"]["kf"], key=ref["turning"]["kf"].get)
    assert ref["straight"]["bestKf"]["q"] == "0.1" and ref["turning"]["bestKf"]["q"] == "100"
    assert ref["noisy"]["raw"] > 2 * ref["straight"]["raw"]


def test_the_reference_uses_the_same_validation_tracks_as_the_recording(recording, tmp_path):
    rec = json.loads(recording.read_text())
    out = tmp_path / "training.js"
    export_demo.write_training_js(recording, out)
    t = load_js(out)
    assert rec["args"]["n"] == 8
    # the tiny recording scored 8 // 4 = 2 validation tracks per scenario; the reference must use the same 2
    np.testing.assert_allclose(t["ref"]["straight"]["raw"], export_demo.reference_rmse(2)["straight"]["raw"])


def test_the_command_line_writes_only_training_js_without_a_model(recording, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_demo, "WEB", tmp_path)
    monkeypatch.setattr(sys, "argv", ["export_demo.py", "--training", str(recording), "--only-training",
                                      "--model", "no_such_model"])
    export_demo.main()
    assert (tmp_path / "training.js").exists() and not (tmp_path / "data.js").exists()
    assert "training.js" in capsys.readouterr().out


def test_missing_recording_and_missing_flag_give_clear_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export_demo.py", "--training", str(tmp_path / "none.json")])
    with pytest.raises(SystemExit, match="train_neural.py --record"):
        export_demo.main()
    monkeypatch.setattr(sys, "argv", ["export_demo.py", "--only-training"])
    with pytest.raises(SystemExit, match="--training"):
        export_demo.main()


def test_the_reference_tracks_start_at_the_training_validation_seed_2000():
    from tracker.metrics import raw_measurement_error
    mode, radar = SCENARIOS["straight"]
    tracks = simulate_batch(2, mode, radar, TARGET, seed=2000)                  # the seed train_neural validates on
    expected = rmse_over_tracks(raw_measurement_error(tracks))
    assert export_demo.reference_rmse(2)["straight"]["raw"] == pytest.approx(expected, abs=0.06)


def test_a_file_over_the_size_budget_gets_a_warning(recording, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_demo, "MAX_TRAINING_KB", 0)
    export_demo.write_training_js(recording, tmp_path / "training.js")
    assert "warning" in capsys.readouterr().out
