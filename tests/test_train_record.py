import json

import numpy as np
import torch

import train_neural
from tracker.dataset import tracks_to_tensors
from tracker.scenarios import SCENARIOS, TARGET
from tracker.simulate import simulate_batch

SCEN = list(SCENARIOS)


def tiny_run(tmp_path, monkeypatch, name, *extra):
    monkeypatch.setattr(train_neural, "OUT_DIR", tmp_path)
    monkeypatch.setattr(train_neural, "checkpoint", lambda n: tmp_path / f"{n}.pt")
    train_neural.main(["--n", "8", "--epochs", "2", "--hidden", "16", "--layers", "1", "--name", name, *extra])
    return tmp_path / f"{name}.pt"


def test_a_tiny_run_writes_a_recording_with_the_expected_structure(tmp_path, monkeypatch):
    rec_path = tmp_path / "rec.json"
    tiny_run(tmp_path, monkeypatch, "rec", "--record", str(rec_path))
    rec = json.loads(rec_path.read_text())
    assert rec["scenarios"] == SCEN
    assert [e["epoch"] for e in rec["epochs"]] == [0, 1, 2]                  # epoch 0 = untrained, then 2 epochs
    assert rec["epochs"][0]["train_loss"] is None and rec["epochs"][0]["saved"] is False
    for e in rec["epochs"][1:]:
        assert isinstance(e["train_loss"], float) and isinstance(e["val_loss"], float)
        assert e["val_rmse"] > 0 and set(e["scenarios"]) == set(SCEN)
        assert set(e["estimates"]) == set(SCEN)
        assert all(len(pts) == TARGET.n_steps and len(pts[0]) == 2 for pts in e["estimates"].values())
    assert rec["best_epoch"] in (1, 2) and any(e["saved"] for e in rec["epochs"])
    assert rec["args"]["epochs"] == 2 and "record" not in rec["args"]


def test_the_demo_tracks_are_the_seed_42_tracks_the_web_page_already_shows(tmp_path, monkeypatch):
    rec_path = tmp_path / "rec.json"
    tiny_run(tmp_path, monkeypatch, "rec", "--record", str(rec_path))
    demo = json.loads(rec_path.read_text())["demo"]
    for name, (mode, radar) in SCENARIOS.items():
        shown = simulate_batch(5, mode, radar, TARGET, seed=42)[0]             # export_demo's first example track
        np.testing.assert_allclose(np.array(demo[name]["truth"]), shown.truth[:, :2], atol=0.06)
        meas = np.array([[np.nan, np.nan] if m is None else m for m in demo[name]["meas"]], dtype=float)
        np.testing.assert_allclose(meas, shown.meas_xy, atol=0.06, equal_nan=True)


def test_epoch_zero_is_exactly_the_untrained_base_position(tmp_path, monkeypatch):
    rec_path = tmp_path / "rec.json"
    tiny_run(tmp_path, monkeypatch, "rec", "--record", str(rec_path))
    rec = json.loads(rec_path.read_text())
    tracks = [simulate_batch(1, mode, radar, TARGET, seed=42)[0] for mode, radar in SCENARIOS.values()]
    _, base, _ = tracks_to_tensors(tracks, hybrid=True)                        # the default model is hybrid
    for i, name in enumerate(SCEN):
        np.testing.assert_allclose(np.array(rec["epochs"][0]["estimates"][name]), base[i].numpy(), atol=0.06)
        np.testing.assert_allclose(np.array(rec["demo"][name]["base"]), base[i].numpy(), atol=0.06)


def test_the_estimates_change_as_the_model_trains(tmp_path, monkeypatch):
    rec_path = tmp_path / "rec.json"
    tiny_run(tmp_path, monkeypatch, "rec", "--record", str(rec_path))
    e = json.loads(rec_path.read_text())["epochs"]
    first, last = np.array(e[0]["estimates"]["turning"]), np.array(e[2]["estimates"]["turning"])
    assert np.abs(first - last).max() > 0.5                                    # the model moved off its base


def test_recording_does_not_change_the_training_itself(tmp_path, monkeypatch):
    plain = tiny_run(tmp_path, monkeypatch, "plain")
    recorded = tiny_run(tmp_path, monkeypatch, "recorded", "--record", str(tmp_path / "r.json"))
    a, b = torch.load(plain, weights_only=True)["state"], torch.load(recorded, weights_only=True)["state"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_without_record_no_file_is_written(tmp_path, monkeypatch):
    tiny_run(tmp_path, monkeypatch, "norec")
    assert not list(tmp_path.glob("*.json"))
    assert train_neural.build_parser().parse_args([]).record is None


def test_missed_scans_are_stored_as_null_and_the_file_is_valid_json(tmp_path, monkeypatch):
    rec_path = tmp_path / "rec.json"
    tiny_run(tmp_path, monkeypatch, "rec", "--record", str(rec_path))
    text = rec_path.read_text()
    assert "NaN" not in text and "Infinity" not in text                        # strict JSON, readable by a browser
    meas = json.loads(text)["demo"]["missed_30"]["meas"]
    track = simulate_batch(1, *[SCENARIOS["missed_30"][i] for i in (0, 1)], TARGET, seed=42)[0]
    assert [m is None for m in meas] == list(~track.detected)
    assert any(m is None for m in meas)                                        # this demo track really misses scans


def test_estimates_are_rounded_to_a_tenth_of_a_metre_to_keep_the_file_small(tmp_path, monkeypatch):
    import re
    rec_path = tmp_path / "rec.json"
    tiny_run(tmp_path, monkeypatch, "rec", "--record", str(rec_path))
    estimates = json.dumps(json.loads(rec_path.read_text())["epochs"][1]["estimates"])
    assert not re.search(r"\d\.\d{2,}", estimates)


def test_the_best_epoch_is_the_latest_one_that_was_saved():
    rec = train_neural.TrainingRecorder(hybrid=True)
    rec.epochs = [{"epoch": 0, "saved": False}, {"epoch": 1, "saved": True}, {"epoch": 2, "saved": False},
                  {"epoch": 3, "saved": True}, {"epoch": 4, "saved": False}]
    assert rec.to_dict()["best_epoch"] == 3
    rec.epochs = [{"epoch": 0, "saved": False}]
    assert rec.to_dict()["best_epoch"] is None
