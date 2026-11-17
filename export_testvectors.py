"""Write web/testvectors.json: fixed answers from the Python code, for web/test_js.mjs to check the JavaScript port.

    python export_testvectors.py
    node web/test_js.mjs

The file holds one fixed flight with misses (the radar plots, the true path), what the Kalman filter
(q = 0.1 and 100) and the network's input features come out as in Python, and a small fixed GRU
(random weights, so the test does not depend on a training run) with its Python output.
"""
import json
from pathlib import Path

import numpy as np
import torch

from export_model import model_payload
from tracker.dataset import n_features, track_to_arr
from tracker.gru_numpy import forward, weights_from_torch
from tracker.kalman import ConstantVelocityKF
from tracker.neural import GRUTracker
from tracker.simulate import RadarConfig, TargetConfig, simulate_track

WEB = Path(__file__).parent / "web"
SEED = 7


def _r(a, digits=6):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def fixed_model() -> GRUTracker:
    """A small hybrid GRU with random, non-zero weights (a fresh model would output exactly its base)."""
    torch.manual_seed(3)
    m = GRUTracker(n_feat=n_features(True), hidden=16, layers=2, hybrid=True)
    with torch.no_grad():
        m.head.weight.normal_(0, 0.5)
        m.head.bias.normal_(0, 0.2)
    return m.eval()


def build_vectors() -> dict:
    radar = RadarConfig(sigma_r=80.0, sigma_theta=0.008, p_detect=0.8)
    track = simulate_track(np.random.default_rng(SEED), "random", radar, TargetConfig())
    feats, base, _ = track_to_arr(track, hybrid=True)
    model = fixed_model()
    payload = model_payload(model)
    # the expected output is computed from the same rounded weights the browser gets
    weights = {"config": payload["config"],
               "layers": [{k: np.array(v) for k, v in layer.items()} for layer in payload["layers"]],
               "head": {k: np.array(v) for k, v in payload["head"].items()}}
    return {
        "radar": {"sigmaR": radar.sigma_r, "sigmaTheta": radar.sigma_theta, "pDetect": radar.p_detect, "dt": radar.dt},
        "truth": _r(track.truth, 3),
        "polar": [None if not d else _r(p, 9) for p, d in zip(track.meas_polar, track.detected)],
        "xy": [None if not d else _r(p, 6) for p, d in zip(track.meas_xy, track.detected)],
        "detected": track.detected.tolist(),
        "kf": {"0.1": _r(ConstantVelocityKF(q=0.1).run(track), 4), "100": _r(ConstantVelocityKF(q=100.0).run(track), 4)},
        "features": _r(feats, 6),
        "base": _r(base, 4),
        "model": payload,
        "gru": _r(forward(weights, feats, base), 4),
    }


def main():
    WEB.mkdir(exist_ok=True)
    out = WEB / "testvectors.json"
    out.write_text(json.dumps(build_vectors(), separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
