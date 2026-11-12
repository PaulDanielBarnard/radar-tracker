"""Export tracks and results for the website demo.

    python train_neural.py     # once, to create the model
    python export_demo.py      # writes web/data.js

The page loads web/data.js as a <script>, so it works from file:// and from any
static host with no server code.
"""
import json

import numpy as np
import torch

from tracker.kalman import ConstantVelocityKF
from tracker.metrics import BURN_IN, evaluate, raw_measurement_error, rmse_over_tracks
from tracker.neural import GRUTracker, gru_estimates
from tracker.paths import CHECKPOINT, ROOT
from tracker.scenarios import SCENARIOS, TARGET
from tracker.simulate import simulate_batch

WEB = ROOT / "web"
DEMO_Q = [0.1, 10.0, 100.0]   # Kalman settings shown on the page
N_SHOW = 8                    # example tracks per scenario
N_SCORE = 500                 # tracks used for the RMSE bars (same as run_compare)

BLURBS = {
    "straight":  "The target flies dead straight. Noise is the only problem.",
    "turning":   "A steady turn. A filter that expects straight flight lags behind.",
    "maneuvers": "Random turns and speed changes, like a real aircraft.",
    "noisy":     "Same maneuvers, but the radar is 3x noisier.",
    "missed_30": "Same maneuvers, but 30% of scans return nothing at all.",
}


def pts(a):
    """Round to 0.1 m and turn NaN into null to keep the file small and valid JSON."""
    return [None if np.isnan(x) else [round(float(x), 1), round(float(y), 1)] for x, y in a]


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"No trained model at {CHECKPOINT}. Run `python train_neural.py` first.")
    model = GRUTracker()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.eval()

    scenarios = []
    for name, (mode, radar) in SCENARIOS.items():
        tracks = simulate_batch(N_SCORE, mode, radar, TARGET, seed=42)
        truth = np.stack([t.truth[:, :2] for t in tracks])
        gru = gru_estimates(model, tracks)

        rmse = {
            "raw": rmse_over_tracks(raw_measurement_error(tracks)),
            "kf": {f"{q:g}": rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q))) for q in DEMO_Q},
            "gru": rmse_over_tracks(np.linalg.norm(gru - truth, axis=2)),
        }
        rmse["raw"] = round(rmse["raw"], 1)
        rmse["gru"] = round(rmse["gru"], 1)
        rmse["kf"] = {k: round(v, 1) for k, v in rmse["kf"].items()}

        shown = []
        for i, t in enumerate(tracks[:N_SHOW]):
            shown.append({
                "truth": pts(t.truth[:, :2]),
                "meas": pts(t.meas_xy),
                "kf": {f"{q:g}": pts(ConstantVelocityKF(q=q).run(t)[:, :2]) for q in DEMO_Q},
                "gru": pts(gru[i]),
            })
        scenarios.append({"id": name, "blurb": BLURBS[name], "rmse": rmse, "tracks": shown})
        print(f"{name:<12} exported {len(shown)} tracks, RMSE over {N_SCORE}: {rmse}")

    demo = {"dt": 1.0, "burnIn": BURN_IN, "qs": [f"{q:g}" for q in DEMO_Q],
            "nScore": N_SCORE, "scenarios": scenarios}
    WEB.mkdir(exist_ok=True)
    out = WEB / "data.js"
    out.write_text("window.DEMO = " + json.dumps(demo, separators=(",", ":")) + ";\n")
    print(f"\nWrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
