"""Export tracks and results for the website demo.

    python train_neural.py     # once, to create the model
    python export_demo.py      # writes web/data.js

To also export a recorded training run for the "How the GRU learns" panel:

    python train_neural.py --record outputs/training_run.json
    python export_demo.py --training outputs/training_run.json   # also writes web/training.js
    python export_demo.py --training outputs/training_run.json --only-training   # just training.js

The page loads web/data.js as a <script>, so it works from file:// and from any
static host with no server code.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tracker.imm import default_imm
from tracker.kalman import ConstantVelocityKF
from tracker.kalmannet import estimate as knet_estimate, load_kalmannet
from tracker.metrics import BURN_IN, evaluate, raw_measurement_error, rmse_over_tracks
from tracker.neural import gru_estimates, load_model
from tracker.paths import ROOT, checkpoint
from tracker.scenarios import Q_VAL, SCENARIOS, TARGET
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


VAL_SEED = 2000            # the validation tracks the training run scored itself on
MAX_TRAINING_KB = 1500     # the page loads this file, keep it small


def reference_rmse(n_val: int) -> dict:
    """RMSE of the raw radar and of each Kalman setting on the validation tracks, per scenario.

    These are the lines the training replay's bars are compared with. They use the same tracks
    (seed 2000 + scenario index, n_val tracks each) and the same burn-in as the per-epoch
    validation numbers in the recording, so the numbers are directly comparable.
    """
    ref = {}
    imm = default_imm()
    for i, (name, (mode, radar)) in enumerate(SCENARIOS.items()):
        tracks = simulate_batch(n_val, mode, radar, TARGET, seed=VAL_SEED + i)
        kf = {f"{q:g}": round(rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q))), 1) for q in Q_VAL}
        best_q = min(kf, key=kf.get)
        ref[name] = {"raw": round(rmse_over_tracks(raw_measurement_error(tracks)), 1), "kf": kf,
                     "bestKf": {"q": best_q, "rmse": kf[best_q]},
                     "imm": round(rmse_over_tracks(evaluate(tracks, imm)), 1)}
    return ref


def demo_imm_lines() -> dict:
    """The IMM filter's estimate on each demo track (the first seed-42 track of every scenario)."""
    imm = default_imm()
    return {name: pts(imm.run(simulate_batch(1, mode, radar, TARGET, seed=42)[0])[:, :2])
            for name, (mode, radar) in SCENARIOS.items()}


def training_payload(rec: dict, ref: dict) -> dict:
    """The recording, reshaped for the page (short keys, no training arguments the page does not need)."""
    return {
        "scenarios": rec["scenarios"],
        "bestEpoch": rec["best_epoch"],
        "nEpochs": len(rec["epochs"]) - 1,
        "dt": 1.0,
        "burnIn": BURN_IN,
        "demo": rec["demo"],
        "ref": ref,
        "epochs": [{"epoch": e["epoch"], "trainLoss": e["train_loss"], "valLoss": e["val_loss"],
                    "valRmse": e["val_rmse"], "saved": e["saved"], "rmse": e["scenarios"],
                    "est": e["estimates"]} for e in rec["epochs"]],
    }


def write_training_js(recording_path, out_path) -> int:
    """Convert a training recording (JSON) into web/training.js. Returns the file size in bytes."""
    rec = json.loads(open(recording_path, encoding="utf-8").read())
    n_val = max(int(rec["args"].get("n", 400)) // 4, 1)
    for name, line in demo_imm_lines().items():          # the recording has no IMM: add it for comparison
        rec["demo"][name]["imm"] = line
    payload = training_payload(rec, reference_rmse(n_val))
    out_path.write_text("window.TRAINING = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    size = out_path.stat().st_size
    if size > MAX_TRAINING_KB * 1024:
        print(f"warning: {out_path.name} is {size / 1024:.0f} KB, above the {MAX_TRAINING_KB} KB budget")
    return size


def extra_estimates(tracks, imm=None, knet=None, gru_imm=None) -> dict:
    """Estimates (n, T, 2) of the other trackers the page can show, keyed by the names the page uses.

    imm: the tuned IMM filter (always); knet: a learned-gain model or None; gru_imm: a GRU trained
    with --imm or None. Missing models are simply left out and the page hides them.
    """
    imm = imm or default_imm()
    out = {"imm": np.stack([imm.run(t)[:, :2] for t in tracks])}
    if knet is not None:
        out["knet"] = knet_estimate(knet, tracks)[:, :, :2]
    if gru_imm is not None:
        out["gruimm"] = gru_estimates(gru_imm, tracks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru_tracker", help="checkpoint name in outputs/")
    ap.add_argument("--imm-model", default="gru_imm", help="checkpoint of a GRU trained with --imm (skipped if missing)")
    ap.add_argument("--kalmannet", default="kalmannet", help="learned-gain checkpoint (skipped if missing)")
    ap.add_argument("--training", default=None, metavar="PATH",
                    help="a recording made with train_neural.py --record; also writes web/training.js")
    ap.add_argument("--only-training", action="store_true", help="write only web/training.js (needs no trained model)")
    args = ap.parse_args()

    if args.training:
        if not Path(args.training).exists():
            raise SystemExit(f"No recording at {args.training}. Make one with `python train_neural.py --record {args.training}`.")
        WEB.mkdir(exist_ok=True)
        size = write_training_js(args.training, WEB / "training.js")
        print(f"Wrote {WEB / 'training.js'} ({size / 1024:.0f} KB)")
        if args.only_training:
            return
    elif args.only_training:
        raise SystemExit("--only-training needs --training PATH")

    ckpt = checkpoint(args.model)
    if not ckpt.exists():
        raise SystemExit(f"No trained model at {ckpt}. Run `python train_neural.py` first.")
    model = load_model(ckpt)
    imm_model = load_model(checkpoint(args.imm_model)) if checkpoint(args.imm_model).exists() else None
    knet = load_kalmannet(checkpoint(args.kalmannet)) if checkpoint(args.kalmannet).exists() else None
    print("also exporting:", ", ".join(["IMM"] + (["KalmanNet"] if knet else []) + (["GRU fed the IMM"] if imm_model else [])))

    scenarios = []
    for name, (mode, radar) in SCENARIOS.items():
        tracks = simulate_batch(N_SCORE, mode, radar, TARGET, seed=42)
        truth = np.stack([t.truth[:, :2] for t in tracks])
        gru = gru_estimates(model, tracks)
        extra = extra_estimates(tracks, knet=knet, gru_imm=imm_model)

        rmse = {
            "raw": rmse_over_tracks(raw_measurement_error(tracks)),
            "kf": {f"{q:g}": rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q))) for q in DEMO_Q},
            "gru": rmse_over_tracks(np.linalg.norm(gru - truth, axis=2)),
        }
        for key, est in extra.items():
            rmse[key] = rmse_over_tracks(np.linalg.norm(est - truth, axis=2))
        rmse["raw"] = round(rmse["raw"], 1)
        rmse["gru"] = round(rmse["gru"], 1)
        for key in extra:
            rmse[key] = round(rmse[key], 1)
        rmse["kf"] = {k: round(v, 1) for k, v in rmse["kf"].items()}

        shown = []
        for i, t in enumerate(tracks[:N_SHOW]):
            shown.append({
                "truth": pts(t.truth[:, :2]),
                "meas": pts(t.meas_xy),
                "kf": {f"{q:g}": pts(ConstantVelocityKF(q=q).run(t)[:, :2]) for q in DEMO_Q},
                "gru": pts(gru[i]),
                **{key: pts(est[i]) for key, est in extra.items()},
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
