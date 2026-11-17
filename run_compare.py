"""Compare Raw Radar, Kalman Filters, the IMM filter and the GRU on every scenario"""
import argparse

import numpy as np
import torch

from tracker.imm import default_imm
from tracker.kalmannet import estimate as kalmannet_estimate
from tracker.kalmannet import load_kalmannet
from tracker.kalman import ConstantVelocityKF
from tracker.metrics import evaluate, raw_measurement_error, rmse_over_tracks
from tracker.neural import gru_estimates, load_model
from tracker.paths import checkpoint
from tracker.scenarios import Q_VAL, SCENARIOS, TARGET
from tracker.simulate import simulate_batch

def gru_errors(model, tracks):

    truth = np.stack([t.truth[:, :2] for t in tracks])
    return np.linalg.norm(gru_estimates(model, tracks) - truth, axis=2)  # (n_tracks, T)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type = int, default = 500)
    ap.add_argument("--model", default = "gru_tracker", help = "checkpoint name in outputs/")
    ap.add_argument("--kalmannet", default = "kalmannet", help = "learned-gain checkpoint name (skipped if missing)")
    args = ap.parse_args()

    ckpt = checkpoint(args.model)

    if not ckpt.exists():
        raise SystemExit(f"No trained model at {ckpt}. Run `python train_neural.py` first.")
    
    model = load_model(ckpt)

    knet_path = checkpoint(args.kalmannet)
    knet = load_kalmannet(knet_path) if knet_path.exists() else None
    lbl = ["raw radar"] + [f"KF q = {q:g}" for q in Q_VAL] + ["IMM", "GRU"] + (["KalmanNet"] if knet else [])
    imm = default_imm()
    hdr = f"{'scenario':<12}" + "".join(f"{lab:>12}" for lab in lbl)
    print("Position RMSE in metres (lower is better)\n")
    print(hdr)
    print("-" * len(hdr))

    for name, (mode, radar) in SCENARIOS.items():
    
        tracks = simulate_batch(args.tracks, mode, radar, TARGET, seed=42)
        row = [rmse_over_tracks(raw_measurement_error(tracks))]
        row += [rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q))) for q in Q_VAL]
        row += [rmse_over_tracks(evaluate(tracks, imm))]
        row += [rmse_over_tracks(gru_errors(model, tracks))]
        if knet is not None:
            truth = np.stack([t.truth[:, :2] for t in tracks])
            row += [rmse_over_tracks(np.linalg.norm(kalmannet_estimate(knet, tracks)[:, :, :2] - truth, axis=2))]
        best = int(np.argmin(row))
        cells = "".join(f"{v:>11.1f}{'*' if i == best else ' '}" for i, v in enumerate(row))
        print(f"{name:<12}{cells}")
    print("\n* = best in that scenario")


if __name__ == "__main__":
    main()    