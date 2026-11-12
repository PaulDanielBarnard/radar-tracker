"""Compare Raw Radar, Kalman Filters and GRU on every scenario"""
import argparse

import numpy as np
import torch

from tracker.dataset import tracks_to_tensors
from tracker.kalman import ConstantVelocityKF
from tracker.metrics import evaluate, raw_measurement_error, rmse_over_tracks
from tracker.neural import GRUTracker
from tracker.paths import CHECKPOINT
from tracker.scenarios import Q_VAL, SCENARIOS, TARGET
from tracker.simulate import simulate_batch

def gru_errors( model, tracks):

    feats, base, truth = tracks_to_tensors(tracks)

    with torch.no_grad():
        pred = model(feats, base)

    return torch.linalg.norm(pred - truth, dim = 2).numpy() #(n_tracks, T)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type = int, default = 500)
    args = ap.parse_args()

    model = GRUTracker()
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()

    lbl = ["raw radar"] + [f"KF q = {q:g}" for q in Q_VAL] + ["GRU"]
    hdr = f"{'scenario':<12}" + "".join(f"{lab:>12}" for lab in lbl)
    print("Position RMSE in metres (lower is better)\n")
    print(hdr)
    print("-" * len(hdr))

    for name, (mode, radar) in SCENARIOS.items():
        tracks = simulate_batch(args.tracks, mode, radar, TARGET, seed=42)
        row = [rmse_over_tracks(raw_measurement_error(tracks))]
        row += [rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q))) for q in Q_VAL]
        row += [rmse_over_tracks(gru_errors(model, tracks))]
        best = int(np.argmin(row))
        cells = "".join(f"{v:>11.1f}{'*' if i == best else ' '}" for i, v in enumerate(row))
        print(f"{name:<12}{cells}")
    print("\n* = best in that scenario")


if __name__ == "__main__":
    main()    