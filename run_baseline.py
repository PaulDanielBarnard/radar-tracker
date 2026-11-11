"""Run the Kalman baseline across scenarios and show where it struggles.

    python run_baseline.py            # results table + plots in outputs/
    python run_baseline.py --tracks 1000
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.kalman import ConstantVelocityKF
from tracker.metrics import position_error, rmse_over_tracks
from tracker.simulate import RadarConfig, TargetConfig, simulate_batch

OUT = Path("outputs")

BASE_RADAR = RadarConfig()
TARGET = TargetConfig()

SCENARIOS = {
    # name: (motion mode, radar config)
    "straight":   ("straight", BASE_RADAR),
    "turning":    ("turn",     BASE_RADAR),
    "maneuvers":  ("random",   BASE_RADAR),
    "noisy":      ("random",   replace(BASE_RADAR, sigma_r=150.0, sigma_theta=0.015)),
    "missed 30%": ("random",   replace(BASE_RADAR, p_detect=0.7)),
}
Q_VALUES = [0.1, 1.0, 10.0, 100.0]


def evaluate(tracks, kf):
    errors = np.stack([position_error(kf.run(t), t.truth) for t in tracks])
    return errors


def raw_measurement_error(tracks):
    """Baseline-of-baselines: just use the latest radar plot, no filtering."""
    out = []
    for t in tracks:
        last = t.meas_xy[0].copy()
        est = np.zeros((len(t.detected), 2))
        for k in range(len(t.detected)):
            if t.detected[k]:
                last = t.meas_xy[k]
            est[k] = last
        out.append(np.linalg.norm(est - t.truth[:, :2], axis=1))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    header = f"{'scenario':<12}{'raw radar':>11}" + "".join(f"{f'KF q={q:g}':>12}" for q in Q_VALUES)
    print("Position RMSE in metres (lower is better, first 10 scans ignored)\n")
    print(header)
    print("-" * len(header))

    results = {}
    for name, (mode, radar) in SCENARIOS.items():
        tracks = simulate_batch(args.tracks, mode, radar, TARGET, seed=42)
        row = {"raw": rmse_over_tracks(raw_measurement_error(tracks))}
        for q in Q_VALUES:
            row[q] = rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q)))
        results[name] = row
        print(f"{name:<12}{row['raw']:>11.1f}" + "".join(f"{row[q]:>12.1f}" for q in Q_VALUES))

    print("\nBest filter setting per scenario:")
    for name, row in results.items():
        best_q = min(Q_VALUES, key=lambda q: row[q])
        print(f"  {name:<12} q={best_q:g}  ({row[best_q]:.0f} m)")
    print("A single fixed q is a compromise: small q suits straight flight, large q suits "
          "maneuvers.\nThat tuning tradeoff is what a learned tracker has to beat.")

    # --- plot 1: one example maneuvering track ---
    rng_tracks = simulate_batch(1, "random", BASE_RADAR, TARGET, seed=7)
    t = rng_tracks[0]
    est = ConstantVelocityKF(q=10.0).run(t)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(t.truth[:, 0] / 1000, t.truth[:, 1] / 1000, "k-", label="truth")
    ax.plot(t.meas_xy[:, 0] / 1000, t.meas_xy[:, 1] / 1000, ".", color="tab:red", alpha=0.5, label="radar plots")
    ax.plot(est[:, 0] / 1000, est[:, 1] / 1000, "-", color="tab:blue", label="Kalman (q=10)")
    ax.plot(0, 0, "^", color="green", label="radar")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_aspect("equal")
    ax.set_title("Example maneuvering track"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "example_track.png", dpi=130); plt.close(fig)

    # --- plot 2: grouped bars of the table ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(results)
    labels = ["raw radar"] + [f"KF q={q:g}" for q in Q_VALUES]
    width = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        key = "raw" if i == 0 else Q_VALUES[i - 1]
        ax.bar(np.arange(len(names)) + i * width, [results[n][key] for n in names], width, label=lab)
    ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2); ax.set_xticklabels(names)
    ax.set_ylabel("position RMSE (m)"); ax.set_title("Kalman baseline by scenario")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "baseline_rmse.png", dpi=130); plt.close(fig)
    print(f"\nSaved plots to {OUT.resolve()}")


if __name__ == "__main__":
    main()
