"""Evaluate every method over many random seeds, with confidence intervals.

One seed = one fresh batch of simulated tracks. Every method is scored on the same
tracks within a seed, so differences between methods are paired and small real gaps
show up that a single test set would hide.
"""
import argparse

import numpy as np

from tracker.kalman import ConstantVelocityKF
from tracker.metrics import evaluate, raw_measurement_error, rmse_over_tracks
from tracker.neural import gru_estimates, load_model
from tracker.paths import checkpoint
from tracker.scenarios import Q_VAL, SCENARIOS, TARGET
from tracker.simulate import simulate_batch
from tracker.stats import mean_ci, paired_diff_ci

# Test seeds start high so they never overlap with training (1000+, 2000+)
# or with the seeds used in earlier experiments (42, 99, 500).
SEED_BASE = 10_000


def gru_errors(model, tracks):
    truth = np.stack([t.truth[:, :2] for t in tracks])
    return np.linalg.norm(gru_estimates(model, tracks) - truth, axis=2)   # (n_tracks, T)


def method_rmses(tracks, model):
    """One RMSE per method on one batch of tracks (the same tracks for every method)."""
    scores = {"raw radar": rmse_over_tracks(raw_measurement_error(tracks))}
    for q in Q_VAL:
        scores[f"KF q={q:g}"] = rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q)))
    scores["GRU"] = rmse_over_tracks(gru_errors(model, tracks))
    return scores


def run_scenario(mode, radar, seeds, n_tracks, model):
    """Returns {method: array of one RMSE per seed}."""
    per_seed = []
    for s in range(seeds):
        tracks = simulate_batch(n_tracks, mode, radar, TARGET, seed=SEED_BASE + s)
        per_seed.append(method_rmses(tracks, model))
    return {m: np.array([row[m] for row in per_seed]) for m in per_seed[0]}


def best_kalman(results):
    """Name of the Kalman setting with the lowest mean RMSE in this scenario."""
    kalmans = [m for m in results if m.startswith("KF")]
    return min(kalmans, key=lambda m: results[m].mean())


def verdict(diff, half):
    """Is the GRU - Kalman difference clearly away from zero? (negative = GRU has less error)"""
    if abs(diff) <= half:
        return "no clear difference"
    return "GRU better" if diff < 0 else "Kalman better"


def print_table(all_results, seeds, n_tracks):
    methods = list(next(iter(all_results.values())))
    print(f"Position RMSE in metres, mean +/- 95% CI over {seeds} seeds x {n_tracks} tracks "
          "(lower is better, * = lowest mean)\n")
    header = f"{'scenario':<12}" + "".join(f"{m:>16}" for m in methods)
    print(header)
    print("-" * len(header))
    for name, results in all_results.items():
        stats = {m: mean_ci(results[m]) for m in methods}
        best = min(methods, key=lambda m: stats[m][0])
        cells = ""
        for m in methods:
            mean, half = stats[m]
            cells += f"{f'{mean:.1f}+/-{half:.1f}' + ('*' if m == best else ' '):>16}"
        print(f"{name:<12}{cells}")


def print_verdicts(all_results):
    print("\nGRU vs the best Kalman setting (paired by seed; negative = GRU has less error)\n")
    print(f"{'scenario':<12}{'best Kalman':>13}{'GRU - KF (m)':>20}   verdict")
    print("-" * 66)
    for name, results in all_results.items():
        kf = best_kalman(results)
        diff, half = paired_diff_ci(results["GRU"], results[kf])
        print(f"{name:<12}{kf:>13}{f'{diff:+.1f}+/-{half:.1f}':>20}   {verdict(diff, half)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="independent test batches per scenario")
    ap.add_argument("--tracks", type=int, default=200, help="tracks per batch")
    ap.add_argument("--model", default="gru_tracker", help="checkpoint name in outputs/")
    args = ap.parse_args()
    if args.seeds < 2:
        raise SystemExit("--seeds must be at least 2 (a confidence interval needs a spread).")

    ckpt = checkpoint(args.model)
    if not ckpt.exists():
        raise SystemExit(f"No trained model at {ckpt}. Run `python train_neural.py` first.")
    model = load_model(ckpt)

    all_results = {}
    for name, (mode, radar) in SCENARIOS.items():
        print(f"running {name} ...", flush=True)
        all_results[name] = run_scenario(mode, radar, args.seeds, args.tracks, model)

    print()
    print_table(all_results, args.seeds, args.tracks)
    print_verdicts(all_results)


if __name__ == "__main__":
    main()
