"""Evaluate every method over many random seeds, with confidence intervals.

One seed = one fresh batch of simulated tracks. Every method is scored on the same
tracks within a seed, so differences between methods are paired and small real gaps
show up that a single test set would hide.
"""
import argparse

import numpy as np

from tracker.imm import default_imm
from tracker.kalmannet import estimate as kalmannet_estimate
from tracker.kalmannet import load_kalmannet
from tracker.kalman import ConstantVelocityKF
from tracker.metrics import BURN_IN, evaluate, raw_measurement_error, rmse_over_tracks
from tracker.neural import gru_estimates, load_model
from tracker.paths import checkpoint
from tracker.scenarios import Q_VAL, SCENARIOS, STRESS, TARGET
from tracker.simulate import simulate_batch
from tracker.stats import chi2_bounds, mean_ci, nees, paired_diff_ci

# Test seeds start high so they never overlap with training (1000+, 2000+)
# or with the seeds used in earlier experiments (42, 99, 500).
SEED_BASE = 10_000

# An honest filter has mean NEES equal to the state size (4) and mean NIS equal to the
# measurement size (2). "Consistent" here means within 15% of those ideals.
IDEAL_NEES, IDEAL_NIS, TOLERANCE = 4.0, 2.0, 0.15


def gru_errors(model, tracks):

    truth = np.stack([t.truth[:, :2] for t in tracks])
    return np.linalg.norm(gru_estimates(model, tracks) - truth, axis=2)   # (n_tracks, T)


def kalmannet_errors(knet, tracks):
    truth = np.stack([t.truth[:, :2] for t in tracks])
    return np.linalg.norm(kalmannet_estimate(knet, tracks)[:, :, :2] - truth, axis=2)   # (n_tracks, T)


def method_rmses(tracks, model, knet=None):
    """One RMSE per method on one batch of tracks (the same tracks for every method)."""
    scores = {"raw radar": rmse_over_tracks(raw_measurement_error(tracks))}
    for q in Q_VAL:
        scores[f"KF q={q:g}"] = rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q)))
    scores["IMM"] = rmse_over_tracks(evaluate(tracks, default_imm()))
    scores["GRU"] = rmse_over_tracks(gru_errors(model, tracks))
    if knet is not None:
        scores["KalmanNet"] = rmse_over_tracks(kalmannet_errors(knet, tracks))
    return scores


def run_scenario(mode, radar, seeds, n_tracks, model, knet=None):
    """Returns {method: array of one RMSE per seed}."""
    per_seed = []
    for s in range(seeds):
        tracks = simulate_batch(n_tracks, mode, radar, TARGET, seed=SEED_BASE + s)
        per_seed.append(method_rmses(tracks, model, knet))
    return {m: np.array([row[m] for row in per_seed]) for m in per_seed[0]}


def best_kalman(results):
    """Name of the Kalman setting with the lowest mean RMSE in this scenario."""
    kalmans = [m for m in results if m.startswith("KF")]
    return min(kalmans, key=lambda m: results[m].mean())


def verdict(diff, half, a="GRU", b="Kalman"):
    """Is the a - b difference clearly away from zero? (negative = a has less error)"""
    if abs(diff) <= half:
        return "no clear difference"
    return f"{a} better" if diff < 0 else f"{b} better"


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


def print_imm_verdicts(all_results):
    """The IMM filter against the best single Kalman filter, and the GRU against the IMM."""
    print("\nIMM vs the best Kalman setting, and GRU vs the IMM (paired by seed; negative = first has less error)\n")
    print(f"{'scenario':<12}{'IMM - best KF (m)':>20}   {'verdict':<20}{'GRU - IMM (m)':>16}   verdict")
    print("-" * 84)
    for name, results in all_results.items():
        kf = best_kalman(results)
        d1, h1 = paired_diff_ci(results["IMM"], results[kf])
        d2, h2 = paired_diff_ci(results["GRU"], results["IMM"])
        print(f"{name:<12}{f'{d1:+.1f}+/-{h1:.1f} ({kf[3:]})':>20}   {verdict(d1, h1, 'IMM', 'Kalman'):<20}"
              f"{f'{d2:+.1f}+/-{h2:.1f}':>16}   {verdict(d2, h2, 'GRU', 'IMM')}")


def print_kalmannet_verdicts(all_results):
    """The learned-gain filter against the IMM, the GRU and the best single Kalman filter."""
    print("\nKalmanNet (learned gain) vs the IMM, the GRU and the best Kalman setting "
          "(paired by seed; negative = KalmanNet has less error)\n")
    print(f"{'scenario':<12}{'vs IMM (m)':>20}   {'verdict':<22}{'vs GRU (m)':>18}   {'verdict':<22}{'vs best KF (m)':>20}   verdict")
    print("-" * 140)
    for name, results in all_results.items():
        kf = best_kalman(results)
        cells = []
        for other, label in (("IMM", "IMM"), ("GRU", "GRU"), (kf, "Kalman")):
            d, h = paired_diff_ci(results["KalmanNet"], results[other])
            cells.append((f"{d:+.1f}+/-{h:.1f}", verdict(d, h, "KalmanNet", label)))
        print(f"{name:<12}{cells[0][0]:>20}   {cells[0][1]:<22}{cells[1][0]:>18}   {cells[1][1]:<22}"
              f"{cells[2][0]:>20}   {cells[2][1]}")


def run_stress(seeds, n_tracks, model, knet=None):
    """The stress scenario (a radar noisier than any training radar): {method: RMSE per seed}."""
    mode, radar = STRESS
    return run_scenario(mode, radar, seeds, n_tracks, model, knet)


def consistency_stats(tracks, filt):
    """Mean NEES and mean NIS of one filter (anything with run(track, return_diag=True)).

    Averaged over tracks first, then over time, skipping the first BURN_IN scans.
    """
    nees_runs, nis_runs = [], []
    for t in tracks:
        est, diag = filt.run(t, return_diag=True)
        nees_runs.append(nees(t.truth, est, diag["P"]))
        nis_runs.append(diag["nis"])                      # NaN on missed scans
    mean_nees = float(np.mean(np.array(nees_runs)[:, BURN_IN:]))
    mean_nis = float(np.nanmean(np.array(nis_runs)[:, BURN_IN:]))
    return mean_nees, mean_nis


def consistency_label(value, ideal):
    """Above the ideal = the filter claims more accuracy than it has; below = too cautious."""
    if abs(value - ideal) <= TOLERANCE * ideal:
        return "consistent"
    return "overconfident" if value > ideal else "too cautious"


def print_consistency(n_tracks):
    lo, hi = chi2_bounds(4, n_tracks)
    print(f"\nFilter consistency on one batch of {n_tracks} tracks per scenario (scans after burn-in)")
    print(f"ideal mean NEES = {IDEAL_NEES:g} (95% band for a single scan: {lo:.2f} to {hi:.2f}), "
          f"ideal mean NIS = {IDEAL_NIS:g}; 'consistent' = within {TOLERANCE:.0%} of the ideal\n")
    header = f"{'scenario':<12}{'filter':>9}{'NEES':>12}  {'NEES verdict':<14}{'NIS':>8}  NIS verdict"
    print(header)
    print("-" * 68)
    filters = [(f"q={q:g}", ConstantVelocityKF(q=q)) for q in Q_VAL] + [("IMM", default_imm())]
    for name, (mode, radar) in SCENARIOS.items():
        tracks = simulate_batch(n_tracks, mode, radar, TARGET, seed=SEED_BASE)
        for label, filt in filters:
            mean_nees, mean_nis = consistency_stats(tracks, filt)
            print(f"{name:<12}{label:>9}{mean_nees:>12.1f}  {consistency_label(mean_nees, IDEAL_NEES):<14}"
                  f"{mean_nis:>8.2f}  {consistency_label(mean_nis, IDEAL_NIS)}")
    print("IMM: its NIS is computed from the combined (mixture) prediction.")
    print("GRU, KalmanNet: n/a (they predict no covariance, so NEES and NIS cannot be computed)")


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="independent test batches per scenario")
    ap.add_argument("--tracks", type=int, default=200, help="tracks per batch")
    ap.add_argument("--model", default="gru_tracker", help="checkpoint name in outputs/")
    ap.add_argument("--kalmannet", default="kalmannet", help="learned-gain checkpoint name in outputs/ (skipped if missing)")
    ap.add_argument("--stress", action="store_true",
                    help="only run the very noisy radar (300 m, 0.03 rad) that is beyond the training range")
    args = ap.parse_args()
    
    if args.seeds < 2:
        raise SystemExit("--seeds must be at least 2 (a confidence interval needs a spread).")

    ckpt = checkpoint(args.model)
    
    if not ckpt.exists():
        raise SystemExit(f"No trained model at {ckpt}. Run `python train_neural.py` first.")
    
    model = load_model(ckpt)
    knet_path = checkpoint(args.kalmannet)
    knet = load_kalmannet(knet_path) if knet_path.exists() else None
    if knet is None:
        print(f"(no learned-gain model at {knet_path}; run `python train_kalmannet.py` to include KalmanNet)\n")

    if args.stress:
        print(f"Stress test: radar noise 300 m and 0.03 rad, beyond the training range ({args.seeds} seeds)")
        all_results = {"very_noisy": run_stress(args.seeds, args.tracks, model, knet)}
        print()
        print_table(all_results, args.seeds, args.tracks)
        if knet is not None:
            print_kalmannet_verdicts(all_results)
        return

    all_results = {}
    
    for name, (mode, radar) in SCENARIOS.items():
    
        print(f"running {name} ...", flush=True)
        all_results[name] = run_scenario(mode, radar, args.seeds, args.tracks, model, knet)

    print()
    print_table(all_results, args.seeds, args.tracks)
    print_verdicts(all_results)
    print_imm_verdicts(all_results)
    if knet is not None:
        print_kalmannet_verdicts(all_results)
    print_consistency(args.tracks)


if __name__ == "__main__":
    main()
