"""Evaluate the trackers on REAL flights (ADS-B) seen through a simulated radar.

The truth is a recorded flight; only the radar is simulated (range and bearing noise, missed
plots, flight placed 20-60 km from the radar). You supply the flights as a CSV, by default
data/opensky_sample.csv (see the README, section on ADS-B, for how to get one).

How to read the numbers honestly: the windows of real flight are the same in every repeat.
What changes between repeats is where the flight is placed and the radar noise, so the
+/- intervals only reflect radar randomness, not how different other flights might be.
Windows do not overlap, so they are separate pieces of the flight, but they come from only
a few aircraft.

  python run_adsb.py                       # data/opensky_sample.csv, normal radar
  python run_adsb.py --radar-noise noisy   # the noisy radar of the five scenarios
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_eval import gru_errors, kalmannet_errors
from tracker.adsb import load_opensky_csv, load_tracks, split_flight
from tracker.imm import default_imm
from tracker.kalman import ConstantVelocityKF
from tracker.kalmannet import load_kalmannet
from tracker.metrics import evaluate, raw_measurement_error, rmse_over_tracks
from tracker.neural import load_model
from tracker.paths import ROOT, checkpoint
from tracker.scenarios import BASE_RADAR, Q_VAL
from tracker.stats import mean_ci, paired_diff_ci

DEFAULT_CSV = ROOT / "data" / "opensky_sample.csv"
RADARS = {
    "normal": BASE_RADAR,
    "noisy": replace(BASE_RADAR, sigma_r=150.0, sigma_theta=0.015),
    "missed": replace(BASE_RADAR, p_detect=0.7),
}

MISSING_HELP = """No ADS-B file found at {path}.

Put a CSV of recorded flights there (or pass --csv PATH). It needs these columns:
  time      epoch seconds
  icao24    the aircraft's address
  lat, lon  degrees
and optionally velocity (m/s) and heading (degrees).
OpenSky Network (https://opensky-network.org) provides such data; its public sample files are listed at
https://opensky-network.org/datasets/ and a free registration gives access to historical state vectors.
Flights that are on the ground, or whose reports are more than 5 seconds apart, are cut out automatically."""


def method_scores(tracks, model=None, knet=None) -> dict:
    """One RMSE per method on one set of tracks (the same tracks for every method)."""
    scores = {"raw radar": rmse_over_tracks(raw_measurement_error(tracks))}
    for q in Q_VAL:
        scores[f"KF q={q:g}"] = rmse_over_tracks(evaluate(tracks, ConstantVelocityKF(q=q)))
    scores["IMM"] = rmse_over_tracks(evaluate(tracks, default_imm()))
    if knet is not None:
        scores["KalmanNet"] = rmse_over_tracks(kalmannet_errors(knet, tracks))
    if model is not None:
        scores["GRU"] = rmse_over_tracks(gru_errors(model, tracks))
    return scores


def evaluate_adsb(csv_path, radar=BASE_RADAR, repeats: int = 10, stride: int = 120, max_tracks=None,
                  model=None, knet=None, seed_base: int = 0):
    """Score every method on the flights in `csv_path`, over several radar placements.

    Returns ({method: array of one RMSE per repeat}, number of tracks per repeat).
    """
    per_repeat, n_tracks = [], 0
    for r in range(repeats):
        tracks = load_tracks(csv_path, radar, seed=seed_base + r, stride=stride, max_tracks=max_tracks)
        if not tracks:
            raise ValueError("No usable flight segments found: need at least 120 consecutive seconds "
                             "in the air with reports less than 5 seconds apart.")
        n_tracks = len(tracks)
        per_repeat.append(method_scores(tracks, model, knet))
    return {m: np.array([row[m] for row in per_repeat]) for m in per_repeat[0]}, n_tracks


def paired_rows(results, best_kf):
    """(a, b, a - b, half-width, verdict) for the pairs worth comparing; negative = a has less error."""
    rows = []
    for a, b in (("IMM", best_kf), ("GRU", best_kf), ("GRU", "IMM"), ("KalmanNet", best_kf), ("KalmanNet", "IMM")):
        if a in results and b in results:
            d, h = paired_diff_ci(results[a], results[b])
            verdict = "no clear difference" if abs(d) <= h else (f"{a} better" if d < 0 else f"{b} better")
            rows.append((a, b, d, h, verdict))
    return rows


def describe_flights(csv_path):
    flights = load_opensky_csv(csv_path)
    segments = [s for f in flights for s in split_flight(f)]
    seconds = sum(len(s.t) for s in segments)
    return len(flights), len(segments), seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV of ADS-B reports (default data/opensky_sample.csv)")
    ap.add_argument("--radar-noise", choices=sorted(RADARS), default="normal",
                    help="radar to simulate: normal, noisy (150 m, 0.015 rad) or missed (30%% missed plots)")
    ap.add_argument("--repeats", type=int, default=10, help="independent radar placements / noise draws")
    ap.add_argument("--max-tracks", type=int, default=None, help="use at most this many 120-scan windows")
    ap.add_argument("--model", default="gru_tracker", help="GRU checkpoint name in outputs/")
    ap.add_argument("--kalmannet", default="kalmannet", help="learned-gain checkpoint name in outputs/")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(MISSING_HELP.format(path=path))
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 (a confidence interval needs a spread).")

    model = load_model(checkpoint(args.model)) if checkpoint(args.model).exists() else None
    knet = load_kalmannet(checkpoint(args.kalmannet)) if checkpoint(args.kalmannet).exists() else None
    for name, m in (("GRU", model), ("KalmanNet", knet)):
        if m is None:
            print(f"(no trained {name} found in outputs/; it is left out. Train it with the matching train script.)")

    n_aircraft, n_segments, seconds = describe_flights(path)
    results, n_tracks = evaluate_adsb(path, RADARS[args.radar_noise], args.repeats,
                                      max_tracks=args.max_tracks, model=model, knet=knet)
    print(f"\n{path.name}: {n_aircraft} aircraft, {n_segments} gap-free segments, {seconds} s of flying; "
          f"{n_tracks} non-overlapping 120-scan windows per repeat; radar '{args.radar_noise}'.")
    print(f"Position RMSE in metres, mean +/- 95% CI over {args.repeats} radar placements (lower is better).")
    print("The flight itself is the same every repeat, so the interval shows radar luck only.\n")
    methods = list(results)
    stats = {m: mean_ci(results[m]) for m in methods}
    best = min(methods, key=lambda m: stats[m][0])
    for m in methods:
        mean, half = stats[m]
        print(f"  {m:<10}{mean:8.1f} +/- {half:5.1f}{'  *' if m == best else ''}")

    kfs = [m for m in methods if m.startswith("KF")]
    best_kf = min(kfs, key=lambda m: stats[m][0])
    print(f"\nPaired differences (negative = first has less error); best single Kalman = {best_kf}")
    for a, b, d, h, verdict in paired_rows(results, best_kf):
        print(f"  {a} - {b:<9}{d:+8.1f} +/- {h:5.1f}   {verdict}")


if __name__ == "__main__":
    main()
