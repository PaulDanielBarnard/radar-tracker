"""Evaluate the multi-target tracker over a grid of aircraft counts and clutter rates.

Each cell runs the tracker on several freshly simulated scenes (aircraft flying random
manoeuvres among false alarms) and scores every scan with the OSPA distance between the
true aircraft positions and the confirmed tracks. The first scans are skipped because
tracks need a few scans to be confirmed. Scenes use high seeds (never seen in training).

  python run_multitarget.py                  # the full grid
  python run_multitarget.py --scenes 3       # a quick look
"""
import argparse

import numpy as np

from tracker.metrics import BURN_IN
from tracker.multitarget import MultiTargetTracker, run_scene
from tracker.ospa import ospa
from tracker.scene import simulate_scene_batch
from tracker.simulate import RadarConfig, TargetConfig
from tracker.stats import mean_ci

SEED_BASE = 10_000
COUNTS = (1, 3, 6)
CLUTTER = (0.0, 5.0, 20.0)


def score_scene(scene, confirmed_per_scan, c: float = 1000.0, p: float = 2.0, burn_in: int = BURN_IN):
    """Mean OSPA, localisation, cardinality, true count and confirmed count over one scene."""
    rows = []
    for k in range(burn_in, scene.n_scans):
        truth_xy = scene.truth_at(k)[1]
        track_xy = np.array([t.x[:2] for t in confirmed_per_scan[k]]).reshape(-1, 2)
        rows.append((*ospa(truth_xy, track_xy, c, p), len(truth_xy), len(track_xy)))
    return np.mean(rows, axis=0)


def evaluate_cell(n_targets: int, clutter: float, n_scenes: int = 10, seed: int = SEED_BASE,
                  radar: RadarConfig = RadarConfig(p_detect=0.9), target: TargetConfig = TargetConfig(),
                  q: float = 100.0):
    """Per-scene scores for one grid cell, shape (n_scenes, 5): total, loc, card, true, found."""
    scenes = simulate_scene_batch(n_scenes, n_targets, radar, target, clutter_rate=clutter, seed=seed)
    return np.array([score_scene(s, run_scene(MultiTargetTracker(q=q, dt=radar.dt), s)) for s in scenes])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=10, help="scenes per grid cell (default 10)")
    ap.add_argument("--q", type=float, default=100.0, help="tracker process noise")
    args = ap.parse_args()

    print(f"OSPA (m, lower is better; 0 = perfect, 1000 = nothing found), {args.scenes} scenes per cell, "
          f"mean +/- 95% CI. p_detect 0.9.\n")
    print(f"{'aircraft':>8} {'clutter/scan':>12} | {'OSPA':>14} {'localisation':>14} {'cardinality':>14} | "
          f"{'true':>5} {'found':>6}")
    for n in COUNTS:
        for cl in CLUTTER:
            res = evaluate_cell(n, cl, args.scenes, SEED_BASE + 100 * n + int(cl), q=args.q)
            cells = [f"{m:7.1f} +/-{h:5.1f}" for m, h in (mean_ci(res[:, i]) for i in range(3))]
            print(f"{n:>8} {cl:>12.0f} | {cells[0]:>14} {cells[1]:>14} {cells[2]:>14} | "
                  f"{res[:, 3].mean():5.1f} {res[:, 4].mean():6.2f}", flush=True)


if __name__ == "__main__":
    main()
