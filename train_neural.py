"""Train GRU Tracker on Simulated Tracks."""
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from tracker.dataset import OUT_SCALE, n_features, noise_scales, tracks_to_tensors
from tracker.metrics import BURN_IN
from tracker.neural import GRUTracker, load_model, save_model
from tracker.paths import OUT_DIR, checkpoint
from tracker.scenarios import SCENARIOS, TARGET, random_radar
from tracker.simulate import simulate_batch, simulate_track

def make_data(n_per_scenario: int, seed: int, hybrid: bool = False):
    """Simulate n tracks for every scenario. Returns {scenario name: tensors}."""
    sets = {}
    for i, (name, (mode, radar)) in enumerate(SCENARIOS.items()):
        tracks = simulate_batch(n_per_scenario, mode, radar, TARGET, seed = seed + i)
        sets[name] = (*tracks_to_tensors(tracks, hybrid), noise_scales(tracks))
    return sets


def make_random_data(n_tracks: int, seed: int, hybrid: bool = False):
    """Tracks with a random motion type and a random radar each, instead of 5 fixed scenarios."""
    rng = np.random.default_rng(seed)
    tracks = []
    for _ in range(n_tracks):
        mode = ("straight", "turn", "random", "random")[rng.integers(4)]   # favour maneuvers
        tracks.append(simulate_track(rng, mode, random_radar(rng), TARGET))
    return {"random": (*tracks_to_tensors(tracks, hybrid), noise_scales(tracks))}


def pool(sets):
    """Stack every scenario's (feats, base, truth, scale) into one training set."""
    return tuple(torch.cat(parts) for parts in zip(*sets.values()))


def loss_fn(pred, truth, scale = None):
    """Mean squared position error, in units of OUT_SCALE (So numbers stay near 0-1).

    With `scale` (each track's expected radar noise) the error is measured relative to how
    noisy that track's radar is, so easy scenarios count as much as hard ones."""
    sq_dist = (((pred - truth) / OUT_SCALE) ** 2).sum(dim = 2)   # squared distance error per scan
    per_track = sq_dist.mean(dim = 1)
    if scale is not None:
        per_track = per_track / scale ** 2
    return per_track.mean()


def settled_loss(pred, truth, scale = None):
    """Same as loss_fn but skips the first BURN_IN scans, matching rmse_over_tracks."""
    return loss_fn(pred[:, BURN_IN:], truth[:, BURN_IN:], scale)


def rmse_m(pred, truth):
    """Burn-in-adjusted RMSE in metres."""
    return settled_loss(pred, truth).sqrt().item() * OUT_SCALE


def report_by_scenario(model, val_sets):
    """Show where the model beats raw radar and where it does not."""
    model.eval()
    print(f"\n{'scenario':<12}{'start':>11}{'GRU':>10}")
    with torch.no_grad():
        for name, (feats, base, truth, _) in val_sets.items():
            print(f"{name:<12}{rmse_m(base, truth):>11.1f}{rmse_m(model(feats, base), truth):>10.1f}")


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type = int, default = 30)
    ap.add_argument("--n", type = int, default = 400, help = "Training tracks per scenario")
    ap.add_argument("--batch", type = int, default = 64)
    ap.add_argument("--lr", type = float, default = 1e-3)
    ap.add_argument("--hidden", type = int, default = 64, help = "GRU memory size")
    ap.add_argument("--layers", type = int, default = 1, help = "stacked GRU layers")
    ap.add_argument("--wd", type = float, default = 1e-4, help = "weight decay (a small penalty on big weights)")
    ap.add_argument("--patience", type = int, default = 8, help = "stop after this many epochs without improvement (0 = never)")
    ap.add_argument("--hybrid", action = "store_true", help = "start from a Kalman filter and learn a correction")
    ap.add_argument("--randomize", action = "store_true", help = "train on random radar settings")
    ap.add_argument("--name", default = "gru_tracker", help = "checkpoint name in outputs/")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok = True)
    ckpt = checkpoint(args.name)
    torch.manual_seed(0)

    # Train & Validation use different seeds -> different tracks
    # (The Baseline's test seed is 42, so none of these overlap with it)
    if args.randomize:   # same number of tracks as the fixed scenarios, but varied conditions
        train = pool(make_random_data(args.n * len(SCENARIOS), seed = 1000, hybrid = args.hybrid))
    else:
        train = pool(make_data(args.n, seed = 1000, hybrid = args.hybrid))
    val_sets = make_data(args.n // 4, seed = 2000, hybrid = args.hybrid)
    val_feats, val_base, val_truth, val_scale = pool(val_sets)
    loader = DataLoader(TensorDataset(*train), batch_size = args.batch, shuffle = True)

    raw_rmse = settled_loss(val_base, val_truth).sqrt().item() * OUT_SCALE
    print(f"Validation RMSE of the starting point (raw plot, or Kalman with --hybrid): {raw_rmse:.1f} m\n")

    model = GRUTracker(n_feat = n_features(args.hybrid), hidden = args.hidden,
                       layers = args.layers, hybrid = args.hybrid)
    opt = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.wd)
    # halve the learning rate whenever validation stops improving for a few epochs
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "min", factor = 0.5, patience = 3)
    best, stale = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0

        for feats, base, truth, scale in loader:

            opt.zero_grad() # 1. Clear old gradients
            loss = loss_fn(model(feats, base), truth, scale) # 2. Forward pass + loss
            loss.backward() # 3. Compute Gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 4. Stop Huge Steps
            opt.step() # 5. Update Weights

            total += loss.item() * len(feats)

        train_loss = total / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_feats, val_base)
            val_loss = settled_loss(val_pred, val_truth, val_scale).item()   # picks the best epoch
            val_rmse = rmse_m(val_pred, val_truth)                           # metres, for humans

        sched.step(val_loss)
        marker = ""
        if val_loss < best:
            best, stale = val_loss, 0
            save_model(model, ckpt)
            marker = " <- saved"
        else:
            stale += 1

        print(f"epoch {epoch:3d}  train loss {train_loss:6.3f}   val loss {val_loss:6.3f}"
              f"   val RMSE {val_rmse:7.1f} m"
              f"   lr {opt.param_groups[0]['lr']:.0e}{marker}")
        if args.patience and stale >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping early.")
            break

    model = load_model(ckpt)   # reload the best epoch
    report_by_scenario(model, val_sets)


if __name__ == "__main__":
    main()
