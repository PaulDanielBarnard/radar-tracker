"""Train the learned-gain Kalman filter (KalmanNet style) on simulated tracks.

Mirrors train_neural.py: randomised radar settings for training, separate validation tracks
(seeds 2000+) from the five fixed scenarios to pick the best epoch, a noise-balanced position
loss that skips the burn-in scans, AdamW, a learning rate that halves when validation stalls,
and early stopping. The checkpoint goes to outputs/<name>.pt (default name "kalmannet").
"""
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tracker.dataset import noise_scales
from tracker.kalmannet import KalmanNet, load_kalmannet, save_kalmannet, tracks_to_inputs
from tracker.paths import OUT_DIR, checkpoint
from tracker.scenarios import SCENARIOS, TARGET, random_radar
from tracker.simulate import simulate_batch, simulate_track
from train_neural import loss_fn, rmse_m, settled_loss


def tensors(tracks):
    """(z, det, static, truth, noise scale) for a list of tracks."""
    return (*tracks_to_inputs(tracks), noise_scales(tracks))


def make_random_data(n_tracks: int, seed: int):
    """Tracks with a random motion type and a random radar each."""
    rng = np.random.default_rng(seed)
    tracks = []
    for _ in range(n_tracks):
        mode = ("straight", "turn", "random", "random")[rng.integers(4)]
        tracks.append(simulate_track(rng, mode, random_radar(rng), TARGET))
    return tensors(tracks)


def make_scenario_data(n_per_scenario: int, seed: int):
    """{scenario name: tensors} for the five fixed scenarios."""
    return {name: tensors(simulate_batch(n_per_scenario, mode, radar, TARGET, seed=seed + i))
            for i, (name, (mode, radar)) in enumerate(SCENARIOS.items())}


def predict_positions(model, z, det, static):
    return model(z, det, static)[:, :, :2]


def report_by_scenario(model, val_sets):
    """Raw plot, the untrained starting filter and the trained filter, per scenario (RMSE in m)."""
    start = KalmanNet(**model.config)
    model.eval()
    print(f"\n{'scenario':<12}{'raw plot':>11}{'start':>10}{'trained':>10}")
    with torch.no_grad():
        for name, (z, det, static, truth, _) in val_sets.items():
            print(f"{name:<12}{rmse_m(z, truth):>11.1f}"
                  f"{rmse_m(predict_positions(start, z, det, static), truth):>10.1f}"
                  f"{rmse_m(predict_positions(model, z, det, static), truth):>10.1f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the learned-gain Kalman filter.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n", type=int, default=400, help="Training tracks per scenario (x5 random tracks)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64, help="GRU memory size")
    ap.add_argument("--layers", type=int, default=1, help="stacked GRU layers")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    ap.add_argument("--patience", type=int, default=8, help="stop after this many epochs without improvement (0 = never)")
    ap.add_argument("--name", default="kalmannet", help="checkpoint name in outputs/")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(exist_ok=True)
    ckpt = checkpoint(args.name)
    torch.manual_seed(0)

    train = make_random_data(args.n * len(SCENARIOS), seed=1000)
    val_sets = make_scenario_data(max(args.n // 4, 1), seed=2000)
    val = tuple(torch.cat(parts) for parts in zip(*val_sets.values()))
    loader = DataLoader(TensorDataset(*train), batch_size=args.batch, shuffle=True)

    model = KalmanNet(hidden=args.hidden, layers=args.layers)
    with torch.no_grad():
        v_z, v_det, v_static, v_truth, v_scale = val
        print(f"Validation RMSE of the raw plot: {rmse_m(v_z, v_truth):.1f} m, "
              f"of the untrained fixed-gain filter: {rmse_m(predict_positions(model, v_z, v_det, v_static), v_truth):.1f} m\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    best, stale = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for z, det, static, truth, scale in loader:
            opt.zero_grad()
            loss = loss_fn(predict_positions(model, z, det, static), truth, scale)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(z)
        train_loss = total / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = predict_positions(model, v_z, v_det, v_static)
            val_loss = settled_loss(val_pred, v_truth, v_scale).item()
            val_rmse = rmse_m(val_pred, v_truth)
        sched.step(val_loss)
        marker = ""
        if val_loss < best:
            best, stale = val_loss, 0
            save_kalmannet(model, ckpt)
            marker = " <- saved"
        else:
            stale += 1
        print(f"epoch {epoch:3d}  train loss {train_loss:6.3f}   val loss {val_loss:6.3f}"
              f"   val RMSE {val_rmse:7.1f} m   lr {opt.param_groups[0]['lr']:.0e}{marker}", flush=True)
        if args.patience and stale >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping early.")
            break

    report_by_scenario(load_kalmannet(ckpt), val_sets)


if __name__ == "__main__":
    main()
