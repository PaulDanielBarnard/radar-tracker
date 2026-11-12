"""Train GRU Tracker on Simulated Tracks."""
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from tracker.dataset import OUT_SCALE, tracks_to_tensors
from tracker.metrics import BURN_IN
from tracker.neural import GRUTracker
from tracker.paths import CHECKPOINT, OUT_DIR
from tracker.scenarios import SCENARIOS, TARGET
from tracker.simulate import simulate_batch

def make_data(n_per_scenario: int, seed: int):
    """Simulate n tracks for every scenario so the model sees all situations"""
    tracks = []
    
    for i, (mode, radar) in enumerate(SCENARIOS.values()):
        tracks += simulate_batch(n_per_scenario, mode, radar, TARGET, seed = seed + i)

    return tracks_to_tensors(tracks)

def loss_fn(pred, truth):
    """Mean squared pos error, in units of OUT_SCALE (So numbers stay near 0-1)"""
    return (((pred - truth) / OUT_SCALE) ** 2).mean()


def settled_loss(pred, truth):
    """Same as loss_fn but skips the first BURN_IN scans, matching rmse_over_tracks."""
    return loss_fn(pred[:, BURN_IN:], truth[:, BURN_IN:])

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type = int, default = 30)
    ap.add_argument("--n", type = int, default = 400, help = "Training tracks per scenario")
    ap.add_argument("--batch", type = int, default = 64)
    ap.add_argument("--lr", type = float, default = 1e-3)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok = True)
    torch.manual_seed(0)

    # Train & Validation use different seeds -> different tracks
    # (The Baseline's test seed is 42, so none of these overlap with it)
    train = make_data(args.n, seed = 1000)
    val_feats, val_base, val_truth = make_data(args.n // 4, seed = 2000)
    loader = DataLoader(TensorDataset(*train), batch_size = args.batch, shuffle = True)

    raw_rmse = settled_loss(val_base, val_truth).sqrt().item() * OUT_SCALE
    print(f"Validation RMSE of Raw Radar (The Num to beat): {raw_rmse:.1f} m\n")

    model = GRUTracker()
    opt = torch.optim.Adam(model.parameters(), lr = args.lr)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0

        for feats, base, truth in loader:

            opt.zero_grad() # 1. Clear old gradients
            loss = loss_fn(model(feats, base), truth) # 2. Forward pass + loss
            loss.backward() # 3. Compute Gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 4. Stop Huge Steps
            opt.step() # 5. Update Weights

            total += loss.item() * len(feats)

        train_loss = total / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = settled_loss(model(val_feats, val_base), val_truth).item()

        marker = ""
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            marker = " <- saved"

        print(f"epoch {epoch:3d}  train RMSE {train_loss ** 0.5 * OUT_SCALE:7.1f} m"
              f"    val RMSE {val_loss ** 0.5 * OUT_SCALE:7.1f} m{marker}")


if __name__ == "__main__":
    main()
