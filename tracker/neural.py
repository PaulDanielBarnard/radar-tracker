"""GRU tracker that learns a correction to a base position estimate."""
import torch
from torch import nn

from .dataset import N_FEATURES, OUT_SCALE, tracks_to_tensors


class GRUTracker(nn.Module):

    def __init__(self, n_feat: int = N_FEATURES, hidden: int = 64, layers: int = 1,
                 hybrid: bool = False):
        super().__init__()
        # remembered so a saved model can be rebuilt exactly (see save_model/load_model)
        self.config = {"n_feat": n_feat, "hidden": hidden, "layers": layers, "hybrid": hybrid}
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True)  # memory
        self.head = nn.Linear(hidden, 2)    # memory -> (dx, dy)
        nn.init.zeros_(self.head.weight)    # start with zero correction,
        nn.init.zeros_(self.head.bias)      # i.e. exactly the base position

    def forward(self, feats, base):
        """feats: (batch, T, n_feat), base: (batch, T, 2) in m -> (batch, T, 2) m"""
        h, _ = self.gru(feats)                   # one memory vector per scan
        return base + self.head(h) * OUT_SCALE   # base position + learned correction


def save_model(model, path):
    """Save weights together with the settings needed to rebuild the model."""
    torch.save({"config": model.config, "state": model.state_dict()}, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GRUTracker(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def gru_estimates(model, tracks):
    """Run a trained model over tracks; returns (n_tracks, T, 2) positions in metres."""
    feats, base, _ = tracks_to_tensors(tracks, hybrid=model.config["hybrid"])
    model.eval()
    with torch.no_grad():
        return model(feats, base).numpy()
