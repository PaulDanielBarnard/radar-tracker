"""Convert sim Tracks to Tensors"""
import numpy as np
import torch

from .simulate import forward_fill

POS_SCALE = 50_000.0   # metres: puts positions roughly in [-1, 1]
VEL_SCALE = 300.0      # metres per scan: typical movement between scans
N_FEATURES = 5          # columns per scan fed to the network
OUT_SCALE = 1000.0     # metres: the network's output unit (1.0 = 1 km correction)

def track_to_arr(track, hybrid=False):
    filled = forward_fill(track)
    step = np.zeros_like(filled) # How Far plot moved since last scan
    step[1:] = filled[1:] - filled[:-1]
    feat = np.concatenate([
        filled / POS_SCALE,  # 2 col : where the target looks to be
        step / VEL_SCALE,    # 2 col : apparent movement
        track.detected[:, None].astype(float),  # 1 col : did we really see it?
        ], axis = 1)    # shape (T, 5)
    return feat,filled,track.truth[:, :2]

TYPICAL_RANGE = 35_000.0   # metres: a typical target distance, used to size bearing noise


def noise_scale(track):
    """Expected raw radar error of this track, in OUT_SCALE units."""
    r = track.radar
    return float(np.hypot(r.sigma_r, TYPICAL_RANGE * r.sigma_theta)) / OUT_SCALE


def noise_scales(tracks):
    return torch.tensor([noise_scale(t) for t in tracks], dtype=torch.float32)


def tracks_to_tensors(tracks, hybrid=False):
    feats, bases, truths = zip(*(track_to_arr(t, hybrid) for t in tracks))
    to_t = lambda a: torch.tensor( np.stack(a), dtype = torch.float32)
    return to_t(feats), to_t(bases), to_t(truths)