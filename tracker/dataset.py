"""Convert sim Tracks to Tensors"""
import numpy as np
import torch

from .simulate import forward_fill

POS_SCALE = 50_000.0   # metres: puts positions roughly in [-1, 1]
VEL_SCALE = 300.0      # metres per scan: typical movement between scans
N_FEATURES = 11         # columns per scan fed to the network
OUT_SCALE = 1000.0     # metres: the network's output unit (1.0 = 1 km correction)

def track_to_arr(track, hybrid=False):
    filled = forward_fill(track)
    T = len(filled)
    step = np.zeros_like(filled)             # how far the plot moved since the last scan
    step[1:] = filled[1:] - filled[:-1]

    since = np.zeros(T)                      # scans since the last real plot (0 = fresh)
    for k in range(1, T):
        since[k] = 0.0 if track.detected[k] else since[k - 1] + 1.0

    rng = np.hypot(filled[:, 0], filled[:, 1])
    bearing = np.arctan2(filled[:, 1], filled[:, 0])
    radar = track.radar
    noise = np.tile([radar.sigma_r / 50.0, radar.sigma_theta / 0.005], (T, 1))  # 1.0 = normal radar

    feat = np.concatenate([
        filled / POS_SCALE,                      # 2: where the target looks to be
        step / VEL_SCALE,                        # 2: apparent movement
        track.detected[:, None].astype(float),   # 1: did we really see it?
        since[:, None] / 10.0,                   # 1: how stale is the latest plot
        (rng / POS_SCALE)[:, None],              # 1: range from the radar
        np.sin(bearing)[:, None],                # 1: bearing as sin/cos (no wrap-around jump)
        np.cos(bearing)[:, None],                # 1
        noise,                                   # 2: radar noise levels (a Kalman filter is told these too)
    ], axis=1)                                   # shape (T, 11)
    return feat, filled, track.truth[:, :2]


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