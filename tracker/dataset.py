"""Convert sim Tracks to Tensors"""
import numpy as np
import torch

POS_SCALE = 50_000.0   # metres: puts positions roughly in [-1, 1]
VEL_SCALE = 300.0      # metres per scan: typical movement between scans
OUT_SCALE = 1000.0     # metres: the network's output unit (1.0 = 1 km correction)

def forward_fill(track):
    """Latest Radar Plot at every scan. Missed scans reuse the last real plot"""
    last = track.meas_xy[0].copy() #scan 0 is ALWAYS detected
    filled = np.zeros((len(track.detected), 2))

    for k in range(len(track.detected)):

        if track.detected[k]:
            last = track.meas_xy[k]

        filled[k] = last
    
    return filled

def track_to_arr(track):
    filled = forward_fill(track)
    step = np.zeros_like(filled) # How Far plot moved since last scan
    step[1:] = filled[1:] - filled[:-1]
    feat = np.concatenate([
        filled / POS_SCALE,  # 2 col : where the target looks to be
        step / VEL_SCALE,    # 2 col : apparent movement
        track.detected[:, None].astype(float),  # 1 col : did we really see it?
        ], axis = 1)    # shape (T, 5)
    return feat,filled,track.truth[:, :2]

def tracks_to_tensors(tracks):
    feats, bases, truths = zip(*(track_to_arr(t) for t in tracks))
    to_t = lambda a: torch.tensor( np.stack(a), dtype = torch.float32)
    return to_t(feats), to_t(bases), to_t(truths)