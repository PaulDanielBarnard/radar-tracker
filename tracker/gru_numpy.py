"""The trained GRU tracker, run with plain NumPy (no PyTorch needed).

The web demo has to run the network in the browser, so we first check the maths here in Python:
this file re-implements the forward pass of `tracker.neural.GRUTracker` using only NumPy and
the weights, and the tests compare it with PyTorch.

PyTorch's GRU, for one scan with input x and previous memory h (weight rows are stacked [r, z, n]):

    r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)          reset gate
    z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)          update gate
    n  = tanh   (W_in x + b_in + r * (W_hn h + b_hn))    candidate memory
    h' = (1 - z) * n + z * h                             new memory

Layers are stacked (the memory sequence of layer k is the input of layer k+1), then a linear
head turns the last memory into a correction in kilometres, and the answer is
`base + head * OUT_SCALE`.
"""
import numpy as np

from .dataset import OUT_SCALE


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weights_from_torch(model) -> dict:
    """Copy a GRUTracker's weights into plain float64 NumPy arrays (the same layout the web export uses)."""
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    layers = [{"w_ih": sd[f"gru.weight_ih_l{k}"], "w_hh": sd[f"gru.weight_hh_l{k}"],
               "b_ih": sd[f"gru.bias_ih_l{k}"], "b_hh": sd[f"gru.bias_hh_l{k}"]}
              for k in range(model.config["layers"])]
    return {"config": dict(model.config), "layers": layers,
            "head": {"w": sd["head.weight"], "b": sd["head.bias"]}}


def gru_layer(x, w_ih, w_hh, b_ih, b_hh):
    """One GRU layer over a sequence. x: (T, n_in) -> (T, hidden)."""
    hidden = w_hh.shape[1]
    h = np.zeros(hidden)
    out = np.empty((x.shape[0], hidden))
    for t in range(x.shape[0]):
        gi = w_ih @ x[t] + b_ih
        gh = w_hh @ h + b_hh
        r = _sigmoid(gi[:hidden] + gh[:hidden])
        z = _sigmoid(gi[hidden:2 * hidden] + gh[hidden:2 * hidden])
        n = np.tanh(gi[2 * hidden:] + r * gh[2 * hidden:])
        h = (1.0 - z) * n + z * h
        out[t] = h
    return out


def forward(weights: dict, feats, base):
    """feats: (T, n_feat) or (B, T, n_feat), base: (T, 2) or (B, T, 2) in metres -> same shape as base, metres."""
    feats, base = np.asarray(feats, dtype=np.float64), np.asarray(base, dtype=np.float64)
    if feats.ndim == 3:
        return np.stack([forward(weights, f, b) for f, b in zip(feats, base)])
    h = feats
    for layer in weights["layers"]:
        h = gru_layer(h, layer["w_ih"], layer["w_hh"], layer["b_ih"], layer["b_hh"])
    head = weights["head"]
    return base + (h @ head["w"].T + head["b"]) * OUT_SCALE
