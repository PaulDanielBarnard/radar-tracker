"""A learned-gain Kalman filter (KalmanNet style).

The hybrid GRU in `neural.py` learns a free-form correction. This tracker keeps the Kalman
filter's structure fixed and learns only ONE thing: the gain.

    predict   x_pred = F x                          (fixed physics: constant velocity)
    innovate  y      = plot - position of x_pred    (how surprised are we?)
    correct   x      = x_pred + K y                 (K is the learned part)

A small GRU reads what a Kalman filter would know (the innovation, its previous correction,
whether the plot arrived, how stale it is, the range and the radar noise levels) and outputs
the 4x2 gain K for this scan. K is bounded so that the error dynamics cannot blow up: for each
axis it has a position gain alpha in (0, 1) and a velocity gain beta in (0, VMAX) (both from a
sigmoid), K = [[alpha_x, 0], [0, alpha_y], [beta_x, 0], [0, beta_y]]. This is the classic
alpha-beta filter, which is stable for every alpha in (0, 1) and beta in (0, VMAX) (a test checks
the whole range); a free 4x2 gain with coupling between axes diverged in early experiments.
On a missed scan there is no innovation, so the filter just coasts on its prediction.

The output layer starts at zero with its bias set to the steady-state gain of an ordinary
q = 100 Kalman filter, so an untrained network IS a sensible fixed-gain filter and training
can only improve on it.
"""
import numpy as np
import torch
from torch import nn

from .dataset import OUT_SCALE, POS_SCALE, VEL_SCALE, track_to_arr
from .kalman import H_POS, cv_matrices, predict, update

VMAX = 0.5           # largest velocity gain beta (m/s per metre of innovation)
STATIC_COLS = [4, 5, 6, 9, 10]   # from track_to_arr: detected, scans since plot, range, noise x2
N_STATIC = len(STATIC_COLS)
N_IN = 2 + 4 + N_STATIC          # innovation, previous correction, the static columns
TYPICAL_RANGE = 35_000.0


def steady_state_gain(q: float = 100.0, sigma_r: float = 50.0, sigma_theta: float = 0.005,
                      dt: float = 1.0, rng: float = TYPICAL_RANGE) -> np.ndarray:
    """The (4, 2) gain an ordinary constant-velocity Kalman filter settles to (isotropic R)."""
    F, Q = cv_matrices(q, dt)
    R = np.eye(2) * np.hypot(sigma_r, rng * sigma_theta) ** 2
    x, P = np.zeros(4), np.eye(4) * 1e4
    K = None
    for _ in range(500):
        x, P = predict(x, P, F, Q)
        _, P, _, _, K = update(x, P, np.zeros(2), R)
    return K


def logit(p):
    return np.log(p / (1 - p))


class KalmanNet(nn.Module):

    def __init__(self, hidden: int = 64, layers: int = 1, dt: float = 1.0, q_init: float = 100.0):
        super().__init__()
        self.config = {"hidden": hidden, "layers": layers, "dt": dt, "q_init": q_init}
        self.cells = nn.ModuleList([nn.GRUCell(N_IN if i == 0 else hidden, hidden) for i in range(layers)])
        self.head = nn.Linear(hidden, 4)            # -> alpha_x, alpha_y, beta_x, beta_y
        F, _ = cv_matrices(q_init, dt)
        self.register_buffer("F", torch.tensor(F, dtype=torch.float32))
        self.register_buffer("H", torch.tensor(H_POS, dtype=torch.float32))
        K0 = steady_state_gain(q_init, dt=dt)
        a = np.clip([K0[0, 0], K0[1, 1]], 1e-3, 1 - 1e-3)
        b = np.clip(np.array([K0[2, 0], K0[3, 1]]) / VMAX, 1e-3, 1 - 1e-3)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor(np.concatenate([logit(a), logit(b)]), dtype=torch.float32))

    def gain(self, h):
        """Bounded (batch, 4, 2) gain from the network's top-layer memory."""
        a, b = torch.sigmoid(self.head(h)).split(2, dim=1)
        b = b * VMAX
        zero = torch.zeros_like(a[:, 0])
        return torch.stack([torch.stack([a[:, 0], zero], 1), torch.stack([zero, a[:, 1]], 1),
                            torch.stack([b[:, 0], zero], 1), torch.stack([zero, b[:, 1]], 1)], dim=1)

    def forward(self, z, det, static, return_gains: bool = False):
        """Filter a batch of tracks.

        z (B, T, 2) plots in metres (forward-filled where missed), det (B, T) 1 where a plot
        arrived, static (B, T, N_STATIC) the radar-side features. Returns (B, T, 4) estimates
        [x, y, vx, vy] in metres and m/s (and the (B, T, 4, 2) gains if asked).
        """
        B, T, _ = z.shape
        x = torch.cat([z[:, 0], torch.zeros(B, 2)], dim=1)
        hs = [torch.zeros(B, c.hidden_size) for c in self.cells]
        prev_corr = torch.zeros(B, 4)
        est, gains = [x], [self.gain(hs[-1])]
        for k in range(1, T):
            x_pred = x @ self.F.T
            y = (z[:, k] - x_pred @ self.H.T) * det[:, k:k + 1]          # no plot -> no innovation
            inp = torch.cat([y / OUT_SCALE,
                             torch.cat([prev_corr[:, :2] / OUT_SCALE, prev_corr[:, 2:] / VEL_SCALE], 1),
                             static[:, k]], dim=1)
            for i, cell in enumerate(self.cells):
                hs[i] = cell(inp, hs[i])
                inp = hs[i]
            K = self.gain(hs[-1])
            corr = torch.einsum("bij,bj->bi", K, y)                      # zero when y is zero
            x = x_pred + corr
            prev_corr = corr
            est.append(x)
            gains.append(K)
        est = torch.stack(est, dim=1)
        return (est, torch.stack(gains, dim=1)) if return_gains else est


def tracks_to_inputs(tracks):
    """Tensors for `KalmanNet.forward` (and the truth) from simulated tracks."""
    zs, dets, statics, truths = [], [], [], []
    for t in tracks:
        feat, base, truth = track_to_arr(t)
        zs.append(base)                                   # latest plot, forward-filled
        dets.append(t.detected.astype(float))
        statics.append(feat[:, STATIC_COLS])
        truths.append(truth)
    f = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    return f(zs), f(dets), f(statics), f(truths)


def save_kalmannet(model, path):
    torch.save({"config": model.config, "state": model.state_dict()}, path)


def load_kalmannet(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = KalmanNet(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def estimate(model, tracks) -> np.ndarray:
    """Run a trained model over tracks; returns (n_tracks, T, 4) [x, y, vx, vy]."""
    z, det, static, _ = tracks_to_inputs(tracks)
    model.eval()
    with torch.no_grad():
        return model(z, det, static).numpy()
