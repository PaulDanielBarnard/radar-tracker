"""GRU Tracker that learns a correction to the latest Radar plot"""
from torch import nn
from .dataset import OUT_SCALE

class GRUTracker(nn.Module):

    def __init__(self, n_feat: int = 5, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first = True)   #Memory
        self.head = nn.Linear(hidden, 2)    # Memory -> (dx, dy)
        nn.init.zeros_(self.head.weight)    # start with zero correction,
        nn.init.zeros_(self.head.bias)      # i.e. exactly the raw radar plot

    def forward(self, feats, base):
        """feats: (batch, T, 5), base: (batch, T, 2) in m -> (batch, T, 2) m"""
        h, _ = self.gru(feats)  # h : (batch, T, hidden), 1 memory per scan
        return base + self.head(h) * OUT_SCALE  #Latest Plot + Learned Correction
    