"""Single source of truth for where outputs live, independent of the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
CHECKPOINT = OUT_DIR / "gru_tracker.pt"
