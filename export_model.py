"""Export the trained GRU tracker for the web demo: writes web/model.js.

    python export_model.py                      # uses outputs/gru_tracker.pt
    python export_model.py --model long         # another checkpoint in outputs/

The browser cannot load a PyTorch file, so the weights go into a JavaScript file as plain numbers
(rounded to 5 significant digits, plenty for a network whose output is in metres) together with
the constants the input features need. web/lab.js runs the network from this file.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tracker import dataset
from tracker.gru_numpy import weights_from_torch
from tracker.neural import load_model
from tracker.paths import checkpoint

WEB = Path(__file__).parent / "web"
MAX_MODEL_MB = 4          # the page loads this file, keep it small


def _round(a) -> list:
    """Nested lists of floats rounded to 5 significant digits."""
    return np.vectorize(lambda v: float(f"{v:.5g}"), otypes=[float])(a).tolist()


def model_payload(model) -> dict:
    w = weights_from_torch(model)
    return {
        "config": w["config"],
        "scales": {"POS_SCALE": dataset.POS_SCALE, "VEL_SCALE": dataset.VEL_SCALE, "OUT_SCALE": dataset.OUT_SCALE,
                   "HYBRID_QS": list(dataset.HYBRID_QS), "HYBRID_BASE_Q": dataset.HYBRID_BASE_Q},
        "layers": [{k: _round(v) for k, v in layer.items()} for layer in w["layers"]],
        "head": {k: _round(v) for k, v in w["head"].items()},
    }


def write_model_js(model, out_path) -> int:
    """Write window.GRU_MODEL = {...}; returns the file size in bytes."""
    text = "window.GRU_MODEL = " + json.dumps(model_payload(model), separators=(",", ":")) + ";\n"
    Path(out_path).write_text(text, encoding="utf-8")
    size = Path(out_path).stat().st_size
    if size > MAX_MODEL_MB * 1024 * 1024:
        print(f"warning: {Path(out_path).name} is {size / 1e6:.1f} MB, above the {MAX_MODEL_MB} MB budget")
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru_tracker", help="checkpoint name in outputs/")
    args = ap.parse_args()
    ckpt = checkpoint(args.model)
    if not ckpt.exists():
        raise SystemExit(f"No trained model at {ckpt}. Run `python train_neural.py` first.")
    WEB.mkdir(exist_ok=True)
    size = write_model_js(load_model(ckpt), WEB / "model.js")
    print(f"Wrote {WEB / 'model.js'} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
