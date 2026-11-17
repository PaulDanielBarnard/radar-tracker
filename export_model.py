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


def kalmannet_payload(model) -> dict:
    """The learned-gain filter's weights: GRU cells, the gain head and the fixed motion matrix F."""
    from tracker.kalmannet import VMAX
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    cells = [{"w_ih": _round(sd[f"cells.{i}.weight_ih"]), "w_hh": _round(sd[f"cells.{i}.weight_hh"]),
              "b_ih": _round(sd[f"cells.{i}.bias_ih"]), "b_hh": _round(sd[f"cells.{i}.bias_hh"])}
             for i in range(model.config["layers"])]
    return {"config": dict(model.config), "vmax": VMAX, "F": _round(sd["F"]), "cells": cells,
            "head": {"w": _round(sd["head.weight"]), "b": _round(sd["head.bias"])}}


def _write_js(variables: dict, out_path) -> int:
    """Write `window.NAME = {...};` for every entry; returns the file size in bytes."""
    text = "".join(f"window.{name} = " + json.dumps(payload, separators=(",", ":")) + ";\n"
                   for name, payload in variables.items())
    Path(out_path).write_text(text, encoding="utf-8")
    size = Path(out_path).stat().st_size
    if size > MAX_MODEL_MB * 1024 * 1024:
        print(f"warning: {Path(out_path).name} is {size / 1e6:.1f} MB, above the {MAX_MODEL_MB} MB budget")
    return size


def write_model_js(model, out_path) -> int:
    """Write window.GRU_MODEL = {...}; returns the file size in bytes."""
    return _write_js({"GRU_MODEL": model_payload(model)}, out_path)


def write_extra_models_js(out_path, gru_imm=None, knet=None) -> int:
    """Write the optional extra models the Random lab can run: window.GRU_MODEL_IMM and window.KNET_MODEL."""
    variables = {}
    if gru_imm is not None:
        variables["GRU_MODEL_IMM"] = model_payload(gru_imm)
    if knet is not None:
        variables["KNET_MODEL"] = kalmannet_payload(knet)
    return _write_js(variables, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru_tracker", help="checkpoint name in outputs/")
    ap.add_argument("--imm-model", default="gru_imm", help="GRU trained with --imm (skipped if missing)")
    ap.add_argument("--kalmannet", default="kalmannet", help="learned-gain checkpoint (skipped if missing)")
    args = ap.parse_args()
    ckpt = checkpoint(args.model)
    if not ckpt.exists():
        raise SystemExit(f"No trained model at {ckpt}. Run `python train_neural.py` first.")
    WEB.mkdir(exist_ok=True)
    size = write_model_js(load_model(ckpt), WEB / "model.js")
    print(f"Wrote {WEB / 'model.js'} ({size / 1024:.0f} KB)")

    from tracker.kalmannet import load_kalmannet
    p_imm, p_knet = checkpoint(args.imm_model), checkpoint(args.kalmannet)
    gru_imm = load_model(p_imm) if p_imm.exists() else None
    knet = load_kalmannet(p_knet) if p_knet.exists() else None
    if gru_imm is not None or knet is not None:
        size = write_extra_models_js(WEB / "model_extra.js", gru_imm, knet)
        print(f"Wrote {WEB / 'model_extra.js'} ({size / 1024:.0f} KB): "
              + ", ".join(n for n, m in [("GRU fed the IMM", gru_imm), ("KalmanNet", knet)] if m is not None))


if __name__ == "__main__":
    main()
