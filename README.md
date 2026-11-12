# Radar Tracker

Track a maneuvering air target from noisy radar-style measurements, and find out
when a classic Kalman filter beats a learned tracker (and when it doesn't).
Runs entirely on a laptop with simulated data. No downloads, no hardware.

## Status

- [x] Step 1: simulator + Kalman baseline + tests
- [x] Step 2: neural tracker (GRU/LSTM in PyTorch) trained on simulated tracks
- [ ] Step 3: comparison by scenario + failure analysis
- [ ] Step 4: live browser demo with sliders (like the trajectory project)

## Setup (VS Code)

1. Open this folder in VS Code (File > Open Folder).
2. Open a terminal (Ctrl+`) and create an environment:

       python -m venv .venv
       .venv\Scripts\activate          # Windows
       # source .venv/bin/activate     # macOS / Linux
       pip install -r requirements.txt

3. Select the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > `.venv`.
4. Run the baseline: `python run_baseline.py` (or press F5, "Run Kalman baseline").
5. Run the tests: `python -m pytest`
6. Train the neural tracker: `python train_neural.py` (saves `outputs/gru_tracker.pt`,
   which is not committed, so every fresh clone must train first).
7. Compare everything: `python run_compare.py` (raw radar vs Kalman filters vs GRU).

## Layout

    tracker/simulate.py   target motion + noisy radar model (range, bearing, missed scans)
    tracker/kalman.py     constant-velocity Kalman filter
    tracker/metrics.py    position error, RMSE and shared evaluation helpers
    tracker/scenarios.py  the five test scenarios and Kalman q values
    tracker/dataset.py    turns tracks into tensors for the network
    tracker/neural.py     GRU tracker that learns a correction to the latest plot
    tracker/paths.py      where outputs and the model checkpoint live
    run_baseline.py       scenarios x filter settings table + plots in outputs/
    train_neural.py       trains the GRU, reports RMSE per scenario
    run_compare.py        raw radar vs Kalman vs GRU table, best per scenario
    tests/                sanity tests

## What the baseline already shows

A Kalman filter has one tuning knob, the process noise `q`. Small `q` is smooth on
straight flight but lags badly in turns; large `q` follows turns but is noisier.
No single `q` wins everywhere, which is the gap a learned tracker is trying to close.

## Scenarios

| name       | target motion        | radar                         |
|------------|----------------------|-------------------------------|
| straight   | constant velocity    | normal noise                  |
| turning    | constant turn rate   | normal noise                  |
| maneuvers  | random turns/accel   | normal noise                  |
| noisy      | random maneuvers     | 3x range and bearing noise    |
| missed 30% | random maneuvers     | 30% of scans have no plot     |

All units are SI (metres, seconds, radians). The radar sits at the origin.
