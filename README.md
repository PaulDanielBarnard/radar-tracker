# Radar Tracker

A radar sees an aircraft only as a scatter of noisy dots. This project tracks a
maneuvering target from those dots and asks one question: **when does a classic Kalman
filter beat a small learned neural network (a GRU), and when does it lose?**

Everything runs on a laptop with simulated data. No downloads, no hardware.

**Live demo:** https://pauldanielbarnard.github.io/radar-tracker/
(an animated radar screen where you can watch both trackers clean up the same noisy
plots, scenario by scenario)

## Contents

1. [How to read this README](#1-how-to-read-this-readme)
2. [Quick start](#2-quick-start)
3. [Concepts for beginners](#3-concepts-for-beginners)
4. [Project map](#4-project-map)
5. [The five scenarios](#5-the-five-scenarios)
6. [Status](#6-status)

---

## 1. How to read this README

- **New to tracking or machine learning?** Read section 3 top to bottom. Each idea builds
  on the one before it, and every concept is tied to the file that implements it.
- **Just want to run it?** Jump to section 2.
- **Want to find a file?** Section 4 lists every file with one line on what it does.

## 2. Quick start

You need Python 3.10 or newer. Commands below are for Windows PowerShell; on macOS or
Linux use `source .venv/bin/activate` instead of the activate line.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # numpy, matplotlib, pytest, torch, scipy
python -m pytest                       # all tests should pass
```

| Command | What it does | Takes |
|---|---|---|
| `python run_baseline.py` | Kalman filters (4 settings) vs raw radar on every scenario; saves plots to `outputs/` | seconds |
| `python train_neural.py --hybrid --randomize --hidden 128 --layers 2 --epochs 50` | trains the GRU and saves `outputs/gru_tracker.pt` | about 5 minutes on a laptop CPU |
| `python run_compare.py` | one table: raw radar vs Kalman vs GRU (needs the trained model) | a minute or two |
| `python run_eval.py` | the same comparison over 10 random seeds with confidence intervals and a filter-consistency check | roughly 15 minutes (`--seeds 2 --tracks 30` for a quick look) |
| `python export_demo.py` | bakes example tracks and results into `web/data.js` | a minute or two |
| `python -m http.server 8000 -d web` | serves the demo page at http://localhost:8000 | instant |

Notes:
- The trained model (`outputs/*.pt`) and everything else in `outputs/` is **not committed**
  (see `.gitignore`), so a fresh clone must run `train_neural.py` before `run_compare.py`,
  `run_eval.py` or `export_demo.py`.
- Run commands from the project root, the folder that contains `run_baseline.py`.
- `train_neural.py --help` lists every option (epochs, model size, learning rate, ...).
  Use `--name` to save several models side by side and `--model` on the other scripts
  to pick one.

## 3. Concepts for beginners

### 3.1 The problem: hidden truth, noisy measurements

An aircraft flies along some true path. You never see that path. You only get a **plot**
(one radar measurement) once per **scan**, and each plot is off by some random amount.
Sometimes the radar misses the aircraft entirely and gives no plot at all.

A **tracker** is an algorithm that turns the stream of noisy plots into the best possible
guess of where the aircraft really is. This is called **state estimation**.

```
truth (hidden):    ----o----o----o----o----o---->
measurements:      -----x---x------x------x---x     (noisy, sometimes missing)
tracker output:    ----o'---o'---o'---o'---o'--->   (should land close to the truth)
```

The simulator gives us the truth, so we can measure exactly how wrong each tracker is.

### 3.2 The radar model (`tracker/simulate.py`)

- The radar sits at the **origin** (0, 0). All units are SI: metres, seconds, radians.
- Each scan (one per second) it measures the target's **range** (distance) and **bearing**
  (angle). Both get Gaussian noise:
  - range noise: standard deviation `sigma_r` = 50 m
  - bearing noise: `sigma_theta` = 0.005 rad (about 0.29 degrees)
- A bearing error becomes a sideways position error that **grows with distance**:
  at 35 km, 0.005 rad is 35,000 x 0.005 = **175 m** sideways, but only 50 m along the
  line of sight. So the noise blob is long and thin, not round.
- With probability `1 - p_detect` the radar **misses** the target and returns nothing.
  The first scan of every track is always a detection, so trackers can start.
- Plots are converted to x, y: `x = r cos(theta)`, `y = r sin(theta)`.
- Each track is 120 scans long. The target starts 20-50 km away at 150-300 m/s.

### 3.3 The state

A tracker keeps a **state**: its current belief about the target.

```
state = [ x, y, vx, vy ]      position (m) and velocity (m/s) in two dimensions
```

The radar only measures position, yet the filter also estimates velocity. It infers speed
and direction from how the plots move over time.

### 3.4 The Kalman filter (`tracker/kalman.py`)

The Kalman filter is the classic answer. It repeats two steps every scan.

**1. Predict.** Assume the target keeps its velocity (a *constant-velocity model*) and
move the estimate forward one scan. Uncertainty grows, because the target might have
turned or sped up.

```
x_pred = F x              F moves position forward by velocity x dt
P_pred = F P F^T + Q      P is the filter's uncertainty (a covariance matrix)
                          Q is the extra uncertainty added by possible maneuvers
```

**2. Update.** A plot z arrives. Compare it with the prediction and blend the two:

```
y = z - H x_pred          innovation: how far the plot is from where we expected it
S = H P_pred H^T + R      how big an innovation we expect (R = radar noise)
K = P_pred H^T S^-1       the gain: how much to trust the plot vs the prediction
x = x_pred + K y          move the estimate toward the plot by a fraction K of y
P = (I - K H) P_pred      uncertainty shrinks, because we learned something
```

The gain **K** is the heart of it. If the prediction is very uncertain (large `P_pred`),
K is large and the filter trusts the plot. If the plot is very noisy (large `R`), K is
small and the filter trusts its prediction.

**A worked example** (one axis only, so the numbers stay small; this is the same maths
as `ConstantVelocityKF`, and the numbers below were checked by running it). Time step 1 s,
`q = 10`. The filter believes the target is at 1000 m moving at 200 m/s, with
uncertainty 100 m and 50 m/s. Radar noise is 50 m.

| Step | Result |
|---|---|
| Predict | position 1000 + 200 = **1200 m**, velocity 200 m/s. Position variance grows from 10,000 to **12,503** (a spread of about 112 m) |
| A plot arrives at 1260 m | innovation y = 1260 - 1200 = **60 m** |
| Innovation covariance | S = 12,503 + 50^2 = **15,003** |
| Gain | K = [**0.833**, **0.167**] (position, velocity) |
| Update | position 1200 + 0.833 x 60 = **1250 m**, velocity 200 + 0.167 x 60 = **210 m/s** |
| New uncertainty | position spread falls from 100 m to **45.6 m** |

So the filter moved 83% of the way toward the plot, nudged its speed estimate up, and
became more certain than either the prediction or the radar alone.

**Turning the radar noise into R.** Because noise is polar, the filter converts it to an
x, y covariance for each plot with the Jacobian `J` of the polar-to-Cartesian conversion:

```
R = J diag(sigma_r^2, sigma_theta^2) J^T       (_polar_to_cart_cov in kalman.py)
```

**Missed scans.** With no plot there is nothing to update with, so the filter just
predicts and its uncertainty `P` keeps growing until the next plot. This is called
*coasting*.

**Starting up.** The filter starts at the first plot with velocity 0 and a large velocity
uncertainty of 200 m/s, so the first few scans are poor. That is why the metrics ignore
the first 10 scans (the *burn-in*).

**The one knob, `q`.** `Q` is built from `q`, the *process noise intensity*: how much
unplanned acceleration the filter expects.

| | small `q` (0.1) | large `q` (100) |
|---|---|---|
| Believes | the target flies steadily | the target may maneuver at any time |
| Gain | small: leans on its smooth prediction | large: follows the plots closely |
| Straight flight | excellent: noise is averaged away | worse: it passes plot noise through |
| Turns | **lags far behind** | follows the turn |

No single `q` is best everywhere. That tradeoff is exactly the gap a learned tracker is
trying to close.

### 3.5 Measuring error (`tracker/metrics.py`)

- **Position error** at one scan is the straight-line distance between the estimate and
  the truth, in metres.
- **RMSE** (root mean square error) combines all scans of all tracks into one number: square
  every error, average, take the square root. Lower is better. It punishes large errors
  more than small ones.
- The first **10 scans are skipped** (burn-in), because every tracker is still settling.
- The **raw radar baseline** is the dumbest possible tracker: just reuse the latest plot
  (`forward_fill` in `tracker/simulate.py`). Everything should beat it.

### 3.6 The neural tracker (`tracker/neural.py`, `tracker/dataset.py`, `train_neural.py`)

**Why try it?** A Kalman filter has to be *told* how targets move and how noisy the radar
is. A neural network can *learn* those patterns from thousands of simulated tracks.

**A GRU reads the scans in order.** A GRU (Gated Recurrent Unit) is a recurrent network.
At each scan it reads that scan's numbers and updates a **memory vector** (64 or 128
numbers) that summarises everything seen so far: roughly how fast the target is moving,
whether it seems to be turning, how noisy the plots look. Built-in gates decide what to
keep and what to forget.

**What it reads.** Each scan becomes a row of numbers (`track_to_arr` in `tracker/dataset.py`),
scaled so they are near size 1, which networks learn from far better than raw metres:

| Columns | Meaning |
|---|---|
| 2 | latest plot position (the last real plot, reused when a scan is missed) |
| 2 | how far that plot moved since the last scan |
| 1 | was this scan a real detection? |
| 1 | scans since the last real plot |
| 1 | range from the radar |
| 2 | bearing as sin and cos (avoids the jump at +/- 180 degrees) |
| 2 | the radar's noise levels (a Kalman filter is told these too, so it is fair) |

That is **11 columns**. In *hybrid* mode (below) 8 more are added, for **19**.

**Residual learning: output a correction, not a position.** A network asked to output a
position directly would have to be accurate to a tiny fraction of tens of kilometres.
Instead it outputs a small **correction** that is added to a base position:

```
estimate = base position + correction        (correction = network output x 1000 m)
```

The output layer starts at **exactly zero**, so an untrained model returns the base
position unchanged (a test checks this). Training can then only improve on the base.

**Hybrid mode (`--hybrid`).** Normally the base is the latest raw plot. In hybrid mode the
base is a Kalman filter's position (q = 100), and the network also sees the estimates of
two Kalman filters (q = 0.1 and q = 100) as extra inputs. It starts at Kalman quality and
learns where to deviate. This is the best-performing configuration.

**How it learns** (`train_neural.py`):
- *Training data:* 2000 simulated tracks. Normally 400 per scenario; with `--randomize`
  every track gets its own random noise level and miss rate, so the model learns the
  general skill instead of five fixed cases.
- *Loss:* the average squared distance error, but each track's error is divided by that
  track's **expected raw radar error** (about 182 m for the normal radar, 546 m for the
  noisy one). Without this the noisy tracks, which have the largest errors, would drown out
  the quiet ones.
- *Each epoch:* the network sees all training tracks once, in small batches, and the
  AdamW optimiser nudges the weights to reduce the loss.
- *Validation:* after every epoch the model is scored on **separate tracks it never
  trains on** (different random seeds). The best epoch is saved. Scoring only on training
  data would reward memorising instead of learning.
- *Training aids:* the learning rate halves when validation stalls, training stops early
  after 8 epochs without improvement, and a small weight penalty discourages extreme weights.

**Fairness.** The GRU reads scans strictly left to right, so at each moment it knows only
the past, exactly like a Kalman filter. Letting it peek at later scans would be cheating.

**Seeds keep the tests honest.** Training uses seeds 1000 and up, validation 2000 and up,
and the test tracks use seed 42 (`run_baseline.py`, `run_compare.py`, `export_demo.py`)
or seeds 10,000 and up (`run_eval.py`). None overlap, so the models are always judged on
tracks they have never seen.

## 4. Project map

| Path | Purpose |
|---|---|
| `tracker/simulate.py` | the radar and target simulator; `Track` data; `simulate_batch`; `forward_fill` |
| `tracker/kalman.py` | the constant-velocity Kalman filter, with optional covariance and NIS output |
| `tracker/metrics.py` | position error, RMSE (with burn-in), and the shared evaluation helpers |
| `tracker/scenarios.py` | the five test scenarios, the Kalman `q` values, and `random_radar` for training |
| `tracker/dataset.py` | turns tracks into tensors for the network; the input features; hybrid features |
| `tracker/neural.py` | the GRU tracker, plus save and load of models together with their settings |
| `tracker/stats.py` | confidence intervals, paired differences, NEES, chi-squared bounds |
| `tracker/paths.py` | where `outputs/` and the model files live |
| `run_baseline.py` | table of Kalman settings vs raw radar, plus two plots in `outputs/` |
| `train_neural.py` | trains the GRU and prints a per-scenario report |
| `run_compare.py` | one table: raw radar, Kalman, GRU on seed-42 test tracks |
| `run_eval.py` | multi-seed evaluation: confidence intervals, paired verdicts, filter consistency |
| `export_demo.py` | writes `web/data.js` (example tracks and results) for the demo page |
| `web/index.html` | the demo page: radar (PPI) view, close-up view, results bars |
| `web/data.js` | generated by `export_demo.py`; committed so the site works without training |
| `tests/test_tracker.py` | simulator and Kalman filter tests |
| `tests/test_neural.py` | dataset, model save/load, zero-start and hybrid tests |
| `tests/test_stats.py` | tests for the statistics helpers and the Kalman diagnostics |
| `.github/workflows/pages.yml` | publishes `web/` to GitHub Pages when it changes on `main` |
| `requirements.txt` | Python dependencies (numpy, matplotlib, pytest, torch, scipy) |
| `.gitignore` | keeps the virtual environment, caches and `outputs/` out of git |

## 5. The five scenarios

Every script tests on the same five scenarios (`tracker/scenarios.py`):

| name | target motion | radar |
|---|---|---|
| `straight` | constant velocity | normal noise |
| `turning` | one constant turn, 50-100% of the maximum rate (0.05 rad/s, about 2.9 degrees per second), random direction | normal noise |
| `maneuvers` | random segments of 8-25 scans: 30% straight, otherwise a random turn and a random acceleration up to 5 m/s^2 | normal noise |
| `noisy` | random maneuvers | 3x range and bearing noise (150 m, 0.015 rad) |
| `missed_30` | random maneuvers | 30% of scans have no plot |

Speed is kept between 80 and 400 m/s.

## 6. Status

- [x] Step 1: simulator, Kalman baseline, tests
- [x] Step 2: neural tracker (GRU in PyTorch) trained on simulated tracks
- [x] Step 3: comparison by scenario (`run_compare.py`, `run_eval.py`)
- [x] Step 4: browser demo with a traditional radar (PPI) view (`web/`)
